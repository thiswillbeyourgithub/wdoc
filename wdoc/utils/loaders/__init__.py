"""
Called by batch_file_loader.py's threads. Contains many cached function to
load each document.
"""

import copy
import inspect
import json
import os
import re
import sys
import time
import traceback
from functools import wraps
from pathlib import Path

from beartype.typing import Callable, List, Optional, Union
from langchain.docstore.document import Document
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.errors import MissingDocdictArguments, TimeoutPdfLoaderError
from wdoc.utils.loaders.shared import get_url_title
from wdoc.utils.misc import (
    ModelName,
    average_word_length,
    check_docs_tkn_length,
    get_splitter,
    hasher,
    max_token,
    min_lang_prob,
    min_token,
    wpm,
)

# needed in case of buggy unstructured install
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Name of all the filetype than can be loaded. They each correspond to a function
# named `load_{filetype}` (e.g. `load_logseq_markdown`) inside the
# wdoc/utils/loaders directory
LOADABLE_FILETYPE = [
    "url",
    "youtube",
    "pdf",
    "online_pdf",
    "anki",
    "string",
    "txt",
    "text",
    "local_html",
    "logseq_markdown",
    "local_audio",
    "local_video",
    "online_media",
    "epub",
    "powerpoint",
    "word",
    "json_dict",
]

markdownlink_regex = re.compile(r"\[.*?\]\((.*?)\)")  # to find markdown links
# to replace markdown links by their text
# to remove image from jina reader that take a lot of tokens but are not yet used


def wrapper_load_one_doc(func: Callable) -> Callable:
    """Decorator to wrap doc_loader to catch errors cleanly"""

    # # load_one_doc wrapped can also return a str, the error message,
    # # wraps(func) removes it so we readd it:
    newfunc = copy.copy(func)
    newfunc.__annotations__["return"] = Union[List[Document], str]

    @wraps(newfunc)
    def wrapper(*args, **kwargs) -> Union[List[Document], str]:
        # Extract loading_failure from kwargs, default to "warn"
        loading_failure = kwargs.pop("loading_failure", "warn")

        try:
            return func(*args, **kwargs)
        except Exception as err:

            # those crashes can rise right away without more details
            if loading_failure == "crash":
                if isinstance(err, (MissingDocdictArguments, TimeoutPdfLoaderError)):
                    raise

            filetype = kwargs.get("filetype", "unknown")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            formatted_tb = "\n".join(traceback.format_tb(exc_tb))
            if "pdf parser" in str(err).lower() and "to parse" in str(err).lower():
                mess = (
                    f"Error when loading doc with filetype {filetype}: '{err}'. "
                    f"Arguments: {kwargs}"
                )
            else:
                mess = (
                    f"Error when loading doc with filetype {filetype}: '{err}'. "
                    f"Arguments: {kwargs}"
                    f"\nLine number: {exc_tb.tb_lineno}"
                    f"\nFull traceback:\n{formatted_tb}"
                )
            if loading_failure == "crash":
                logger.exception(mess)
                raise Exception(mess) from err
            elif loading_failure == "warn" or env.WDOC_DEBUG:
                logger.warning(mess)
                return str(err)
            else:
                logger.exception(mess)
                raise ValueError(loading_failure) from err

    return wrapper


@wrapper_load_one_doc
def load_one_doc(
    task: str,
    llm_name: ModelName,
    temp_dir: Path,
    filetype: str,
    file_hash: str,
    source_tag: Optional[str] = None,
    doccheck_min_lang_prob: float = min_lang_prob,
    doccheck_min_token: int = min_token,
    doccheck_max_token: int = max_token,
    recur_parent_id: str = None,  # just used to keep track of which document comes from which recursive filetype
    subitem_link: str = None,
    **kwargs,
) -> List[Document]:
    """choose the appropriate loader for a file, then load it,
    split into documents, add some metadata then return.
    The loader is cached"""
    import bs4
    import ftfy

    text_splitter = get_splitter(task, modelname=llm_name)
    assert kwargs, "Received an empty dict of arguments to load. Maybe --path is empty?"
    assert temp_dir.exists(), temp_dir

    # Check if filetype is supported
    if filetype not in LOADABLE_FILETYPE:
        logger.warning(f"Unsupported filetype: '{filetype}'")
        raise Exception(f"Unsupported filetype: '{filetype}'")
    loader_func_name = f"load_{filetype}"

    # Lazy loading the document loader function
    exec(f"from wdoc.utils.loaders.{filetype} import {loader_func_name}")
    loader_func = locals()[loader_func_name] or globals()[loader_func_name] or None

    if loader_func is None:
        raise Exception(
            f"Loader function 'load_{filetype}' not found for filetype '{filetype}'"
        )

    # Get function signature to determine what arguments to pass
    sig = inspect.signature(loader_func)

    # wdoc_global arguments (created by wdoc internally) - these are parameters of load_one_doc
    wdoc_global_args = {
        "task": task,
        "llm_name": llm_name,
        "temp_dir": temp_dir,
        "filetype": filetype,
        "file_hash": file_hash,
        "source_tag": source_tag,
        "doccheck_min_lang_prob": doccheck_min_lang_prob,
        "doccheck_min_token": doccheck_min_token,
        "doccheck_max_token": doccheck_max_token,
        "recur_parent_id": recur_parent_id,
        "text_splitter": text_splitter,
        "loaders_temp_dir": temp_dir,
        "verbose": env.WDOC_VERBOSE,
    }

    # User-provided arguments (from kwargs) - these come from user input
    user_args = kwargs

    # All available arguments
    available_args = {**wdoc_global_args, **user_args}

    # Get the parameter names of load_one_doc to distinguish wdoc_global vs user args
    load_one_doc_sig = inspect.signature(load_one_doc)
    wdoc_global_param_names = set(wdoc_global_args.keys())

    # Build arguments to pass to the loader function
    args_to_pass = {}
    missing_user_args = []
    missing_wdoc_global_args = []

    for param_name, param in sig.parameters.items():
        if param_name in available_args:
            args_to_pass[param_name] = available_args[param_name]
        elif param.default is param.empty:
            # Required parameter that we don't have - determine if it's wdoc_global or user arg
            if param_name in wdoc_global_param_names:
                # This should be provided by wdoc wdoc_global - indicates a bug
                missing_wdoc_global_args.append(param_name)
            else:
                # This should be provided by the user
                missing_user_args.append(param_name)

    # Check for unexpected user arguments that don't match function parameters
    unexpected_user_args = []
    for user_arg in user_args.keys():
        if user_arg not in sig.parameters:
            unexpected_user_args.append(user_arg)

    # Helper function to format arguments with their type hints and default values
    def format_args_with_types(arg_names: List[str]) -> str:
        formatted_lines = []
        for arg_name in arg_names:
            param = sig.parameters.get(arg_name)
            if param:
                # Build the argument description
                parts = [f"- {arg_name}"]

                # Add type hint if available
                if param.annotation != param.empty:
                    type_hint = param.annotation
                    # Always use the full string representation to show complete type hints
                    # like Literal["whisper", "deepgram"] instead of just "Literal"
                    type_str = str(type_hint)
                    parts.append(f": {type_str}")

                # Add default value if not required
                if param.default != param.empty:
                    parts.append(f" (default: {param.default})")

                formatted_lines.append("".join(parts))
            else:
                formatted_lines.append(f"- {arg_name}")

        return "\n".join(formatted_lines) if formatted_lines else ""

    if unexpected_user_args:
        valid_params = [
            param_name
            for param_name in sig.parameters.keys()
            if param_name not in wdoc_global_param_names
        ]
        formatted_valid_params = format_args_with_types(valid_params)
        raise MissingDocdictArguments(
            f"\n\nLoader function 'l{loader_func_name}' for filetype '{filetype}' "
            f"received unexpected arguments: {unexpected_user_args}\n"
            f"Valid user arguments for this loader are: {formatted_valid_params}\n"
            f"Please check the documentation for the correct arguments for this filetype."
        )

    # Get optional arguments with their types for better error messages
    optional_args = []
    for param_name, param in sig.parameters.items():
        if param.default is not param.empty and param_name not in available_args:
            optional_args.append(param_name)
    formatted_optional_args = format_args_with_types(optional_args)

    # Check for missing arguments
    if missing_wdoc_global_args and missing_user_args:
        # Both wdoc_global and user args are missing
        user_arg_names = list(user_args.keys()) if user_args else []
        formatted_wdoc_global_args = format_args_with_types(missing_wdoc_global_args)
        formatted_user_args = format_args_with_types(missing_user_args)
        raise MissingDocdictArguments(
            f"\n\nLoader function '{loader_func_name}' for filetype '{filetype}' "
            f"is missing required arguments from both wdoc wdoc_global and user input:\n"
            f"- Missing wdoc_global arguments (wdoc bug): {formatted_wdoc_global_args}\n"
            f"- Missing user arguments: {formatted_user_args}\n"
            f"You provided these arguments: {user_arg_names}.\n"
            f"Please check the documentation for the required arguments for this filetype and "
            f"create a GitHub issue at https://github.com/wdoc-ai/wdoc/issues with this error message."
        )
    elif missing_wdoc_global_args:
        # Only wdoc_global args are missing (wdoc bug)
        formatted_wdoc_global_args = format_args_with_types(missing_wdoc_global_args)
        optional_msg = (
            f"\n- Optional arguments available: {formatted_optional_args}"
            if formatted_optional_args
            else ""
        )
        raise MissingDocdictArguments(
            f"\n\nnInternal error: Loader function '{loader_func_name}' for filetype '{filetype}' "
            f"is missing required wdoc_global arguments: {formatted_wdoc_global_args}.{optional_msg}\n"
            f"This appears to be a wdoc bug - please create a GitHub issue at "
            f"https://github.com/wdoc-ai/wdoc/issues with this error message and your command."
        )
    elif missing_user_args:
        # Only user args are missing (user error)
        user_arg_names = list(user_args.keys()) if user_args else []
        formatted_user_args = format_args_with_types(missing_user_args)
        optional_msg = (
            f"\n- Optional arguments available: {formatted_optional_args}"
            if formatted_optional_args
            else ""
        )
        raise MissingDocdictArguments(
            f"\n\nLoader function '{loader_func_name}' for filetype '{filetype}' "
            f"is still missing required user arguments: {formatted_user_args}.{optional_msg}"
            f"\nYou provided these arguments: {user_arg_names}.\n"
            f"Please add the missing aguments or check the documentation for the required arguments for this filetype."
        )

    # Call the loader function with the appropriate arguments
    docs = loader_func(**args_to_pass)

    assert (
        docs
    ), f"The loader function returned no documents, something went wrong.\nLoader function: '{loader_func}'\nArguments: '{args_to_pass}'"

    tdocs = text_splitter.transform_documents(docs)

    if not tdocs:
        logger.warning(
            f"text_splitter.transform_documents apparently erased the docs, something went wrong so using original docs.\nLoader function: '{loader_func}'\nArguments: '{args_to_pass}'\nText_splitter: '{text_splitter}'"
        )
    else:
        logger.debug(
            f"Successfuly used text_splitter.transform_documents on {len(docs)} docs"
        )
        docs = tdocs

    if filetype not in ["anki", "pdf"]:
        check_docs_tkn_length(
            docs=docs,
            identifier=filetype,
            min_lang_prob=doccheck_min_lang_prob,
            min_token=doccheck_min_token,
            max_token=doccheck_max_token,
        )

    # add and format metadata
    for i in range(len(docs)):
        # if html, parse it
        soup = bs4.BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = soup.get_text()

        # fix text just in case
        docs[i].page_content = ftfy.fix_text(docs[i].page_content)

        if source_tag:
            if "source_tag" not in docs[i].metadata:
                docs[i].metadata["source_tag"] = source_tag
            else:
                if not isinstance(docs[i].metadata["source_tag"], str):
                    docs[i].metadata["source_tag"] = str(docs[i].metadata["source_tag"])
                docs[i].metadata["source_tag"] = (
                    docs[i].metadata["source_tag"].replace("unset", "").strip()
                )
                docs[i].metadata["source_tag"] += f" {source_tag}"
        else:
            docs[i].metadata["source_tag"] = "unset"
        if "Author" in docs[i].metadata:
            docs[i].metadata["author"] = docs[i].metadata["Author"]
            del docs[i].metadata["Author"]
        if "authors" in docs[i].metadata:
            docs[i].metadata["author"] = docs[i].metadata["authors"]
            del docs[i].metadata["authors"]
        if "Authors" in docs[i].metadata:
            docs[i].metadata["author"] = docs[i].metadata["Authors"]
            del docs[i].metadata["Authors"]
        if "filetype" not in docs[i].metadata:
            docs[i].metadata["filetype"] = filetype
        if "path" not in docs[i].metadata and "path" in kwargs:
            docs[i].metadata["path"] = kwargs["path"]
        if subitem_link and "subitem_link" not in docs[i].metadata:
            docs[i].metadata["subitem_link"] = subitem_link
        if "title" not in docs[i].metadata or docs[i].metadata["title"] == "Untitled":
            if "title" in kwargs and kwargs["title"] and kwargs["title"] != "Untitled":
                docs[i].metadata["title"] = kwargs["title"]
            elif (
                "path" in docs[i].metadata
                and isinstance(docs[i].metadata["path"], str)
                and "http" in docs[i].metadata["path"].lower()
            ):
                docs[i].metadata["title"] = get_url_title(docs[i].metadata["path"])
                if not docs[i].metadata["title"]:
                    docs[i].metadata["title"] = "Untitled"
                    logger.debug(f"Could not get title from url of doc '{kwargs}'")
        if (
            "title" in kwargs
            and kwargs["title"] != docs[i].metadata["title"]
            and kwargs["title"] not in docs[i].metadata["title"]
        ):
            docs[i].metadata["title"] += " - " + kwargs["title"]
        if "playlist_title" in kwargs:
            docs[i].metadata["title"] = (
                kwargs["playlist_title"] + " - " + docs[i].metadata["title"]
            )

        if "doc_reading_time" not in docs[i].metadata:
            reading_length = len(docs[i].page_content) / average_word_length / wpm
            docs[i].metadata["doc_reading_time"] = round(reading_length, 3)
        if "source" not in docs[i].metadata:
            if "path" in docs[i].metadata:
                docs[i].metadata["source"] = docs[i].metadata["path"]
            elif "path" in docs[i].metadata:  # was probably not a path
                docs[i].metadata["source"] = docs[i].metadata["title"]
            else:
                docs[i].metadata["source"] = "undocumented"

        # make sure the filepath are absolute
        try:
            if "path" in docs[i].metadata and Path(docs[i].metadata["path"]).exists():
                docs[i].metadata["path"] = str(
                    Path(docs[i].metadata["path"]).resolve().absolute()
                )
        except Exception:
            pass  # was probably not a path

        docs[i].metadata["indexing_timestamp"] = int(time.time())

        # replace any path to just the filename, to avoid sending privacy
        # revealing information to LLMs
        for k, v in docs[i].metadata.items():
            if isinstance(v, Path):
                docs[i].metadata[k] = v.name

        # set hash
        docs[i].metadata["content_hash"] = hasher(docs[i].page_content)
        docs[i].metadata["file_hash"] = file_hash
        assert docs[i].metadata[
            "content_hash"
        ], f"Empty content_hash for document: {docs[i]}"
        assert docs[i].metadata["file_hash"], f"Empty file_hash for document: {docs[i]}"

        # check if metadata can be dumped, otherwise stringify the culprit
        try:
            meta_dump = json.dumps(docs[i].metadata, ensure_ascii=False)
        except Exception:
            for k, v in docs[i].metadata.items():
                if isinstance(v, Path):
                    docs[i].metadata[k] = v.name
                    continue
                try:
                    json.dumps(v, ensure_ascii=False)
                except Exception:
                    docs[i].metadata[k] = str(v)
            meta_dump = json.dumps(docs[i].metadata, ensure_ascii=False)

        docs[i].metadata["all_hash"] = hasher(
            docs[i].metadata["content_hash"] + meta_dump
        )
        assert docs[i].metadata["all_hash"], f"Empty all_hash for document: {docs[i]}"

    total_reading_length = None
    try:
        total_reading_length = sum(
            [float(d.metadata["doc_reading_time"]) for d in docs]
        )
    except Exception:
        pass
    if total_reading_length is not None:
        assert total_reading_length > 0.1, (
            f"Failing doc: total reading length is {total_reading_length:.3f}"
            "min which is  suspiciously low. Filetype {filetype} with kwargs "
            f"'{kwargs}'"
        )

    assert docs, "empty list of loaded documents!"
    return docs


# import all loader functions
if not env.WDOC_LOADER_LAZY_LOADING:
    logger.debug("Importing all dependencies because WDOC_LOADER_LAZY_LOADING is False")
    from wdoc.utils.loaders.anki import load_anki
    from wdoc.utils.loaders.epub import load_epub
    from wdoc.utils.loaders.local_html import load_local_html
    from wdoc.utils.loaders.json_dict import load_json_dict
    from wdoc.utils.loaders.local_audio import load_local_audio
    from wdoc.utils.loaders.local_video import load_local_video
    from wdoc.utils.loaders.logseq_markdown import load_logseq_markdown
    from wdoc.utils.loaders.online_media import load_online_media
    from wdoc.utils.loaders.pdf import load_online_pdf, load_pdf
    from wdoc.utils.loaders.powerpoint import load_powerpoint
    from wdoc.utils.loaders.string import load_string
    from wdoc.utils.loaders.text import load_text
    from wdoc.utils.loaders.txt import load_txt
    from wdoc.utils.loaders.url import load_url
    from wdoc.utils.loaders.word import load_word
    from wdoc.utils.loaders.youtube import load_youtube

    # Validation: Check that all loader functions exist
    def _validate_loader_functions():
        """Validate that all loader functions referenced in LOADABLE_FILETYPE exist."""
        current_module = sys.modules[__name__]
        for filetype in LOADABLE_FILETYPE:
            if not hasattr(current_module, f"load_{filetype}"):
                raise Exception(
                    f"Loader function 'load_{filetype}' not found in module"
                )

    # Run validation when module is imported
    _validate_loader_functions()
