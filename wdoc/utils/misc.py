"""
Miscellanous functions etc.
"""

import hashlib
import requests
import re
from copy import copy
import platform
import inspect
import json
import os
import socket
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from difflib import get_close_matches
from functools import cache as memoize
from functools import partial, wraps
from pathlib import Path

import bs4
import litellm
from beartype.door import is_bearable
from beartype.typing import (
    Dict,
    Callable,
    List,
    Literal,
    Union,
    get_type_hints,
    Optional,
    Any,
)
from joblib import Memory
from joblib import hash as jhash
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.runnables import chain
from platformdirs import user_cache_dir
from loguru import logger

from wdoc.utils.env import env, is_input_piped
from wdoc.utils.errors import UnexpectedDocDictArgument
from wdoc.utils.typechecker import optional_typecheck

# ignore warnings from beautiful soup that can happen because anki is not exactly html
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="bs4",
    message=".*The input looks more like a filename than markup.*",
)

# additional warnings to ignore
warnings.filterwarnings(
    "ignore", module="litellm", message=".*Counting tokens for OpenAI model=.*"
)

try:
    import ftlangdetect

    @optional_typecheck
    def language_detector(text: str) -> float:
        return ftlangdetect.detect(text.lower())["score"]

    assert isinstance(language_detector("This is a test"), float)
except Exception as err:
    if env.WDOC_VERBOSE:
        logger.warning(
            f"Couldn't import optional package 'ftlangdetect' from 'fasttext-langdetect', trying to import langdetect (but it's much slower): '{err}'"
        )
    if "ftlangdetect" in sys.modules:
        del sys.modules["ftlangdetect"]

    try:
        import langdetect

        @optional_typecheck
        def language_detector(text: str) -> float:
            return langdetect.detect_langs(text.lower())[0].prob

        assert isinstance(language_detector("This is a test"), float)
    except Exception as err:
        if env.WDOC_VERBOSE:
            logger.warning(
                f"Couldn't import optional package 'langdetect' either: '{err}'"
            )

        @optional_typecheck
        def language_detector(text: str) -> None:
            return None


if (
    "OVERRIDE_USER_DIR_PYTEST_WDOC" in os.environ
    and os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] == "true"
):
    cache_dir = Path.cwd() / "wdoc_user_cache_dir"
    if cache_dir.exists():
        logger.debug(
            f"PYTEST detected so using cache_dir '{cache_dir.absolute()}' (already exists)"
        )
    else:
        logger.debug(
            f"PYTEST detected so using cache_dir '{cache_dir.absolute()}' (does not exists)"
        )
else:
    cache_dir = Path(user_cache_dir(appname="wdoc"))

cache_dir.mkdir(parents=True, exist_ok=True)

doc_loaders_cache_dir = cache_dir / "doc_loaders"
doc_loaders_cache_dir.mkdir(exist_ok=True)
doc_loaders_cache = Memory(doc_loaders_cache_dir, verbose=0)
hashdoc_cache_dir = cache_dir / "doc_hashing"
hashdoc_cache_dir.mkdir(exist_ok=True)
hashdoc_cache = Memory(hashdoc_cache_dir, verbose=0)
(cache_dir / "query_eval_llm").mkdir(exist_ok=True)
query_eval_cache = Memory(cache_dir / "query_eval_llm", verbose=0)

# remove cache files older than X days
if env.WDOC_EXPIRE_CACHE_DAYS:
    doc_loaders_cache.reduce_size(
        age_limit=timedelta(days=int(env.WDOC_EXPIRE_CACHE_DAYS))
    )
    hashdoc_cache.reduce_size(age_limit=timedelta(days=int(env.WDOC_EXPIRE_CACHE_DAYS)))
    query_eval_cache.reduce_size(
        age_limit=timedelta(days=int(env.WDOC_EXPIRE_CACHE_DAYS))
    )

# for reading length estimation
wpm = 250
average_word_length = 6

# separators used for the text splitter
recur_separator = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "...", ".", " ", ""]

min_token = 20
max_token = 10_000_000
min_lang_prob = 0.50

# list of available tasks
tasks_list = ["query", "summarize", "parse", "search", "summarize_then_query"]

printed_unexpected_api_keys = [False]  # to print it only once

# loader specific arguments
filetype_arg_types = {
    "pdf_parsers": Union[str, List[str]],
    "anki_deck": str,
    "anki_notetype": str,
    "anki_profile": str,
    "anki_template": str,
    "anki_tag_filter": str,
    "anki_tag_render_filter": str,
    "json_dict_template": str,
    "json_dict_exclude_keys": List,
    "audio_backend": Literal["whisper", "deepgram"],
    "audio_unsilence": bool,
    "whisper_lang": str,
    "whisper_prompt": str,
    "deepgram_kwargs": dict,
    "youtube_language": str,
    "youtube_translation": str,
    "youtube_audio_backend": Literal["youtube", "whisper", "deepgram"],
    "load_functions": List,
    "doccheck_min_token": int,
    "doccheck_max_token": int,
    "doccheck_min_lang_prob": float,
    "online_media_url_regex": str,
    "online_media_resourcetype_regex": str,
    "loading_failure": str,
}

# extra arguments supported when instanciating wdoc
extra_args_types = {
    "path": Union[str, Path],
    "embed_instruct": str,
    "include": str,
    "exclude": str,
    "filter_content": Union[List[str], str],
    "filter_metadata": Union[List[str], str],
    "source_tag": str,
    "pattern": str,
    "recursed_filetype": str,
}
extra_args_types.update(filetype_arg_types)


class DocDict(dict):
    """like dictionnaries but only allows keys that can be used when loading
    a document. Also checks the value type.

    The environnment variable 'WDOC_STRICT_DOCDICT' is a default value
    at instanciation time.
    Depending on WDOC_STRICT_DOCDICT (if not passed manually):
        if True: crash if unexpected arg
        if False: print in red if unexpected arg but add anyway
        if "strip": print in red but don't add
    """

    allowed_keys: set = set(
        sorted(
            [
                "path",
                "filetype",
                "file_hash",
                "source_tag",
                "recur_parent_id",
            ]
            + list(filetype_arg_types.keys())
        )
    )
    allowed_types: dict = filetype_arg_types
    __strict__ = env.WDOC_STRICT_DOCDICT

    def __hash__(self):
        "make it hashable, to check for duplicates"
        keys = sorted(self.keys())

        as_string = ""
        for k in keys:
            as_string += "\n"
            as_string += str(k)
            try:
                as_string += jhash(self[k])
            except Exception:
                as_string += str(self[k])
        return hash(as_string)

    def __check_values__(self, key, value, strict) -> bool:
        if key not in self.allowed_keys:
            mess = (
                f"Cannot set key '{key}' in a DocDict. Allowed keys are "
                f"'{','.join(self.allowed_keys)}'\nYou can use the env "
                "variable WDOC_STRICT_DOCDICT to avoid this issue."
            )
            if strict is True:
                raise UnexpectedDocDictArgument(mess)
            elif strict is False:
                logger.warning(mess)
                return True
            elif strict == "strip":
                logger.warning(mess)
                return False
            else:
                raise ValueError(strict)

        elif (
            (key in self.allowed_types)
            and (value is not None)
            and (not is_bearable(value, self.allowed_types[key]))
        ):
            mess = (
                f"Type of key {key} should be {self.allowed_types[key]},"
                f"not {type(value)}."
                "\nYou can use the env "
                "variable WDOC_STRICT_DOCDICT to avoid this issue."
            )

            if strict is True:
                raise UnexpectedDocDictArgument(mess)
            elif strict is False:
                logger.warning(mess)
                return True
            elif strict == "strip":
                logger.warning(mess)
                return False
            else:
                raise ValueError(strict)
        return True

    def __init__(self, docdict: dict, strict=env.WDOC_STRICT_DOCDICT) -> None:
        assert docdict, "Can't give an empty docdict as argument"
        assert strict in [True, False, "strip"], "Unexpected strict value"
        ignore_kwargs = []
        for k, v in docdict.items():
            if not self.__check_values__(k, v, strict):
                ignore_kwargs.append(k)

        for ik in ignore_kwargs:
            if ik in docdict:
                del docdict[ik]

        if strict != "strip":
            assert docdict, "Can't create DocDict: no args nor kwargs after filtering!"
        super().__init__(docdict)

        self.__strict__ = strict

    def __setitem__(self, key, value) -> None:
        assert self.__strict__ in [True, False, "strip"], "Unexpected strict value"
        self.__check_values__(key, value, self.__strict__)
        super().__setitem__(key, value)


@optional_typecheck
def optional_strip_unexp_args(func: Callable) -> Callable:
    """if the environment variable WDOC_STRICT_DOCDICT is set to 'true'
    then this automatically removes any unexpected argument before calling a
    loader function for a specific filetype."""
    if not env.WDOC_STRICT_DOCDICT:
        return optional_typecheck(func)
    else:
        # find the true function, otherwise func can be a decorated truefunc and might forget the annotations.
        if hasattr(func, "func"):
            truefunc = func.func
        else:
            truefunc = func
        while hasattr(truefunc, "func"):
            truefunc = truefunc.func

        @optional_typecheck
        @wraps(truefunc)
        def wrapper(*args, **kwargs):
            assert (
                not args
            ), f"We are not expecting args here, only kwargs. Received {args}"
            sig = inspect.signature(truefunc)
            bound_args = sig.bind_partial(**kwargs)

            # Remove unexpected positional arguments
            bound_args.arguments = {
                k: v for k, v in bound_args.arguments.items() if k in sig.parameters
            }

            # Remove unexpected keyword arguments
            kwargs2 = {k: v for k, v in kwargs.items() if k in sig.parameters}

            diffkwargs = {k: v for k, v in kwargs.items() if k not in kwargs2}
            if diffkwargs:
                mess = f"Unexpected args or kwargs in func {func}:"
                for kwarg in diffkwargs:
                    mess += f"\n-KWARG: {kwarg}"
                logger.warning(mess)
            assert (
                kwargs2
            ), f"No kwargs2 found for func {func}. There's probably an issue with the decorator"

            return func(**kwargs2)

        return wrapper


@optional_typecheck
def hasher(text: str) -> str:
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@optional_typecheck
def file_hasher(doc: dict) -> str:
    """used to hash a file's content, as describe by a dict
    A caching mechanism is used to avoid recomputing hash of file that
    have the same path and metadata.
    If the doc dict does not contain a path, the hash of the dict will be
    returned.
    """
    if "path" not in doc:
        return hasher(json.dumps(doc, ensure_ascii=False))
    hashable = False
    if "path" in doc and doc["path"] and Path(doc["path"]).exists():
        hashable = True
    if isinstance(doc["path"], str):
        if doc["path"] == "" or (not doc["path"].strip()):
            hashable = False
    if not doc["path"]:
        hashable = False
    if not isinstance(doc["path"], (str, Path)):
        hashable = False

    if hashable:
        file = Path(doc["path"])
        stats = file.stat()
        return _file_hasher(
            abs_path=str(file.resolve().absolute()),
            stats=[stats.st_mtime, stats.st_ctime, stats.st_ino, stats.st_size],
        )
    else:
        return hasher(json.dumps(doc, ensure_ascii=False))


@optional_typecheck
@hashdoc_cache.cache
def _file_hasher(abs_path: str, stats: List[Union[int, float]]) -> str:
    with open(abs_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:20]


@optional_typecheck
def html_to_text(html: str, remove_image: bool = False) -> str:
    """used to strip any html present in the text files"""
    html = html.replace("</li><li>", "<br>")  # otherwise they might get joined
    html = html.replace("</ul><ul>", "<br>")  # otherwise they might get joined
    html = html.replace("<br>", "\n").replace(
        "</br>", "\n"
    )  # otherwise newlines are lost

    soup = bs4.BeautifulSoup(html, "html.parser")
    content = []
    for element in soup.descendants:
        if element.name == "img" and (not remove_image):
            element = str(element)
            if element in html:
                content.append(element)
            elif element[:-2] + ">" in html:
                content.append(element[:-2] + ">")
            else:
                if env.WDOC_VERBOSE:
                    temptext = " ".join(filter(None, content))
                    logger.warning(
                        f"Image not properly parsed from bs4:\n{element}\n{temptext}"
                    )
        elif isinstance(element, bs4.NavigableString):
            content.append(str(element).strip())
    text = " ".join(filter(None, content))
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    if "<img" in text and remove_image:
        logger.warning(f"Failed to remove <img from anki card: {text}")
    return text


@chain
@optional_typecheck
def debug_chain(inputs: Union[dict, List]) -> Union[dict, List]:
    "use it between | pipes | in a chain to open the debugger"
    if hasattr(inputs, "keys"):
        logger.warning(str(inputs.keys()))
    breakpoint()
    return inputs


@optional_typecheck
def wrapped_model_name_matcher(model: str) -> str:
    "find the best match for a modelname (wrapped to make some check)"
    # find the currently set api keys to avoid matching models from
    # unset providers
    all_backends = list(litellm.models_by_provider.keys())
    backends = []
    for k, v in dict(os.environ).items():
        if k.endswith("_API_KEY"):
            backend = k.split("_API_KEY")[0].lower()
            if (
                backend not in all_backends
                and env.WDOC_VERBOSE
                and not printed_unexpected_api_keys[0]
            ):
                logger.debug(
                    f"Found API_KEY for backend {backend} that is not a known backend for litellm."
                )
            else:
                backends.append(backend)
    if env.WDOC_VERBOE:
        printed_unexpected_api_keys[0] = True
    assert backends, "No API keys found in environnment"

    # filter by providers
    backend, modelname = model.split("/", 1)
    if backend not in all_backends:
        raise Exception(
            f"Model {model} with backend {backend}: backend not found in "
            "litellm.\nList of litellm providers/backend:\n"
            f"{all_backends}"
        )
    if backend not in backends:
        raise Exception(
            f"Trying to use backend {backend} but no API KEY was found for it in the environnment."
        )
    candidates = litellm.models_by_provider[backend]
    if modelname in candidates:
        return model
    subcandidates = [m for m in candidates if m.startswith(modelname)]
    if len(subcandidates) == 1:
        good = f"{backend}/{subcandidates[0]}"
        return good
    match = get_close_matches(modelname, candidates, n=1)
    if match:
        return match[0]
    else:
        logger.warning(
            f"Couldn't match the modelname {model} to any known model. "
            "Continuing but this will probably crash wdoc further "
            "down the code."
        )
        return model


@memoize
@optional_typecheck
def model_name_matcher(model: str) -> str:
    """find the best match for a modelname (wrapper that checks if the matched
    model has a known cost and print the matched name)
    Bypassed if env variable WDOC_NO_MODELNAME_MATCHING is 'true'
    """
    assert "testing" not in model
    assert "/" in model, f"expected / in model '{model}'"
    if env.WDOC_NO_MODELNAME_MATCHING:
        # logger.debug(f"Bypassing model name matching for model '{model}'")
        return model

    out = wrapped_model_name_matcher(model)
    if out != model and env.WDOC_VERBOSE:
        logger.debug(f"Matched model name {model} to {out}")
    assert (
        out in litellm.model_cost or out.split("/", 1)[1] in litellm.model_cost
    ), f"Neither {out} nor {out.split('/', 1)[1]} found in litellm.model_cost"
    return out


@memoize
@optional_typecheck
def get_openrouter_metadata() -> dict:
    """fetch the metadata from openrouter, because litellm takes always too much time to add new models."""
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    rep = response.json()
    # put it in a suitable format
    data = {}
    for info in rep["data"]:
        modelid = "openrouter/" + info["id"]
        assert modelid not in data, modelid
        del info["id"]
        pricing = info["pricing"]  # fix pricing is a str originally
        for k, v in pricing.items():
            pricing[k] = float(v)
        data[modelid] = info

        # for models that for example end with ":free", make them appear
        # under their full name too
        while ":" in modelid:
            modelid = modelid[::-1].split(":")[0][::-1]
            if modelid not in data:
                data[modelid] = info

    # Example of output:
    # {'id': 'microsoft/phi-4-reasoning-plus:free',
    #  'name': 'Microsoft: Phi 4 Reasoning Plus (free)',
    #  'created': 1746130961,
    #  'description': REMOVED
    #  'context_length': 32768,
    #  'architecture': {'modality': 'text->text',
    #   'input_modalities': ['text'],
    #   'output_modalities': ['text'],
    #   'tokenizer': 'Other',
    #   'instruct_type': None},
    #  'pricing': {'prompt': '0',
    #   'completion': '0',
    #   'request': '0',
    #   'image': '0',
    #   'web_search': '0',
    #   'internal_reasoning': '0'},
    #  'top_provider': {'context_length': 32768,
    #   'max_completion_tokens': None,
    #   'is_moderated': False},
    #  'per_request_limits': None,
    #  'supported_parameters': ['max_tokens',
    #   'temperature',
    #   'top_p',
    #   'reasoning',
    #   'include_reasoning',
    #   'stop',
    #   'frequency_penalty',
    #   'presence_penalty',
    #   'seed',
    #   'top_k',
    #   'min_p',
    #   'repetition_penalty',
    #   'logprobs',
    #   'logit_bias',
    #   'top_logprobs']}
    return data


@dataclass
class ModelName:
    "Simply stores the different way to phrase a model name"

    original: str
    backend: str = field(init=False)
    model: str = field(init=False)
    sanitized: str = field(init=False)

    def __post_init__(self):
        assert (
            "/" in self.original
        ), f"Modelname must contain a / to distinguish the backend from the model. Received '{self.original}'"
        self.backend, self.model = self.original.split("/", 1)
        self.backend = self.backend.lower()

        # Use a sanitized name for the cache path
        self.sanitized = self.original
        if "/" in self.model:
            try:
                if Path(self.model).exists():
                    with open(
                        Path(self.model).resolve().absolute().__str__(), "rb"
                    ) as f:
                        h = hashlib.sha256(f.read() + str(self.model)).hexdigest()[:15]
                    self.sanitized = Path(self.model).name + "_" + h
            except Exception:
                pass
        self.sanitized = self.sanitized.replace("/", "_")
        if env.WDOC_PRIVATE_MODE:
            self.sanitized = "private_" + self.sanitized

    def is_testing(self) -> bool:
        "Return True if the model is 'testing/testing'."
        if "testing" in self.original.lower():
            return True
        return False

    def __hash__(self):
        # necessary for memoizing
        return (str(self.original.__hash__()) + str("ModelName".__hash__())).__hash__()


@memoize
@optional_typecheck
def get_model_price(model: ModelName) -> Dict[str, Union[float, int]]:
    if env.WDOC_ALLOW_NO_PRICE:
        logger.warning(
            f"Disabling price computation for {model} because env var 'WDOC_ALLOW_NO_PRICE' is 'true'"
        )
        return {"prompt": 0, "completion": 0, "internal_reasoning": 0}

    if model.backend == "ollama":
        return {"prompt": 0, "completion": 0, "internal_reasoning": 0}
    elif model.is_testing():
        return {"prompt": 0, "completion": 0, "internal_reasoning": 0}
    elif model.backend == "openrouter":
        metadata = get_openrouter_metadata()
        assert model.original in metadata, f"Missing {model} from openrouter"
        pricing = metadata[model.original]["pricing"]
        if "request" in pricing and pricing["request"]:
            logger.error(
                f"Found non 0 request for {model}, this is not supported by wdoc so the price will not be accurate"
            )
        return pricing

    for key in ["original", "model", "sanitized"]:
        mod = getattr(model, key)
        if mod in litellm.model_cost:
            pricing = litellm.model_cost[mod]
            output = {}
            output["prompt"] = pricing["input_cost_per_token"]
            output["completion"] = pricing["output_cost_per_token"]
            if "output_cost_per_reasoning_token" in pricing:
                output["internal_reasoning"] = pricing[
                    "output_cost_per_reasoning_token"
                ]
            else:
                output["internal_reasoning"] = 0
            for k, v in pricing.items():
                if k not in output:
                    output[k] = v
            return output
    raise Exception(
        f"Can't find the price of '{model}'\nUpdate litellm or set WDOC_ALLOW_NO_PRICE=True if you still want to use this model."
    )


@memoize
@optional_typecheck
def get_model_max_tokens(modelname: ModelName) -> int:
    if modelname.backend == "openrouter":
        openrouter_data = get_openrouter_metadata()
        assert (
            modelname.original in openrouter_data
        ), f"Missing model {modelname.original} from openrouter metadata"
        return openrouter_data[modelname.original]["context_length"]

    if modelname.original in litellm.model_cost:
        return litellm.model_cost[modelname.original]["max_tokens"]
    elif (trial := modelname.model) in litellm.model_cost:
        return litellm.model_cost[trial]["max_tokens"]
    elif (trial2 := modelname.model.split("/")[-1]) in litellm.model_cost:
        return litellm.model_cost[trial2]["max_tokens"]
    else:
        try:
            return litellm.get_model_info(modelname.original)["max_tokens"]
        except Exception:
            return litellm.get_model_info(modelname.name)[
                "max_tokens"
            ]  # crash if still not found


@optional_typecheck
def get_tkn_length(
    tosplit: str,
    modelname: Union[str, ModelName] = "gpt-4o-mini",
) -> int:
    if isinstance(modelname, ModelName):
        modelname = modelname.original
    modelname = modelname.replace("openrouter/", "")
    return litellm.token_counter(model=modelname, text=tosplit)


text_splitters = {}

DEFAULT_SPLITTER_MODELNAME = ModelName("openai/gpt-4o-mini")


@optional_typecheck
def get_splitter(
    task: str,
    modelname: ModelName = DEFAULT_SPLITTER_MODELNAME,
) -> TextSplitter:
    "we don't use the same text splitter depending on the task"
    # avoid creating many times this object
    if task not in text_splitters:
        text_splitters[task] = {}
    if modelname.original in text_splitters[task]:
        return text_splitters[task][modelname.original]

    if modelname.original == "testing/testing":
        return get_splitter(task=task, modelname=DEFAULT_SPLITTER_MODELNAME)

    try:
        if modelname.model == "gpt-4o-mini":
            # this is not the true limit of 4o-mini but a good placeholder for if we are using the default model anyway, see get_tkn_length above
            max_tokens = 4096
        else:
            max_tokens = get_model_max_tokens(modelname)

    except Exception as err:
        max_tokens = 4096
        logger.warning(
            f"Failed to get max_tokens limit for model {modelname.original}: '{err}'"
        )

    # Cap context sizes
    if task in ["query", "search"] and max_tokens > env.WDOC_MAX_EMBED_CONTEXT:
        logger.warning(
            f"Capping max_tokens for model {modelname} to WDOC_MAX_EMBED_CONTEXT ({env.WDOC_MAX_EMBED_CONTEXT} instead of {max_tokens}) because in query mode and we can only guess the context size of the embedding model."
        )
        max_tokens = min(max_tokens, env.WDOC_MAX_EMBED_CONTEXT)
    if max_tokens > env.WDOC_MAX_CHUNK_SIZE:
        logger.debug(
            f"Capping max_tokens for model {modelname} to the WDOC_MAX_CHUNK_SIZE value ({env.WDOC_MAX_CHUNK_SIZE} instead of {max_tokens})."
        )
        max_tokens = min(max_tokens, env.WDOC_MAX_CHUNK_SIZE)

    model_tkn_length = partial(get_tkn_length, modelname=modelname.original)

    if task in ["query", "search"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=int(3 / 4 * max_tokens),  # default 4000
            chunk_overlap=500,  # default 200
            length_function=model_tkn_length,
        )
    elif task in ["summarize_then_query", "summarize"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=int(1 / 2 * max_tokens),
            chunk_overlap=500,
            length_function=model_tkn_length,
        )
    elif task == "recursive_summary":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=int(1 / 4 * max_tokens),
            chunk_overlap=300,
            length_function=model_tkn_length,
        )
    else:
        raise Exception(task)

    text_splitters[task][modelname.original] = text_splitter
    return text_splitter


@optional_typecheck
def check_docs_tkn_length(
    docs: List[Document],
    identifier: str,
    min_token: int = min_token,
    max_token: int = max_token,
    min_lang_prob: float = min_lang_prob,
    check_language: bool = False,
) -> float:
    """checks that the number of tokens in the document is high enough,
    not too low, and has a high enough language probability,
    otherwise something probably went wrong."""
    size = sum([get_tkn_length(d.page_content) for d in docs])
    nline = len("\n".join([d.page_content for d in docs]).splitlines())
    if size <= min_token:
        logger.warning(
            f"Example of page from document with too few tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{identifier}' is {size} <= {min_token}, probably something went wrong?"
        )
    if size >= max_token:
        logger.warning(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{identifier}' is {size} >= {max_token}, probably something went wrong?"
        )
    if check_language is False:
        return 1.0

    # check if language check is above a threshold and cast as lowercase as it's apparently what it was trained on
    try:
        probs = [language_detector(d.page_content.replace("\n", "<br>")) for d in docs]
        if probs[0] is None or not probs:
            # bypass if language_detector not defined
            return 1.0
        prob = sum(probs) / len(probs)
        if prob <= min_lang_prob:
            raise Exception(
                f"Low language probability for {identifier}: prob={prob:.3f}<{min_lang_prob}.\nExample page: {docs[len(docs)//2]}"
            )
    except Exception as err:
        if str(err).startswith("Low language probability"):
            raise
        else:
            logger.warning(
                f"Error when using language_detector on '{identifier}': {err}. Treating it as valid document."
            )
            return 1.0
    return prob


@optional_typecheck
def unlazyload_modules():
    """make sure no modules are lazy loaded. Useful when we wan't to make
    sure not to loose time and that everything works smoothly. For example
    who knows what happens when multiprocessing with lazy loaded modules."""
    if env.WDOC_IMPORT_TYPE not in ["both", "lazy"]:
        logger.debug("Lazyloading is disabled so not unlazyloading modules.")
        return

    while True:
        found_one = False
        for k, v in sys.modules.items():
            try:
                str(v)
            except Exception as e:
                logger.warning(
                    f"Very weird error when loading a package, consider setting WDOC_IMPORT_TYPE to another value than '{env.WDOC_IMPORT_TYPE}'. Error message was '{e}'"
                )
            if "Lazily-loaded" in str(v):
                try:
                    dir(v)  # this is enough to trigger the loading
                    found_one = True
                except Exception as err:
                    raise Exception(
                        f"Error when unlazyloading module '{k}'. Error: '{err}'"
                        "\nThis can be caused by beartype's typechecking"
                        "\nYou can also try setting the env variable "
                        "WDOC_IMPORT_TYPE to 'native' or 'thread'"
                    ) from err
                break  # otherwise dict size change during iteration
            assert "Lazily-loaded" not in str(v)
        if found_one:
            continue
        else:
            break


@optional_typecheck
def disable_internet(allowed: dict) -> None:
    """
    To be extra sure that no connection goes out of the computer when
    --private is used, we overload the socket module to make it only able to
    reach local connection.
    """
    logger.warning(
        "Disabling outgoing internet because private mode is on. "
        "The only allowed IPs from now on are the ones from the "
        "argument llm_api_bases. Note that this permanently filters "
        "outgoing python connections so might interfere with other "
        "python programs is you are importing wdoc instead "
        "of calling it from the shell"
    )

    # unlazyload all modules as otherwise the overloading can happen too late
    unlazyload_modules()

    # list of certainly allowed IPs
    allowed_IPs = set(
        [
            "localhost",
            "127.0.0.1",
        ]
    )
    vals = [
        v.split("//")[1].split(":")[0] if "//" in v else v.split(":")[0]
        for v in list(allowed.values())
    ]
    [allowed_IPs.add(v) for v in vals]

    # list of probably allowed IPs
    private_ranges = [
        ("10.0.0.0", "10.255.255.255"),
        ("172.16.0", "172.31.255.255"),
        ("192.168.0.0", "192.168.255.255"),
        ("127.0.0.0", "127.255.255.255"),
    ]

    @memoize
    @optional_typecheck
    def is_private(ip: str) -> bool:
        "detect if the connection would go to our computer or to a remote server"
        if ip in allowed_IPs:
            return True
        ip = int.from_bytes(socket.inet_aton(ip), "big")
        if ip in allowed_IPs:
            return True
        for start, end in private_ranges:
            if (
                int.from_bytes(socket.inet_aton(start), "big")
                <= ip
                <= int.from_bytes(socket.inet_aton(end), "big")
            ):
                return True
        return False

    @optional_typecheck
    def create_connection(address, *args, **kwargs):
        "overload socket.create_connection to forbid outgoing connections"
        ip = socket.gethostbyname(address[0])
        if not is_private(ip):
            raise RuntimeError("Network connections to the open internet are blocked")
        return socket._original_create_connection(address, *args, **kwargs)

    socket.socket = lambda *args, **kwargs: None
    socket._original_create_connection = socket.create_connection
    socket.create_connection = create_connection

    # sanity check
    assert is_private("localhost")
    assert is_private("10.0.1.32")
    assert is_private("192.168.2.35")
    assert is_private("127.12.13.15")

    # checking allowed ips are okay
    for v in vals:
        assert is_private(v), f"An address failed to be set as private: '{v}'"
    for al in list(allowed.values()):
        ip = socket.gethostbyname(al)
        assert is_private(ip), f"An address failed to be set as private: '{al}'"
    try:
        ip = socket.gethostbyname("www.google.com")
        skip = False
    except Exception as err:
        logger.warning(
            "Failed to get IP address of www.google.com to check if it is "
            "indeed blocked. You probably did this on purpose so not "
            f"crashing. Error: '{err}'"
        )
        skip = True
    if not skip:
        assert not is_private(
            ip
        ), f"Failed to set www.google.com as unreachable: IP is '{ip}'"


@optional_typecheck
def set_func_signature(func: Callable) -> Callable:
    """dynamically set the extra args of wdoc.__init__ so that
    instead of **cli_kwargs the signature indicates all allowed arguments.
    Needed to get correct behavior from fire.Fire '--completion'"""
    original_sig = inspect.signature(func)
    assert (
        list(original_sig.parameters.values())[-1].kind == inspect.Parameter.VAR_KEYWORD
    )
    new_params = list(original_sig.parameters.values())[:-1]  # Remove **cli_kwargs
    new_params.extend(
        [
            inspect.Parameter(
                name=arg,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=hint,
                default=None,
            )
            for arg, hint in extra_args_types.items()
        ]
    )
    new_sig = original_sig.replace(parameters=new_params)

    @wraps(func)
    def new_func(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    new_func.__signature__ = new_sig
    new_func.__annotations__ = get_type_hints(func) | extra_args_types

    return new_func


# Tag constants
THIN = "<think>"
THINE = "</think>"
ANSW = "<answer>"
ANSWE = "</answer>"

# Pre-compiled regex patterns
_THIN_REGEX = re.compile(f"{re.escape(THIN)}(.*){re.escape(THINE)}", re.DOTALL)
_THIN_SUB_REGEX = re.compile(
    f"{re.escape(THIN)}|{re.escape(THINE)}|{re.escape(ANSW)}|{re.escape(ANSWE)}"
)


@optional_typecheck
def thinking_answer_parser(output: str, strict: bool = False) -> dict:
    """separate the <think> and <answer> tags in an answer"""
    orig = copy(output)
    try:
        # some models like the geminis don't return their thinking output, sometimes
        # by mistake they keep thinking anyway so we get THINE without THIN. Let's just add
        # it at the beginning of output
        if THINE in output and THIN not in output:
            output = THIN + output

        if (THIN not in output) and (ANSW not in output):
            assert (
                THINE not in output
            ), f"Output contains no {THIN} nor {ANSW} but an unexpected {THINE}:\n'''\n{output}\n'''"
            assert (
                ANSWE not in output
            ), f"Output contains no {THIN} nor {ANSW} but an unexpected {ANSWE}:\n'''\n{output}\n'''"

            logger.debug(f"LLM output contained neither {THIN} nor {ANSW}")
            return {"thinking": "", "answer": output}

        thinking = ""
        if (
            THIN in output and THINE in output
        ):  # meaning we found the expected <think> </think> block
            thinking_match = _THIN_REGEX.search(output)
            if thinking_match:
                thinking = thinking_match.group(1)
                if not (THIN not in thinking and THINE not in thinking):
                    logger.warning(
                        f"Found {THINK} or {THINE} inside the thinking block, we don't expect nested thinkings but will proceed anyway."
                    )
        else:
            # check we don't have only one of the xml sides
            assert (
                THIN not in output and THINE not in output
            ), f"Found only one of '{THIN}' or '{THINE}' in LLM output"
            logger.debug("LLM output contained no thinking block")

        answer = ""
        if (
            ANSW in output and ANSWE in output
        ):  # meaning we found the expected <answer> </answer> block
            # Create a version without the thinking part
            answer_text = output
            if thinking:
                answer_text = re.sub(re.escape(thinking), "", answer_text)
            # Remove the xml sides
            answer = _THIN_SUB_REGEX.sub("", answer_text)
            logger.debug("LLM output contained answer block")
        else:
            # check we don't have only one of the xml sides
            assert (
                ANSW not in output and ANSWE not in output
            ), f"Found only one of '{ANSW}' or '{ANSWE}' in LLM output"
            if thinking:
                logger.debug(
                    "LLM output contained no answer block, assuming it's all but the thinking"
                )
                answer = (
                    output.replace(thinking, "").replace(THIN, "").replace(THINE, "")
                )
            else:
                logger.debug(
                    "LLM output contained no answer block, assuming it's all the output"
                )
                answer = output

        output = output.strip()
        thinking = thinking.strip()
        answer = answer.rstrip()

        assert (
            THIN not in answer
        ), f"Parsed answer contained unexpected {THIN}:\n'''\n{output}\n'''"
        assert (
            THINE not in answer
        ), f"Parsed answer contained unexpected {THIN}:\n'''\n{output}\n'''"
        assert (
            THIN not in thinking
        ), f"Parsed thinking contained unexpected {THIN}:\n'''\n{output}\n'''"
        assert (
            THINE not in thinking
        ), f"Parsed thinking contained unexpected {THIN}:\n'''\n{output}\n'''"
        assert (
            ANSW not in answer
        ), f"Parsed answer contained unexpected {ANSW}:\n'''\n{output}\n'''"
        assert (
            ANSWE not in answer
        ), f"Parsed answer contained unexpected {ANSW}:\n'''\n{output}\n'''"

        assert answer, f"No answer could be parsed from LLM output: '{output}'"

        return {"thinking": thinking, "answer": answer}
    except Exception as err:
        if (
            strict
        ):  # otherwise combining answers could snowball into losing lots of text
            raise
        logger.warning(
            f"Error when parsing LLM output to get thinking and answer part.\nError: '{err}'\nOriginal output: '{orig}'\nNote: if the output seems fine but ends abruptly instead of by </answer> you might want to tweak the max_token settings.\nWill continue if not using --debug"
        )
        if env.WDOC_DEBUG:
            raise
        else:
            assert output.strip(), "LLM output was empty"
            return {
                "thinking": "",
                "answer": f"""
<note>
The following LLM answer might have had a problem during parsing
</note>
<output>
{orig}
</output>
""".strip(),
            }


# this will contain wdoc's version to be used by langfuse's callback without circular imports
langfuse_callback_holder = []


@optional_typecheck
def create_langfuse_callback(version: str) -> None:
    assert not env.WDOC_PRIVATE_MODE
    # replace langfuse's env variable if set for wdoc, this is already done in env.py but doing it here also at runtime
    for k in [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]:
        newk = "WDOC_" + k
        if newk in os.environ and os.environ[newk]:
            os.environ[k] = os.environ[newk]
    if (
        "LANGFUSE_PUBLIC_KEY" in os.environ
        and "LANGFUSE_SECRET_KEY" in os.environ
        and "LANGFUSE_HOST" in os.environ
    ):
        logger.debug("Activating langfuse callbacks")
        try:
            import langfuse
        except ImportError as e:
            if (
                "WDOC_LANGFUSE_PUBLIC_KEY" in os.environ
                and "redacted" not in os.environ.get("WDOC_LANGFUSE_PUBLIC_KEY", "")
            ):
                raise Exception(
                    f"Couldn't import langfuse even though WDOC_LANGFUSE environment variables appear set. Crashing."
                ) from e
            else:
                logger.warning(
                    f"Failed to setup langfuse callback because of ImportError, make sure package 'langfuse' is installed. The error was: '{e}'"
                )
        try:
            # use litellm's callbacks for chatlitellm backend
            import litellm

            litellm.success_callback.append("langfuse")
            litellm.failure_callback.append("langfuse")

            # # and use langchain's callback for openai's backend
            # BUT as of october 2024 it seems buggy with chatlitellm, the modelname does not seem to be passed?
            from langfuse.callback import CallbackHandler as LangfuseCallback

            langfuse_callback = LangfuseCallback(
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                host=os.environ["LANGFUSE_HOST"],
                session_id=str(uuid.uuid4()),
                version=version,
            )
            langfuse_callback_holder.append(langfuse_callback)
        except Exception as e:
            logger.warning(
                f"Failed to setup langfuse callback, make sure package 'langfuse' is installed. The error was: '{e}'"
            )


@optional_typecheck
def seconds_to_timecode(inp: Union[str, float, int]) -> str:
    "used for vtt subtitle conversion"
    second = float(inp)
    minute = second // 60
    second = second % 60
    hour = minute // 60
    minute = minute % 60
    hour, minute, second = int(hour), int(minute), int(second)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


@optional_typecheck
def timecode_to_second(inp: str) -> int:
    "turns a vtt timecode into seconds"
    hour, minute, second = map(int, inp.split(":"))
    return hour * 3600 + minute * 60 + second


@optional_typecheck
def is_timecode(inp: Union[float, str]) -> bool:
    try:
        timecode_to_second(inp)
        return True
    except Exception:
        return False


@memoize
@optional_typecheck
def get_supported_model_params(modelname: ModelName) -> list:
    if modelname.backend == "testing":
        return []
    if modelname.backend == "openrouter":
        metadata = get_openrouter_metadata()
        assert (
            modelname.original in metadata
        ), f"Missing {modelname.original} from openrouter"
        return metadata[modelname.original]["supported_parameters"]

    for test in [
        modelname.original,
        modelname.model,
        model_name_matcher(modelname.original),
    ]:
        params = litellm.get_supported_openai_params(test)
        if params:
            return params
    for test in [
        modelname.original,
        modelname.model,
        model_name_matcher(modelname.original),
    ]:
        params = litellm.get_supported_openai_params(
            test, custom_llm_provider=modelname.backend
        )
        if params:
            return params
    for test in [
        modelname.original,
        modelname.model,
        model_name_matcher(modelname.original),
    ]:
        params = litellm.get_supported_openai_params(
            test, custom_llm_provider="openrouter"
        )
        if params:
            return params
    return []


@optional_typecheck
def cache_file_in_memory(file_path: Path, recursive: bool = False) -> bool:
    """
    Advise the Linux kernel to cache the given file in memory.

    Args:
        file_path: Path to the file or directory to cache
        recursive: If True and file_path is a directory, cache all files within it

    Returns:
        bool: True if caching was successful, False otherwise
    """
    # Check if we're on Linux
    if platform.system() != "Linux":
        # This function only works on Linux systems.
        return False

    files_to_cache: List[Path] = []

    # Handle directory case
    if file_path.is_dir():
        if not recursive:
            # Warning: {file_path} is a directory. Set recursive=True to cache all files.
            return False
        # Collect all files recursively
        files_to_cache = [f for f in file_path.rglob("*") if f.is_file()]
    elif file_path.is_file():
        files_to_cache = [file_path]
    else:
        # Error: {file_path} does not exist.
        return False

    success = True
    for file in files_to_cache:
        try:
            fd = os.open(str(file), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            os.close(fd)
        except Exception as e:
            # logger.warning(f"Failed to cache {file}: {e}")
            success = False

    return success


def log_and_time_fn(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        logger.debug(f"Enterring {fn}")
        val = fn(*args, **kwargs)
        logger.debug(f"Exiting {fn}")
        return val

    wrapped = wraps(fn)(wrapper)
    return wrapped


@optional_typecheck
def get_piped_input() -> Optional[Any]:
    """
    Read data from stdin/pipes.
    This is done when importing wdoc, to avoid any issues with parallelism
    and threads etc.
    The content is added to the commandline starting wdoc directly in
    __main__.py.
    """
    # Check if data is being piped (stdin is not a terminal)
    if not is_input_piped:
        return None
    # Save a copy of the original stdin for debugging
    # original_stdin = sys.stdin

    # Read the piped data
    piped_input = sys.stdin.buffer.read()
    try:
        piped_input = piped_input.decode()
    except Exception:
        pass

    # Create a new file descriptor for stdin from /dev/tty if available
    # This allows breakpoint() to work later
    try:
        if os.name != "nt":  # Unix-like systems
            sys.stdin = open("/dev/tty")
        else:  # Windows
            # On Windows this is trickier, consider using a different approach
            pass

    except:
        # If we can't reopen stdin, at least return the data
        pass

    logger.debug("Loaded piped data")
    return piped_input
