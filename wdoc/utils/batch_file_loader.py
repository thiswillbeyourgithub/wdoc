"""
called at wdoc instance creation. It parsed the combined filetype
into an individual list of DocDict describing each a document (or in some cases
a list of documents for example a whole anki database).
This list is then processed in loaders.py, multithreading or multiprocessing
is used.
"""

import inspect
import json
import random
import re
import shutil
import time
import uuid
from collections import Counter
from functools import cache as memoizer
from multiprocessing.context import TimeoutError as MultiprocessTimeoutError
from pathlib import Path

import dill
import rtoml
import uuid6
from beartype.typing import List, Optional, Tuple, Union, Literal
from joblib import Parallel, delayed
from langchain.docstore.document import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from tqdm import tqdm
from loguru import logger

from wdoc.utils.env import env, is_out_piped
from wdoc.utils.loaders import (
    load_one_doc,
    load_youtube_playlist,
    markdownlink_regex,
    yt_link_regex,
)
from wdoc.utils.misc import (
    DocDict,
    ModelName,
    cache_dir,
    file_hasher,
    unlazyload_modules,
)
from wdoc.utils.errors import NoInferrableFiletype

assert env.WDOC_BEHAVIOR_EXCL_INCL_USELESS in [
    "warn",
    "crash",
], "Unexpected value of WDOC_BEHAVIOR_EXCL_INCL_USELESS"


# rules used to attribute input to proper filetype. For example
# any link containing youtube will be treated as a youtube link
inference_rules = {
    # format:
    # key is output filtype, value is list of regex that if match
    # will return the key
    # the order of the keys is important
    "youtube_playlist": ["youtube.*playlist"],
    "youtube": ["youtube", "invidi", r"youtu\."],
    "logseq_markdown": [".*logseq.*.md"],
    "txt": [".txt$", ".md$"],
    "online_pdf": ["^http.*pdf.*"],
    "pdf": [".*pdf$"],
    "url": ["^http"],
    "local_html": [r"^(?!http).*\.html?$"],
    "local_audio": [r".*(mp3|m4a|ogg|flac)$"],
    "epub": [".epub$"],
    "powerpoint": [".ppt$", ".pptx$", ".odp$"],
    "word": [".doc$", ".docx$", ".odt$"],
    "local_video": [".mp4", ".avi", ".mkv"],
    "json_entries": [".*.json"],
    "toml_entries": [".*.toml"],
}

# compile the inference rules as regex
for k, v in inference_rules.items():
    for i, vv in enumerate(v):
        inference_rules[k][i] = re.compile(vv)


def infer_filetype(path: str) -> str:
    """
    Heuristics to infer the 'filetype' argument of a --path given to wdoc.
    """
    for k, v in inference_rules.items():
        for vv in inference_rules[k]:
            if vv.search(path):
                logger.debug(f"Regex infered that path '{path}' is of filetype '{k}'")
                return k
    fp = Path(path)
    if not fp.exists():
        raise NoInferrableFiletype(
            f"Failed to detect 'auto' filetype for '{fp}' with regex, and it's not a file (does not exist)"
        )
    try:
        import magic

        info = magic.from_file(fp).lower()
        # # instead of passing the file, pass only the
        # # headers of the file because otherwise
        # # it seems to have issues with some files
        # with open(fp, "rb") as temp:
        #     start = temp.read(1024)
        # info = magic.from_buffer(start).lower()
    except Exception as err:
        raise NoInferrableFiletype(
            f"Failed to detect 'auto' filetype for '{fp}' with regex and even python-magic. Error: '{err}'"
        ) from err
    if "pdf" in info:
        logger.debug(f"Magic infered that path '{path}' is of filetype 'pdf'")
        return "pdf"
    elif "mpeg" in info or "mp3" in info:
        logger.debug(f"Magic infered that path '{path}' is of filetype 'local_audio'")
        return "local_audio"
    elif "epub" in info:
        logger.debug(f"Magic infered that path '{path}' is of filetype 'epub'")
        return "epub"
    else:
        raise NoInferrableFiletype(
            f"No more python magic heuristics to try for path '{path}'"
        )


def batch_load_doc(
    llm_name: ModelName,
    filetype: str,
    task: str,
    backend: str,
    n_jobs: int,
    **cli_kwargs,
) -> List[Document]:
    """
    Receives the arguments from the wdoc instanciation. Here we turn the input
    arguments into a list of DocDict. Then `load_one_doc` is called on each DocDict to
    get the (langchain) Documents expected by LLMs.
    A simple example of such DocDict could be '{path: some/file.pdf, filetype: pdf}'
    Note that if the `filetype` value is a recursive_filetype, one of the
    parser_* function is called on it to replace it with multiple DocDict.
    For example '{path: some/file.json, filetype: json_entries}' will trigger
    a call to `parse_json_entries` to load the dict stored in `some/file.json`,
    cast them as DocDict then send them to `load_one_doc`.
    """

    # just in case, make sure all modules are loaded
    unlazyload_modules()

    if "path" in cli_kwargs and isinstance(cli_kwargs["path"], str):
        cli_kwargs["path"] = cli_kwargs["path"].strip()

    # used to make sure all source tags were indeed parsed
    asked_source_tags = []
    if "source_tag" in cli_kwargs:
        asked_source_tags.append(cli_kwargs["source_tag"])

    # expand the list of document to load as long as there are recursive types
    to_load = [cli_kwargs.copy()]
    to_load[-1]["filetype"] = filetype.lower()
    new_doc_to_load = []
    loop_counter = 0
    while any(d["filetype"] in recursive_types_func_mapping for d in to_load):
        loop_counter += 1
        if loop_counter > 5:
            culprit_elements = [
                d for d in to_load if d["filetype"] in recursive_types_func_mapping
            ]
            raise Exception(
                f"Infinite loop detected in recursive file type processing after {loop_counter} iterations. "
                f"Culprit elements still in to_load: {culprit_elements}"
            )
        for ild, load_kwargs in enumerate(to_load):

            if "source_tag" in load_kwargs:
                if load_kwargs["source_tag"] not in asked_source_tags:
                    asked_source_tags.append(load_kwargs["source_tag"])

            to_load[ild]["filetype"] = to_load[ild]["filetype"].lower()
            if not ("path" in load_kwargs and load_kwargs["path"]):
                continue
            load_filetype = load_kwargs["filetype"]

            # guess the appropriate 'filetype' argument based on the path because
            # the user gave us filetype='auto'
            if load_filetype == "auto":
                load_filetype = infer_filetype(load_kwargs["path"])

                assert (
                    load_filetype != "auto"
                ), f"Could not infer the filetype of '{load_kwargs['path']}', please specify a value for the 'filetype' argument."
                if load_filetype not in recursive_types_func_mapping:
                    to_load[ild]["filetype"] = load_filetype

            if load_filetype not in recursive_types_func_mapping:
                continue
            del load_kwargs["filetype"]

            if (
                load_filetype in recursive_types_func_mapping
                and load_filetype != "auto"
            ):
                func_to_use = recursive_types_func_mapping[load_filetype]
                to_add = func_to_use(
                    cli_kwargs=cli_kwargs,
                    **load_kwargs,
                )

                # remove the arguments that were expected by the recursive parsing function and are returned unchanged. For example arguments like ddg_max_result
                the_func_kwargs = dict(inspect.signature(func_to_use).parameters)
                logger.warning(str(to_add))
                for inewdoc, newdoc in enumerate(to_add):
                    for k, v in newdoc.copy().items():
                        if (
                            k in the_func_kwargs
                            and cli_kwargs[k] == load_kwargs[k]
                            and cli_kwargs[k] == v
                        ):
                            del to_add[inewdoc][k]
                logger.warning(str(to_add))
                new_doc_to_load.extend(to_add)
                break

        if new_doc_to_load:
            for indtl, ndtl in enumerate(new_doc_to_load):
                assert (
                    ndtl
                ), f"Args for document #{indtl} from recursive_types '{load_filetype}' is empty."

            assert load_filetype in recursive_types_func_mapping
            to_load.remove(to_load[ild])
            to_load.extend(new_doc_to_load)
            new_doc_to_load = []
            continue

    try:
        to_load = [d if isinstance(d, DocDict) else DocDict(d) for d in to_load]
    except Exception as err:
        raise Exception(f"Expected to have only DocDict at this point: {err}'")

    for d in to_load:
        assert d, "One document to load has no arguments, double check your inputs"

    # if is not a file, check if it's not just because spaces are not escaped or something dumb like that
    for idoc, doc in enumerate(to_load):
        if "path" not in doc:
            continue

        p = Path(doc["path"])
        if p.exists():
            continue
        alternatives = [
            Path(p.expanduser()),
        ]
        for ialt, alt in enumerate(alternatives):
            if Path(alt).exists():
                if isinstance(doc["path"], Path):
                    doc["path"] = alt
                elif isinstance(doc["path"], str):
                    doc["path"] = str(alt.absolute())
                else:
                    raise ValueError(
                        f"At this point path should only be str or Path: {doc['path']}"
                    )
                break

    # remove duplicate documents
    temp = []
    for d in to_load:
        if d in temp:
            logger.warning(f"Removed document {d} (duplicate)")
        else:
            temp.append(d)
    to_load = temp

    assert to_load, f"empty list of documents to load from filetype '{filetype}'"

    # look for unexpected keys that are not relevant to doc loading, because that would
    # skip the cache
    all_unexp_keys = set()
    for doc in to_load:
        to_del = [k for k in doc if k not in DocDict.allowed_keys]
        for k in to_del:
            all_unexp_keys.add(k)
            del doc[k]
            assert k not in [
                "include",
                "exclude",
            ], "Include or exclude arguments should be reomved at this point"

    if "summar" not in task:
        # shuffle the list of files to load to make
        # the hashing progress bar more representative
        to_load = sorted(to_load, key=lambda x: random.random())

    # store the file hash in the doc kwarg
    doc_hashes = Parallel(
        n_jobs=-1 if len(to_load) > 1 else 1,
        backend=backend,
        verbose=0 if not env.WDOC_VERBOSE else 51,
    )(
        delayed(file_hasher)(doc=doc)
        for doc in tqdm(
            to_load,
            desc="Hashing files",
            unit="doc",
            colour="magenta",
            disable=len(to_load) <= 10_000 or is_out_piped,
        )
    )
    for i, h in enumerate(doc_hashes):
        to_load[i]["file_hash"] = doc_hashes[i]

    if "summar" not in task:
        # shuffle the list of files again to be random but deterministic:
        # keeping only the digits of each hash, then multiplying by the
        # index of the filetype by size. This makes sure the doc dicts are
        # sorted by increasing order of filetype frequency, so if there's
        # an error with the code of this filetype of its args the user knows
        # it quickly instead of after waiting a super long time
        bins = {}
        for d in to_load:
            if d["filetype"] not in bins:
                bins[d["filetype"]] = 1
            else:
                bins[d["filetype"]] += 1

        to_load = sorted(
            to_load,
            key=lambda d: hash(d["file_hash"]),
        )

    # load_functions are slow to load so loading them here in advance for every file
    if any(("load_functions" in doc and doc["load_functions"]) for doc in to_load):
        logger.info("Preloading load_functions")
        for idoc, doc in enumerate(to_load):
            if "load_functions" in doc:
                if doc["load_functions"]:
                    to_load[idoc]["load_functions"] = parse_load_functions(
                        tuple(doc["load_functions"])
                    )

    to_load = list(set(to_load))  # remove duplicates docdicts

    if len(to_load) > 1:
        for tl in to_load:
            assert (
                tl["filetype"] != "text"
            ), "You shouldn't not be using filetype 'text' with other kind of documents normally. Please open an issue on github and explain me your usecase to see how I can fix that for you!"

    # dir name where to store temporary files
    load_temp_name = "file_load_" + str(uuid6.uuid6())
    # delete previous temp dir if it's several days old
    for f in cache_dir.iterdir():
        f = f.resolve()
        if (
            f.is_dir()
            and f.name.startswith("file_load_")
            and (abs(time.time() - f.stat().st_mtime) > 2 * 86400)
        ):
            assert str(cache_dir.absolute()) in str(f.absolute())
            shutil.rmtree(f)
    temp_dir = cache_dir / load_temp_name
    temp_dir.mkdir(exist_ok=False)

    loader_max_timeout = env.WDOC_MAX_LOADER_TIMEOUT
    if loader_max_timeout <= 0:
        loader_max_timeout = None

    sharedmem = None
    if len(to_load) == 1:
        n_jobs = 1
        to_load[0][
            "loading_failure"
        ] = "crash"  # crash if loading fails when only one document is to be loaded and fails anyway
    else:
        if backend == "loky":
            if env.WDOC_VERBOSE:
                logger.warning("Using loky backend so not using 'require=sharedmem'")
        else:
            sharedmem = "sharedmem"

    # # early stopping if all the documents of a recursive filetype failed
    # expected_recur_nb = {}
    # for d in to_load:
    #     if d["recur_parent_id"] not in expected_recur_nb:
    #         expected_recur_nb[d["recur_parent_id"]] = 1
    #     else:
    #         expected_recur_nb[d["recur_parent_id"]] += 1
    # found_recur_nb = {k: 0 for k in expected_recur_nb.keys()}

    docs = []
    t_load = time.time()
    try:
        generator_doc_lists = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=0 if not env.WDOC_VERBOSE else 51,
            timeout=loader_max_timeout,
            return_as="generator",  # try to reduce memory footprint
            require=sharedmem,
        )(
            delayed(load_one_doc)(
                llm_name=llm_name,
                task=task,
                temp_dir=temp_dir,
                **d,
            )
            for d in tqdm(
                to_load,
                desc="Loading",
                unit="doc",
                colour="magenta",
                disable=is_out_piped,
            )
        )
        doc_lists = []
        for idoc, o in enumerate(generator_doc_lists):
            doc_lists.append(o)
            # # detect errors, used for early stopping if all the doc from the same recursive parent failed:
            # if isinstance(o, str):
            #     d = to_load[idoc]
            #     if "recur_parent_id" in d:
            #         assert d["recur_parent_id"] in expected_recur_nb, expected_recur_nb
            #         found_recur_nb[d["recur_parent_id"]] -= 1
            #         if found_recur_nb[d["recur_parent_id"]] <= 0 and expected_recur_nb[d['recur_parent_id']] > 1:
            #             mess = f"All document from a recursive file with parent id {d['recur_parent_id']} failed so crashing."
            #             mess += "\nFailed documents:"
            #             cnt = 0
            #             for ifd, fd in enumerate(to_load):
            #                 if fd['recur_parent_id'] == d["recur_parent_id"]:
            #                     cnt += 1
            #                     er = doc_lists[ifd]
            #                     assert isinstance(er, str), er
            #                     mess += f"\n- #{cnt}: {fd}: '{er}'"
            #             mess += f"\nLatest error was: '{o}'"
            #             logger.warning(mess)
            #             raise Exception(mess)
    except MultiprocessTimeoutError as e:
        raise Exception(
            logger.warning(
                f"Timed out when loading batch files after {loader_max_timeout}s"
            )
        ) from e

    logger.info(
        f"Done loading all {len(to_load)} documents in {time.time()-t_load:.2f}s"
    )
    missing_docargs = []
    for idoc, d in tqdm(
        enumerate(doc_lists),
        total=len(doc_lists),
        desc="Concatenating results",
        disable=not env.WDOC_VERBOSE or is_out_piped,
    ):
        if isinstance(d, list):
            docs.extend(d)
        else:
            assert isinstance(d, str)
            missing_docargs.append(
                dict(to_load[idoc])
            )  # must be cast as dict to set error message
            missing_docargs[-1]["error_message"] = d
    assert not any(isinstance(d, str) for d in docs)

    if missing_docargs:
        missing_docargs = sorted(
            missing_docargs, key=lambda x: json.dumps(x, ensure_ascii=False)
        )
        logger.warning(f"Number of failed documents: {len(missing_docargs)}:")
    else:
        logger.debug("No document failed to load!")

    if len(missing_docargs) == len(doc_lists):
        raise Exception("All documents failed to load. The errors appear above.")

    if asked_source_tags:
        no_st = 0
        st = {t: 0 for t in asked_source_tags}
        extra = {}
        for doc in docs:
            if "source_tag" in doc.metadata:
                s = doc.metadata["source_tag"]
                if s not in st:
                    if s in extra:
                        extra[s] += 1
                    else:
                        extra[s] = 1
                else:
                    st[s] += 1
            else:
                no_st += 1
        should_crash = False
        logger.warning("Found the following source_tag after loading all documents:")
        for n, s in st.items():
            logger.warning(f"- {s}: {n}")
            if n == 0:
                should_crash = True
        if extra:
            logger.warning(
                "Found the following EXTRA source_tag after loading all documents:"
            )
            logger.warning("(This can happen after merging identical documents though)")
            should_crash = True
            for n, s in extra.items():
                logger.warning(f"- {s}: {n}")
        logger.warning(f"Found {no_st} documents with no source_tag")

        if should_crash:
            logger.warning(
                "Something might have gone wrong given those source tags.\nAnswer 'crash' to crash, 'd' to debug, anything else to continue."
            )
            ans = input(">")
            if ans == "d":
                breakpoint()
            elif ans == "crash":
                raise Exception("Probable error given the source tags")
            else:
                pass

    # smart deduplication before embedding:
    # find the document with the same content_hash, merge their metadata and keep only one
    if "summar" not in task and len(docs) > 1:
        logger.debug("Deduplicating...")
        logger.debug("Getting all hash")
        content_hash = [d.metadata["content_hash"] for d in docs]
        logger.debug("Counting them")
        counts = Counter(content_hash)
        dupes = set()
        [dupes.add(h) for h, c in counts.items() if c > 1]
        deduped = {}
        lenbefore = len(docs)
        for idoc, doc in enumerate(
            tqdm(docs, desc="Deduplicating", unit="doc", disable=is_out_piped)
        ):
            ch = doc.metadata["content_hash"]
            if not dupes:
                logger.debug("No duplicates!")
                break
            if ch in deduped:
                assert doc.page_content == deduped[ch].page_content
                for k, v in doc.metadata.items():
                    if "hash" in k:
                        continue
                    elif k == "source_tag":
                        if "source_tag" in deduped[ch].metadata:
                            deduped[ch].metadata[k] += " " + v
                        else:
                            deduped[ch].metadata[k] += v
                        continue
                    elif k in deduped[ch].metadata:
                        if v == deduped[ch].metadata[k]:
                            continue
                    elif k not in deduped[ch].metadata:
                        deduped[ch].metadata[k] = v
                    elif isinstance(v, list) and isinstance(
                        deduped[ch].metadata[k], list
                    ):
                        deduped[ch].metadata[k] += deduped[ch].metadata[k]
                    elif env.WDOC_VERBOSE:
                        logger.warning(f"UNEXPECTED METADATA TYPE: '{k}:{v}'")
                docs[idoc] = None

            if ch in dupes:
                deduped[ch] = doc
                docs[idoc] = None
                assert counts[ch] > 1, doc
                counts[ch] -= 1
                if counts[ch] == 1:
                    dupes.remove(ch)
        if deduped:
            assert None in docs
        assert not dupes, dupes
        docs = [d for d in docs if d is not None]
        if deduped:
            docs += list(deduped.values())
        assert (
            len(docs) <= lenbefore
        ), f"Removing duplicates seems to have added documents: {lenbefore} -> {len(docs)}. Something went wrong."

    assert docs, "No documents were succesfully loaded!"

    # delete temp dir
    shutil.rmtree(temp_dir)
    assert not temp_dir.exists()

    return docs


def parse_recursive_paths(
    cli_kwargs: dict,
    path: Union[str, Path],
    pattern: str,
    recursed_filetype: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **extra_args,
) -> List[Union[DocDict, dict]]:
    """
    Turn a DocDict that has `filetype==recursive_paths` into the DocDict of
    individual files in that path.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The directory path to search recursively
        pattern: Glob pattern to match files (e.g., "*.pdf", "**/*.txt")
        recursed_filetype: The filetype to assign to found files
        include: Optional list of regex patterns to include files, default=None
        exclude: Optional list of regex patterns to exclude files, default=None
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict or dict objects, each representing a found file
    """
    logger.info(f"Parsing recursive load_filetype: '{path}'")
    assert recursed_filetype not in [
        "recursive_paths",
        "json_entries",
        "youtube",
        "anki",
    ], "'recursed_filetype' cannot be 'recursive_paths', 'json_entries', 'anki' or 'youtube'"

    if not Path(path).exists() and Path(path.replace(r"\ ", " ")).exists():
        logger.info(r"File was not found so replaced '\ ' by ' '")
        path = path.replace(r"\ ", " ")
    assert Path(path).exists(), f"not found: {path}"
    doclist = [p for p in Path(path).rglob(pattern)]
    assert doclist, f"No document found by pattern {pattern}"
    doclist = [str(p).strip() for p in doclist if p.is_file()]
    assert doclist, "No document after filtering by file"
    doclist = [p for p in doclist if p]
    assert doclist, "No document after removing nonemtpy"
    doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
    recur_parent_id = str(uuid.uuid4())

    if include:
        for iinc, inc in enumerate(include):
            if isinstance(inc, str):
                if inc == inc.lower():
                    inc = re.compile(inc, flags=re.IGNORECASE)
                else:
                    inc = re.compile(inc)
                include[iinc] = inc
        ndoclist = len(doclist)
        doclist = [d for d in doclist if any(inc.search(d) for inc in include)]
        if not len(doclist) < ndoclist:
            mess = f"Include rules were useless and didn't filter out anything.\nInclude rules: '{include}'"
            if env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "warn":
                logger.warning(mess)
            elif env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "crash":
                logger.warning(mess)
                raise Exception(mess)

    if exclude:
        for iexc, exc in enumerate(exclude):
            if isinstance(exc, str):
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                exclude[iexc] = exc
            ndoclist = len(doclist)
            doclist = [d for d in doclist if not exc.search(d)]
            if not len(doclist) < ndoclist:
                mess = f"Exclude rule '{exc}' was useless and didn't filter out anything.\nExclude rules: '{exclude}'"
                if env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "warn":
                    logger.warning(mess)
                elif env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "crash":
                    logger.warning(mess)
                    raise Exception(mess)

    for i, d in enumerate(doclist):
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = recursed_filetype
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        if doc_kwargs["filetype"] not in recursive_types_func_mapping:
            doclist[i] = DocDict(doc_kwargs)
        else:
            doclist[i] = doc_kwargs
    return doclist


def parse_json_entries(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[Union[DocDict, dict]]:
    """
    Turn a DocDict that has `filetype==json_entries` into the individual
    DocDict mentionned inside the json file.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The path to the JSON file containing document entries
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict or dict objects, each representing an entry from the JSON file
    """
    logger.info(f"Loading json_entries: '{path}'")
    doclist = str(Path(path).read_text()).splitlines()
    doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
    doclist = [
        p.strip() for p in doclist if p.strip() and not p.strip().startswith("#")
    ]
    recur_parent_id = str(uuid.uuid4())

    for i, d in enumerate(doclist):
        meta = cli_kwargs.copy()
        meta["filetype"] = "auto"
        meta.update(json.loads(d.strip()))
        for k, v in cli_kwargs.copy().items():
            if k not in meta:
                meta[k] = v
        if meta["path"] == path:
            del meta["path"]
        meta.update(extra_args)
        meta["recur_parent_id"] = recur_parent_id
        if meta["filetype"] not in recursive_types_func_mapping:
            doclist[i] = DocDict(meta)
        else:
            doclist[i] = meta
    return doclist


def parse_toml_entries(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[Union[DocDict, dict]]:
    """
    Turn a DocDict that has `filetype==toml_entries` into the individual
    DocDict mentionned inside the toml file.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The path to the TOML file containing document entries
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict or dict objects, each representing an entry from the TOML file
    """
    logger.info(f"Loading toml_entries: '{path}'")
    content = rtoml.load(toml=Path(path))
    assert isinstance(content, dict)
    doclist = list(content.values())
    assert all(len(d) == 1 for d in doclist)
    doclist = [d[0] for d in doclist]
    assert all(isinstance(d, dict) for d in doclist)
    recur_parent_id = str(uuid.uuid4())

    for i, d in enumerate(doclist):
        meta = cli_kwargs.copy()
        meta["filetype"] = "auto"
        meta.update(d)
        for k, v in cli_kwargs.items():
            if k not in meta:
                meta[k] = v
        if meta["path"] == path:
            del meta["path"]
        meta.update(extra_args)
        meta["recur_parent_id"] = recur_parent_id
        if meta["filetype"] not in recursive_types_func_mapping:
            doclist[i] = DocDict(meta)
        else:
            doclist[i] = meta
    return doclist


def parse_link_file(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==link_file` into the individual
    DocDict of each url, where there is one url per line inside the
    `link_file` file. Note that bullet points are stripped (i.e. "- [the url]" is
    treated the same as "the url"), and commented lines (i.e. starting with "#")
    are ignored.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The path to the link file containing URLs
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each representing a URL from the link file
    """
    logger.info(f"Loading link_file: '{path}'")
    doclist = str(Path(path).read_text()).splitlines()
    doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
    doclist = [
        p.strip()
        for p in doclist
        if p.strip() and not p.strip().startswith("#") and "http" in p
    ]
    doclist = [
        matched.group(0)
        for d in doclist
        if (matched := markdownlink_regex.search(d).strip())
    ]

    recur_parent_id = str(uuid.uuid4())
    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["subitem_link"] = d
        doc_kwargs["filetype"] = "auto"
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        doclist[i] = DocDict(doc_kwargs)
    return doclist


def parse_youtube_playlist(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==youtube_playlist` into the individual
    DocDict of each youtube video part of that playlist.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The YouTube playlist URL
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each representing a YouTube video from the playlist
    """
    if "\\" in path:
        logger.warning(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")
    logger.info(f"Loading youtube playlist: '{path}'")
    video = load_youtube_playlist(path)

    playlist_title = video["title"].strip().replace("\n", "")
    assert (
        "duration" not in video
    ), f'"duration" found when loading youtube playlist. This might not be a playlist: {path}'
    doclist = [ent["webpage_url"] for ent in video["entries"]]
    doclist = [li for li in doclist if yt_link_regex.search(li)]

    recur_parent_id = str(uuid.uuid4())
    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = "youtube"
        doc_kwargs["subitem_link"] = d
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        doclist[i] = DocDict(doc_kwargs)

    assert doclist, f"No video found in youtube playlist: {path}"
    for idoc, doc in enumerate(doclist):
        if "playlist_title" in doc.metadata and doc.metadata["playlist_title"]:
            if playlist_title not in doc.metadata["playlist_title"]:
                doc.metadata["playlist_title"] += " - " + playlist_title
        else:
            doc.metadata["playlist_title"] = playlist_title
        doclist[idoc]
    return doclist


def parse_ddg_search(
    cli_kwargs: dict,
    path: Union[str, Path],
    ddg_max_results: int = 50,
    ddg_region: str = "",
    ddg_safesearch: Literal["on", "off", "moderate"] = "off",
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==ddg` into the individual
    DocDict of the webpage of each DuckDuckGo search result, treating the
    `path` as a search query.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The search query string
        ddg_max_results: Maximum number of search results to return, default=50
        ddg_region: DuckDuckGo search region, default=''
        ddg_safesearch: SafeSearch setting ("on", "moderate", "off"), default='off'
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each representing a URL from search results
    """
    query = str(path).strip()
    logger.info(f"Performing DuckDuckGo search: '{query}'")

    # Set up DuckDuckGo search
    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=ddg_max_results,
        safesearch=ddg_safesearch,
        source="text",
        region=ddg_region,
    )

    tool = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        response_format="content_and_artifact",
        output_format="list",
        max_results=ddg_max_results,
        region=ddg_region,
    )
    tool.max_results = ddg_max_results

    # Perform the search
    try:
        search_results = tool.invoke(query)
    except Exception as err:
        raise Exception(f"DuckDuckGo search failed for query '{query}': {err}") from err

    if not search_results:
        logger.warning(f"No search results found for query: '{query}'")
        return []

    logger.info(f"Found {len(search_results)} search results for: '{query}'")

    # Create a unique parent ID for tracking this search batch
    recur_parent_id = str(uuid.uuid4())

    doclist = []
    for i, result in enumerate(search_results):
        if "link" not in result:
            logger.warning(f"Search result #{i+1} missing 'link' field: {result}")
            continue

        url = result["link"].strip()
        if not url or not url.startswith("http"):
            logger.warning(f"Invalid URL in search result #{i+1}: '{url}'")
            continue

        # Create document kwargs
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = url
        doc_kwargs["subitem_link"] = url
        doc_kwargs["filetype"] = "auto"  # Let wdoc auto-detect the URL content type
        doc_kwargs["recur_parent_id"] = recur_parent_id

        # Add DuckDuckGo search metadata
        # doc_kwargs["ddg_search_query"] = query
        # doc_kwargs["ddg_search_rank"] = i + 1
        # if "title" in result:
        #     doc_kwargs["ddg_title"] = result["title"]
        # if "snippet" in result:
        #     doc_kwargs["ddg_snippet"] = result["snippet"]

        # Apply any extra arguments
        doc_kwargs.update(extra_args)

        doclist.append(DocDict(doc_kwargs))

    if not doclist:
        raise Exception(
            f"No valid URLs found in DuckDuckGo search results for query: '{query}'"
        )

    # it seems sometimes ddg returns duplicate result
    assert len(doclist) == len(set(doclist)), f"Duplicate elements: {doclist}"

    return doclist


@memoizer
def parse_load_functions(load_functions: Tuple[str, ...]) -> bytes:
    load_functions = list(load_functions)
    assert isinstance(load_functions, list), "load_functions must be a list"
    assert all(
        isinstance(lf, str) for lf in load_functions
    ), "load_functions elements must be strings"

    try:
        for ilf, lf in enumerate(load_functions):
            load_functions[ilf] = eval(lf)
    except Exception as err:
        raise Exception(f"Error when evaluating load_functions #{ilf}: {lf} '{err}'")
    load_functions = tuple(load_functions)
    pickled = dill.dumps(load_functions)
    return pickled


recursive_types_func_mapping = {
    "recursive_paths": parse_recursive_paths,
    "json_entries": parse_json_entries,
    "toml_entries": parse_toml_entries,
    "link_file": parse_link_file,
    "youtube_playlist": parse_youtube_playlist,
    "ddg": parse_ddg_search,
    "auto": None,
}
