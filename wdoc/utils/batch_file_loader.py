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
import uuid6
from collections import Counter
from multiprocessing.context import TimeoutError as MultiprocessTimeoutError
from pathlib import Path

from beartype.typing import List
from joblib import Parallel, delayed
from langchain.docstore.document import Document
from tqdm import tqdm
from loguru import logger

from wdoc.utils.env import env, is_out_piped
from wdoc.utils.tasks.types import wdocTask
from wdoc.utils.loaders import load_one_doc
from wdoc.utils.misc import (
    DocDict,
    ModelName,
    cache_dir,
    file_hasher,
    hasher,
    unlazyload_modules,
)
from wdoc.utils.errors import NoInferrableFiletype
from wdoc.utils.load_recursive import (
    recursive_types_func_mapping,
    parse_load_functions,
)

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
    task: wdocTask,
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
    seen_hashes = set()
    while any(d["filetype"] in recursive_types_func_mapping for d in to_load):
        # Create a hash of the current state to detect infinite loops
        # Using sorted JSON to ensure consistent hashing regardless of dict ordering
        current_state = json.dumps(
            sorted([json.dumps(tl) for tl in to_load]),
            ensure_ascii=False,
            sort_keys=True,
        )
        state_hash = hasher(current_state)

        if state_hash in seen_hashes:
            culprit_elements = [
                d for d in to_load if d["filetype"] in recursive_types_func_mapping
            ]
            raise Exception(
                f"Infinite loop detected in recursive file type processing. "
                f"The same state has been encountered twice. "
                f"Culprit elements still in to_load: {culprit_elements}"
            )
        seen_hashes.add(state_hash)

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

                assert load_filetype != "auto", (
                    f"Could not infer the filetype of '{load_kwargs['path']}', please specify a value for the 'filetype' argument."
                )
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
                assert ndtl, (
                    f"Args for document #{indtl} from recursive_types '{load_filetype}' is empty."
                )
                if (
                    "filetype" not in ndtl
                ):  # fix if the filetype has not been set after recursive loading
                    ndtl["filetype"] = "auto"

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

    if not task.summarize:
        # shuffle the list of files to load to make
        # the hashing progress bar more representative
        to_load = sorted(to_load, key=lambda x: random.random())

    # store the file hash in the doc kwarg
    doc_hashes = Parallel(
        n_jobs=-1 if len(to_load) > 1 else 1,
        backend=backend,
        verbose=0,
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

    if not task.summarize:
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
            assert tl["filetype"] != "text", (
                "You shouldn't not be using filetype 'text' with other kind of documents normally. Please open an issue on github and explain me your usecase to see how I can fix that for you!"
            )

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
        to_load[0]["loading_failure"] = (
            "crash"  # crash if loading fails when only one document is to be loaded and fails anyway
        )
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
        f"Done loading all {len(to_load)} documents in {time.time() - t_load:.2f}s"
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

        # Count failures per source tag
        st_failed = {t: 0 for t in asked_source_tags}
        for failed_doc in missing_docargs:
            if "source_tag" in failed_doc:
                tag = failed_doc["source_tag"]
                if tag in st_failed:
                    st_failed[tag] += 1

        should_crash = False
        logger.warning("Found the following source_tag after loading all documents:")
        for n, s in st.items():
            # n is the tag name, s is the successful count
            n_failed = st_failed.get(n, 0)
            if n_failed > 0:
                total = s + n_failed
                success_rate = (s / total * 100) if total > 0 else 0
                logger.warning(
                    f"- {s}: {n} ({n_failed} failed, {success_rate:.1f}% success rate)"
                )
            else:
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

    if task.summarize and len(docs) > 1:
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
        assert len(docs) <= lenbefore, (
            f"Removing duplicates seems to have added documents: {lenbefore} -> {len(docs)}. Something went wrong."
        )

    assert docs, "No documents were succesfully loaded!"

    # delete temp dir
    shutil.rmtree(temp_dir)
    assert not temp_dir.exists()

    return docs
