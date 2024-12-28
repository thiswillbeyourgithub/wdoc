"""
called at wdoc instance creation. It parsed the combined filetype
into an individual list of DocDict describing each a document (or in some cases
a list of documents for example a whole anki database).
This list is then processed in loaders.py, multithreading or multiprocessing
is used.
"""

import json
import random
import re
import shutil
import sys
import time
import uuid
from collections import Counter
from functools import cache as memoizer
from multiprocessing.context import TimeoutError as MultiprocessTimeoutError
from pathlib import Path

import dill
import rtoml
import uuid6
from beartype.typing import List, Optional, Tuple, Union
from joblib import Parallel, delayed
from langchain.docstore.document import Document
from tqdm import tqdm

from .env import WDOC_BEHAVIOR_EXCL_INCL_USELESS, WDOC_MAX_LOADER_TIMEOUT
from .flags import is_debug, is_verbose
from .loaders import (
    load_one_doc_wrapped,
    load_youtube_playlist,
    loaders_temp_dir_file,
    markdownlink_regex,
    yt_link_regex,
)
from .logger import logger, red, whi
from .misc import (
    DocDict,
    cache_dir,
    doc_loaders_cache,
    file_hasher,
    get_tkn_length,
    min_token,
    unlazyload_modules,
)
from .typechecker import optional_typecheck

assert WDOC_BEHAVIOR_EXCL_INCL_USELESS in [
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

recursive_types = [
    "recursive_paths",
    "json_entries",
    "toml_entries",
    "link_file",
    "youtube_playlist",
    "auto",
]

# compile the inference rules as regex
for k, v in inference_rules.items():
    for i, vv in enumerate(v):
        inference_rules[k][i] = re.compile(vv)


@optional_typecheck
def batch_load_doc(
    llm_name: str, filetype: str, task: str, backend: str, n_jobs: int, **cli_kwargs
) -> List[Document]:
    """load the input"""

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
    while any(d["filetype"] in recursive_types for d in to_load):
        for ild, load_kwargs in enumerate(to_load):

            if "source_tag" in load_kwargs:
                if load_kwargs["source_tag"] not in asked_source_tags:
                    asked_source_tags.append(load_kwargs["source_tag"])

            to_load[ild]["filetype"] = to_load[ild]["filetype"].lower()
            if not ("path" in load_kwargs and load_kwargs["path"]):
                continue
            load_filetype = load_kwargs["filetype"]

            # auto parse filetype if infer
            if load_filetype == "auto":
                for k, v in inference_rules.items():
                    for vv in inference_rules[k]:
                        if vv.search(load_kwargs["path"]):
                            load_filetype = k
                            break
                    if load_filetype != "auto":
                        break
                if load_filetype == "auto":
                    try:
                        fp = Path(load_kwargs["path"])
                        if fp.exists():
                            try:
                                import magic

                                info = magic.from_file(fp).lower()
                            except Exception as err:
                                raise Exception(
                                    f"Failed to run python-magic as a last resort heuristic: '{err}'"
                                ) from err
                            if "pdf" in info:
                                load_filetype = "pdf"
                                break
                            elif "mpeg" in info:
                                load_filetype = "local_audio"
                                break
                            elif "epub" in info:
                                load_filetype = "epub"
                                break
                            else:
                                raise Exception(
                                    "No more python magic heuristics to try"
                                )
                    except Exception as err:
                        red(
                            f"Failed to detect 'auto' filetype for '{fp}' with regex and even python-magic. Error: '{err}'"
                        )

                assert (
                    load_filetype != "auto"
                ), f"Could not infer filetype of {load_kwargs['path']}. Use the 'filetype' argument."
                if load_filetype not in recursive_types:
                    to_load[ild]["filetype"] = load_filetype

            if load_filetype not in recursive_types:
                continue
            del load_kwargs["filetype"]

            if load_filetype == "recursive_paths":
                new_doc_to_load.extend(
                    parse_recursive_paths(cli_kwargs=cli_kwargs, **load_kwargs)
                )
                break

            elif load_filetype == "json_entries":
                new_doc_to_load.extend(
                    parse_json_entries(cli_kwargs=cli_kwargs, **load_kwargs)
                )
                break

            elif load_filetype == "toml_entries":
                new_doc_to_load.extend(
                    parse_toml_entries(cli_kwargs=cli_kwargs, **load_kwargs)
                )
                break

            elif load_filetype == "link_file":
                new_doc_to_load.extend(
                    parse_link_file(
                        cli_kwargs=cli_kwargs,
                        **load_kwargs,
                    )
                )
                break

            elif load_filetype == "youtube_playlist":
                new_doc_to_load.extend(
                    parse_youtube_playlist(cli_kwargs=cli_kwargs, **load_kwargs)
                )
                break

        if new_doc_to_load:
            for indtl, ndtl in enumerate(new_doc_to_load):
                assert (
                    ndtl
                ), f"Args for document #{indtl} from recursive_types '{load_filetype}' is empty."

            assert load_filetype in recursive_types
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
            red(f"Removed document {d} (duplicate)")
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
        verbose=0 if not is_verbose else 51,
    )(
        delayed(file_hasher)(doc=doc)
        for doc in tqdm(
            to_load,
            desc="Hashing files",
            unit="doc",
            colour="magenta",
            disable=len(to_load) <= 10_000,
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
        sorted_filetypes = sorted(bins.keys(), key=lambda x: bins[x])

        @optional_typecheck
        def deterministic_sorter(doc_dict: DocDict) -> int:
            h = doc_dict["file_hash"]
            h2 = "".join(filter(str.isdigit, h))
            h_ints = int(h2) if h2.isdigit() else int(random.random() * 1000)
            h_ordered = h_ints * (
                10 ** (sorted_filetypes.index(doc_dict["filetype"]) + 1)
            )
            return h_ordered

        to_load = sorted(
            to_load,
            key=deterministic_sorter,
        )

    # load_functions are slow to load so loading them here in advance for every file
    if any(("load_functions" in doc and doc["load_functions"]) for doc in to_load):
        whi("Preloading load_functions")
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
    loaders_temp_dir_file.write_text(str(temp_dir.absolute().resolve()))

    loader_max_timeout = WDOC_MAX_LOADER_TIMEOUT
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
            if is_verbose:
                red("Using loky backend so not using 'require=sharedmem'")
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
            verbose=0 if not is_verbose else 51,
            timeout=loader_max_timeout,
            return_as="generator",  # try to reduce memory footprint
            require=sharedmem,
        )(
            delayed(load_one_doc_wrapped)(
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
            #             raise Exception(red(mess))
    except MultiprocessTimeoutError as e:
        raise Exception(
            red(f"Timed out when loading batch files after {loader_max_timeout}s")
        ) from e

    # erases content that links to the loaders temporary files at startup
    loaders_temp_dir_file.write_text("")

    red(f"Done loading all {len(to_load)} documents in {time.time()-t_load:.2f}s")
    missing_docargs = []
    for idoc, d in tqdm(
        enumerate(doc_lists),
        total=len(doc_lists),
        desc="Concatenating results",
        disable=not is_verbose,
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
        red(f"Number of failed documents: {len(missing_docargs)}:")
    else:
        red("No document failed to load!")

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
        red("Found the following source_tag after loading all documents:")
        for n, s in st.items():
            red(f"- {s}: {n}")
            if n == 0:
                should_crash = True
        if extra:
            red("Found the following EXTRA source_tag after loading all documents:")
            red("(This can happen after merging identical documents though)")
            should_crash = True
            for n, s in extra.items():
                red(f"- {s}: {n}")
        red(f"Found {no_st} documents with no source_tag")

        if should_crash:
            red(
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
        red("Deduplicating...")
        whi("Getting all hash")
        content_hash = [d.metadata["content_hash"] for d in docs]
        whi("Counting them")
        counts = Counter(content_hash)
        dupes = set()
        [dupes.add(h) for h, c in counts.items() if c > 1]
        deduped = {}
        lenbefore = len(docs)
        for idoc, doc in enumerate(tqdm(docs, desc="Deduplicating", unit="doc")):
            ch = doc.metadata["content_hash"]
            if not dupes:
                whi("No duplicates!")
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
                    elif is_verbose:
                        red(f"UNEXPECTED METADATA TYPE: '{k}:{v}'")
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


@optional_typecheck
def parse_recursive_paths(
    cli_kwargs: dict,
    path: Union[str, Path],
    pattern: str,
    recursed_filetype: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **extra_args,
) -> List[Union[DocDict, dict]]:
    whi(f"Parsing recursive load_filetype: '{path}'")
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
            if WDOC_BEHAVIOR_EXCL_INCL_USELESS == "warn":
                red(mess)
            elif WDOC_BEHAVIOR_EXCL_INCL_USELESS == "crash":
                raise Exception(red(mess))

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
                if WDOC_BEHAVIOR_EXCL_INCL_USELESS == "warn":
                    red(mess)
                elif WDOC_BEHAVIOR_EXCL_INCL_USELESS == "crash":
                    raise Exception(red(mess))

    for i, d in enumerate(doclist):
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = recursed_filetype
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        if doc_kwargs["filetype"] not in recursive_types:
            doclist[i] = DocDict(doc_kwargs)
        else:
            doclist[i] = doc_kwargs
    return doclist


@optional_typecheck
def parse_json_entries(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[Union[DocDict, dict]]:
    whi(f"Loading json_entries: '{path}'")
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
        if meta["filetype"] not in recursive_types:
            doclist[i] = DocDict(meta)
        else:
            doclist[i] = meta
    return doclist


@optional_typecheck
def parse_toml_entries(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[Union[DocDict, dict]]:
    whi(f"Loading toml_entries: '{path}'")
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
        if meta["filetype"] not in recursive_types:
            doclist[i] = DocDict(meta)
        else:
            doclist[i] = meta
    return doclist


@optional_typecheck
def parse_link_file(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[DocDict]:
    whi(f"Loading link_file: '{path}'")
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


@optional_typecheck
def parse_youtube_playlist(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[DocDict]:
    if "\\" in path:
        red(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")
    whi(f"Loading youtube playlist: '{path}'")
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
    return doclist


@optional_typecheck
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
