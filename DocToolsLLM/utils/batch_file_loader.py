"""
called at DocToolsLLM instance creation. It parsed the combined filetype
into an individual list of dict describing each a document (or in some cases
a list of documents for example a whole anki database).
This list is then processed in loaders.py, multithreading or multiprocessing
is used.
"""

import shutil
import uuid
import re
import sys
import traceback
from tqdm import tqdm
from functools import cache as memoizer
import time
from typing import List, Tuple, Union
import random

from langchain.docstore.document import Document
from joblib import Parallel, delayed
from pathlib import Path
import json
import dill

from .misc import doc_loaders_cache, file_hasher, min_token, get_tkn_length, unlazyload_modules, doc_kwargs_keys, cache_dir
from .typechecker import optional_typecheck
from .logger import red, whi, log
from .loaders import load_one_doc, yt_link_regex, load_youtube_playlist, markdownlink_regex, loaders_temp_dir_file
from .flags import is_debug


# rules used to attribute input to proper filetype. For example
# any link containing youtube will be treated as a youtube link
inference_rules = {
    # format:
    # key is output filtype, value is list of regex that if match
    # will return the key
    # the order of the keys is important
    "youtube_playlist": ["youtube.*playlist"],
    "youtube": ["youtube", "invidi"],
    "logseq_markdown": [".*logseq.*.md"],
    "txt": [".txt$", ".md$"],
    "text": [".txt$", ".md$"],
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
}

recursive_types = [
    "recursive_paths",
    "json_entries",
    "link_file",
    "youtube_playlist",
    "infer"
]

# compile the inference rules as regex
for k, v in inference_rules.items():
    for i, vv in enumerate(v):
        inference_rules[k][i] = re.compile(vv)


@optional_typecheck
def batch_load_doc(
        filetype: str,
        task: str,
        backend: str,
        **cli_kwargs) -> List[Document]:
    """load the input"""
    # # remove cache files older than 90 days
    # try:
    #     doc_loaders_cache.reduce_size(age_limit=timedelta(90))
    # except Exception as err:
    #     # red(f"Error when reducing cache size: '{err}'")
    #     pass

    # just in case, make sure all modules are loaded
    unlazyload_modules()

    if "path" in cli_kwargs and isinstance(cli_kwargs["path"], str):
        cli_kwargs["path"] = cli_kwargs["path"].strip()

    loading_failure = cli_kwargs["loading_failure"] if "loading_failure" in cli_kwargs else "warn"
    assert loading_failure in [
        "crash", "warn"], f"loading_failure must be either crash or warn. Not {loading_failure}"

    # expand the list of document to load as long as there are recursive types
    to_load = [cli_kwargs.copy()]
    to_load[-1]["filetype"] = filetype
    new_doc_to_load = []
    while any(d["filetype"] in recursive_types for d in to_load):
        for ild, load_kwargs in enumerate(to_load):
            if not ("path" in load_kwargs and load_kwargs["path"]):
                continue
            load_filetype = load_kwargs["filetype"]

            # auto parse filetype if infer
            if load_filetype == "infer":
                for k, v in inference_rules.items():
                    for vv in inference_rules[k]:
                        if vv.search(load_kwargs["path"]):
                            load_filetype = k
                            break
                    if load_filetype != "infer":
                        break
                assert (
                    load_filetype != "infer"
                ), f"Could not infer load_filetype of {load_kwargs['path']}. Use the 'load_filetype' argument."
                to_load[ild]["filetype"] = load_filetype

            if load_filetype == "recursive_paths":
                new_doc_to_load.extend(
                    parse_recursive_paths(load_kwargs)
                )
                break

            elif load_filetype == "json_entries":
                new_doc_to_load.extend(
                    parse_json_entries(load_kwargs)
                )
                break

            elif load_filetype == "link_file":
                new_doc_to_load.extend(
                    parse_link_file(load_kwargs, task=task)
                )
                break

            elif load_filetype == "youtube_playlist":
                new_doc_to_load.extend(
                    parse_youtube_playlist(load_kwargs)
                )
                break

        if new_doc_to_load:
            assert to_load[ild]["filetype"] in recursive_types
            to_load.remove(to_load[ild])
            to_load.extend(new_doc_to_load)
            new_doc_to_load = []
            continue

    # remove duplicate documents
    temp = []
    for d in to_load:
        if d in temp:
            red(f"Removed document {d} (duplicate)")
        else:
            temp.append(d)
    to_load = temp

    assert to_load, f"empty list of documents to load from filetype '{filetype}'"

    if "file_loader_n_jobs" in cli_kwargs:
        n_jobs = cli_kwargs["file_loader_n_jobs"]
        del cli_kwargs["file_loader_n_jobs"]
    else:
        if is_debug:
            n_jobs = 1
        else:
            n_jobs = 10

    # look for unexpected keys that are not relevant to doc loading, because that would
    # skip the cache
    all_unexp_keys = set()
    for doc in to_load:
        to_del = [k for k in doc if k not in doc_kwargs_keys]
        for k in to_del:
            all_unexp_keys.add(k)
            del doc[k]
    # filter out the usual unexpected
    all_unexp_keys = [a for a in all_unexp_keys if a not in [
        "out_file", "file_loader_n_jobs", "loading_failure",
    ]]
    if all_unexp_keys:
        red(f"Found unexpected keys in doc_kwargs: '{all_unexp_keys}'")

    if "summar" not in task:
        # shuffle the list of files to load to make
        # the hashing progress bar more representative
        to_load = sorted(to_load, key=lambda x: random.random())

    # store the file hash in the doc kwarg
    doc_hashes = Parallel(
        n_jobs=-1,
        backend="loky",
    )(delayed(file_hasher)(doc=doc) for doc in tqdm(
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
        # shuffle the list of files again to be random but deterministic: keeping only the digits of each hash
        to_load = sorted(
            to_load,
            key=lambda x: int(
                ''.join(
                    filter(
                        str.isdigit,
                        doc_hashes[to_load.index(x)],
                    )
                )
            )
        )

        # deduplicate files based on hash
        whi("Deduplicating files")
        doc_hash_counts = {h: doc_hashes.count(h) for h in doc_hashes}
        assert len(doc_hashes) == len(to_load)
        n_dupl = 0
        for i, h in enumerate(doc_hashes):
            if doc_hash_counts[h] > 1:
                doc_hash_counts[h] -= 1
                to_load[i] = None
                n_dupl += 1
            else:
                assert doc_hash_counts[h] == 1
        to_load = [tl for tl in to_load if tl is not None]
        if n_dupl:
            red(f"Ignored '{n_dupl}' duplicate files")

    # load_functions are slow to load so loading them here in advance for every file
    if any(
        ("load_functions" in doc and doc["load_functions"])
            for doc in to_load):
        whi("Preloading load_functions")
        for idoc, doc in enumerate(to_load):
            if "load_functions" in doc:
                if doc["load_functions"]:
                    to_load[idoc]["load_functions"] = parse_load_functions(
                        tuple(doc["load_functions"]))

    # wrap doc_loader to cach errors cleanly
    @optional_typecheck
    def load_one_doc_wrapped(**doc_kwargs) -> Union[List[Document], str]:
        try:
            out = load_one_doc(**doc_kwargs)
            return out
        except Exception as err:
            filetype = doc_kwargs["filetype"]
            exc_type, exc_obj, exc_tb = sys.exc_info()
            formatted_tb = '\n'.join(traceback.format_tb(exc_tb))
            red(f"Error when loading doc with filetype {filetype}: '{err}'. "
                f"Arguments: {doc_kwargs}"
                f"\nLine number: {exc_tb.tb_lineno}"
                f"\nFull traceback:\n{formatted_tb}")
            if loading_failure == "crash" or is_debug:
                raise
            elif loading_failure == "warn":
                return str(err)
            else:
                raise ValueError(loading_failure)

    if len(to_load) > 1:
        for tl in to_load:
            assert tl["filetype"] != "string", "You shouldn't not be using filetype 'string' with other kind of documents normally. Please open an issue on github and explain me your usecase to see how I can fix that for you!"

    # dir name where to store temporary files
    load_temp_name = "file_load_" + str(uuid.uuid4())
    # delete previous temp dir if it's several days old
    for f in cache_dir.iterdir():
        f = f.resolve()
        if f.is_dir() and f.name.startswith("file_load_") and (abs(time.time() - f.stat().st_mtime) > 2 * 86400):
            assert str(cache_dir.absolute()) in str(f.absolute())
            shutil.rmtree(f)
    temp_dir = cache_dir / load_temp_name
    temp_dir.mkdir(exist_ok=False)
    loaders_temp_dir_file.write_text(str(temp_dir.absolute().resolve()))

    docs = []
    t_load = time.time()
    if len(to_load) == 1:
        n_jobs = 1
    doc_lists = Parallel(
        n_jobs=n_jobs,
        backend=backend,
    )(delayed(load_one_doc_wrapped)(
        task=task,
        temp_dir=temp_dir,
        **d,
    ) for d in tqdm(
        to_load,
        desc="Loading",
        unit="doc",
        colour="magenta",
    )
    )

    # erases content that links to the loaders temporary files at startup
    loaders_temp_dir_file.write_text("")

    red(f"Done loading all {len(to_load)} documents in {time.time()-t_load:.2f}s")
    missing_docargs = []
    for idoc, d in tqdm(enumerate(doc_lists), total=len(doc_lists), desc="Concatenating results"):
        if isinstance(d, list):
            docs.extend(d)
        else:
            assert isinstance(d, str)
            missing_docargs.append(to_load[idoc])
            missing_docargs[-1]["error_message"] = d
    assert not any(isinstance(d, str) for d in docs)

    if missing_docargs:
        missing_docargs = sorted(missing_docargs, key=lambda x: json.dumps(x))
        red(f"Number of failed documents: {len(missing_docargs)}:")
        missed_recur = []
        for imissed, missed in enumerate(missing_docargs):
            if len(missing_docargs) > 99:
                red(f"- {imissed + 1:03d}]: '{missed}'")
            else:
                red(f"- {imissed + 1:02d}]: '{missed}'")
            if missed["filetype"] in recursive_types:
                missed_recur.append(missed)

        if missed_recur:
            missed_recur = sorted(missed_recur, key=lambda x: json.dumps(x))
            red("Crashing because some recursive filetypes failed:")
            for imr, mr in enumerate(missed_recur):
                red(f"- {imr + 1}]: '{mr}'")
            raise Exception(
                f"{len(missed_recur)} recursive filetypes failed to load.")
    else:
        red("No document failed to load!")

    assert docs, "No documents were succesfully loaded!"

    size = sum(
        [
            get_tkn_length(d.page_content)
            for d in tqdm(
                docs,
                desc="Computing token size",
                unit="doc"
            )
        ]
    )
    if size <= min_token:
        raise Exception(
            f"The number of token is {size} <= {min_token} tokens, probably something went wrong?"
        )

    # delete temp dir
    shutil.rmtree(temp_dir)
    assert not temp_dir.exists()

    # check that the hash are unique
    if len(docs) > 1:
        whi(f"Checking document uniqueness using hash")
        ids = [id(d.metadata) for d in docs]
        assert len(ids) == len(set(ids)), (
            "Same metadata object is used to store information on "
            "multiple documents!")

        hashes = [d.metadata["all_hash"] for d in docs]
        uniq_hashes = list(set(hashes))
        removed_paths = []
        removed_docs = []
        counter = {h: hashes.count(h) for h in uniq_hashes}
        if len(hashes) != len(uniq_hashes):
            red("Found duplicate hashes after loading documents:")

            for i, doc in enumerate(tqdm(docs, desc="Looking for duplicates")):
                h = doc.metadata['all_hash']
                n = counter[h]
                if n > 1:
                    removed_docs.append(docs[i])
                    docs[i] = None
                    counter[h] -= 1
                assert counter[h] > 0
            red(f"Removed {len(removed_docs)}/{len(hashes)} documents because they had the same hash")

            docs = [d for d in docs if d is not None]
    return docs


@optional_typecheck
def parse_recursive_paths(load_kwargs: dict) -> List[dict]:
    load_path = load_kwargs["path"]
    whi(f"Parsing recursive load_filetype: '{load_path}'")
    assert "pattern" in load_kwargs, "missing 'pattern' key in args"
    assert "recursed_filetype" in load_kwargs, "missing 'recursed_filetype' in args"
    assert (
        load_kwargs["recursed_filetype"]
        not in [
            "recursive_paths",
            "json_entries",
            "youtube",
            "anki",
        ]
    ), "'recursed_filetype' cannot be 'recursive_paths', 'json_entries', 'anki' or 'youtube'"
    pattern = load_kwargs["pattern"]

    if not Path(load_path).exists() and Path(load_path.replace(r"\ ", " ")).exists():
        log.info(r"File was not found so replaced '\ ' by ' '")
        load_path = load_path.replace(r"\ ", " ")
    assert Path(load_path).exists, f"not found: {load_path}"
    doclist = [p for p in Path(load_path).rglob(pattern)]
    assert doclist, f"No document found by pattern {pattern}"
    doclist = [str(p).strip() for p in doclist if p.is_file()]
    assert doclist, f"No document after filtering by file"
    doclist = [p for p in doclist if p]
    assert doclist, f"No document after removing nonemtpy"
    doclist = [
        p[1:].strip() if p.startswith("-") else p.strip() for p in doclist
    ]

    if "include" in load_kwargs:
        for i, d in enumerate(doclist):
            keep = True
            for iinc, inc in enumerate(load_kwargs["include"]):
                if isinstance(inc, str):
                    if inc == inc.lower():
                        inc = re.compile(inc, flags=re.IGNORECASE)
                    else:
                        inc = re.compile(inc)
                    load_kwargs["include"][iinc] = inc
                if not inc.search(d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for iexc, exc in enumerate(load_kwargs["exclude"]):
            if isinstance(exc, str):
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                load_kwargs["exclude"][iexc] = exc
            doclist = [d for d in doclist if not exc.search(d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        doc_kwargs = load_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = doc_kwargs["recursed_filetype"]
        del doc_kwargs["recursed_filetype"]
        if "pattern" in doc_kwargs:
            del doc_kwargs["pattern"]
        doclist[i] = doc_kwargs
    return doclist


@optional_typecheck
def parse_json_entries(load_kwargs: dict) -> List[dict]:
    load_path = load_kwargs["path"]
    whi(f"Loading json_entries: '{load_path}'")
    doclist = str(Path(load_path).read_text()).splitlines()
    doclist = [
        p[1:].strip() if p.startswith("-") else p.strip() for p in doclist
    ]
    doclist = [
        p.strip()
        for p in doclist
        if p.strip() and not p.strip().startswith("#")
    ]

    if "include" in load_kwargs:
        for i, d in enumerate(doclist):
            keep = True
            for iinc, inc in enumerate(load_kwargs["include"]):
                if isinstance(inc, str):
                    if inc == inc.lower():
                        inc = re.compile(inc, flags=re.IGNORECASE)
                    else:
                        inc = re.compile(inc)
                    load_kwargs["include"][iinc] = inc
                if not inc.search(d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for iexc, exc in enumerate(load_kwargs["exclude"]):
            if isinstance(exc, str):
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                load_kwargs["exclude"][iexc] = exc
            doclist = [d for d in doclist if not exc.search(d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        meta = json.loads(d.strip())
        assert isinstance(
            meta, dict
        ), f"meta from line '{d}' is not dict but '{type(meta)}'"
        assert "filetype" in meta, "no key 'filetype' in meta"
        for k, v in load_kwargs.items():
            if k not in meta:
                meta[k] = v
        if meta["path"] == load_path:
            del meta["path"]
        doclist[i] = meta
    return doclist


@optional_typecheck
def parse_link_file(load_kwargs: dict, task: str) -> List[dict]:
    load_path = load_kwargs["path"]
    whi(f"Loading link_file: '{load_path}'")
    doclist = str(Path(load_path).read_text()).splitlines()
    doclist = [
        p[1:].strip() if p.startswith("-") else p.strip() for p in doclist
    ]
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

    if "done_links" in load_kwargs:
        # discard any links that are already present in the output
        doclist = [
            d.strip() for d in doclist if d.strip() not in load_kwargs["done_links"]
        ][: load_kwargs["n_summaries_target"]]
        del load_kwargs["done_links"]

    if "include" in load_kwargs:
        for i, d in enumerate(doclist):
            keep = True
            for iinc, inc in enumerate(load_kwargs["include"]):
                if isinstance(inc, str):
                    if inc == inc.lower():
                        inc = re.compile(inc, flags=re.IGNORECASE)
                    else:
                        inc = re.compile(inc)
                    load_kwargs["include"][iinc] = inc
                if not inc.search(d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for iexc, exc in enumerate(load_kwargs["exclude"]):
            if isinstance(exc, str):
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                load_kwargs["exclude"][iexc] = exc
            doclist = [d for d in doclist if not exc.search(d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = load_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["subitem_link"] = d
        doc_kwargs["filetype"] = "infer"
        doclist[i] = doc_kwargs
    return doclist


@optional_typecheck
def parse_youtube_playlist(load_kwargs: dict) -> List[dict]:
    assert "path" in load_kwargs, "missing 'path' key in args"
    path = load_kwargs["path"]
    if "\\" in path:
        red(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")
    whi(f"Loading youtube playlist: '{path}'")
    video = load_youtube_playlist(path)

    load_kwargs["playlist_title"] = video["title"].strip().replace("\n", "")
    assert (
        "duration" not in video
    ), f'"duration" found when loading youtube playlist. This might not be a playlist: {path}'
    doclist = [ent["webpage_url"] for ent in video["entries"]]
    doclist = [li for li in doclist if yt_link_regex.search(li)]

    if "include" in load_kwargs:
        for i, d in enumerate(doclist):
            keep = True
            for iinc, inc in enumerate(load_kwargs["include"]):
                if isinstance(inc, str):
                    if inc == inc.lower():
                        inc = re.compile(inc, flags=re.IGNORECASE)
                    else:
                        inc = re.compile(inc)
                    load_kwargs["include"][iinc] = inc
                if not inc.search(d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for iexc, exc in enumerate(load_kwargs["exclude"]):
            if isinstance(exc, str):
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                load_kwargs["exclude"][iexc] = exc
            doclist = [d for d in doclist if not exc.search(d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = load_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = "youtube"
        doc_kwargs["subitem_link"] = d
        doclist[i] = doc_kwargs

    assert doclist, f"No video found in youtube playlist: {load_kwargs}"
    return doclist


@optional_typecheck
@memoizer
def parse_load_functions(load_functions: Tuple[str, ...]) -> bytes:
    load_functions = list(load_functions)
    assert isinstance(load_functions, list), "load_functions must be a list"
    assert all(isinstance(lf, str)
               for lf in load_functions), "load_functions elements must be strings"

    try:
        for ilf, lf in enumerate(load_functions):
            load_functions[ilf] = eval(lf)
    except Exception as err:
        raise Exception(
            f"Error when evaluating load_functions #{ilf}: {lf} '{err}'")
    load_functions = tuple(load_functions)
    pickled = dill.dumps(load_functions)
    return pickled
