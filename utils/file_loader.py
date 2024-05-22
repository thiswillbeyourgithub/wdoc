from typing import List
from functools import wraps
import random

from langchain.docstore.document import Document
from joblib import Parallel, delayed
from pathlib import Path
import json

from .misc import loaddoc_cache, file_hasher
from .typechecker import optional_typecheck
from .logger import red, whi, log
from .loaders import load_one_doc, yt_link_regex, load_youtube_playlist, markdownlink_regex, min_token, get_tkn_length

import re
from tqdm import tqdm

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
    "online_pdf": ["^http.*pdf.*"],
    "pdf": [".*pdf$"],
    "url": ["^http"],
    "local_html": [r"^(?!http).*\.html?$"],
    "local_audio": [r".*(mp3|m4a|ogg|flac)$"],
    "json_list": [".*.json"],
}

# compile the inference rules as regex
for k, v in inference_rules.items():
    for i, vv in enumerate(v):
        inference_rules[k][i] = re.compile(vv)

doc_kwargs_keys = [
    "path",
    "filetype",
    "file_hash",
    "anki_profile",
    "anki_notetype",
    "anki_fields",
    "anki_deck",
    "anki_mode",
    "whisper_lang",
    "whisper_prompt",
    "youtube_language",
    "youtube_translation",
    "load_functions",
]


@optional_typecheck
def load_doc(filetype: str, debug: bool, task: str, **kwargs) -> List[Document]:
    """load the input"""
    # remove cache files older than 90 days
    try:
        loaddoc_cache.reduce_size(age_limit=timedelta(90))
    except Exception as err:
        # red(f"Error when reducing cache size: '{err}'")
        pass

    if "path" in kwargs and isinstance(kwargs["path"], str):
        kwargs["path"] = kwargs["path"].strip()

    # expand the list of document to load as long as there are recursive types
    recurs_type = ["recursive", "json_list", "link_file", "youtube_playlist", "infer"]
    to_load = [kwargs.copy()]
    to_load[-1]["filetype"] = filetype
    new_doc_to_load = []
    while any(d["filetype"] in recurs_type for d in to_load):
        for ild, load_kwargs in enumerate(to_load):
            if not ("path" in load_kwargs or load_kwargs["path"]):
                continue
            load_filetype = load_kwargs["filetype"]

            # auto parse filetype if infer
            if load_filetype == "infer":
                for k, v in inference_rules.items():
                    for vv in inference_rules[k]:
                        if re.search(vv, load_kwargs["path"]):
                            load_filetype = k
                            break
                    if load_filetype != "infer":
                        break
                assert (
                    load_filetype != "infer"
                ), f"Could not infer load_filetype of {load_kwargs['path']}. Use the 'load_filetype' argument."
                to_load[ild]["filetype"] = load_filetype


            if load_filetype == "recursive":
                new_doc_to_load.extend(
                    parse_recursive(load_kwargs)
                )
                break

            elif load_filetype == "json_list":
                new_doc_to_load.extend(
                    parse_json_list(load_kwargs)
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
            assert to_load[ild]["filetype"] in recurs_type
            to_load.remove(to_load[ild])
            ild_done = None
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

    if "file_loader_n_jobs" in kwargs:
        n_jobs = kwargs["file_loader_n_jobs"]
    else:
        n_jobs = 20
    if len(to_load) == 1 or debug:
        n_jobs = 1

    # look for unexpected keys that are not relevant to doc loading, because that would
    # skip the cache
    all_unexp_keys = set()
    for doc in to_load:
        to_del = [k for k in doc if k not in doc_kwargs_keys]
        for k in to_del:
            all_unexp_keys.add(k)
            del doc[k]
    if all_unexp_keys:
        red(f"Found unexpected keys in doc kwargs: '{all_unexp_keys}'")

    # shuffle the list of files to load to make
    # the progress bar more representative
    to_load = sorted(to_load, key=lambda x: random.random())

    # store the file hash in the doc kwarg
    doc_hashes = Parallel(
        #n_jobs=10,
        backend="threading",
    )(delayed(file_hasher)(doc=doc) for doc in tqdm(
      to_load,
      desc="Hashing files",
      unit="doc",
      colour="magenta",
      )
    )

    # deduplicate files based on hash
    doc_hash_counts = {h: doc_hashes.count(h) for h in doc_hashes}
    assert len(doc_hashes) == len(to_load)
    for i, h in enumerate(doc_hashes):
        if doc_hash_counts[h] > 1:
            doc_hash_counts[h] -= 1
            to_load[i] = None
        else:
            assert doc_hash_counts[h] in [0, 1]
            to_load[i]["file_hash"] = doc_hashes[i]
    to_load = [tl for tl in to_load if tl is not None]

    # wrap doc_loader to cach errors cleanly
    @wraps(load_one_doc)
    def load_one_doc_wrapped(*args, **kwargs):
        try:
            return load_one_doc(*args, **kwargs)
        except Exception as err:
            red(f"Error when loading doc: '{err}'\nArguments: {args}\n{kwargs}")
            return None

    docs = []
    doc_lists = Parallel(
        n_jobs=n_jobs,
        backend="threading",
    )(delayed(load_one_doc_wrapped)(
        debug=debug,
        task=task,
        **d,
        ) for d in tqdm(
            to_load,  # TODO
            desc="Loading",
            unit="doc",
            colour="magenta",
        )
    )
    n_failed = len([d for d in doc_lists if d is None])
    if n_failed:
        red(f"Number of failed documents: {n_failed}")
    [docs.extend(d) for d in doc_lists if d is not None]

    size = sum([get_tkn_length(d.page_content) for d in docs])
    if size <= min_token:
        raise Exception(
            f"The number of token is {size} <= {min_token} tokens, probably something went wrong?"
        )

    return docs

@optional_typecheck
def parse_recursive(load_kwargs: dict) -> List[dict]:
    load_path = load_kwargs["path"]
    whi(f"Parsing recursive load_filetype: '{load_path}'")
    assert "pattern" in load_kwargs, "missing 'pattern' key in args"
    assert "recursed_filetype" in load_kwargs, "missing 'recursed_filetype' in args"
    assert (
        load_kwargs["recursed_filetype"]
        not in [
            "recursive",
            "json_list",
            "youtube",
            "anki",
        ]
    ), "'recursed_filetype' cannot be 'recursive', 'json_list', 'anki' or 'youtube'"
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
            for inc in load_kwargs["include"]:
                if not re.search(inc, d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for exc in load_kwargs["exclude"]:
            doclist = [d for d in doclist if not re.search(exc, d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        doc_kwargs = load_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = doc_kwargs["recursed_filetype"]
        del doc_kwargs["recursed_filetype"]
        if "file_loader_n_jobs" in doc_kwargs:
            del doc_kwargs["file_loader_n_jobs"]
        if "pattern" in doc_kwargs:
            del doc_kwargs["pattern"]
        doclist[i] = doc_kwargs
    return doclist

@optional_typecheck
def parse_json_list(load_kwargs: dict) -> List[dict]:
    load_path = load_kwargs["path"]
    whi(f"Loading json_list: '{load_path}'")
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
            for inc in load_kwargs["include"]:
                if not re.search(inc, d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for exc in load_kwargs["exclude"]:
            doclist = [d for d in doclist if not re.search(exc, d)]
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
        if "file_loader_n_jobs" in meta:
            del meta["file_loader_n_jobs"]
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
        re.findall(markdownlink_regex, d)[0]
        if re.search(markdownlink_regex, d)
        else d
        for d in doclist
    ]
    if task == "summarize_link_file":
        # if summarize, start from bottom
        doclist.reverse()

    if "done_links" in load_kwargs:
        # discard any links that are already present in the output
        doclist = [
            d.strip() for d in doclist if d.strip() not in load_kwargs["done_links"]
        ][: load_kwargs["n_summaries_target"]]
        del load_kwargs["done_links"]

    if "include" in load_kwargs:
        for i, d in enumerate(doclist):
            keep = True
            for inc in load_kwargs["include"]:
                if not re.search(inc, d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for exc in load_kwargs["exclude"]:
            doclist = [d for d in doclist if not re.search(exc, d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = load_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["subitem_link"] = d
        doc_kwargs["filetype"] = "infer"
        if "file_loader_n_jobs" in doc_kwargs:
            del doc_kwargs["file_loader_n_jobs"]
        doclist[i] = doc_kwargs
    return doclist

@optional_typecheck
def parse_youtube_playlist(load_kwargs: dict) -> List[dict]:
    assert "path" in load_kwargs, "missing 'path' key in args"
    path = load_kwargs["path"]
    whi(f"Loading youtube playlist: '{path}'")
    video = load_youtube_playlist(path)

    load_kwargs["playlist_title"] = video["title"].strip().replace("\n", "")
    assert (
        "duration" not in video
    ), f'"duration" found when loading youtube playlist. This might not be a playlist: {path}'
    doclist = [ent["webpage_url"] for ent in video["entries"]]
    doclist = [li for li in doclist if re.search(yt_link_regex, li)]

    if "include" in load_kwargs:
        for i, d in enumerate(doclist):
            keep = True
            for inc in load_kwargs["include"]:
                if not re.search(inc, d):
                    keep = False
            if not keep:
                doclist[i] = None
        doclist = [d for d in doclist if d]
        del load_kwargs["include"]

    if "exclude" in load_kwargs:
        for exc in load_kwargs["exclude"]:
            doclist = [d for d in doclist if not re.search(exc, d)]
        del load_kwargs["exclude"]

    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = load_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = "youtube"
        doc_kwargs["subitem_link"] = d
        if "file_loader_n_jobs" in doc_kwargs:
            del doc_kwargs["file_loader_n_jobs"]
        doclist[i] = doc_kwargs
    return doclist

