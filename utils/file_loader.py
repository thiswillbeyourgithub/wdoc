from functools import partial
import tldextract
import uuid
import threading
import queue
import pdb
import time
import tempfile
import requests
import youtube_dl
from youtube_dl.utils import DownloadError, ExtractorError
import random
import shutil
import ankipandas as akp
import ftfy
from bs4 import BeautifulSoup
from goose3 import Goose
from pathlib import Path
import re
from tqdm import tqdm
import json
from prompt_toolkit import prompt
import tiktoken

try:
    from ftlangdetect import detect as language_detect
except Exception as err:
    print(f"Couldn't import ftlangdetect: '{err}'")
try:
    import pdftotext
except Exception as err:
    print(f"Failed to import pdftotext: '{err}'")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import WebBaseLoader

from unstructured.cleaners.core import clean_extra_whitespace

import LogseqMarkdownParser

from .misc import loaddoc_cache, html_to_text, hasher
from .logger import whi, yel, red, log
from .llm import transcribe

import os
# needed in case of buggy unstructured install
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
        "local_audio": [r".*(mp3|m4a|ogg|flac)$"],
        "json_list": [".*.json"],
        }

# compile the inference rules as regex
for k, v in inference_rules.items():
    for i, vv in enumerate(v):
        inference_rules[k][i] = re.compile(vv)

# for reading length estimation
wpm = 250
average_word_length = 6

clozeregex = re.compile(r"{{c\d+::|}}")  # for removing clozes in anki
markdownlink_regex = re.compile(r'\[.*?\]\((.*?)\)')  # to parse markdown links"
yt_link_regex = re.compile("youtube.*watch")  # to check that a youtube link is valid
emptyline_regex = re.compile(r'^\s*$', re.MULTILINE)
emptyline2_regex = re.compile(r'\n\n+', re.MULTILINE)
linebreak_before_letter = re.compile(r'\n([a-záéíóúü])', re.MULTILINE)  # match any linebreak that is followed by a lowercase letter

tokenize = tiktoken.encoding_for_model("gpt-3.5-turbo").encode  # used to get token length estimation

max_threads = 5
threads = {}
lock = threading.Lock()
n_recursive = 0  # global var to keep track of the number of recursive loading threads. If there are many recursions they can actually get stuck

min_token = 200
max_token = 1_000_000
max_lines = 100_000
min_lang_prob = 0.50


def get_tkn_length(tosplit):
    return len(tokenize(tosplit))


def get_splitter(task):
    "we don't use the same text splitter depending on the task"
    if task in ["query", "search"]:
        text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
                chunk_size=3000,  # default 4000
                chunk_overlap=386,  # default 200
                length_function=get_tkn_length,
                )
    elif task in ["summarize_link_file", "summarize_then_query", "summarize"]:
        text_splitter = RecursiveCharacterTextSplitter(
                separators=[".\n", ". ", " ", ""],
                chunk_size=2000,
                chunk_overlap=300,
                length_function=get_tkn_length,
                )
    elif task == "recursive_summary":
        text_splitter = RecursiveCharacterTextSplitter(
                separators=[".\n", ". ", " ", ""],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=get_tkn_length,
                )
    else:
        raise Exception(task)
    return text_splitter


def cloze_stripper(clozed):
    clozed = re.sub(clozeregex, " ", clozed)
    return clozed


def check_docs_tkn_length(docs, name):
    """checks that the number of tokens in the document is high enough,
    not too low, and has a high enough language probability,
    otherwise something probably went wrong."""
    size = sum([get_tkn_length(d.page_content) for d in docs])
    nline = len("\n".join([d.page_content for d in docs]).splitlines())
    if nline > max_lines:
        red(f"Example of page from document with too many lines : {docs[len(docs)//2].page_content}")
        raise Exception(f"The number of lines from '{name}' is {nline} > {max_lines}, probably something went wrong?")
    if size <= min_token:
        red(f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}")
        raise Exception(f"The number of token from '{name}' is {size} <= {min_token}, probably something went wrong?")
    if size >= max_token:
        red(f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}")
        raise Exception(f"The number of token from '{name}' is {size} >= {max_token}, probably something went wrong?")

    # check if language check is above a threshold
    if "language_detect" not in globals():
        # bypass if language_detect not imported
        return 1
    prob = language_detect(docs[0].page_content.replace("\n", "<br>"))["score"]
    if len(docs) > 1:
        prob += language_detect(docs[-1].page_content.replace("\n", "<br>"))["score"]
        if len(docs) > 2:
            prob += language_detect(docs[len(docs)//2].page_content.replace("\n", "<br>"))["score"]
            prob /= 3
        else:
            prob /= 2
    if prob <= min_lang_prob:
        red(f"Low language probability for {name}: prob={prob}<{min_lang_prob}.\nExample page: {docs[len(docs)//2]}")
        raise Exception(f"Low language probability for {name}: prob={prob}.\nExample page: {docs[len(docs)//2]}")
    return prob


def get_url_title(url):
    """if the title of the url is not loaded from the loader, trying as last
    resort with this one"""
    loader = WebBaseLoader(url, raise_for_status=True)
    docs = loader.load()
    if "title" in docs[0].metadata and docs[0].metadata["title"]:
        return docs[0].metadata["title"]
    else:
        return None


def load_doc(filetype, debug, task, **kwargs):
    """load the input"""
    text_splitter = get_splitter(task)

    if "path" in kwargs and isinstance(kwargs["path"], str):
        kwargs["path"] = kwargs["path"].strip()

    if filetype == "infer":
        assert "path" in kwargs, "if filetype is infer, path should be supplied"
        for k, v in inference_rules.items():
            for vv in inference_rules[k]:
                if re.search(vv, kwargs["path"]):
                    filetype = k
                    break
            if filetype != "infer":
                break
        assert filetype != "infer", f"Could not infer filetype of {kwargs['path']}. Use the 'filetype' argument."

    if filetype in ["json_list", "recursive", "link_file", "youtube_playlist"]:
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]

        if filetype == "recursive":
            whi(f"Loading recursive filetype: '{path}'")
            assert "pattern" in kwargs, "missing 'pattern' key in args"
            assert "recursed_filetype" in kwargs, "missing 'recursed_filetype' in args"
            assert kwargs["recursed_filetype"] not in [
                    "recursive", "json_list", "youtube", "anki",
                    ], "'recursed_filetype' cannot be 'recursive', 'json_list', 'anki' or 'youtube'"
            pattern = kwargs["pattern"]

            doclist = [p for p in Path(path).rglob(pattern)]
            doclist = [str(p).strip() for p in doclist if p.is_file()]
            doclist = [p for p in doclist if p]
            doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]

            # randomize order to even out the progress bar
            doclist = sorted(doclist, key=lambda x: random.random())

            def threaded_load_item(filetype, item, kwargs, pbar, q, lock):
                kwargs["path"] = item
                kwargs["filetype"] = kwargs["recursed_filetype"]
                assert Path(kwargs["path"]).exists(), f"file '{item}' does not exist"
                del kwargs["pattern"]
                try:
                    res = load_doc(
                            task=task,
                            debug=debug,
                            **kwargs,
                            )
                    with lock:
                        pbar.update(1)
                        q.put(res)
                    return res
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    if debug:
                        pdb.post_mortem()
                    else:
                        with lock:
                            pbar.update(1)
                            q.put(f"{item}: {err}")
                        return item

        elif filetype == "json_list":
            whi(f"Loading json_list: '{path}'")
            doclist = str(Path(path).read_text()).splitlines()
            doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
            doclist = [p.strip() for p in doclist if p.strip() and not p.strip().startswith("#")]

            def threaded_load_item(filetype, item, kwargs, pbar, q, lock):
                meta = json.loads(item.strip())
                for k, v in kwargs.items():
                    if k not in meta:
                        meta[k] = v
                assert isinstance(meta, dict), f"meta from line '{item}' is not dict but '{type(meta)}'"
                assert "filetype" in meta, "no key 'filetype' in meta"
                try:
                    res = load_doc(
                            task=task,
                            debug=debug,
                            **meta,
                            )
                    with lock:
                        pbar.update(1)
                        q.put(res)
                    return res
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    if debug:
                        pdb.post_mortem()
                    else:
                        with lock:
                            pbar.update(1)
                            q.put(f"{item}: {err}")
                        return item

        elif filetype == "link_file":
            whi(f"Loading link_file: '{path}'")
            doclist = str(Path(path).read_text()).splitlines()
            doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
            doclist = [p.strip() for p in doclist if p.strip() and not p.strip().startswith("#") and "http" in p]
            doclist = [re.findall(markdownlink_regex, d)[0] if re.search(markdownlink_regex, d) else d for d in doclist]
            if task == "summarize_link_file":
                # if summarize, start from bottom
                doclist.reverse()

            if "done_links" in kwargs:
                # discard any links that are already present in the output
                doclist = [d.strip() for d in doclist if d.strip() not in kwargs["done_links"]][:kwargs["n_summaries_target"]]
                del kwargs["done_links"]

            def threaded_load_item(filetype, item, kwargs, pbar, q, lock):
                kwargs["path"] = item
                if "http" not in item:
                    red(f"item does not appear to be a link: '{item}'")
                    q.put(f"{item}: does not appear to be a link")
                    return item
                kwargs["filetype"] = "infer"
                kwargs["subitem_link"] = item
                try:
                    res = load_doc(
                            task=task,
                            debug=debug,
                            **kwargs,
                            )
                    with lock:
                        pbar.update(1)
                        q.put(res)
                    return res
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    if debug:
                        pdb.post_mortem()
                    else:
                        with lock:
                            pbar.update(1)
                            q.put(f"{item}: {err}")
                        return item

        elif filetype == "youtube_playlist":
            assert "path" in kwargs, "missing 'path' key in args"
            path = kwargs["path"]
            whi(f"Loading youtube playlist: '{path}'")
            video = load_youtube_playlist(path)

            kwargs["playlist_title"] = video['title'].strip().replace("\n", "")
            assert "duration" not in video, f'"duration" found when loading youtube playlist. This might not be a playlist: {path}'
            doclist = [ent["webpage_url"] for ent in video["entries"]]
            doclist = [li for li in doclist if re.search(yt_link_regex, li)]

            def threaded_load_item(filetype, item, kwargs, pbar, q, lock):
                kwargs["path"] = item
                assert "http" in item, f"item does not appear to be a link: '{item}'"
                kwargs["filetype"] = "youtube"
                kwargs["subitem_link"] = item
                try:
                    res = load_doc(
                            task=task,
                            debug=debug,
                            **kwargs,
                            )
                    with lock:
                        pbar.update(1)
                        q.put(res)
                    return res
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    if debug:
                        pdb.post_mortem()
                    else:
                        with lock:
                            pbar.update(1)
                            q.put(f"{item}: {err}")
                        return item

        else:
            raise ValueError(filetype)

        if "include" in kwargs:
            for i, d in enumerate(doclist):
                keep = True
                for inc in kwargs["include"]:
                    if not re.search(inc, d):
                        keep = False
                if not keep:
                    doclist[i] = None
            doclist = [d for d in doclist if d]
            del kwargs["include"]

        if "exclude" in kwargs:
            for exc in kwargs["exclude"]:
                doclist = [d for d in doclist if not re.search(exc, d)]
            del kwargs["exclude"]

        # remove duplicate documents
        temp = []
        for d in doclist:
            if d in temp:
                red(f"Removed document {d} (duplicate)")
            else:
                temp.append(d)
        doclist = temp

        assert doclist, f"empty list of documents to load from filetype '{filetype}'"

        q = queue.Queue()
        global threads, lock, n_recursive

        if "depth" in kwargs:
            depth = kwargs["depth"]
            kwargs["depth"] += 1
        else:
            depth = 0
            kwargs["depth"] = 1

        # if debugging, don't multithread
        if (not debug) and (depth > 0):
            message = f"Loading documents using {max_threads} threads (depth={depth})"
            pbar = tqdm(total=len(doclist), desc=message)
            recursion_id = str(uuid.uuid4())
            with lock:
                n_recursive += 1

            class thread_args(dict):
                """used to store the arguments used to create the thread and
                create it at the last minute"""
                _is_started = False
                _recursion_id = recursion_id
                _depth = depth

            for doc in doclist:
                thread = thread_args()
                thread["target"] = threaded_load_item
                thread["args"] = (filetype, doc, kwargs.copy(), pbar, q, lock)
                thread["daemon"] = True  # exit when the main program exits
                thread._recursion_id = recursion_id
                assert doc not in threads, f"{doc} already present as thread"
                threads[doc] = thread

            # waiting for threads to finish
            with lock:
                n_threads_alive = sum([t.is_alive() for t in threads.values() if t._is_started])
                n_subthreads_alive = sum([t.is_alive() for t in threads.values() if t._is_started and t._recursion_id == recursion_id])
                n_subthreads_todo = len([t for t in threads.values() if not t._is_started and t._recursion_id == recursion_id])
            i = 0
            while n_subthreads_alive or n_subthreads_todo:

                with lock:
                    n_subthreads_alive = sum([t.is_alive() for t in threads.values() if t._is_started and t._recursion_id == recursion_id])
                    n_threads_alive = sum([t.is_alive() for t in threads.values() if t._is_started])
                    n_subthreads_todo = len([t for t in threads.values() if not t._is_started and t._recursion_id == recursion_id])

                    if n_threads_alive < max_threads + n_recursive and n_subthreads_todo:
                        # launch one more thread
                        docid = [docid for docid, t in threads.items() if not t._is_started and t._recursion_id == recursion_id][0]
                        assert isinstance(threads[docid], dict)
                        threads[docid] = threading.Thread(**threads[docid])
                        threads[docid].start()
                        threads[docid]._is_started = True
                        threads[docid]._recursion_id = recursion_id
                        threads[docid]._depth = depth
                        continue

                time.sleep(0.1)

                # display progress every 10s
                i += 1
                if i % 100 == 0:
                    with lock:
                        doc_print = []
                        to_del = []
                        for k, v in threads.items():
                            if v._recursion_id != recursion_id:
                                continue
                            if v._is_started and v.is_alive():
                                doc_print.append(k)
                            elif v._is_started and not v.is_alive():
                                to_del.append(k)
                        for k in to_del:
                            del threads[k]
                    for ii, d in enumerate(doc_print):
                        d = d.strip()
                        if d.startswith("http"):  # print only domain name
                            doc_print[ii] = tldextract.extract(d).registered_domain
                            continue
                        if d.startswith("{") and d.endswith("}"):
                            # print only path if recursive
                            try:
                                doc_print[ii] = json.loads(d)["path"].replace("../", "")
                                continue
                            except:
                                try:  # for other recursion, show all key:values
                                    temp = json.loads(d)
                                    doc_print[ii] = ""
                                    for k, v in temp.items():
                                        doc_print[ii] += f"{k}:{v},"
                                    doc_print[ii] = doc_print[ii][:-1]  # remove comma
                                except:
                                    pass
                        if "/" in d:
                            # print filename
                            try:
                                doc_print[ii] = Path(d).name
                                continue
                            except:
                                pass
                    whi(f"(Depth={depth}) Waiting for {n_subthreads_alive} / {n_subthreads_todo + n_subthreads_alive} subthreads to finish: {','.join(doc_print)}")

            # check that all its subthreads are done
            with lock:
                assert sum([t.is_alive() for t in threads.values() if t._is_started and t._recursion_id == recursion_id]) == 0
                assert len([t for t in threads.values() if not t._is_started and t._recursion_id == recursion_id]) == 0

                # remove old finished threads
                threads = {k: t for k, t in threads.items() if t._recursion_id != recursion_id}

            # get the values from the queue
            results = []
            failed = []
            while not q.empty():
                doc = q.get()
                if not isinstance(doc, str):
                    results.append(doc)
                else:
                    # when failed: we returned the name of the item
                    failed.append(doc)
        else:
            if debug:
                message = "Loading documents without multithreading because debug is on"
            else:
                message = f"Loading documents without multithreading (depth={depth})"
            pbar = tqdm(total=len(doclist), desc=message)
            temp = []
            for doc in doclist:
                res = threaded_load_item(
                        filetype,
                        doc,
                        kwargs.copy(),
                        pbar,
                        q,
                        lock,
                        )
                temp.append(res)

            # get the values from the queue
            results = []
            failed = []
            for doc in temp:
                if not isinstance(doc, str):
                    results.append(doc)
                else:
                    # when failed: we returned the name of the item
                    failed.append(doc)

        n = len(doclist) - len(results)
        if depth == 0 and failed:
            red(f"List of {n} failed documents:\n")
            for f in sorted(failed):
                red(f"* {f}")
        assert results, "Empty results after loading documents"
        assert n == len(failed), "Unexpected number of failed documents"
        docs = []
        [docs.extend(x) for x in results if x]

        pbar.close()

        size = sum([get_tkn_length(d.page_content) for d in docs])
        if size <= min_token:
            raise Exception(f"The number of token from '{path}' is {size} <= {min_token} tokens, probably something went wrong?")

    elif filetype == "youtube":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        if "\\" in path:
            red(f"Removed backslash found in '{path}'")
            path = path.replace("\\", "")
        assert re.search(yt_link_regex, path), f"youtube link is not valid: '{path}'"
        if "language" not in kwargs:
            lang = ["fr-FR", "fr", "en", "en-US", "en-UK"]
        else:
            lang = kwargs["language"]
        if "translation" in kwargs:
            transl = kwargs["translation"]
        else:
            transl = None

        whi(f"Loading youtube: '{path}'")
        fyu = YoutubeLoader.from_youtube_url
        docs = cached_yt_loader(
                loader=fyu,
                path=path,
                add_video_info=True,
                language=lang,
                translation=transl,
                )
        docs = text_splitter.transform_documents(docs)

    elif filetype == "online_pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading online pdf: '{path}'")

        try:
            loader = OnlinePDFLoader(path)
            docs = loader.load()
            docs = text_splitter.transform_documents(docs)
            check_docs_tkn_length(docs, path)

        except Exception as err:
            red(f"Failed parsing online PDF {path} using only OnlinePDFLoader. Will try downloading it directly.")

            response = requests.get(path)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()

            meta = kwargs.copy()
            meta["filetype"] = "pdf"
            meta["path"] = temp_file.name
            try:
                return load_doc(
                        task=task,
                        debug=debug,
                        **meta,
                        )
            except Exception as err:
                red(f"Error when parsing online pdf from {path} downloaded to {temp_file.name}: '{err}'")
                raise

    elif filetype == "pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading pdf: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"

        docs = cached_pdf_loader(
                path=path,
                text_splitter=text_splitter,
                splitter_chunk_size=text_splitter._chunk_size,
                debug=debug
                )

    elif filetype == "anki":
        for nk in ["anki_deck", "anki_notetype", "anki_profile", "anki_fields"]:
            assert nk in kwargs, f"Missing '{nk}' in arguments from load_doc"
        profile = kwargs["anki_profile"]
        deck = kwargs["anki_deck"]
        notetype = kwargs["anki_notetype"]
        fields = kwargs["anki_fields"]
        if "anki_mode" not in kwargs:
            anki_mode = "window_single_note"
        else:
            anki_mode = kwargs["anki_mode"]
        assert anki_mode.replace("window", "").replace("concatenate", "").replace("single_note", "").replace("_", "") == "", f"Unexpected anki_mode: {anki_mode}"

        whi(f"Loading anki profile: '{profile}'")
        original_db = akp.find_db(user=profile)
        name = f"{profile}".replace(" ", "_")
        random_val = str(uuid.uuid4()).split("-")[-1]
        new_db_path = f"./.cache/anki_collection_{name.replace('/', '_')}_{random_val}"
        assert not Path(new_db_path).exists(), f"{new_db_path} already existing!"
        shutil.copy(original_db, new_db_path)
        col = akp.Collection(path=new_db_path)
        cards = col.cards.merge_notes()

        cards.loc[cards['codeck']=="", 'codeck'] = cards['cdeck'][cards['codeck']==""]
        cards["codeck"] = cards["codeck"].apply(lambda x: x.replace("\x1f", "::"))
        cards = cards[cards["codeck"].str.startswith(deck)]
        cards["nmodel"] = cards["nmodel"].apply(lambda x: x.lower())
        cards = cards[cards["nmodel"].str.startswith(notetype)]

        cards["mid"] = col.cards.mid.loc[cards.index]
        mid2fields = akp.raw.get_mid2fields(col.db)
        mod2mid = akp.raw.get_model2mid(col.db)
        cards["fields_name"] = cards["mid"].apply(lambda x: mid2fields[x])
        assert cards.index.tolist(), "empty dataframe!"
        cards["fields_dict"] = cards.apply(
                lambda x: {
                    k: html_to_text(cloze_stripper(v), issoup=False).strip()
                    for k, v in zip(x["fields_name"], x["nflds"])
                    if k.lower() in fields
                    },
                axis=1)
        cards["text"] = cards["fields_dict"].apply(
            lambda x: "\n".join(
                f"{k}: {x[k]}" for k in fields
                if x[k]
                ))
        cards = cards[~cards["text"].str.contains("[IMAGE]")]
        cards["text"] = cards["text"].apply(lambda x: x.strip())
        cards.drop_duplicates(subset="text", inplace=True)

        cards = cards.sort_index()

        docs = []

        if "single_note" in anki_mode:
            # load each card as a single document
            for cid in cards.index:
                c = cards.loc[cid, :]
                docs.append(
                        Document(
                            page_content=c["text"],
                            metadata={
                                "anki_tags": " ".join(c["ntags"]),
                                "anki_cid": str(cid),
                                }
                            )
                        )

        if "concatenate" in anki_mode:
            # # turn all cards into a single wall of text then use text_splitter
            # pro: fill the context window as much I possible I guess
            # con: - editing cards will force re-embedding a lot of cards
            #      - ignores tags
            chunksize = text_splitter._chunk_size
            full_text = ""
            spacer = "\n\n#####\n\n"
            metadata = {"anki_tags": "", "anki_cid": "", "anki_deck": ""}
            for cid in sorted(cards.index):
                c = cards.loc[cid, :]
                cid = str(cid)
                tags = c["ntags"]
                text = ftfy.fix_text(c["text"].strip())
                card_deck = c["codeck"]
                assert card_deck, f"empty card_deck for cid {cid}"

                if not full_text:  # always add first
                    full_text = text
                    metadata = {"anki_tags": " ".join(tags), "anki_cid": str(cid), "anki_deck": card_deck}
                    continue

                # if too many token, add the current chunk of text and start
                # the next chunk with this card
                if get_tkn_length(full_text + spacer + text) >= chunksize:
                    assert full_text, f"An anki card is too large for the text splitter: {text}"
                    assert metadata["anki_cid"], "No anki_cid in metadata"
                    docs.append(
                            Document(
                                page_content=full_text,
                                metadata=metadata,
                                )
                            )

                    metadata = {"anki_tags": " ".join(tags), "anki_cid": str(cid), "anki_deck": card_deck}
                    full_text = text
                else:
                    for t in tags:
                        if t not in metadata["anki_tags"]:
                            metadata["anki_tags"] += f" {t}"
                    metadata["anki_cid"] += " " + str(cid)
                    if card_deck not in metadata["anki_deck"]:
                        metadata["anki_deck"] += " " + card_deck
                    full_text += spacer + text

            if full_text:  # add latest chunk
                docs.append(
                        Document(
                            page_content=full_text,
                            metadata=metadata,
                            )
                        )

        if "window" in anki_mode:
            # # set window_size to X turn each X cards into one document, overlapping
            window_size = 5
            index_list = cards.index.tolist()
            n = len(index_list)
            cards["text_concat"] = ""
            cards["tags_concat"] = ""
            cards["cids"] = ""
            cards["ntags_t"] = cards["ntags"].apply(lambda x: " ".join(x))
            for i in tqdm(range(len(index_list)), desc="combining anki cards"):
                text_concat = ""
                tags_concat = ""
                cids = ""
                skip = 0
                for w in range(0, window_size):
                    if i + window_size + skip >= n:
                        s = -1  # when at the end of the list, apply the window in reverse
                        # s for 'sign'
                    else:
                        s = 1
                    if cards.at[index_list[i+w*s], "text"] in cards.at[index_list[i], "text_concat"]:
                        # skipping this card because it's a duplicate
                        skip += 1
                    text_concat += "\n\n" + cards.at[index_list[i+(w+skip)*s], "text"]
                    tags_concat += cards.at[index_list[i+(w+skip)*s], "ntags_t"]
                    cids += f"{index_list[i+(w+skip)*s]} "
                cards.at[index_list[i], "text_concat"] = text_concat
                cards.at[index_list[i], "tags_concat"] = tags_concat
                cards.at[index_list[i], "cids"] = cids

            for cid in sorted(cards.index):
                c = cards.loc[cid, ]
                docs.append(
                        Document(
                            page_content=c["text_concat"].strip(),
                            metadata={
                                "anki_tags": " ".join(list(set(c["tags_concat"].split(" ")))),
                                "anki_cid": c["cids"].strip(),
                                }
                            )
                        )

        assert docs, "List of loaded anki document is empty!"

        path = f"Anki_profile='{profile}',deck='{deck}'notetype={notetype},fields={','.join(fields)}"
        for i in range(len(docs)):
            docs[i].metadata["anki_profile"] = profile
            docs[i].metadata["anki_topdeck"] = deck
            docs[i].metadata["anki_notetype"] = notetype
            docs[i].metadata["path"] = path
            docs[i].metadata["anki_tags"] = " ".join(sorted(list(set(docs[i].metadata["anki_tags"].split(" ")))))
            docs[i].metadata["anki_cid"] = " ".join(sorted(docs[i].metadata["anki_cid"].split(" ")))

        # try:
        #     check_docs_tkn_length(docs, f"{filetype}: {profile}")
        # except Exception as err:
        #     red(f"Number of token in anki document is surprising. Not quitting because anki can cause this: '{err}'")

        # delete temporary db file
        Path(new_db_path).unlink()
        Path(new_db_path + "-shm").unlink(missing_ok=True)
        Path(new_db_path + "-wal").unlink(missing_ok=True)

    elif filetype == "string":
        whi("Loading string")
        content = prompt(
                "Paste your text content here then press esc+enter or meta+enter:\n>",
                multiline=True,
                )
        log.info(f"Pasted string input:\n{content}")
        texts = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in texts]
        path = "user_string"

    elif filetype == "txt":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading txt: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        with open(path) as f:
            content = f.read()
        texts = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in texts]
        check_docs_tkn_length(docs, path)

    elif filetype == "logseq_markdown":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading logseq markdown file: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        parsed = LogseqMarkdownParser.parse_file(path, verbose=debug)
        blocks = parsed.blocks

        # group blocks by parent block
        pblocks = []
        for b in blocks:
            if b.indentation_level == 0:
                pblocks.append([b])
            else:
                pblocks[-1].append(b)
        whi(f"Found {len(pblocks)} parent blocks")

        page_props = parsed.page_property
        if not page_props:
            page_props = {}
        else:
            lines = page_props.splitlines()
            page_props = {}
            for li in lines:
                li = li.strip()
                li = li.replace("- ", "", 1)
                li = li.split(":: ")
                page_props[li[0]] = li[1]

        docs = []
        for grou in pblocks:
            # store in metadata the properties of the blocks inside a given
            # parent block
            meta = page_props.copy()
            for b in grou:
                print(b)
                for k, v in b.get_properties().items():
                    meta[k] = v
            doc = Document(
                    page_content="\n".join([b.content for b in grou]),
                    metadata=meta,
                    )
            docs.append(doc)

        check_docs_tkn_length(docs, path)

    elif filetype == "local_audio":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        assert Path(path).exists(), f"file not found: '{path}'"
        cache_transcribe = loaddoc_cache.cache(transcribe, ignore=["audio_path"])
        assert "whisper_lang" in kwargs, (
            f"No whisper_lang argument found in kwargs but is needed "
            f"to transcribe '{path}'")
        assert "whisper_prompt" in kwargs, (
            f"No whisper_prompt argument found in kwargs but is needed "
            f"to transcribe '{path}'")

        # get audio hash
        with open(path, "rb") as f:
            audio_hash = hasher(str(f.read()))

        content = cache_transcribe(
                audio_path=path,
                audio_hash=audio_hash,
                language=kwargs["whisper_lang"],
                prompt=kwargs["whisper_prompt"],
                )
        texts = text_splitter.split_text(content["text"])
        docs = [
                Document(
                    page_content=t,
                    metadata={
                        "duration": content["duration"],
                        "language": content["language"],
                        "whisper_task": content["task"],
                        "source": path,
                        },
                    )
                for t in texts]
        check_docs_tkn_length(docs, path)

    elif filetype == "url":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading url: '{path}'")

        # even if loading fails the title might be found so trying to keep
        # the first working title across trials
        if "title" not in kwargs or kwargs["title"] == "Untitled":
            title = None
        else:
            title = kwargs["title"]

        # try with playwright
        try:
            loader = PlaywrightURLLoader(urls=[path], remove_selectors=["header", "footer"])
            docs = text_splitter.transform_documents(loader.load())
            assert docs, f"Empty docs when using playwright"
            if not title and "title" in docs[0].metadata:
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)

        # try with selenium firefox
        except Exception as err:
            red(f"Exception when using playwright to parse text: '{err}'\nUsing selenium firefox as fallback")
            time.sleep(1)
            try:
                loader = SeleniumURLLoader(urls=[path], browser="firefox")
                docs = text_splitter.transform_documents(loader.load())
                assert docs, f"Empty docs when using selenium firefox"
                if not title and "title" in docs[0].metadata and docs[0].metadata["title"] != "No title found.":
                    title = docs[0].metadata["title"]
                check_docs_tkn_length(docs, path)

            # try with selenium chrome
            except Exception as err:
                red(f"Exception when using selenium firefox to parse text: '{err}'\nUsing selenium chrome as fallback")
                time.sleep(1)
                try:
                    loader = SeleniumURLLoader(urls=[path], browser="chrome")
                    docs = text_splitter.transform_documents(loader.load())
                    assert docs, f"Empty docs when using selenium chrome"
                    if not title and "title" in docs[0].metadata and docs[0].metadata["title"] != "No title found.":
                        title = docs[0].metadata["title"]
                    check_docs_tkn_length(docs, path)

                # try with goose
                except Exception as err:
                    red(f"Exception when using selenium chrome to parse text: '{err}'\nUsing goose as fallback")
                    time.sleep(1)
                    try:
                        g = Goose()
                        article = g.extract(url=path)
                        text = article.cleaned_text
                        texts = text_splitter.split_text(text)
                        docs = [Document(page_content=t) for t in texts]
                        assert docs, f"Empty docs when using goose"
                        if not title:
                            if "title" in docs[0].metadata and docs[0].metadata["title"]:
                                title = docs[0].metadata["title"]
                            elif article.title:
                                title = article.title
                        check_docs_tkn_length(docs, path)

                    # try with html
                    except Exception as err:
                        red(f"Exception when using goose to parse text: '{err}'\nUsing html as fallback")
                        time.sleep(1)
                        loader = WebBaseLoader(path, raise_for_status=True)
                        docs = text_splitter.transform_documents(loader.load())
                        assert docs, f"Empty docs when using html"
                        if not title and "title" in docs[0].metadata and docs[0].metadata["title"]:
                            title = docs[0].metadata["title"]
                        check_docs_tkn_length(docs, path)

        # last resort, try to get the title from the most basic loader
        if not title:
            title = get_url_title(path)

        # store the title as metadata if missing
        if title:
            for d in docs:
                if "title" not in d.metadata or not d.metadata["title"]:
                    d.metadata["title"] = title
                else:
                    if d.metadata["title"] != title:
                        d.metadata["title"] = f"{title} - {d.metadata['title']}"

    else:
        raise Exception(red(f"Unsupported filetype: '{filetype}'"))

    # add and format metadata
    total_reading_length = None
    for i in range(len(docs)):
        # if html, parse it
        soup = BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = html_to_text(soup, issoup=True)

        # fix text just in case
        docs[i].page_content = ftfy.fix_text(docs[i].page_content)

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
        if "path" not in docs[i].metadata and "path" in locals():
            docs[i].metadata["path"] = path
        if "subitem_link" in kwargs and "subitem_link" not in docs[i].metadata:
            docs[i].metadata["subitem_link"] = kwargs["subitem_link"]
        if "title" not in docs[i].metadata or docs[i].metadata["title"] == "Untitled":
            if "title" in kwargs and kwargs["title"] and kwargs["title"] != "Untitled":
                docs[i].metadata["title"] = kwargs["title"]
            elif "http" in docs[i].metadata["path"].lower():
                docs[i].metadata["title"] = get_url_title(docs[i].metadata["path"])
                if not docs[i].metadata["title"]:
                    docs[i].metadata["title"] = "Untitled"
                    red(f"Could not get title from {path}")
        if "title" in kwargs and kwargs["title"] != docs[i].metadata["title"] and kwargs["title"] not in docs[i].metadata["title"]:
            docs[i].metadata["title"] += " - " + kwargs["title"]
        if "playlist_title" in kwargs:
            docs[i].metadata["title"] = kwargs["playlist_title"] + " - " + docs[i].metadata["title"]

        if "docs_reading_time" not in docs[i].metadata:
            if not total_reading_length:
                total_reading_length = sum([len(d.page_content) for d in docs]) / average_word_length / wpm
                assert total_reading_length > 0.5, f"Failing doc: total reading length is suspiciously low for {docs[i].metadata}"
            docs[i].metadata["docs_reading_time"] = total_reading_length
        if "source" not in docs[i].metadata:
            if "path" in docs[i].metadata:
                docs[i].metadata["source"] = docs[i].metadata["path"]
            else:
                docs[i].metadata["source"] = docs[i].metadata["title"]

        if "hash" not in docs[i].metadata:
            docs[i].metadata["hash"] = hasher(docs[i].page_content + json.dumps(docs[i].metadata))
        assert docs[i].metadata["hash"], f"Invalid hash for document: {docs[i]}"

    assert docs, "empty list of loaded documents!"
    docs = [d for d in docs if d.page_content]
    assert docs, "empty list of loaded documents after removing empty docs!"
    return docs


@loaddoc_cache.cache
def load_youtube_playlist(playlist_url):
    with youtube_dl.YoutubeDL({"quiet": False}) as ydl:
        try:
            loaded = ydl.extract_info(playlist_url, download=False)
        except (KeyError, DownloadError, ExtractorError) as e:
            raise Exception(red(f"ERROR: Youtube playlist link skipped because : error during information \
        extraction from {playlist_url} : {e}"))
    return loaded


@loaddoc_cache.cache(ignore=["loader"])
def cached_yt_loader(loader, path, add_video_info, language, translation):
    yel(f"Not using cache for youtube {path}")
    docs = loader(
            path,
            add_video_info=add_video_info,
            language=language,
            translation=translation,
            ).load()
    return docs


@loaddoc_cache.cache(ignore=["text_splitter"])
def cached_pdf_loader(path, text_splitter, splitter_chunk_size, debug):
    assert splitter_chunk_size == text_splitter._chunk_size, "unexpected error"
    loaders = {
            "pdftotext": None,
            "PDFMiner": PDFMinerLoader,
            "PyPDFLoader": PyPDFLoader,
            "Unstructured_elements_hires": partial(
                UnstructuredPDFLoader,
                mode="elements",
                strategy="hi_res",
                post_processors=[clean_extra_whitespace],
                infer_table_structure=True,
                # languages=["fr"],
                ),
            "Unstructured_elements_fast": partial(
                UnstructuredPDFLoader,
                mode="elements",
                strategy="fast",
                post_processors=[clean_extra_whitespace],
                infer_table_structure=True,
                # languages=["fr"],
                ),
            "Unstructured_hires": partial(
                UnstructuredPDFLoader,
                strategy="hi_res",
                post_processors=[clean_extra_whitespace],
                infer_table_structure=True,
                # languages=["fr"],
                ),
            "Unstructured_fast": partial(
                UnstructuredPDFLoader,
                strategy="fast",
                post_processors=[clean_extra_whitespace],
                infer_table_structure=True,
                # languages=["fr"],
                ),
            "PyPDFium2": PyPDFium2Loader,
            "PyMuPDF": PyMuPDFLoader,
            "PdfPlumber": PDFPlumberLoader,
            }
    # optionnal support for pdftotext as windows is terrible shit
    try:
        loaders["pdftotext"] = pdftotext.PDF
    except:
        del loaders["pdftotext"]
    loaded_docs = {}
    # using language detection to keep the parsing with the highest lang
    # probability
    probs = {}
    for loader_name, loader_func in loaders.items():
        try:
            if debug:
                red(f"Trying to parse {path} using {loader_name}")

            if loader_name == "pdftotext":
                with open(path, "rb") as f:
                    loader = loader_func(f)
                content = "\n\n".join(loader)
            else:
                loader = loader_func(path)
                content = loader.load()

            if "Unstructured" in loader_name:
                content = "\n".join([d.page_content.strip() for d in content])
                # remove empty lines. frequent in pdfs
                content = re.sub(emptyline_regex, '', content)
                content = re.sub(emptyline2_regex, '\n', content)
                content = re.sub(linebreak_before_letter, r'\1', content)

            texts = text_splitter.split_text(content)
            docs = [Document(page_content=t) for t in texts]

            prob = check_docs_tkn_length(docs, path)

            if prob >= 0.5:
                # only consider it okay if decent quality
                probs[loader_name] = prob
                loaded_docs[loader_name] = docs
                if prob > 0.90:
                    # select this one as its bound to be okay
                    whi(f"Early stopping of PDF parsing because {loader_name} has prob {prob} for {path}")
                    break
            else:
                whi(f"Ignore parsing by {loader_name} of '{path}' as it seems of poor quality: prob={prob}")
                continue

            if len(probs.keys()) >= 3:
                # if more than 3 worked, take the best amon them to save
                # time on running all the others
                break
        except Exception as err:
            yel(f"Error when parsing '{path}' with {loader_name}: {err}")

    assert probs.keys(), f"No pdf parser succedded to parse {path}"

    # no loader worked, exiting
    if not loaded_docs:
        raise Exception(f"No pdf parser worked for {path}")

    max_prob = max([v for v in probs.values()])

    if debug:
        red(f"Language probability after parsing {path}: {probs}")

    return loaded_docs[[name for name in probs if probs[name] == max_prob][0]]
