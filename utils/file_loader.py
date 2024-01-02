from functools import partial
import tldextract
import uuid
import threading
import queue
import copy
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
from joblib import Parallel, delayed
import tiktoken

from ftlangdetect import detect as language_detect

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFium2Loader
from langchain.document_loaders import PyMuPDFLoader
# from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from .misc import loaddoc_cache, html_to_text, hasher
from .logger import whi, yel, red, log
from .llm import RollingWindowEmbeddings, transcribe

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

max_threads = 20
threads = {}
lock = threading.Lock()
n_recursive = 0  # global var to keep track of the number of recursive loading threads. If there are many recursions they can actually get stuck

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
                chunk_size=3000,
                chunk_overlap=300,
                length_function=get_tkn_length,
                )
    else:
        raise Exception(task)
    return text_splitter


def cloze_stripper(clozed):
    clozed = re.sub(clozeregex, " ", clozed)
    return clozed

min_token = 200
max_token = 1_000_000
max_lines = 100_000
min_lang_prob = 0.50

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
        if not debug:
            message = f"Loading documents using {max_threads} threads (depth={depth})"
            pbar = tqdm(total=len(doclist), desc=message)
            recursion_id = str(uuid.uuid4())

            class thread_args(dict):
                """used to store the arguments used to create the thread and
                create it at the last minute"""
                _is_started = False
                _recursion_id = recursion_id

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
                    whi(f"(Depth={depth}) Waiting for {n_subthreads_alive} subthreads to finish: {','.join(doc_print)}")

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
            message = "Loading documents without multithreading because debug is on"
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
        if "translation" not in kwargs:
            transl = "en"
        else:
            transl = kwargs["translation"]

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
            for cid in cards.index:
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

            for cid in cards.index:
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


def load_embeddings(embed_model, loadfrom, saveas, debug, loaded_docs, kwargs):
    """loads embeddings for each document"""

    if embed_model == "openai":
        red("Using openai embedding model")
        assert Path("API_KEY.txt").exists(), "No API_KEY.txt found"

        embeddings = OpenAIEmbeddings(
                openai_api_key = str(Path("API_KEY.txt").read_text()).strip()
                )

    else:
        embeddings = RollingWindowEmbeddings(
                model_name=embed_model,
                encode_kwargs={
                    "batch_size": 1,
                    "show_progress_bar": True,
                    "normalize_embeddings": True,
                    },
                )

    lfs = LocalFileStore(f".cache/embeddings/{embed_model}")
    cache_content = list(lfs.yield_keys())
    red(f"Found {len(cache_content)} embeddings in local cache")

    # cached_embeddings = embeddings
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            lfs,
            namespace=embed_model,
            )

    # reload passed embeddings
    if loadfrom:
        red("Reloading documents and embeddings from file")
        path = Path(loadfrom)
        assert path.exists(), f"file not found at '{path}'"
        db = FAISS.load_local(str(path), cached_embeddings)
        return db, cached_embeddings

    red("\nLoading embeddings.")

    docs = loaded_docs
    if len(docs) >= 50:
        docs = sorted(docs, key=lambda x: random.random())

    Path(".cache").mkdir(exist_ok=True)
    Path(".cache/faiss_embeddings").mkdir(exist_ok=True)
    embeddings_cache = Path(f".cache/faiss_embeddings/{embed_model}")
    embeddings_cache.mkdir(exist_ok=True)
    t = time.time()
    whi(f"Creating FAISS index for {len(docs)} documents")

    in_cache = [p for p in embeddings_cache.iterdir()]
    whi(f"Found {len(in_cache)} embeddings in cache")
    db = None
    to_embed = []

    # load previous faiss index from cache
    for doc in tqdm(docs, desc="Loading embeddings from cache"):
        fi = embeddings_cache / str(doc.metadata["hash"] + ".faiss_index")
        if fi.exists():
            temp = FAISS.load_local(fi, cached_embeddings)
            if not db and temp:
                db = temp
            else:
                try:
                    db.merge_from(temp)
                except Exception as err:
                    red(f"Error when loading cache from {fi}: {err}\nDeleting {fi}")
                    [p.unlink() for p in fi.iterdir()]
                    fi.rmdir()
        else:
            to_embed.append(doc)

    whi(f"Docs left to embed: {len(to_embed)}")

    # check price of embedding
    full_tkn = sum([get_tkn_length(doc.page_content) for doc in to_embed])
    red(f"Total number of tokens in documents (not checking if already present in cache): '{full_tkn}'")
    if embed_model == "openai":
        dol_price = full_tkn * 0.0001 / 1000
        red(f"With OpenAI embeddings, the total cost for all tokens is ${dol_price:.4f}")
        if dol_price > 1:
            ans = input(f"Do you confirm you are okay to pay this? (y/n)\n>")
            if ans.lower() not in ["y", "yes"]:
                red("Quitting.")
                raise SystemExit()

    # create a faiss index for batch of documents, then save them
    # as 1 document faiss index to cache
    if to_embed:
        batch_size = 1000
        batches = [
                [i * batch_size, (i + 1) * batch_size]
                for i in range(len(to_embed) // batch_size + 1)
                ]
        pbar = tqdm(total=len(to_embed), desc="Saving to cache")
        for batch in tqdm(batches, desc="Embedding by batch"):
            temp = FAISS.from_documents(
                    to_embed[batch[0]:batch[1]],
                    cached_embeddings,
                    normalize_L2=True
                    )

            recursive_faiss_saver(temp, to_embed[batch[0]:batch[1]], embeddings_cache, 0, pbar)

            if not db:
                db = temp
            else:
                db.merge_from(temp)

    # to get vectors from a faiss index
    # vecs = faiss.rev_swig_ptr(temp.index.get_xb(), len(to_embed) * temp.index.d).reshape(len(to_embed), temp.index.d)

    whi(f"Done creating index in {time.time()-t:.2f}s")

    # saving embeddings
    if saveas:
        db.save_local(saveas)

    return db, cached_embeddings

def recursive_faiss_saver(index, documents, path, depth, pbar):
    """split the faiss index by hand into 1 docstore index and save
    it to cache. To split it, as the copy.deepcopy is long we
    use a recursive call to only copy fewer times the full index"""
    doc_ids = [k for k in index.docstore._dict.keys()]
    assert doc_ids, "unexpected empty doc_ids"
    n = 10
    threads = []
    le = len(doc_ids)
    nn = len(doc_ids) // n
    if depth:
        spacer = " " * depth * 2
    else:
        spacer = ""
    info = f"(n={n}, nn={nn}, le={le}, d={depth})"
    if nn > n:  # more than 1 order of magnitude
        for i in range(len(doc_ids) // nn + 1):
            whi(f"{spacer}Creating larger subindex #{i} {info}")
            sub_index = copy.deepcopy(index)
            sub_docids = doc_ids[i * nn: (i + 1) * nn]
            to_del = [d for d in doc_ids if d not in sub_docids]
            if not to_del or not sub_docids:
                continue
            sub_index.delete(to_del)
            threads.extend(
                    recursive_faiss_saver(
                        sub_index, documents[i * nn:(i + 1) * nn], path, depth + 1, pbar)
                    )

    elif len(doc_ids) > n:
        for i in range(len(doc_ids) // n + 1):
            whi(f"{spacer}Creating subindex #{i} {info}")
            sub_index = copy.deepcopy(index)
            sub_docids = doc_ids[i * n: (i + 1) * n]
            to_del = [d for d in doc_ids if d not in sub_docids]
            if not to_del or not sub_docids:
                continue
            sub_index.delete(to_del)
            threads.extend(
                    recursive_faiss_saver(
                        sub_index, documents[i * n:(i + 1) * n], path, depth + 1, pbar)
                    )
            while sum([t.is_alive() for t in threads]) > 3 * n:
                time.sleep(0.1)
    else:
        for i, did in enumerate(doc_ids):
            whi(f"{spacer}Saving {documents[i].metadata['hash']}.faiss_index {info}")
            to_del = [d for d in doc_ids if d != did]
            if not to_del:
                continue
            file = (path / str(documents[i].metadata["hash"] + ".faiss_index"))
            assert not file.exists(), "cache file already exists!"
            thread = threading.Thread(
                    target=save_one_index,
                    args=(copy.deepcopy(index), to_del, file, pbar),
                    )
            thread.start()
            threads.append(thread)
        return threads
    [t.join() for t in threads]
    return []

def save_one_index(index, to_del, file, pbar):
    index.delete(to_del)
    index.save_local(file)
    pbar.update(1)

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
            "PDFMiner": PDFMinerLoader,
            "PyPDFLoader": PyPDFLoader,
            "Unstructured": partial(UnstructuredPDFLoader, mode="elements", strategy="hi_res"),
            "PyPDFium2": PyPDFium2Loader,
            "PyMuPDF": PyMuPDFLoader,
            # "PdfPlumber": PDFPlumberLoader,
            }
    loaded_docs = {}
    # using language detection to keep the parsing with the highest lang
    # probability
    probs = {}
    for loader_name, loader_func in loaders.items():
        try:
            if debug:
                red(f"Trying to parse {path} using {loader_name}")
            loader = loader_func(path)
            content = loader.load()

            if loader_name != "Unstructured":
                content = "\n".join([d.page_content.strip() for d in content])
                # remove empty lines. frequent in pdfs
                content = re.sub(emptyline_regex, '', content)
                content = re.sub(emptyline2_regex, '\n', content)
                content = re.sub(linebreak_before_letter, r'\1', content)

            texts = text_splitter.split_text(content)
            docs = [Document(page_content=t) for t in texts]

            prob = check_docs_tkn_length(docs, path)
            probs[loader_name] = prob
            loaded_docs[loader_name] = docs
        except Exception as err:
            red(f"Error when parsing '{path}' with {loader_name}: {err}")

    # no loader worked, exiting
    if not loaded_docs:
        raise Exception(f"No pdf parser worked for {path}")

    max_prob = max([v for v in probs.values()])

    if debug:
        red(f"Language probability after parsing {path}: {probs}")

    return loaded_docs[[name for name in probs if probs[name] == max_prob][0]]

def create_hyde_retriever(
        query,
        filetype,

        llm,
        top_k,
        relevancy,

        embeddings_engine,
        embeddings,
        ):
    """
    create a retriever only for the subset of documents from the
    loaded_embeddings that were found using HyDE technique (i.e. asking
    the llm to create a hypothetical answer and use the embedding of this
    answer to search similar content)

    The code is a little strange because it actually reloads only a portion
    of the embeddings from cache if possible.

    https://python.langchain.com/docs/use_cases/question_answering/how_to/hyde
    """

    HyDE_template = """Please imagine the answer to the user's question about a document:
    Document type: [[filetype]]
    User question: {question}
    Answer:""".replace("[[filetype]]", filetype)
    hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template=HyDE_template,
            )

    hyde_chain = LLMChain(
            llm=llm,
            prompt=hyde_prompt,
            )

    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=hyde_chain,
        base_embeddings=embeddings_engine,
        )
    db = FAISS.from_documents(
            documents=[Document(page_content="")],
            embedding=hyde_embeddings,
            )
    # get id of the dummy doc
    dummy = list(db.docstore._dict.keys())
    db.delete(dummy)
    db.merge_from(embeddings)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "distance_metric": "cos",
            "score_threshold": relevancy,
            }
        )
    return retriever


def create_parent_retriever(
        task,
        loaded_embeddings,
        loaded_docs,
        top_k,
        relevancy,
        ):
    "https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever"
    csp = get_splitter(task)
    psp = get_splitter(task)
    psp._chunk_size *= 4
    parent = ParentDocumentRetriever(
            vectorstore=loaded_embeddings,
            docstore=LocalFileStore(".cache/parent_retriever"),
            child_splitter=csp,
            parent_splitter=psp,
            search_type="similarity",
            search_kwargs={
                "k": top_k,
                "distance_metric": "cos",
                "score_threshold": relevancy,
                }
            )
    parent.add_documents(loaded_docs)
    return parent
