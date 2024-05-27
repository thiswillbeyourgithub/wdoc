import os
from typing import List, Union, Any, Optional, Callable

from langchain.docstore.document import Document
try:
    from ftlangdetect import detect as language_detect
except Exception as err:
    print(f"Couldn't import ftlangdetect: '{err}'")
try:
    import pdftotext
except Exception as err:
    print(f"Failed to import pdftotext: '{err}'")
from .misc import loaddoc_cache, html_to_text, hasher, cache_dir, file_hasher
from .typechecker import optional_typecheck
from .logger import whi, yel, red, log
from .llm import transcribe

#exec(lazy_import_statements("""
import tiktoken
from textwrap import dedent
from functools import partial
import uuid
import tempfile
import requests
import youtube_dl
from youtube_dl.utils import DownloadError, ExtractorError
import shutil
import ankipandas as akp
import ftfy
from bs4 import BeautifulSoup
from goose3 import Goose
from pathlib import Path
import re
from tqdm import tqdm
import json
import dill
from prompt_toolkit import prompt

from langchain.text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
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

# needed in case of buggy unstructured install
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

min_token = 50
max_token = 1_000_000
max_lines = 100_000
min_lang_prob = 0.50

# separators used for the text splitter
recur_separator = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "...", ".", " ", ""]


# for reading length estimation
wpm = 250
average_word_length = 6

clozeregex = re.compile(r"{{c\d+::|}}")  # for removing clozes in anki
markdownlink_regex = re.compile(r"\[.*?\]\((.*?)\)")  # to find markdown links
markdownlinkparser_regex = pattern = re.compile(r'\[([^\]]+)\]\(http[s]?://[^)]+\)')  # to replace markdown links by their text
# to check that a youtube link is valid
yt_link_regex = re.compile("youtube.*watch")
emptyline_regex = re.compile(r"^\s*$", re.MULTILINE)
emptyline2_regex = re.compile(r"\n\n+", re.MULTILINE)
linebreak_before_letter = re.compile(
    r"\n([a-záéíóúü])", re.MULTILINE
)  # match any linebreak that is followed by a lowercase letter

tokenize = tiktoken.encoding_for_model(
    "gpt-3.5-turbo"
).encode  # used to get token length estimation

pdf_loaders = {
    "pdftotext": None,  # optional support
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

# pdftotext is kinda weird to install on windows so support it
# only if it's correctly imported
if "pdftotext" in globals():
    class pdftotext_loader_class:
        "simple wrapper for pdftotext to make it load by pdf_loader"
        def __init__(self, path):
            self.path = path

        def load(self):
            with open(self.path, "rb") as f:
                return "\n\n".join(pdftotext.PDF(f))
    pdf_loaders["pdftotext"] = pdftotext_loader_class



@optional_typecheck
def load_one_doc(
    task: str,
    debug: bool,
    filetype: str,
    **kwargs,
    ) -> List[Document]:
    """choose the appropriate loader for a file, then load it,
    split into documents, add some metadata then return.
    The loader is cached"""
    text_splitter = get_splitter(task)

    if filetype == "youtube":
        docs = load_youtube_video(**kwargs)

    elif filetype == "online_pdf":
        docs = load_online_pdf(
            debug=debug,
            task=task,
            **kwargs,
        )

    elif filetype == "pdf":
        assert kwargs['file_hash']
        docs = load_pdf(
            debug=debug,
            text_splitter=text_splitter,
            **kwargs,
        )

    elif filetype == "anki":
        random_val = str(uuid.uuid4()).split("-")[-1]
        try:
            docs = load_anki(random_val=random_val, **kwargs)
        except:
            # delete the failed db files from cache
            name = f"{anki_profile}".replace(" ", "_")
            new_db_path = cache_dir / f"anki_collection_{name.replace('/', '_')}_{random_val}"
            Path(new_db_path).unlink(missing_ok=True)
            Path(new_db_path + "-shm").unlink(missing_ok=True)
            Path(new_db_path + "-wal").unlink(missing_ok=True)
            raise

    elif filetype == "string":
        assert not kwargs, f"Received unexpected arguments for filetype 'string': {kwargs}"
        docs = load_string()

    elif filetype == "txt" or filetype == "text":
        assert kwargs['file_hash']
        docs = load_txt(**kwargs)

    elif filetype == "local_html":
        assert kwargs['file_hash']
        docs = load_local_html(**kwargs)

    elif filetype == "logseq_markdown":
        assert kwargs['file_hash']
        docs = load_logseq_markdown(debug=debug, **kwargs,)

    elif filetype == "local_audio":
        assert kwargs['file_hash']
        docs = load_local_audio(**kwargs)

    elif filetype == "epub":
        assert kwargs['file_hash']
        docs = load_epub(**kwargs)

    elif filetype == "url":
        docs = load_url(**kwargs)

    else:
        raise Exception(red(f"Unsupported filetype: '{filetype}'"))

    docs = text_splitter.transform_documents(docs)

    if filetype not in ["logseq_markdown", "anki"]:
        check_docs_tkn_length(docs, filetype)

    # add and format metadata
    total_reading_length = None
    for i in range(len(docs)):
        # if html, parse it
        soup = BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = soup.get_text()

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
        if "path" not in docs[i].metadata and "path" in kwargs:
            docs[i].metadata["path"] = kwargs["path"]
        if "subitem_link" in kwargs and "subitem_link" not in docs[i].metadata:
            docs[i].metadata["subitem_link"] = kwargs["subitem_link"]
        if "title" not in docs[i].metadata or docs[i].metadata["title"] == "Untitled":
            if "title" in kwargs and kwargs["title"] and kwargs["title"] != "Untitled":
                docs[i].metadata["title"] = kwargs["title"]
            elif "http" in docs[i].metadata["path"].lower():
                docs[i].metadata["title"] = get_url_title(
                    docs[i].metadata["path"])
                if not docs[i].metadata["title"]:
                    docs[i].metadata["title"] = "Untitled"
                    red(f"Could not get title from {path}")
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

        if "docs_reading_time" not in docs[i].metadata:
            if not total_reading_length:
                total_reading_length = (
                    sum([len(d.page_content)
                        for d in docs]) / average_word_length / wpm
                )
                assert (
                    total_reading_length > 0.1
                ), (
                    "Failing doc: total reading length is suspiciously low "
                    f"for {docs[i].metadata}: '{total_reading_length:.3f} minutes'"
                )
            docs[i].metadata["docs_reading_time"] = total_reading_length
        if "source" not in docs[i].metadata:
            if "path" in docs[i].metadata:
                docs[i].metadata["source"] = docs[i].metadata["path"]
            else:
                docs[i].metadata["source"] = docs[i].metadata["title"]

        if "hash" not in docs[i].metadata:
            docs[i].metadata["hash"] = hasher(
                docs[i].page_content + json.dumps(docs[i].metadata)
            )
        assert docs[i].metadata["hash"], f"Invalid hash for document: {docs[i]}"

        # make sure the filepath are absolute
        if "path" in docs[i].metadata and Path(docs[i].metadata["path"]).exists():
           docs[i].metadata["path"] = str(Path(docs[i].metadata["path"]).absolute())

    assert docs, "empty list of loaded documents!"
    docs = [d for d in docs if d.page_content]
    assert docs, "empty list of loaded documents after removing empty docs!"
    return docs

# Convenience functions #########################

@optional_typecheck
def get_url_title(url: str) -> Union[str, type(None)]:
    """if the title of the url is not loaded from the loader, trying as last
    resort with this one"""
    loader = WebBaseLoader(url, raise_for_status=True)
    docs = loader.load()
    if "title" in docs[0].metadata and docs[0].metadata["title"]:
        return docs[0].metadata["title"]
    else:
        return None


@optional_typecheck
def check_docs_tkn_length(docs: List[Document], name: str) -> float:
    """checks that the number of tokens in the document is high enough,
    not too low, and has a high enough language probability,
    otherwise something probably went wrong."""
    size = sum([get_tkn_length(d.page_content) for d in docs])
    nline = len("\n".join([d.page_content for d in docs]).splitlines())
    if nline > max_lines:
        red(
            f"Example of page from document with too many lines : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of lines from '{name}' is {nline} > {max_lines}, probably something went wrong?"
        )
    if size <= min_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{name}' is {size} <= {min_token}, probably something went wrong?"
        )
    if size >= max_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{name}' is {size} >= {max_token}, probably something went wrong?"
        )

    # check if language check is above a threshold
    if "language_detect" not in globals():
        # bypass if language_detect not imported
        return 1
    prob = [language_detect(docs[0].page_content.replace("\n", "<br>"))["score"]]
    if len(docs) > 1:
        prob.append(language_detect(docs[1].page_content.replace("\n",
                                "<br>"))["score"])
        if len(docs) > 2:
            prob.append(
                    language_detect(
                        docs[len(docs) // 2].page_content.replace("\n", "<br>")
                    )["score"]
            )
    prob = max(prob)
    if prob <= min_lang_prob:
        red(
            f"Low language probability for {name}: prob={prob:.3f}<{min_lang_prob}.\nExample page: {docs[len(docs)//2]}"
        )
        raise Exception(
            f"Low language probability for {name}: prob={prob:.3f}.\nExample page: {docs[len(docs)//2]}"
        )
    return prob

@optional_typecheck
def get_tkn_length(tosplit: str) -> int:
    return len(tokenize(tosplit))


@optional_typecheck
def get_splitter(task: str) -> TextSplitter:
    "we don't use the same text splitter depending on the task"
    if task in ["query", "search"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=3000,  # default 4000
            chunk_overlap=386,  # default 200
            length_function=get_tkn_length,
        )
    elif task in ["summarize_link_file", "summarize_then_query", "summarize"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=2000,
            chunk_overlap=300,
            length_function=get_tkn_length,
        )
    elif task == "recursive_summary":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=1000,
            chunk_overlap=200,
            length_function=get_tkn_length,
        )
    else:
        raise Exception(task)
    return text_splitter


@optional_typecheck
def cloze_stripper(clozed: str) -> str:
    clozed = re.sub(clozeregex, " ", clozed)
    return clozed

# loaders #######################################

@optional_typecheck
def load_youtube_video(path: str, youtube_language: Optional[str] = None, youtube_translation: Optional[str] = None) -> List[Document]:
    if "\\" in path:
        red(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")
    assert re.search(
        yt_link_regex, path), f"youtube link is not valid: '{path}'"

    whi(f"Loading youtube: '{path}'")
    fyu = YoutubeLoader.from_youtube_url
    docs = cached_yt_loader(
        loader=fyu,
        path=path,
        add_video_info=True,
        language=youtube_language if youtube_language else ["fr-FR", "fr", "en", "en-US", "en-UK"],
        translation=youtube_translation if youtube_translation else None,
    )
    return docs

@optional_typecheck
@loaddoc_cache.cache
def load_online_pdf(debug: bool, task: str, path: str, **kwargs) -> List[Document]:
    whi(f"Loading online pdf: '{path}'")

    try:
        loader = OnlinePDFLoader(path)
        docs = loader.load()

    except Exception as err:
        red(
            f"Failed parsing online PDF {path} using only OnlinePDFLoader. Will try downloading it directly."
        )

        response = requests.get(path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

        meta = kwargs.copy()
        meta["filetype"] = "pdf"
        meta["path"] = temp_file.name
        meta["file_hash"] = file_hasher(temp_file.name)
        try:
            return load_one_doc(
                task=task,
                debug=debug,
                **meta,
            )
        except Exception as err:
            red(
                f"Error when parsing online pdf from {path} downloaded to {temp_file.name}: '{err}'"
            )
            raise
    return docs


@optional_typecheck
def load_anki(
    anki_profile: str,
    random_val: str,
    anki_mode: str = "window_single_note",
    anki_deck: Optional[str] = None,
    anki_fields: Optional[List[str]] = None,
    anki_notetype: Optional[str] = None,
    ) -> List[Document]:
    assert (
        anki_mode.replace("window", "")
        .replace("concatenate", "")
        .replace("single_note", "")
        .replace("_", "")
        == ""
    ), f"Unexpected anki_mode: {anki_mode}"

    whi(f"Loading anki profile: '{anki_profile}'")
    original_db = akp.find_db(user=anki_profile)
    name = f"{anki_profile}".replace(" ", "_")
    new_db_path = cache_dir / f"anki_collection_{name.replace('/', '_')}_{random_val}"
    assert not Path(new_db_path).exists(
    ), f"{new_db_path} already existing!"
    shutil.copy(original_db, new_db_path)
    col = akp.Collection(path=new_db_path)
    cards = col.cards.merge_notes()

    cards.loc[cards["codeck"] == "", "codeck"] = cards["cdeck"][
        cards["codeck"] == ""
    ]
    cards["codeck"] = cards["codeck"].apply(
        lambda x: x.replace("\x1f", "::"))
    if anki_deck:
        cards = cards[cards["codeck"].str.startswith(anki_deck)]
    cards["nmodel"] = cards["nmodel"].apply(lambda x: x.lower())
    if anki_notetype:
        cards = cards[cards["nmodel"].str.contains(anki_notetype, case=False)]

    cards["mid"] = col.cards.mid.loc[cards.index]
    mid2fields = akp.raw.get_mid2fields(col.db)
    mod2mid = akp.raw.get_model2mid(col.db)
    cards["fields_name"] = cards["mid"].apply(lambda x: mid2fields[x])
    assert cards.index.tolist(), "empty dataframe!"
    if anki_fields:
        cards["fields_dict"] = cards.apply(
            lambda x: {
                k: html_to_text(cloze_stripper(v)).strip()
                for k, v in zip(x["fields_name"], x["nflds"])
                if k.lower() in anki_fields
            },
            axis=1,
        )
        cards["text"] = cards["fields_dict"].apply(
            lambda x: "\n".join(f"{k}: {x[k]}" for k in anki_fields if x[k].strip())
        )
    else:
        cards["text"] = cards.apply(
                lambda x: (lambda d: "\n".join([f"{k}: {v}" for k, v in d.items()]))(
                    {
                        k: html_to_text(cloze_stripper(v)).strip()
                        for k, v in zip(x["fields_name"], x["nflds"])
                    }
                ),
            axis=1,
        )
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
                        "anki_mode": "single_note",
                    },
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
                metadata = {
                    "anki_tags": " ".join(tags),
                    "anki_cid": str(cid),
                    "anki_deck": card_deck,
                    "anki_mode": "concatenate",
                }
                continue

            # if too many token, add the current chunk of text and start
            # the next chunk with this card
            if get_tkn_length(full_text + spacer + text) >= chunksize:
                assert (
                    full_text
                ), f"An anki card is too large for the text splitter: {text}"
                assert metadata["anki_cid"], "No anki_cid in metadata"
                docs.append(
                    Document(
                        page_content=full_text,
                        metadata=metadata,
                    )
                )

                metadata = {
                    "anki_tags": " ".join(tags),
                    "anki_cid": str(cid),
                    "anki_deck": card_deck,
                }
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
                    s = (
                        -1
                    )  # when at the end of the list, apply the window in reverse
                    # s for 'sign'
                else:
                    s = 1
                if (
                    cards.at[index_list[i + w * s], "text"]
                    in cards.at[index_list[i], "text_concat"]
                ):
                    # skipping this card because it's a duplicate
                    skip += 1
                text_concat += (
                    "\n\n" +
                    cards.at[index_list[i + (w + skip) * s], "text"]
                )
                tags_concat += cards.at[index_list[i +
                                                    (w + skip) * s], "ntags_t"]
                cids += f"{index_list[i+(w+skip)*s]} "
            cards.at[index_list[i], "text_concat"] = text_concat
            cards.at[index_list[i], "tags_concat"] = tags_concat
            cards.at[index_list[i], "cids"] = cids

        for cid in sorted(cards.index):
            c = cards.loc[cid,]
            docs.append(
                Document(
                    page_content=c["text_concat"].strip(),
                    metadata={
                        "anki_tags": " ".join(
                            list(set(c["tags_concat"].split(" ")))
                        ),
                        "anki_cid": c["cids"].strip(),
                        "anki_mode": f"window_{window_size}",
                    },
                )
            )

    assert docs, "List of loaded anki document is empty!"

    path = f"Anki_profile='{anki_profile}',deck='{anki_deck}'notetype={anki_notetype},fields={','.join(anki_fields)}"
    for i in range(len(docs)):
        docs[i].metadata["anki_profile"] = anki_profile
        docs[i].metadata["anki_topdeck"] = anki_deck
        docs[i].metadata["anki_notetype"] = anki_notetype
        docs[i].metadata["path"] = path
        docs[i].metadata["anki_tags"] = " ".join(
            sorted(list(set(docs[i].metadata["anki_tags"].split(" "))))
        )
        docs[i].metadata["anki_cid"] = " ".join(
            sorted(docs[i].metadata["anki_cid"].split(" "))
        )


    # delete temporary db file
    Path(new_db_path).unlink()
    Path(new_db_path + "-shm").unlink(missing_ok=True)
    Path(new_db_path + "-wal").unlink(missing_ok=True)
    return docs

@optional_typecheck
@loaddoc_cache.cache
def load_string() -> List[Document]:
    whi("Loading string")
    content = prompt(
        "Paste your text content here then press esc+enter or meta+enter:\n>",
        multiline=True,
    )
    log.info(f"Pasted string input:\n{content}")
    docs = [Document(page_content=content)]
    return docs

@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_txt(path: str, file_hash: str) -> List[Document]:
    whi(f"Loading txt: '{path}'")
    assert Path(path).exists(), f"file not found: '{path}'"
    with open(path) as f:
        content = f.read()
    docs = [Document(page_content=content, metadata={})]
    return docs

@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_local_html(
    path: str,
    file_hash: str,
    load_functions: Optional[bytes] = None,
    ) -> List[Document]:
    whi(f"Loading local html: '{path}'")
    assert Path(path).exists(), f"file not found: '{path}'"

    with open(path) as f:
        content = f.read()

    if load_functions:
        # the functions must be pickled because joblib can't
        # cache string that would declare as lambda functions

        try:
            load_functions = dill.loads(load_functions)
        except Exception as err:
            raise Exception(f"Error when unpickling load_functions: '{err}'")
        assert isinstance(load_functions, tuple), (
            f"load_functions must be a tuple, not {type(load_functions)}")
        assert all(callable(lf) for lf in load_functions), (
            f"load_functions element must be a callable, not {[type(lf) for lf in load_functions]}")

        for ifunc, func in enumerate(load_functions):
            try:
                content = func(content)
            except Exception as err:
                raise Exception(
                    f"load_functions #{ifunc}: '{func}' failed with "
                    f"error : '{err}'")
        assert isinstance(content, str), (
            f"output of function #{ifunc}: '{func}' is not a "
            f"string: {content}")
    try:
        soup = BeautifulSoup(content, "html.parser")
    except Exception as err:
        raise Exception(f"Error when parsing html: {err}")

    text = soup.get_text().strip()
    assert text, "Empty text after loading from html"

    docs = [
        Document(
            page_content=text,
        )
    ]
    return docs

@loaddoc_cache.cache
def eval_load_functions(
    load_functions: str,
    ) -> List[Callable]:
    assert isinstance(load_functions, list), "load_functions must be of type list"
    assert all(isinstance(lf, str) for lf in load_functions), "elements of load_functions must be of type str"

    try:
        for ilf, lf in enumerate(load_functions):
            load_functions[ilf] = eval(lf)
    except Exception as err:
        raise Exception(f"Error when evaluating load_functions #{ilf}: {lf} '{err}'")
    assert all(callable(lf) for lf in load_functions), (
            f"Some load_functions are not callable: {load_functions}")


@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_logseq_markdown(debug: bool, path: str, file_hash: str) -> List[Document]:
    whi(f"Loading logseq markdown file: '{path}'")
    assert Path(path).exists(), f"file not found: '{path}'"
    try:
        parsed = LogseqMarkdownParser.parse_file(path, verbose=debug)
    except Exception as err:
        raise Exception(f"Error when parsing {path} LogseqMarkdownParser: '{err}'")

    if not parsed.blocks:
        raise Exception(f"No logseq blocks loaded for {path} (file size: {Path(path).stat().st_size})")

    blocks = parsed.blocks

    # group blocks by parent block
    pblocks = [[blocks[0]]]
    for b in blocks[1:]:
        if b.indentation_level == 0:
            pblocks.append([b])
        else:
            pblocks[-1].append(b)
    whi(f"Found {len(pblocks)} parent blocks")
    assert sum([len(pb) for pb in pblocks]) == len(blocks), f"Unexpected number of blocks after grouping by parent"

    page_props = parsed.page_properties

    docs = []
    for grou in pblocks:
        # store in metadata the properties of the blocks inside a given
        # parent block
        meta = page_props.copy()
        content = ""  # and remove the metadata from the page content
        for b in grou:
            cont = b.content
            for k, v in b.properties.items():
                meta[k] = v
                cont = cont.replace(f"{k}:: {v}", "").strip()
            cont = dedent(cont)
            # note: should it be dedented? that saves a lot of token
            # use tabs instead of spaces to save tokens and avoid confusion the LLM
            lines = cont.splitlines()
            lines = [li.expandtabs(tabsize=2) for li in lines if li.strip()]
            lines = [li.replace("  ", "\t") for li in lines]
            cont = "\n".join(lines)
            if cont.count("\t") * 2 ==  cont.count("\t\t"):
                cont = cont.replace("\t\t", "\t")
            content += "\n" + cont

        doc = Document(
            page_content=content,
            metadata=meta,
        )
        docs.append(doc)

    return docs

@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_local_audio(
    path: str,
    file_hash: str,
    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,
    ) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"
    cache_transcribe = loaddoc_cache.cache(
        transcribe, ignore=["audio_path"])

    content = cache_transcribe(
        audio_path=path,
        audio_hash=file_hash,
        language=whisper_lang,
        prompt=whisper_prompt,
    )
    docs = [
        Document(
            page_content=content,
            metadata={
                "duration": content["duration"],
                "language": content["language"],
                "source": path,
            },
        )
    ]
    return docs

@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_epub(
    path: str,
    file_hash: str,
    ) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"
    loader = UnstructuredEPubLoader(path)
    content = loader.load()

    docs = [
        Document(
            page_content=content,
            metadata={},
        )
    ]
    return docs

@optional_typecheck
@loaddoc_cache.cache
def load_url(path: str, title=None) -> List[Document]:
    whi(f"Loading url: '{path}'")

    # even if loading fails the title might be found so trying to keep
    # the first working title across trials
    if title == "Untitled":
        title = None

    loaded_success = False
    if not loaded_success:
        try:
            loader = WebBaseLoader(
                "https://r.jina.ai/" + path,
                raise_for_status=True)
            text = "\n".join([doc.page_content for doc in loader.load()]).strip()
            assert text, "Empty text"
            if not title:
                if text.splitlines()[0].startswith("Title: "):
                    title = text.splitlines()[0].replace("Title: ", "", 1)
            text = text.split("Markdown Content:", 1)[1]
            text = markdownlinkparser_regex.sub(r'\1', text)  # remove links
            docs = [
                Document(
                    page_content=text,
                    metadata={
                    "parser": "jinareader",
                    }
                )
            ]
            if title:
                for doc in docs:
                    doc.metadata["title"] = title
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            red(
                f"Exception when using jina reader to parse url: '{err}'"
            )

    if not loaded_success:
        try:
            loader = PlaywrightURLLoader(
                urls=[path], remove_selectors=["header", "footer"]
            )
            docs = loader.load()
            assert docs, "Empty docs when using playwright"
            if not title and "title" in docs[0].metadata:
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            red(
                f"Exception when using playwright to parse url: '{err}'"
            )

    if not loaded_success:
        try:
            loader = SeleniumURLLoader(urls=[path], browser="firefox")
            docs = loader.load()
            assert docs, "Empty docs when using selenium firefox"
            if (
                not title
                and "title" in docs[0].metadata
                and docs[0].metadata["title"] != "No title found."
            ):
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            red(
                f"Exception when using selenium firefox to parse url: '{err}'"
            )

    if not loaded_success:
        try:
            loader = SeleniumURLLoader(urls=[path], browser="chrome")
            docs = loader.load()
            assert docs, "Empty docs when using selenium chrome"
            if (
                not title
                and "title" in docs[0].metadata
                and docs[0].metadata["title"] != "No title found."
            ):
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            red(
                f"Exception when using selenium chrome to parse url: '{err}'\nUsing goose as fallback"
            )

    if not loaded_success:
        try:
            g = Goose()
            article = g.extract(url=path)
            text = article.cleaned_text
            docs = [Document(page_content=text)]
            assert docs, "Empty docs when using goose"
            if not title:
                if (
                    "title" in docs[0].metadata
                    and docs[0].metadata["title"]
                ):
                    title = docs[0].metadata["title"]
                elif article.title:
                    title = article.title
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            red(
                f"Exception when using goose to parse url: '{err}'"
            )

    if not loaded_success:
        try:
            loader = WebBaseLoader(path, raise_for_status=True)
            docs = loader.load()
            assert docs, "Empty docs when using html"
            if (
                not title
                and "title" in docs[0].metadata
                and docs[0].metadata["title"]
            ):
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            red(
                f"Exception when using html as LAST RESORT to parse url: '{err}'"
            )


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

    return docs


@optional_typecheck
@loaddoc_cache.cache
def load_youtube_playlist(playlist_url: str) -> Any:
    with youtube_dl.YoutubeDL({"quiet": False}) as ydl:
        try:
            loaded = ydl.extract_info(playlist_url, download=False)
        except (KeyError, DownloadError, ExtractorError) as e:
            raise Exception(
                red(f"ERROR: Youtube playlist link skipped because : error during information \
        extraction from {playlist_url} : {e}")
            )
    return loaded


@optional_typecheck
@loaddoc_cache.cache(ignore=["loader"])
def cached_yt_loader(
        loader: Any,
        path: str,
        add_video_info: bool,
        language: List[str],
        translation: List[str]) -> List[Document]:
    yel(f"Not using cache for youtube {path}")
    docs = loader(
        path,
        add_video_info=add_video_info,
        language=language,
        translation=translation,
    ).load()
    return docs


@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def _pdf_loader(loader_name: str, path: str, file_hash: str) -> str:
    loader = pdf_loaders[loader_name](path)
    content = loader.load()
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        if isinstance(content[0], str):
            return "\n".join(content)
        elif hasattr(content[0], "page_content"):
            return "\n".join([d.page_content for d in content])
        else:
            raise ValueError(f"Unexpected type of content[0]: '{content}'")
    elif hasattr(content, "page_content"):
        return content.page_content
    raise ValueError(f"Unexpected type of content: '{content}'")


@optional_typecheck
def load_pdf(
    path: str,
    text_splitter: TextSplitter,
    debug: bool,
    file_hash: str,
    ) -> List[Document]:
    whi(f"Loading pdf: '{path}'")
    assert Path(path).exists(), f"file not found: '{path}'"

    loaded_docs = {}
    # using language detection to keep the parsing with the highest lang
    # probability
    probs = {}
    for loader_name in pdf_loaders:
        try:
            if debug:
                red(f"Trying to parse {path} using {loader_name}")

            content = _pdf_loader(loader_name, path, file_hash)

            if "unstructured" in loader_name.lower():
                # remove empty lines. frequent in pdfs
                content = re.sub(emptyline_regex, "", content)
                content = re.sub(emptyline2_regex, "\n", content)
                content = re.sub(linebreak_before_letter, r"\1", content)

            content = ftfy.fix_text(content)

            texts = text_splitter.split_text(content)
            docs = [Document(page_content=t) for t in texts]

            prob = check_docs_tkn_length(docs, path)

            if prob >= 0.5:
                # only consider it okay if decent quality
                probs[loader_name] = prob
                loaded_docs[loader_name] = docs
                if prob > 0.90:
                    # select this one as its bound to be okay
                    whi(
                        f"Early stopping of PDF parsing because {loader_name} has prob {prob} for {path}"
                    )
                    break
            else:
                whi(
                    f"Ignore parsing by {loader_name} of '{path}' as it seems of poor quality: prob={prob}"
                )
                continue

            if len(probs.keys()) >= 3:
                # if more than 3 worked, take the best among them to save
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
