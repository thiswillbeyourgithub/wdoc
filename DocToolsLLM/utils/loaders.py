"""
Called by batch_file_loader.py's threads. Contains many cached function to
load each document.
The imports are taking a substantial amount of time so loaders.py is
lazily loaded.
"""

import sys
import os
import time
from typing import List, Union, Any, Optional, Callable
from textwrap import dedent
from functools import partial
import uuid
import tempfile
import requests
import shutil
from pathlib import Path, PosixPath
import re
from tqdm import tqdm
import json
import dill
import httpx

import lazy_import

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import WebBaseLoader

from unstructured.cleaners.core import clean_extra_whitespace

from .misc import (loaddoc_cache, html_to_text, hasher,
                   file_hasher, get_splitter, check_docs_tkn_length,
                   average_word_length, wpm)
from .typechecker import optional_typecheck
from .logger import whi, yel, red, log
from .flags import is_verbose, is_linux

# lazy loading of modules
Document = lazy_import.lazy_class('langchain.docstore.document.Document')
TextSplitter = lazy_import.lazy_class('langchain.text_splitter.TextSplitter')
RecursiveCharacterTextSplitter = lazy_import.lazy_class('langchain.text_splitter.RecursiveCharacterTextSplitter')
youtube_dl = lazy_import.lazy_module('youtube_dl')
DownloadError = lazy_import.lazy_class('youtube_dl.utils.DownloadError')
ExtractorError = lazy_import.lazy_class('youtube_dl.utils.ExtractorError')
akp = lazy_import.lazy_module('ankipandas')
ftfy = lazy_import.lazy_module('ftfy')
BeautifulSoup = lazy_import.lazy_class('bs4.BeautifulSoup')
Goose = lazy_import.lazy_class('goose3.Goose')
prompt = lazy_import.lazy_function('prompt_toolkit.prompt')
LogseqMarkdownParser = lazy_import.lazy_module('LogseqMarkdownParser')
litellm = lazy_import.lazy_module("litellm")
deepgram = lazy_import.lazy_module("deepgram")
pydub = lazy_import.lazy_module("pydub")
ffmpeg = lazy_import.lazy_module("ffmpeg")


try:
    import pdftotext
except Exception as err:
    if is_verbose:
        red(f"Failed to import optional package 'pdftotext': '{err}'")
        if is_linux:
            red(
                "On linux, you can try to install pdftotext with :\nsudo "
                "apt install build-essential libpoppler-cpp-dev pkg-config "
                "python3-dev\nThen:\npython -m pip install pdftotext"
            )


# needed in case of buggy unstructured install
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


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

pdf_loaders = {
    "pdftotext": None,  # optional support, see below
    "PDFMiner": PDFMinerLoader,
    "PyPDFLoader": PyPDFLoader,
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
    "Unstructured_elements_hires": partial(
        UnstructuredPDFLoader,
        mode="elements",
        strategy="hi_res",
        post_processors=[clean_extra_whitespace],
        infer_table_structure=True,
        # languages=["fr"],
    ),
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

global_temp_dir = [None]  # will be replaced when load_one_doc is called


@optional_typecheck
def load_one_doc(
    task: str,
    debug: bool,
    temp_dir: PosixPath,
    filetype: str,
    file_hash: str,
    source_tag: Optional[str] = None,
    **kwargs,
    ) -> List[Document]:
    """choose the appropriate loader for a file, then load it,
    split into documents, add some metadata then return.
    The loader is cached"""
    text_splitter = get_splitter(task)

    assert global_temp_dir[0] is temp_dir

    if filetype == "youtube":
        docs = load_youtube_video(**kwargs)

    elif filetype == "online_pdf":
        docs = load_online_pdf(
            debug=debug,
            task=task,
            **kwargs,
        )

    elif filetype == "pdf":
        docs = load_pdf(
            debug=debug,
            text_splitter=text_splitter,
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "anki":
        docs = load_anki(**kwargs)

    elif filetype == "string":
        assert not kwargs, f"Received unexpected arguments for filetype 'string': {kwargs}"
        docs = load_string()

    elif filetype == "txt" or filetype == "text":
        docs = load_txt(file_hash=file_hash, **kwargs)

    elif filetype == "local_html":
        docs = load_local_html(file_hash=file_hash, **kwargs)

    elif filetype == "logseq_markdown":
        docs = load_logseq_markdown(debug=debug, file_hash=file_hash, **kwargs,)

    elif filetype == "local_audio":
        docs = load_local_audio(file_hash=file_hash, **kwargs)

    elif filetype == "local_video":
        docs = load_local_video(file_hash=file_hash, **kwargs)

    elif filetype == "epub":
        docs = load_epub(file_hash=file_hash, **kwargs)

    elif filetype == "powerpoint":
        docs = load_powerpoint(file_hash=file_hash, **kwargs)

    elif filetype == "word":
        docs = load_word_document(file_hash=file_hash, **kwargs)

    elif filetype == "url":
        docs = load_url(**kwargs)

    else:
        raise Exception(red(f"Unsupported filetype: '{filetype}'"))

    docs = text_splitter.transform_documents(docs)

    if filetype not in ["logseq_markdown", "anki", "pdf"]:
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

        if source_tag:
            if "source_tag" not in docs[i].metadata:
                docs[i].metadata["source_tag"] = source_tag
            else:
                docs[i].metadata["source_tag"] = docs[i].metadata["source_tag"].replace("unset", "").strip()
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
        if "subitem_link" in kwargs and "subitem_link" not in docs[i].metadata:
            docs[i].metadata["subitem_link"] = kwargs["subitem_link"]
        if "title" not in docs[i].metadata or docs[i].metadata["title"] == "Untitled":
            if "title" in kwargs and kwargs["title"] and kwargs["title"] != "Untitled":
                docs[i].metadata["title"] = kwargs["title"]
            elif "path" in docs[i].metadata and isinstance(docs[i].metadata["path"], str) and "http" in docs[i].metadata["path"].lower():
                docs[i].metadata["title"] = get_url_title(
                    docs[i].metadata["path"])
                if not docs[i].metadata["title"]:
                    docs[i].metadata["title"] = "Untitled"
                    red(f"Could not get title from doc '{kwargs}'")
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
            elif "path" in docs[i].metadata:
                docs[i].metadata["source"] = docs[i].metadata["title"]
            else:
                docs[i].metadata["source"] = "undocumented"

        if "hash" not in docs[i].metadata:
            docs[i].metadata["hash"] = hasher(
                docs[i].page_content + json.dumps(docs[i].metadata)
            )
        assert docs[i].metadata["hash"], f"Invalid hash for document: {docs[i]}"

        # make sure the filepath are absolute
        if "path" in docs[i].metadata and Path(docs[i].metadata["path"]).exists():
           docs[i].metadata["path"] = str(Path(docs[i].metadata["path"]).resolve().absolute())

    assert docs, "empty list of loaded documents!"
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
def cloze_stripper(clozed: str) -> str:
    clozed = clozeregex.sub(" ", clozed)
    return clozed

# loaders #######################################

@optional_typecheck
def load_youtube_video(
    path: str,
    youtube_language: Optional[str] = None,
    youtube_translation: Optional[str] = None,
    youtube_audio_backend: Optional[str] = "youtube",

    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,

    deepgram_kwargs: Optional[dict] = None,
    ) -> List[Document]:
    assert youtube_audio_backend in [
        "youtube", "whisper", "deepgram"
    ], f"Invalid value for youtube_audio_backend. Must be either youtube, whisper or deepgram, not '{youtube_audio_backend}'"

    if "\\" in path:
        red(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")
    assert yt_link_regex.search(path), f"youtube link is not valid: '{path}'"

    if youtube_audio_backend == "youtube":
        whi(f"Loading youtube: '{path}'")
        fyu = YoutubeLoader.from_youtube_url
        docs = cached_yt_loader(
            loader=fyu,
            path=path,
            add_video_info=True,
            language=youtube_language if youtube_language else ["fr-FR", "fr", "en", "en-US", "en-UK"],
            translation=youtube_translation if youtube_translation else None,
        )
    else:
        whi("Downloading audio from youtube")
        file_name = global_temp_dir[0] / f"youtube_audio_{uuid.uuid4()}"  # without extension!
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{file_name.absolute().resolve()}.%(ext)s'  # with extension
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([path])
        candidate = []
        for f in global_temp_dir[0].iterdir():
            if file_name.name in f.name:
                candidate.append(f)
        assert len(candidate), f"Audio file of {path} failed to download?"
        assert len(candidate) == 1, f"Multiple audio file found for video: '{candidate}'"
        audio_file = str(candidate[0].absolute())
        audio_hash=file_hasher({"path": audio_file})

        if youtube_audio_backend == "whisper":
            content = transcribe_audio_whisper(
                audio_path=audio_file,
                audio_hash=audio_hash,
                language=whisper_lang,
                prompt=whisper_prompt,
            )

            docs = [
                Document(
                    page_content=content["text"],
                    metadata={
                        "source": "youtube_whisper",
                    },
                )
            ]
            if "duration" in content:
                docs[-1].metadata["duration"] = content["duration"]
            if "language" in content:
                docs[-1].metadata["language"] = content["language"]
            elif whisper_lang:
                docs[-1].metadata["language"] = whisper_lang

        elif youtube_audio_backend == "deepgram":
            content = transcribe_audio_deepgram(
                audio_path=audio_file,
                audio_hash=audio_hash,
                deepgram_kwargs=deepgram_kwargs,
            )
            assert len(content["results"]["channels"]) == 1, "unexpected deepgram output"
            assert len(content["results"]["channels"][0]["alternatives"]) == 1, "unexpected deepgram output"
            text = content["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"].strip()
            assert text, "Empty text from deepgram transcription"

            docs = [
                Document(
                    page_content=text,
                    metadata={
                        "source": "youtube_deepgram",
                    },
                )
            ]
            docs[-1].metadata.update(content["metadata"])
            docs[-1].metadata["deepgram_kwargs"] = deepgram_kwargs

        else:
            raise ValueError(youtube_audio_backend)

        for f in Path(audio_file).parent.iterdir():
            if str(file_name.name) in f.stem:
                f.unlink()
        assert not Path(audio_file).exists()

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
    random_val = str(uuid.uuid4()).split("-")[-1]
    new_db_path = global_temp_dir[0] / f"anki_collection_{name.replace('/', '_')}_{random_val}"
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
    # mod2mid = akp.raw.get_model2mid(col.db)
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
    new_db_path.unlink()
    Path(str(new_db_path.absolute()) + "-shm").unlink(missing_ok=True)
    Path(str(new_db_path.absolute()) + "-wal").unlink(missing_ok=True)
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
    docs = [
        Document(
            page_content=content,
            metadata={"path": "user_string"},
        )
    ]
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
    audio_backend: str,

    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,

    deepgram_kwargs: Optional[dict] = None,
    ) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"

    if audio_backend == "whisper":
        assert deepgram_kwargs is None, "Found kwargs for deepgram but selected whisper backend for local_audio"
        content = transcribe_audio_whisper(
            audio_path=path,
            audio_hash=file_hash,
            language=whisper_lang,
            prompt=whisper_prompt,
        )
        docs = [
            Document(
                page_content=content["text"],
                metadata={
                    "source": path,
                },
            )
        ]
        if "duration" in content["metadata"]:
            docs[-1].metadata["duration"] = content["metadata"]["duration"]
        if "language" in content:
            docs[-1].metadata["language"] = content["language"]
        elif whisper_lang:
            docs[-1].metadata["language"] = whisper_lang

    elif audio_backend == "deepgram":
        assert whisper_prompt is None and whisper_lang is None, f"Found args whisper_prompt or whisper_lang but selected deepgram backend for local_audio"
        content = transcribe_audio_deepgram(
            audio_path=path,
            audio_hash=file_hash,
            deepgram_kwargs=deepgram_kwargs,
        )
        assert len(content["results"]["channels"]) == 1, "unexpected deepgram output"
        assert len(content["results"]["channels"][0]["alternatives"]) == 1, "unexpected deepgram output"
        text = content["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"].strip()
        assert text, "Empty text from deepgram transcription"

        docs = [
            Document(
                page_content=text,
                metadata={
                    "source": "local_audio_deepgram",
                },
            )
        ]
        docs[-1].metadata.update(content["metadata"])
        docs[-1].metadata["deepgram_kwargs"] = deepgram_kwargs

    else:
        raise ValueError(f"Invalid audio backend: must be either 'deepgram' or 'whisper'. Not '{audio_backend}'")

    return docs

@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_local_video(
    path: str,
    file_hash: str,
    audio_backend: str,

    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,

    deepgram_kwargs: Optional[dict] = None,
    ) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"

    audio_path = global_temp_dir[0] / f"audio_from_video_{uuid.uuid4()}.mp3"
    assert not Path(audio_path).exists()

    # extract audio from video
    try:
        whi(f"Exporting audio from {path} to {audio_path} (this can take some time)")
        t = time.time()
        stream = ffmpeg.input(path)
        stream = ffmpeg.output(stream, audio_path)
        ffmpeg.run(stream)
        whi(f"Done extracting audio in {time.time()-t:.2f}s")
    except Exception as err:
        red(f"Error when getting audio from video using ffmpeg. Retrying with pydub. Error: '{err}'")

        try:
            Path(audio_path).unlink(missing_ok=True)
            audio = pydub.AudioSegment.from_file(path)
            # extract audio from video
            whi(f"Extracting audio from {path} to {audio_path} (this can take some time)")
            t = time.time()
            audio.export(audio_path, format="mp3")
            whi(f"Done extracting audio in {time.time()-t:.2f}s")
        except Exception as err:
            raise Exception(
                f"Error when getting audio from video using ffmpeg: '{err}'")

    assert Path(audio_path).exists(), f"FileNotFound: {audio_path}"

    # need the hash from the mp3, not video
    audio_hash=file_hasher({"path": audio_path})

    return load_local_audio(
        path=audio_path,
        file_hash=audio_hash,
        audio_backend=audio_backend,
        whisper_lang=whisper_lang,
        whisper_prompt=whisper_prompt,
        deepgram_kwargs=deepgram_kwargs,
    )


@optional_typecheck
@loaddoc_cache.cache(ignore=["audio_path"])
def transcribe_audio_deepgram(
    audio_path: str,
    audio_hash: str,
    deepgram_kwargs: Optional[dict] = None,
    ) -> dict:
    "Use whisper to transcribe an audio file"
    whi(f"Calling deepgram to transcribe {audio_path}")
    assert os.environ["DOCTOOLS_PRIVATEMODE"] == "false", (
        "Private mode detected, aborting before trying to use deepgram's API"
    )
    assert "DEEPGRAM_API_KEY" in os.environ and not os.environ["DEEPGRAM_API_KEY"] == "REDACTED_BECAUSE_DOCTOOLSLLM_IN_PRIVATE_MODE", "No environment variable DEEPGRAM_API_KEY found"

    # client
    try:
        client = deepgram.DeepgramClient()
    except Exception as err:
        raise Exception(f"Error when creating deepgram client: '{err}'")

    # set options
    options = dict(
        # docs: https://playground.deepgram.com/?endpoint=listen&smart_format=true&language=en&model=nova-2
        model="nova-2",

        detect_language=True,
        # not all features below are available for all languages

        # intelligence
        summarize=False,
        topics=False,
        intents=False,
        sentiment=False,

        # transcription
        smart_format=True,
        punctuate=True,
        paragraphs=True,
        utterances=True,
        diarize=True,

        # redact=None,
        # replace=None,
        # search=None,
        # keywords=None,
        # filler_words=False,
    )
    if deepgram_kwargs is None:
        deepgram_kwargs = {}
    if "language" in deepgram_kwargs and deepgram_kwargs["language"]:
        del options["detect_language"]
    options.update(deepgram_kwargs)
    options = deepgram.PrerecordedOptions(**options)

    # load file
    with open(audio_path, "rb") as f:
        payload = {"buffer": f.read()}

    # get content
    t = time.time()
    content = client.listen.prerecorded.v("1").transcribe_file(
        payload,
        options,
        timeout=httpx.Timeout(300.0, connect=10.0)  # timeout for large files
    )
    whi(f"Done deepgram transcribing {audio_path} in {int(time.time()-t)}s")
    d = content.to_dict()
    return d

@optional_typecheck
@loaddoc_cache.cache(ignore=["audio_path"])
def transcribe_audio_whisper(
    audio_path: str,
    audio_hash: str,
    language: str,
    prompt: str) -> dict:
    "Use whisper to transcribe an audio file"
    whi(f"Calling openai's whisper to transcribe {audio_path}")
    assert os.environ["DOCTOOLS_PRIVATEMODE"] == "false", (
        "Private mode detected, aborting before trying to use openai's whisper"
    )

    assert "OPENAI_API_KEY" in os.environ and not os.environ["OPENAI_API_KEY"] == "REDACTED_BECAUSE_DOCTOOLSLLM_IN_PRIVATE_MODE", "No environment variable OPENAI_API_KEY found"

    t = time.time()
    with open(audio_path, "rb") as audio_file:
        transcript = litellm.transcription(
            model="whisper-1",
            file=audio_file,
            prompt=prompt,
            language=language,
            temperature=0,
            response_format="verbose_json",
            ).json()
    whi(f"Done transcribing {audio_path} in {int(time.time()-t)}s")
    return transcript

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
@loaddoc_cache.cache(ignore=["path"])
def load_powerpoint(
    path: str,
    file_hash: str,
    ) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"
    loader = UnstructuredPowerPointLoader(path)
    content = loader.load()

    docs = [
        Document(
            page_content=content,
            metadata={},
        )
    ]
    return docs
@optional_typecheck
@loaddoc_cache.cache(ignore=["path"])
def load_word_document(
    path: str,
    file_hash: str,
    ) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"
    try:
        loader = Docx2txtLoader(path)
        content = loader.load()
        docs = [Document(page_content=content)]
        check_docs_tkn_length(docs, path)
    except Exception as err:
        red(f"Error when loading word document with docx2txt, trying with unstructured: '{err}'")
        loader = UnstructuredWordDocumentLoader(path)
        content = loader.load()
        docs = [Document(page_content=content)]

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
    name = Path(path).name
    if len(name) > 30:
        name = name[:15] + "..." + name[-15:]

    loaded_docs = {}
    # using language detection to keep the parsing with the highest lang
    # probability
    probs = {}

    pbar = tqdm(total=len(pdf_loaders), desc=f"Parsing PDF {name}", unit="loader")
    for loader_name in pdf_loaders:
        pbar.desc = f"Parsing PDF {name} with {loader_name}"
        try:
            if debug:
                red(f"Trying to parse {path} using {loader_name}")

            content = _pdf_loader(loader_name, path, file_hash)
            pbar.update(1)

            if "unstructured" in loader_name.lower():
                # remove empty lines. frequent in pdfs
                content = emptyline_regex.sub("", content)
                content = emptyline2_regex.sub("\n", content)
                content = linebreak_before_letter.sub(r"\1", content)

            content = ftfy.fix_text(content)

            texts = text_splitter.split_text(content)
            docs = [Document(page_content=t) for t in texts]

            prob = check_docs_tkn_length(docs, path)

            if prob >= 0.5:
                # only consider it okay if decent quality
                probs[loader_name] = prob
                loaded_docs[loader_name] = docs
                if prob > 0.95:
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
            if "content" not in locals():
                pbar.update(1)

    pbar.close()
    assert probs.keys(), f"No pdf parser succedded to parse {path}"

    # no loader worked, exiting
    if not loaded_docs:
        raise Exception(f"No pdf parser worked for {path}")

    max_prob = max([v for v in probs.values()])

    if debug:
        red(f"Language probability after parsing {path}: {probs}")

    return loaded_docs[[name for name in probs if probs[name] == max_prob][0]]
