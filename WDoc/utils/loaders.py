"""
Called by batch_file_loader.py's threads. Contains many cached function to
load each document.
The imports are taking a substantial amount of time so loaders.py is
lazily loaded.
"""

import copy
import sys
import os
import time
from typing import List, Union, Any, Optional, Callable, Dict, Tuple
import signal
from contextlib import contextmanager
import traceback
from functools import partial, wraps
import uuid6
import tempfile
import requests
import shutil
from pathlib import Path, PosixPath
import re
from tqdm import tqdm
import json
import dill
import httpx
import magic

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# from unstructured.cleaners.core import clean_extra_whitespace
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document

import lazy_import
# from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
TextSplitter = lazy_import.lazy_class('langchain.text_splitter.TextSplitter')
RecursiveCharacterTextSplitter = lazy_import.lazy_class('langchain.text_splitter.RecursiveCharacterTextSplitter')
import youtube_dl
import youtube_dl.utils
import ankipandas as akp
import pandas as pd
import ftfy
import bs4
import goose3
from prompt_toolkit import prompt
import LogseqMarkdownParser
import litellm
import deepgram
import pydub
import ffmpeg
import torchaudio
import playwright.sync_api
import openparse

from .misc import (doc_loaders_cache, html_to_text, hasher,
                   file_hasher, get_splitter, check_docs_tkn_length,
                   average_word_length, wpm, loaders_temp_dir_file,
                   min_lang_prob, min_token, max_token, max_lines,
                   optional_strip_unexp_args,
                   )
from .typechecker import optional_typecheck
from .logger import whi, yel, red, logger
from .flags import is_verbose, is_linux, is_debug
from .errors import TimeoutPdfLoaderError
from .env import WDOC_MAX_PDF_LOADER_TIMEOUT, WDOC_EMPTY_LOADER

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
# to replace markdown links by their text
markdownlinkparser_regex = re.compile(r'\[([^\]]+)\]\(http[s]?://[^)]+\)')
# to remove image from jina reader that take a lot of tokens but are not yet used
markdownimage_regex = re.compile(
    r'!\[([^\]]*)\]\s*(\([^\)]+\)|\[[^\]]+\])', flags=re.MULTILINE)


@optional_typecheck
def md_shorten_image_name(md_image: re.Match) -> str:
    "turn a markdown image link into just the name"
    name = md_image.group(1)
    if len(name) <= 16:
        return name
    else:
        return name[:8] + "…" + name[-8:]


# to check that a youtube link is valid
yt_link_regex = re.compile("youtube.*watch")
emptyline_regex = re.compile(r"^\s*$", re.MULTILINE)
emptyline2_regex = re.compile(r"\n\n+", re.MULTILINE)
linebreak_before_letter = re.compile(
    r"\n([a-záéíóúü])", re.MULTILINE
)  # match any linebreak that is followed by a lowercase letter
anki_replacements_regex = re.compile(r'\{([^}]*)\}')

@optional_typecheck
class OpenparseDocumentParser:
    def __init__(
        self,
        path: Union[str, PosixPath],
        table_args: Optional[dict] = {
            "parsing_algorithm": "pymupdf",
            "table_output_format": "markdown",
        },
        # table_args: Optional[dict] = None,
        ) -> None:
        self.path = path
        self.table_args = table_args

    def load(self) -> List[Document]:
        parser = openparse.DocumentParser(table_args=self.table_args)
        self.parsed = parser.parse(self.path)

        base_metadata = self.parsed.dict()
        nodes = base_metadata["nodes"]
        del base_metadata["nodes"]

        docs = []
        for node in nodes:
            meta = base_metadata.copy()
            meta.update(node)
            meta["page"] = meta["bbox"][0]["page"]
            text = meta["text"]
            del meta["text"], meta["bbox"], meta["node_id"], meta["tokens"]
            if meta["embedding"] is None:
                del meta["embedding"]

            doc = Document(
                page_content=text,
                metadata=meta,
            )

            if not docs:
                docs.append(doc)
            elif docs[-1].metadata["page"] != meta["page"]:
                docs.append(doc)
            else:
                docs[-1].page_content += "\n" + doc.page_content
                for k, v in doc.metadata.items():
                    if k not in docs[-1].metadata:
                        docs[-1].metadata[k] = v
                    else:
                        val = docs[-1].metadata[k]
                        if v == val:
                            continue
                        elif isinstance(val, list):
                            if v not in val:
                                if isinstance(v, list):
                                    docs[-1].metadata[k].extend(v)
                                else:
                                    docs[-1].metadata[k].append(v)
                        else:
                            docs[-1].metadata[k] = [val, v]
        self.docs = docs
        return docs

pdf_loaders = {
    "openparse": OpenparseDocumentParser,  # gets page number too, finds individual elements, kinda slow but good, optional table support
    "PDFMiner": PDFMinerLoader,  # little metadata
    "PyPDFLoader": PyPDFLoader,  # little metadata
    "PyPDFium2": PyPDFium2Loader,  # little metadata
    "PyMuPDF": PyMuPDFLoader,  # good for metadata
    "PdfPlumber": PDFPlumberLoader,  # good for metadata
    "pdftotext": None,  # optional support, see below
    "Unstructured_fast": partial(
        UnstructuredPDFLoader,
        strategy="fast",
        # post_processors=[clean_extra_whitespace],
        # infer_table_structure=True,
        # languages=["fr"],
    ),
    # "Unstructured_elements_fast": partial(
    #     UnstructuredPDFLoader,
    #     mode="elements",
    #     strategy="fast",
    #     # post_processors=[clean_extra_whitespace],
    #     # infer_table_structure=True,
    #     # languages=["fr"],
    # ),
    "Unstructured_hires": partial(
        UnstructuredPDFLoader,
        strategy="hi_res",
        # post_processors=[clean_extra_whitespace],
        # infer_table_structure=True,
        # languages=["fr"],
    ),
    # "Unstructured_elements_hires": partial(
    #     UnstructuredPDFLoader,
    #     mode="elements",
    #     strategy="hi_res",
    #     # post_processors=[clean_extra_whitespace],
    #     # infer_table_structure=True,
    #     # languages=["fr"],
    # ),
}

# pdftotext is kinda weird to install on windows so support it
# only if it's correctly imported
if "pdftotext" in sys.modules:
    @optional_typecheck
    class pdftotext_loader_class:
        "simple wrapper for pdftotext to make it load by pdf_loader"
        def __init__(self, path: Union[str, PosixPath]) -> None:
            self.path = path

        def load(self) -> List[Document]:
            with open(self.path, "rb") as f:
                docs = [
                    Document(
                        page_content=d,
                        metadata={"page": idoc}
                    )
                    for idoc, d in enumerate(pdftotext.PDF(f))
                ]
                return docs
    pdf_loaders["pdftotext"] = pdftotext_loader_class

# unsilence audio
sox_effects = [
    # ["norm"],  # normalize audio
    # isolate voice frequency
    # ["highpass", "-1", "100"],
    # ["lowpass", "-1", "3000"],
    # -2 is for a steeper filtering: removes high frequency and very low ones
    # ["highpass", "-2", "50"],
    # ["lowpass", "-2", "5000"],
    ["norm"],  # normalize audio
    # max silence should be 3s
    ["silence", "-l", "1", "0", "1%", "-1", "3.0", "1%"],
    ["norm"],
]

def debug_return_empty(func: Callable) -> Callable:
    if WDOC_EMPTY_LOADER:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metadata = {
                "debug_empty": True,
                "content_hash": str(uuid6.uuid6()),
                "all_hash": str(uuid6.uuid6()),
            }
            metadata.update(kwargs)
            out = [
                Document(
                    page_content="Lorem Ipsum",
                    metadata=metadata,
                )
            ]
            return out
        return wrapper
    else:
        return func

pdf_loader_max_timeout = WDOC_MAX_PDF_LOADER_TIMEOUT

@contextmanager
def signal_timeout(timeout: int, exception: Exception):
    "disabled in some joblib backend"
    assert timeout > 0
    def signal_handler(signum, frame):
        raise exception("Timeout occurred")


    # Set the signal handler and an alarm
    disabled = False
    try:
        signal.signal(signal.SIGALRM, signal_handler)
    except Exception:
        disabled = True

    if disabled:
        yield
    else:
        signal.alarm(timeout)

        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)


@optional_typecheck
def load_one_doc_wrapped(
    loading_failure: str = "warn",
    **doc_kwargs,
) -> Union[List[Document], str]:
    """wrap doc_loader to cach errors cleanly"""
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
        if loading_failure == "crash":
            raise
        elif loading_failure == "warn" or is_debug:
            return str(err)
        else:
            raise ValueError(loading_failure)

@optional_typecheck
def load_one_doc(
    task: str,
    llm_name: str,
    temp_dir: PosixPath,
    filetype: str,
    file_hash: str,
    source_tag: Optional[str] = None,
    doccheck_min_lang_prob: float = min_lang_prob,
    doccheck_min_token: int = min_token,
    doccheck_max_token: int = max_token,
    doccheck_max_lines: int = max_lines,
    **kwargs,
) -> List[Document]:
    """choose the appropriate loader for a file, then load it,
    split into documents, add some metadata then return.
    The loader is cached"""
    debug = is_debug
    text_splitter = get_splitter(task, modelname=llm_name)

    expected_global_dir = loaders_temp_dir_file.read_text().strip()
    assert expected_global_dir, f"Empty loaders_temp_dir_file at {loaders_temp_dir_file}"
    expected_global_dir = Path(expected_global_dir)
    assert expected_global_dir.exists(
    ), f"File loaders_temp_dir_file not found in {loaders_temp_dir_file} pointing at '{expected_global_dir}'"
    assert expected_global_dir == temp_dir, f"Error handling temp dir: temp_dir is {temp_dir} but loaders_temp_dir is {expected_global_dir}"

    if filetype == "url":
        docs = load_url(**kwargs)

    elif filetype == "youtube":
        docs = load_youtube_video(
            loaders_temp_dir=temp_dir,
            **kwargs,
        )

    elif filetype == "pdf":
        docs = load_pdf(
            debug=debug,
            text_splitter=text_splitter,
            file_hash=file_hash,
            doccheck_min_lang_prob=doccheck_min_lang_prob,
            doccheck_min_token=doccheck_min_token,
            doccheck_max_token=doccheck_max_token,
            doccheck_max_lines=doccheck_max_lines,
            **kwargs,
        )

    elif filetype == "online_pdf":
        docs = load_online_pdf(
            debug=debug,
            task=task,
            **kwargs,
        )

    elif filetype == "anki":
        docs = load_anki(
            verbose=is_verbose,
            text_splitter=text_splitter,
            loaders_temp_dir=temp_dir,
            **kwargs,
        )

    elif filetype == "string":
        assert not kwargs, f"Received unexpected arguments for filetype 'string': {kwargs}"
        docs = load_string()

    elif filetype == "txt":
        docs = load_txt(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "text":
        docs = load_text_input(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "local_html":
        docs = load_local_html(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "logseq_markdown":
        docs = load_logseq_markdown(
            debug=debug,
            file_hash=file_hash,
            text_splitter=text_splitter,
            **kwargs,
        )

    elif filetype == "local_audio":
        docs = load_local_audio(
            loaders_temp_dir=temp_dir,
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "local_video":
        docs = load_local_video(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "online_media":
        docs = load_online_media(
            loaders_temp_dir=temp_dir,
            **kwargs,
        )

    elif filetype == "epub":
        docs = load_epub(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "powerpoint":
        docs = load_powerpoint(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "word":
        docs = load_word_document(
            file_hash=file_hash,
            **kwargs,
        )

    elif filetype == "json_dict":
        docs = load_json_dict(
            file_hash=file_hash,
            **kwargs,
        )

    else:
        raise Exception(red(f"Unsupported filetype: '{filetype}'"))

    docs = text_splitter.transform_documents(docs)

    if filetype not in ["anki", "pdf"]:
        check_docs_tkn_length(
            docs=docs,
            identifier=filetype,
            min_lang_prob = doccheck_min_lang_prob,
            min_token = doccheck_min_token,
            max_token = doccheck_max_token,
            max_lines = doccheck_max_lines,
        )

    # add and format metadata
    for i in range(len(docs)):
        # if html, parse it
        soup = bs4.BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = soup.get_text()

        # fix text just in case
        docs[i].page_content = ftfy.fix_text(docs[i].page_content)

        if source_tag:
            if "source_tag" not in docs[i].metadata:
                docs[i].metadata["source_tag"] = source_tag
            else:
                if not isinstance(docs[i].metadata["source_tag"], str):
                    docs[i].metadata["source_tag"] = str(docs[i].metadata["source_tag"])
                docs[i].metadata["source_tag"] = docs[i].metadata["source_tag"].replace(
                    "unset", "").strip()
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

        if "doc_reading_time" not in docs[i].metadata:
            reading_length = len(docs[i].page_content) / \
                average_word_length / wpm
            docs[i].metadata["doc_reading_time"] = round(reading_length, 3)
        if "source" not in docs[i].metadata:
            if "path" in docs[i].metadata:
                docs[i].metadata["source"] = docs[i].metadata["path"]
            elif "path" in docs[i].metadata:  # was probably not a path
                docs[i].metadata["source"] = docs[i].metadata["title"]
            else:
                docs[i].metadata["source"] = "undocumented"

        # make sure the filepath are absolute
        try:
            if "path" in docs[i].metadata and Path(docs[i].metadata["path"]).exists():
                docs[i].metadata["path"] = str(
                    Path(docs[i].metadata["path"]).resolve().absolute())
        except Exception:
            pass  # was probably not a path

        docs[i].metadata["indexing_timestamp"] = int(time.time())

        # replace any path to just the filename, to avoid sending privacy
        # revealing information to LLMs
        for k, v in docs[i].metadata.items():
            if isinstance(v, PosixPath):
                docs[i].metadata[k] = v.name

        # set hash
        docs[i].metadata["content_hash"] = hasher(docs[i].page_content)
        docs[i].metadata["file_hash"] = file_hash
        assert docs[i].metadata[
            "content_hash"], f"Empty content_hash for document: {docs[i]}"
        assert docs[i].metadata["file_hash"], f"Empty file_hash for document: {docs[i]}"

        # check if metadata can be dumped, otherwise stringify the culprit
        try:
            meta_dump = json.dumps(docs[i].metadata)
        except Exception:
            for k, v in docs[i].metadata.items():
                if isinstance(v, PosixPath):
                    docs[i].metadata[k] = v.name
                    continue
                try:
                    json.dumps(v)
                except Exception:
                    docs[i].metadata[k] = str(v)
            meta_dump = json.dumps(docs[i].metadata)


        docs[i].metadata["all_hash"] = hasher(
            docs[i].metadata["content_hash"] + meta_dump
        )
        assert docs[i].metadata["all_hash"], f"Empty all_hash for document: {docs[i]}"

    total_reading_length =  None
    try:
        total_reading_length = sum([float(d.metadata["doc_reading_time"]) for d in docs])
    except Exception:
        pass
    if total_reading_length is not None:
        assert total_reading_length > 0.1, (
            f"Failing doc: total reading length is {total_reading_length:.3f}"
            "min which is  suspiciously low. Filetype {filetype} with kwargs "
            f"'{kwargs}'"
        )

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


@debug_return_empty
@optional_strip_unexp_args
def load_youtube_video(
    path: str,
    loaders_temp_dir: PosixPath,
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

    if not yt_link_regex.search(path):
        whi(f"Not a youtube link but trying anyway: '{path}'")

    if youtube_audio_backend == "youtube":
        whi(f"Using youtube.com loader: '{path}'")
        fyu = YoutubeLoader.from_youtube_url
        docs = cached_yt_loader(
            loader=fyu,
            path=path,
            add_video_info=True,
            language=youtube_language if youtube_language else [
                "fr-FR", "fr", "en", "en-US", "en-UK"],
            translation=youtube_translation if youtube_translation else None,
        )
    else:
        whi(f"Downloading audio from url: '{path}'")
        file_name = loaders_temp_dir / \
            f"youtube_audio_{uuid6.uuid6()}"  # without extension!
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            # with extension
            'outtmpl': f'{file_name.absolute().resolve()}.%(ext)s',
            'verbose': is_verbose,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([path])
        candidate = []
        for f in loaders_temp_dir.iterdir():
            if file_name.name in f.name:
                candidate.append(f)
        assert len(candidate), f"Audio file of {path} failed to download?"
        assert len(
            candidate) == 1, f"Multiple audio file found for video: '{candidate}'"
        audio_file = str(candidate[0].absolute())
        audio_hash = file_hasher({"path": audio_file})

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
            assert len(content["results"]["channels"]
                       ) == 1, "unexpected deepgram output"
            assert len(content["results"]["channels"][0]
                       ["alternatives"]) == 1, "unexpected deepgram output"
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


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
def load_online_pdf(
    debug: bool,
    task: str,
    path: str,
    doccheck_min_lang_prob: float = min_lang_prob,
    doccheck_min_token: int = min_token,
    doccheck_max_token: int = max_token,
    doccheck_max_lines: int = max_lines,
    **kwargs,
    ) -> List[Document]:
    whi(f"Loading online pdf: '{path}'")

    try:
        loader = OnlinePDFLoader(path)
        docs = loader.load()

    except Exception as err:
        red(
            f"Failed parsing online PDF {path} using only OnlinePDFLoader. Will try downloading it directly. Error message: '{err}'"
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


@debug_return_empty
@optional_strip_unexp_args
def load_anki(
    verbose: bool,
    text_splitter: TextSplitter,
    loaders_temp_dir: PosixPath,
    anki_profile: Optional[str] = None,
    anki_deck: Optional[str] = None,
    anki_notetype: Optional[str] = None,
    anki_template: Optional[str] = "{allfields}\n{image_ocr_alt}",
    anki_tag_render_filter: Optional[str] = None,
) -> List[Document]:
    if anki_tag_render_filter:
        assert "{tags}" in anki_template, "Can't use anki_tag_render_filter without using {tags} in anki_template"
        try:
            anki_tag_render_filter = re.compile(anki_tag_render_filter)
        except Exception as err:
            raise Exception(f"Failed to compile anki_tag_render_filter: '{err}'")

    if not anki_profile:
        original_db = akp.find_db()
        anki_profile = original_db.parent.name
        whi(f"Detected anki profile: '{anki_profile}'")

    whi(f"Loading anki profile: '{anki_profile}'")
    original_db = akp.find_db(user=anki_profile)
    name = f"{anki_profile}".replace(" ", "_")
    random_val = str(uuid6.uuid6())
    new_db_path = loaders_temp_dir / \
        f"anki_collection_{name.replace('/', '_')}_{random_val}"
    assert not Path(new_db_path).exists(
    ), f"{new_db_path} already existing!"
    shutil.copy(original_db, new_db_path)
    col = akp.Collection(path=new_db_path)
    cards = col.cards.merge_notes()

    if verbose:
        tqdm.pandas()
        pbar = lambda *x, **y: tqdm.pandas(*x, **y)
    else:
        pd.DataFrame.progress_apply = pd.DataFrame.apply
        pd.Series.progress_apply = pd.Series.apply
        pbar = lambda *x, **y: None

    cards.loc[cards["codeck"] == "", "codeck"] = cards["cdeck"][
        cards["codeck"] == ""
    ]

    cards["codeck"] = cards["codeck"].progress_apply(
        lambda x: x.replace("\x1f", "::"))
    if anki_deck:
        cards = cards[cards["codeck"].str.startswith(anki_deck)]
    cards["nmodel"] = cards["nmodel"].progress_apply(lambda x: x.lower())
    if anki_notetype:
        cards = cards[cards["nmodel"].str.contains(anki_notetype, case=False)]

    # remove suspended
    cards = cards[cards["cqueue"] != "suspended"]

    # merge models and fields for easy handling
    cards["mid"] = col.cards.mid.loc[cards.index]
    mid2fields = akp.raw.get_mid2fields(col.db)
    # make the model fields lowercase
    mid2fields = {
        k: (lambda x: [y.lower() for y in x])(v)
        for k, v in mid2fields.items()
    }
    # mod2mid = akp.raw.get_model2mid(col.db)
    cards["fields_name"] = cards["mid"].progress_apply(lambda x: mid2fields[x])
    assert not cards.empty, "empty dataframe!"

    # remove duplicate, essentially making cards the same thing as notes
    cards = cards.drop_duplicates(subset='nid', keep='first')
    notes = cards.reset_index().set_index("nid")

    # check placeholders validity
    placeholders = [ph.lower() for ph in anki_replacements_regex.findall(anki_template)]
    assert placeholders, f"No placeholder found in anki_template '{anki_template}'"
    for ph in placeholders:
        for ic, c in notes.iterrows():
            if ph not in c["fields_name"] + ["allfields", "tags", "image_ocr_alt"]:
                raise Exception(
                    "A placeholder in anki template didn't match fields of "
                    f"a card.\nCulprit placeholder: {ph}\nTemplate: "
                    f"{anki_template}\nExample card: {c}"
                )

    # prepare field values
    if "{allfields}" in anki_template:
        useallfields = True
        pbar(desc="Parsing allfields value")
        notes["allfields"] = notes.progress_apply(
            lambda x: "\n\n".join([
                f"{k.lower()}: '{html_to_text(cloze_stripper(v)).strip()}'"
                for k, v in zip(x["fields_name"], x["nflds"])
                if v.strip()
            ]),
            axis=1,
        )
    else:
        useallfields = False

    if "{image_ocr_alt}" in anki_template:
        useimageocr = True
    else:
        useimageocr = False

    if "{tags}" in anki_template:
        usetags = True
        pbar(desc="Formatting tags")
        notes["tags_formatted"] = notes.progress_apply(
            lambda x: ("Anki tags:\n'''\n" +  "\n".join([
                f"{t}"
                for t in x["ntags"]
                if (
                    anki_tag_render_filter is None or anki_tag_render_filter.match(t)
                )
            ]).strip() + "\n'''") if x["ntags"] else "",
            axis=1,
        )
        if notes["ntags"].notnull().any():
            assert notes["tags_formatted"].notnull().any(), "No tags were extracted because of your filter. Crashing to let you recheck your setup."
        # remove the tags formatting if it didn't match anything
        notes["tags_formatted"] = notes["tags_formatted"].str.replace(
            "Anki tags:\n'''\n\n'''",
            "",
        )
    else:
        usetags = False


    @optional_typecheck
    def placeholder_replacer(row: pd.Series) -> Tuple[str, dict]:
        text = anki_template

        if useallfields:
            text = text.replace("{allfields}", row["allfields"])
        if usetags:
            text = text.replace("{tags}", row["tags_formatted"])

        for ph in placeholders:
            if ph == "tags" or ph == "allfields" or ph == "image_ocr_alt":
                continue
            field_val = row["nflds"][row["fields_name"].index(ph)]
            text = text.replace(
                "{" + ph + "}",
                html_to_text(
                    cloze_stripper(
                        field_val
                    )
                )
            )
        text = text.replace("\\n", "\n").replace("\\xa0", " ")

        # replace media
        text, medias = replace_media(
            content=text,
            media=None,
            mode="remove_media",
            strict=False,
            replace_links=False,
        )
        if useimageocr:
            for img in [k for k in medias.keys() if "IMAGE" in k]:
                img = bs4.BeautifulSoup(medias[k], 'html.parser')
                title = img.get('title').strip() if img.has_attr('title') else ""
                alt = img.get('alt').strip() if img.has_attr('alt') else ""
                ocr_alt = ""
                if title:
                    ocr_alt += f"\nTitle: '{title}'"
                if alt:
                    ocr_alt += f"\nAlt: '{alt}'"
                ocr_alt = ocr_alt.strip()
                if ocr_alt:
                    text = text.replace(
                            "{image_ocr_alt}",
                            f"\n----\nOCR or alt text of {k}:\n{ocr_alt}\n''''\n" + "{image_ocr_alt}"
                    )
            text = text.replace("{image_ocr_alt}", "").strip()

        return text, medias

    pbar(desc="Formatting all cards")
    notes["medias"] = {}
    out = notes.progress_apply(placeholder_replacer, axis=1)
    notes["text"] = [t[0] for t in out]
    notes["medias"] = [t[1] for t in out]

    notes["text"] = notes["text"].progress_apply(lambda x: x.strip())
    notes = notes[notes["text"].ne('')]  # remove empty text

    # remove notes that contain an image, sound or link
    # notes = notes[~notes["text"].str.contains("\[IMAGE_")]
    # notes = notes[~notes["text"].str.contains("\[SOUND_")]
    # notes = notes[~notes["text"].str.contains("\[LINK_")]

    notes["text"] = notes["text"].apply(lambda x: x.strip())
    notes = notes[notes["text"].ne('')]  # remove empty text
    notes.drop_duplicates(subset="text", inplace=True)

    notes = notes.sort_index()

    docs = []

    # load each card as a single document
    for nid, c in notes.iterrows():
        assert c["codeck"], f"empty card_deck for nid {nid}"
        # turn the media into absolute paths
        medias = c["medias"]
        for k, v in medias.items():
            try:
                v = Path(original_db).parent / "collection.media" / v
                if v.exists():
                    medias[k] = v
            except Exception:
                # it was probably not a file
                continue
            assert k in c["text"], f"missing media '{k}' in text '{c['text']}' of card '{c}'"
        # better formatting for tags
        ntags = [
                nt
                # bettter for the tokenizer I guess
                # nt.replace("_", " ").replace("-", " ").replace("::", " > ")
                for nt in c["ntags"]
        ]
        docs.append(
            Document(
                page_content=c["text"],
                metadata={
                    "anki_tags": " ; ".join(ntags),
                    "anki_nid": str(nid),
                    "anki_deck": c["codeck"],
                    "anki_modtime": int(c["cmod"]),
                    "anki_media": json.dumps(medias),
                },
            )
        )

    assert docs, "List of loaded anki document is empty!"

    path = f"Anki_profile='{anki_profile}',deck='{anki_deck}',notetype='{anki_notetype}'"
    for i in range(len(docs)):
        docs[i].metadata["anki_profile"] = anki_profile
        docs[i].metadata["anki_topdeck"] = anki_deck
        docs[i].metadata["anki_notetype"] = anki_notetype
        docs[i].metadata["path"] = path
        docs[i].metadata["anki_nid"] = " ".join(
            sorted(docs[i].metadata["anki_nid"].split(" "))
        )

    # delete temporary db file
    new_db_path.unlink()
    Path(str(new_db_path.absolute()) + "-shm").unlink(missing_ok=True)
    Path(str(new_db_path.absolute()) + "-wal").unlink(missing_ok=True)
    return docs


REG_IMG = re.compile(
    r'<img src="[^"]+"(?:[^>]*?)?>',
    flags=re.MULTILINE | re.DOTALL)

REG_SOUNDS = re.compile(
    r'\[sound:\w+\.\w{2,3}\]',
)
REG_LINKS = re.compile(
    r'[A-Za-z0-9]+://[A-Za-z0-9%-_]+(?:/[A-Za-z0-9%-_])*(?:#|\\?)[A-Za-z0-9%-_&=]*',
)


@optional_typecheck
def replace_media(
    content: str,
    media: Union[None, Dict],
    mode: str,
    strict: bool = True,
    replace_image: bool = True,
    replace_links: bool = True,
    replace_sounds: bool = True,
) -> Tuple[str, Dict]:
    """
    Else: exclude any note that contains in the content:
        * an image (<img...)
        * or a sound [sound:...
        * or a link href / http
    This is because:
        1 as LLMs are non deterministic I preferred
            to avoid taking the risk of botching the content
        2 it costs less token

    The intended use is to call it first to replace
    each media by a simple string like [IMAGE_1] and check if it's
    indeed present in the output of the LLM then replace it back.

    It uses both bs4 and regex to be sure of itself
    """
    assert mode in ["add_media", "remove_media"]
    assert content.strip()
    if media is None:
        media = {}
    assert isinstance(media, dict)
    assert any(rule for rule in [replace_sounds, replace_links, replace_image])

    if mode == "remove_media":
        assert not media
        images = []
        sounds = []
        links = []

        if replace_links:
            # fix links common issues
            content = content.replace(":// ", "://")
            content = content.replace("http ://", "http://")
            content = content.replace("https ://", "http://")

        # Images
        if replace_image and "<img" in content:
            soup = bs4.BeautifulSoup(content, 'html.parser')
            images_bs4 = [str(img) for img in soup.find_all('img')]
            images_reg = re.findall(REG_IMG, content)
            if len(images_bs4) != len(images_reg):
                red(f"Different images found:\nbs4: {images_bs4}\nregex: {images_reg}\nContent: {content}")
                if images_bs4 and not images_reg:
                    images = [str(img) for img in images_bs4]
                elif (not images_bs4) and images_reg:
                    images = [str(img) for img in images_reg]
            else:
                images = [str(img) for img in images_bs4]
            assert images, f"no image found but should have. Text is '{content}'"
            for iimg, img in enumerate(images):
                try:
                    assert img in content, f"missing img from content:\nimg: {img}\ncontent: {content}"
                    assert re.search(REG_IMG, img), f"Regex couldn't identify img: {img}"
                    assert not re.search(REG_SOUNDS, img), f"Sound regex identifier img: {img}"
                except AssertionError as err:
                    if strict:
                        raise
                    else:
                        red(err)
                    images[iimg] = None
            images = [i for i in images if i is not None]

        # Sounds
        if replace_sounds and "[sounds:" in content:
            sounds = re.findall(REG_SOUNDS, content)
            assert sounds, f"No sounds found but should have. Content: {content}"
            for isound, sound in enumerate(sounds):
                try:
                    assert sound in content, f"Sound is not in content: {sound}"
                    assert not re.search(REG_IMG, sound), f"Image regex identified this sound: {sound}"
                    assert re.search(REG_SOUNDS, sound), f"Regex didn't identify this sound: {sound}"
                except AssertionError as err:
                    if strict:
                        raise
                    else:
                        red(err)
                    sounds[isound] = None
            sounds = [s for s in sounds if s is not None]

        # links
        if replace_links and "://" in content:
            links = re.findall(REG_LINKS, content)
            links = [
                link
                for link in links
                if not any(
                    other != link
                    and
                    other in link
                    for other in links
                )
            ]
            if strict:
                assert links, "No links found"
            for ilink, link in enumerate(links):
                try:
                    assert link in content, f"Link not in content:\nlink: {link}\ncontent: {content}"
                    assert re.search(REG_LINKS, link), f"Regex couldn't identify link: {link}"
                except AssertionError as err:
                    if strict:
                        raise
                    else:
                        red(err)
                    links[ilink] = None
            links = [li for li in links if li is not None]

        if not images + sounds + links:
            return content, {}

        new_content = content

        # do the replacing
        for i, img in enumerate(images):
            assert replace_image
            assert img in content
            assert img in new_content
            assert img not in media.keys() and img not in media.values()
            replaced = f"[IMAGE_{i+1}]"
            assert replaced not in media.keys() and replaced not in media.values()
            assert replaced not in content
            assert replaced not in new_content
            new_content = new_content.replace(img, replaced)
            media[replaced] = img
            assert img not in new_content
            assert replaced in new_content

        for i, sound in enumerate(sounds):
            assert replace_sounds
            assert sound in content
            assert sound in new_content
            assert sound not in media.keys() and sound not in media.values()
            replaced = f"[SOUND_{i+1}]"
            assert replaced not in media.keys() and replaced not in media.values()
            assert replaced not in content
            assert replaced not in new_content
            new_content = new_content.replace(sound, replaced)
            media[replaced] = sound
            assert sound not in new_content
            assert replaced in new_content

        for i, link in enumerate(links):
            assert replace_links
            assert link in content
            assert link not in media.keys()
            replaced = f"[LINK_{i+1}]"
            assert replaced not in media.keys() and replaced not in media.values()
            assert replaced not in content
            assert replaced not in new_content
            assert link in new_content or len(
                [
                    val for val in media.values()
                    if link in val
                ]
            )
            if link not in new_content:
                continue
            else:
                new_content = new_content.replace(link, replaced)
                media[replaced] = link
                assert link not in new_content
                assert replaced in new_content

        # check no media can be found anymore
        if replace_image:
            assert not re.findall(REG_IMG, new_content), new_content
            assert not bs4.BeautifulSoup(
                new_content, 'html.parser').find_all('img'), new_content
            if strict:
                assert "<img" not in new_content, new_content
            elif "<img" in new_content:
                red(f"AnkiMediaReplacer: Found '<img' in '{new_content}'")
        if replace_sounds:
            assert not re.findall(REG_SOUNDS, new_content), new_content
            if strict:
                assert "[sound:" not in new_content, new_content
            elif "[sound:" in new_content:
                red(f"AnkiMediaReplacer: Found '[sound:' in '{new_content}'")
        if replace_links:
            assert not re.findall(REG_LINKS, new_content), new_content
            if strict:
                assert "://" not in new_content, new_content
            elif "://" in new_content:
                red(f"AnkiMediaReplacer: Found '://' in '{new_content}'")

        # check non empty
        temp = new_content
        for med, val in media.items():
            temp = temp.replace(med, "")
        assert temp.strip()

        # ignore warnings from beautiful soup that can happen because anki is not exactly html
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

        # recursive check:
        assert replace_media(
            content=new_content,
            media=media,
            mode="add_media",
            strict=strict,
            replace_image=replace_image,
            replace_links=replace_links,
            replace_sounds=replace_sounds,
        )[0] == content

        return new_content, media

    elif mode == "add_media":
        assert media

        # TODO check that all media are found
        new_content = content
        for med, val in media.items():
            assert med in content
            assert val not in content
            assert val not in new_content
            new_content = new_content.replace(med, val)
            assert med not in new_content
            assert val in new_content

        return new_content, {}

    else:
        raise ValueError(mode)


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
def load_string() -> List[Document]:
    whi("Loading string")
    content = prompt(
        "Paste your text content here then press esc+enter or meta+enter:\n>",
        multiline=True,
    )
    logger.info(f"Pasted string input:\n{content}")
    docs = [
        Document(
            page_content=content,
            metadata={"path": "user_string"},
        )
    ]
    return docs

@debug_return_empty
@optional_strip_unexp_args
def load_txt(path: str, file_hash: str) -> List[Document]:
    whi(f"Loading txt: '{path}'")
    assert Path(path).exists(), f"file not found: '{path}'"
    with open(path) as f:
        content = f.read()
    docs = [Document(page_content=content, metadata={})]
    return docs

@debug_return_empty
@optional_strip_unexp_args
def load_text_input(
        path: str,
        metadata: Optional[Union[str, dict]],
        file_hash: str,
) -> List[Document]:
    whi(f"Loading text input: '{path}'")
    text = path.strip()
    assert text, "Empty text"
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    docs = [
        Document(
            page_content=text,
            metadata=metadata,
        )
    ]
    return docs

@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
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
        soup = bs4.BeautifulSoup(content, "html.parser")
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


@optional_typecheck
@doc_loaders_cache.cache
def eval_load_functions(
    load_functions: str,
) -> List[Callable]:
    assert isinstance(
        load_functions, list), "load_functions must be of type list"
    assert all(isinstance(lf, str)
               for lf in load_functions), "elements of load_functions must be of type str"

    try:
        for ilf, lf in enumerate(load_functions):
            load_functions[ilf] = eval(lf)
    except Exception as err:
        raise Exception(
            f"Error when evaluating load_functions #{ilf}: {lf} '{err}'")
    assert all(callable(lf) for lf in load_functions), (
        f"Some load_functions are not callable: {load_functions}")


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_logseq_markdown(
    debug: bool,
    path: str,
    file_hash: str,
    text_splitter: TextSplitter,
) -> List[Document]:
    whi(f"Loading logseq markdown file: '{path}'")
    assert Path(path).exists(), f"file not found: '{path}'"
    try:
        parsed = LogseqMarkdownParser.parse_file(path, verbose=False)
    except Exception as err:
        raise Exception(
            f"Error when parsing {path} LogseqMarkdownParser: '{err}'")

    if not parsed.blocks:
        raise Exception(
            f"No logseq blocks loaded for {path} (file size: {Path(path).stat().st_size})")

    blocks = parsed.blocks
    page_props = parsed.page_properties

    content = parsed.content
    content = content.replace("\t", "    ")
    content, medias = replace_media(
        content=content,
        media=None,
        mode="remove_media",
        strict=False,
        replace_image=True,
        replace_links=True,
        replace_sounds=False,
    )

    # create a single document then for each document add the properties of each block found in the doc
    docs = text_splitter.transform_documents([
        Document(
            page_content=content,
            metadata=page_props,
        )
    ])

    failed_blocks = []
    for b in blocks:
        b = copy.copy(b)
        props = b.properties.copy()
        for k, v in props.items():
            b.del_property(key=k)
            b.content = b.content.strip()
        cont = b.content.replace("\t", "    ")
        cont, _ = replace_media(
            content=cont,
            media=None,
            mode="remove_media",
            strict=False,
            replace_image=True,
            replace_links=True,
            replace_sounds=False,
        )
        if not cont:
            continue
        found = False
        for i, d in enumerate(docs):
            if i + 1 >= len(docs):
                next = ""
            else:
                next = docs[i+1].page_content
            if cont.strip() in d.page_content or (cont not in next and cont in d.page_content + next):

                # merge metadata dictionnaries
                for k, v in props.items():
                    if not v:
                        continue
                    if k not in docs[i].metadata:
                        docs[i].metadata[k] = v
                    elif docs[i].metadata[k] == v:
                        continue
                    elif isinstance(docs[i].metadata[k], list):
                        if isinstance(v, list):
                            docs[i].metadata[k].extend(v)
                        else:
                            docs[i].metadata[k].append(v)
                    else:
                        assert k in docs[i].metadata
                        assert not isinstance(docs[i].metadata[k], list)
                        assert docs[i].metadata[k] != v
                        if isinstance(v, list):
                            docs[i].metadata[k] = [docs[i].metadata[k]] + v
                        else:
                            docs[i].metadata[k] = [docs[i].metadata[k], v]
                found = True
                break
        if not found:
            failed_blocks.append(b)

    if failed_blocks:
        mess = f"Couldn't find {len(failed_blocks)} blocks after splitting the document"
        if len(failed_blocks) >= 0.5 * len(blocks):
            raise Exception(mess)
        else:
            red(mess)

    # sort and deduplicate metadata
    for i, d in enumerate(docs):
        for k, v in d.metadata.items():
            if isinstance(v, list):
                d.metadata[k] = list(sorted(list(set(v))))
            assert d.metadata[k], f"There shouldn't be any empty metadata value but key '{k}' of doc '{d}' is empty."

    return docs


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_local_audio(
    path: Union[str, PosixPath],
    file_hash: str,
    audio_backend: str,
    loaders_temp_dir: PosixPath,
    audio_unsilence: Optional[bool] = None,

    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,

    deepgram_kwargs: Optional[dict] = None,
) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"

    if audio_unsilence:
        red(f"Removing silence from audio file {path.name}")
        waveform, sample_rate = torchaudio.load(path)

        dur = waveform.shape[1] / sample_rate
        start = time.time()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            sox_effects,
        )
        elapsed = time.time() - start
        new_dur = waveform.shape[1] / sample_rate
        assert new_dur < dur, (
            f"Failed to remove silence for {path.name}:\n"
            f"Original duration: {dur:.1f}\n"
            f"New duration: {new_dur:.1f}\n"
        )
        assert new_dur > 10, (
            f"Silence removal ended up with a suspiciously short audio for {path.name}:\n"
            f"Original duration: {dur:.1f}\n"
            f"New duration: {new_dur:.1f}\n"
        )
        red(f"Removed silence from {path.name}: {dur:.1f} -> {new_dur:.1f} in {elapsed:.1f}s")

        unsilenced_path_wav = loaders_temp_dir / \
            f"unsilenced_audio_{uuid6.uuid6()}.wav"
        unsilenced_path_ogg = loaders_temp_dir / \
            f"unsilenced_audio_{uuid6.uuid6()}.ogg"
        assert not unsilenced_path_wav.exists()
        assert not unsilenced_path_ogg.exists()
        torchaudio.save(
            uri=str(unsilenced_path_wav.resolve().absolute()),
            src=waveform,
            sample_rate=sample_rate,
            format="wav",
        )
        # turn the .wav into .ogg
        ffmpeg.input(str(unsilenced_path_wav.resolve().absolute())).output(
            str(unsilenced_path_ogg.resolve().absolute())).run()
        unsilenced_hash = file_hasher({"path": unsilenced_path_ogg})

        old_path = path
        old_hash = file_hash
        path = unsilenced_path_ogg
        file_hash = unsilenced_hash

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
                    "source": str(Path(path)),
                },
            )
        ]
        if "duration" in content:
            docs[-1].metadata["duration"] = content["duration"]
        if "language" in content:
            docs[-1].metadata["language"] = content["language"]
        elif whisper_lang:
            docs[-1].metadata["language"] = whisper_lang

    elif audio_backend == "deepgram":
        assert whisper_prompt is None and whisper_lang is None, "Found args whisper_prompt or whisper_lang but selected deepgram backend for local_audio"
        content = transcribe_audio_deepgram(
            audio_path=path,
            audio_hash=file_hash,
            deepgram_kwargs=deepgram_kwargs,
        )
        assert len(content["results"]["channels"]
                   ) == 1, "unexpected deepgram output"
        assert len(content["results"]["channels"][0]
                   ["alternatives"]) == 1, "unexpected deepgram output"
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
        raise ValueError(
            f"Invalid audio backend: must be either 'deepgram' or 'whisper'. Not '{audio_backend}'")

    return docs


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_local_video(
    path: str,
    file_hash: str,
    audio_backend: str,
    loaders_temp_dir: PosixPath,
    audio_unsilence: Optional[bool] = None,

    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,

    deepgram_kwargs: Optional[dict] = None,
) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"

    audio_path = loaders_temp_dir / f"audio_from_video_{uuid6.uuid6()}.mp3"
    assert not audio_path.exists()

    # extract audio from video
    try:
        whi(f"Exporting audio from {path} to {audio_path} (this can take some time)")
        t = time.time()
        ffmpeg.input(
            path
        ).output(
            str(audio_path.resolve().absolute())
        ).run()
        whi(f"Done extracting audio in {time.time()-t:.2f}s")
    except Exception as err:
        red(
            f"Error when getting audio from video using ffmpeg. Retrying with pydub. Error: '{err}'")

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
    audio_hash = file_hasher({"path": audio_path})

    sub_loaders_temp_dir = loaders_temp_dir / "local_audio"
    sub_loaders_temp_dir.mkdir()

    return load_local_audio(
        path=audio_path,
        loaders_temp_dir=sub_loaders_temp_dir,
        file_hash=audio_hash,
        audio_backend=audio_backend,
        whisper_lang=whisper_lang,
        whisper_prompt=whisper_prompt,
        deepgram_kwargs=deepgram_kwargs,
        audio_unsilence=audio_unsilence,
    )


@optional_typecheck
@doc_loaders_cache.cache(ignore=["audio_path"])
def transcribe_audio_deepgram(
    audio_path: Union[str, PosixPath],
    audio_hash: str,
    deepgram_kwargs: Optional[dict] = None,
) -> dict:
    "Use whisper to transcribe an audio file"
    whi(f"Calling deepgram to transcribe {audio_path}")
    assert os.environ["WDOC_PRIVATEMODE"] == "false", (
        "Private mode detected, aborting before trying to use deepgram's API"
    )
    assert "DEEPGRAM_API_KEY" in os.environ and not os.environ[
        "DEEPGRAM_API_KEY"] == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE", "No environment variable DEEPGRAM_API_KEY found"

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
@doc_loaders_cache.cache(ignore=["audio_path"])
def transcribe_audio_whisper(
        audio_path: Union[PosixPath, str],
        audio_hash: str,
        language: Optional[str],
        prompt: Optional[str]) -> dict:
    "Use whisper to transcribe an audio file"
    whi(f"Calling openai's whisper to transcribe {audio_path}")
    assert os.environ["WDOC_PRIVATEMODE"] == "false", (
        "Private mode detected, aborting before trying to use openai's whisper"
    )

    assert "OPENAI_API_KEY" in os.environ and not os.environ[
        "OPENAI_API_KEY"] == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE", "No environment variable OPENAI_API_KEY found"

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


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
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


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
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


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
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
        red(
            f"Error when loading word document with docx2txt, trying with unstructured: '{err}'")
        loader = UnstructuredWordDocumentLoader(path)
        content = loader.load()
        docs = [Document(page_content=content)]

    return docs

@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_json_dict(
    path: str,
    json_dict_template: str,
    file_hash: str,
    metadata: Optional[Union[str, dict]] = None,
    json_dict_exclude_keys: Optional[List[str]] = None,
) -> List[Document]:
    assert Path(path).exists(), f"file not found: '{path}'"

    assert "{key}" in json_dict_template, "json_dict_template must contain '{key}'"
    assert "{value}" in json_dict_template, "json_dict_template must contain '{value}'"

    with Path(path).open("r") as f:
        d = json.load(f)
    assert d, "dict is empty"

    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not metadata:
        metadata = {}
    if json_dict_exclude_keys is None:
        json_dict_exclude_keys = []

    docs = []
    for k, v in d.items():
        if k in json_dict_exclude_keys:
            continue
        doc = Document(
            page_content=json_dict_template.replace("{key}", k).replace("{value}", v),
            metadata=metadata,
        )
        docs.append(doc)
    assert docs, "No document found in json_dict"

    return docs


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
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
            text = "\n".join(
                [doc.page_content for doc in loader.load()]).strip()
            assert text, "Empty text"
            if not title:
                if text.splitlines()[0].startswith("Title: "):
                    title = text.splitlines()[0].replace("Title: ", "", 1)
            text = text.split("Markdown Content:", 1)[1]
            text = markdownlinkparser_regex.sub(r'\1', text)  # remove links
            # remove markdown images for now as caption is disabled so it's just base64 or something like that, keep only a shorten image name
            text = markdownimage_regex.sub(md_shorten_image_name, text)
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
            g = goose3.Goose()
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
            loader = UnstructuredURLLoader([path])
            docs = loader.load()
            assert docs, "Empty docs when using UnstructuredURLLoader"
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
                f"Exception when using UnstructuredURLLoader to parse url: '{err}'"
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


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
def load_youtube_playlist(playlist_url: str) -> Any:
    with youtube_dl.YoutubeDL({"quiet": False}) as ydl:
        try:
            loaded = ydl.extract_info(playlist_url, download=False)
        except (KeyError, youtube_dl.utils.DownloadError, youtube_dl.utils.ExtractorError) as e:
            raise Exception(
                red(f"ERROR: Youtube playlist link skipped because : error during information \
        extraction from {playlist_url} : {e}")
            )
    return loaded


@optional_typecheck
@doc_loaders_cache.cache(ignore=["loader"])
def cached_yt_loader(
        loader: Any,
        path: str,
        add_video_info: bool,
        language: List[str],
        translation: Optional[List[str]]) -> List[Document]:
    yel(f"Not using cache for youtube {path}")
    docs = loader(
        path,
        add_video_info=add_video_info,
        language=language,
        translation=translation,
    ).load()
    return docs


@optional_typecheck
@doc_loaders_cache.cache(ignore=["path"])
def _pdf_loader(loader_name: str, path: str, file_hash: str) -> List[Document]:
    loader = pdf_loaders[loader_name](path)
    content = loader.load()
    assert isinstance(
        content, list), f"Output of {loader_name} is of type {type(content)}"
    assert all(isinstance(d, Document)
               for d in content), f"Output of {loader_name} contains elements that are not Documents: {[type(c) for c in docs]}"
    return content


@debug_return_empty
@optional_strip_unexp_args
def load_pdf(
    path: str,
    text_splitter: TextSplitter,
    debug: bool,
    file_hash: str,
    doccheck_min_lang_prob: float = min_lang_prob,
    doccheck_min_token: int = min_token,
    doccheck_max_token: int = max_token,
    doccheck_max_lines: int = max_lines,
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

    info = magic.from_file(path)
    if "pdf" not in info.lower():
        yel(f"WARNING: magic says that your PDF is not a PDF:\npath={path}\nMagic info={info}")

    pbar = tqdm(total=len(pdf_loaders),
                desc=f"Parsing PDF {name}", unit="loader")
    for loader_name in pdf_loaders:
        pbar.desc = f"Parsing PDF {name} with {loader_name}"
        try:
            if debug:
                red(f"Trying to parse {path} using {loader_name}")

            with signal_timeout(
                timeout=pdf_loader_max_timeout,
                exception=TimeoutPdfLoaderError,
                ):
                docs = _pdf_loader(loader_name, path, file_hash)

            pbar.update(1)

            for i, d in enumerate(docs):
                docs[i].page_content = ftfy.fix_text(d.page_content)
                if "pdf_loader_name" not in docs[i].metadata:
                    docs[i].metadata["pdf_loader_name"] = loader_name

            prob = check_docs_tkn_length(
                docs=docs,
                identifier=path,
                check_language=True,
                min_lang_prob=doccheck_min_lang_prob,
                min_token=doccheck_min_token,
                max_token=doccheck_max_token,
                max_lines=doccheck_max_lines,
            )

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
            yel(f"Error when parsing '{path}' with {loader_name}: {err}\nMagic info: {info}")
            if "content" not in locals():
                pbar.update(1)

    pbar.close()
    assert probs.keys(), f"No pdf parser succedded to parse {path}"

    # no loader worked, exiting
    if not loaded_docs:
        raise Exception(f"No pdf parser worked for {path}")

    max_prob = max([v for v in probs.values()])

    if debug:
        yel(f"Language probability after parsing {path}: {probs}")

    return loaded_docs[[name for name in probs if probs[name] == max_prob][0]]


@optional_typecheck
def find_online_media(
    url: str,
    online_media_url_regex: Optional[str] = None,
    online_media_resourcetype_regex: Optional[str] = None,
    headless: bool = True,
    ) -> dict:

    @optional_typecheck
    def check_browser_installation(browser_type: str) -> bool:
        try:
            with playwright.sync_api.sync_playwright() as p:
                browser = getattr(p, browser_type).launch()
                browser.close()
            return True
        except Exception as err:
            if "p" in locals():
                red(str(p))
            red(str(err))
            return False

    # the media request will be stored in this dict
    video_urls = {
        "url_regex": [],
        "resourcetype_regex": [],
        "media": [],
        "mpeg": [],
        "mp4": [],
        "mp3": [],
        "m3u": [],
    }
    if online_media_url_regex:
        online_media_url_regex = re.compile(online_media_url_regex)
    if online_media_resourcetype_regex:
        online_media_resourcetype_regex = re.compile(online_media_resourcetype_regex)
    nonmedia_urls = []

    @optional_typecheck
    def request_filter(req) -> None:
        if online_media_url_regex is not None and online_media_url_regex.match(req.url):
            video_urls["url_regex"].append(req.url)
        elif online_media_resourcetype_regex is not None and online_media_resourcetype_regex.match(req.resource_type):
            video_urls["resourcetype_regex"].append(req.url)
        elif req.resource_type == "media":
            video_urls["media"].append(req.url)
        elif "media" in req.resource_type:
            video_urls["media"].append(req.url)
        elif "mpeg" in req.resource_type:
            video_urls["mpeg"].append(req.url)
        elif "m3u" in req.resource_type or ".m3u" in req.url:
            video_urls["m3u"].append(req.url)
        elif "mp3" in req.resource_type or ".mp3" in req.url:
            video_urls["mp3"].append(req.url)
        elif "mp4" in req.resource_type or ".mp4" in req.url:
            video_urls["mp4"].append(req.url)
        else:
            nonmedia_urls.append(req.url)

    if check_browser_installation("firefox"):
        installed = "firefox"
    elif check_browser_installation("chromium"):
        installed = "chromium"
    else:
        raise Exception("Couldn't launch either firefox or chromium using playwright. Maybe try running 'playwright install'")

    with playwright.sync_api.sync_playwright() as p:
        browser = getattr(p, installed).launch(headless=headless)

        context = browser.new_context(
            java_script_enabled=True,
            geolocation={
                "latitude": 38.8954381,
                "longitude": -77.0312812,
            },
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            ignore_https_errors=True,

        )
        page = context.new_page()

        # start logging requests
        page.on("request", lambda request: request_filter(request))
        browser.on("request", lambda request: request_filter(request))
        context.on("request", lambda request: request_filter(request))

        # load page
        page.goto(url)
        page.wait_for_load_state('networkidle')

        # Scroll the page to trigger lazy-loaded content
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)  # Wait for X seconds after scrolling
        page.evaluate("window.scrollTo(0, 0)")

        # Try to click on video play buttons
        for trial in [
            '[class*="play-button"]',
            '[class*="playback"]',
            '[class*="play-back"]',
        ]:
            playback_elements = page.query_selector_all(trial)
            for element in playback_elements:
                try:
                    element.click(timeout=500)
                    print(f"Clicked element: {element.evaluate('el => el.outerHTML')}")
                    page.wait_for_timeout(1000)  # Wait for X seconds after each click
                except Exception:
                    pass
        play_button_selectors = [
            '[aria-label="Play"]',
            '.ytp-play-button',
            '.play-button',
            '[aria-label="播放"]',
            'div.avp-icon.avp-icon-playback',
        ]
        for selector in play_button_selectors:
            try:
                page.click(selector, timeout=500)
            except Exception:
                pass


        if not any(v for v in video_urls.values()):
            # Wait a bit more for any video to start loading
            page.wait_for_timeout(10000)

        browser.close()

    # deduplicate urls
    for k, v in video_urls.items():
        video_urls[k] = list(set(v))

    return video_urls

@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_online_media(
    path: str,
    audio_backend: str,
    loaders_temp_dir: PosixPath,

    audio_unsilence: Optional[bool] = None,
    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,

    deepgram_kwargs: Optional[dict] = None,

    online_media_url_regex: Optional[str] = None,
    online_media_resourcetype_regex: Optional[str] = None,
    ) -> List[Document]:

    urls_to_try = [path]
    extra_media = find_online_media(
        url=path,
        online_media_url_regex=online_media_url_regex,
        online_media_resourcetype_regex=online_media_resourcetype_regex,
    )
    for k in [
        "url_regex",
        "resourcetype_regex",
        "media",
        "mpeg",
        "mp4",
        "mp3",
        "m3u",
    ]:
        urls_to_try.extend(extra_media[k])
    urls_to_try = list(set(urls_to_try))
    whi(f"Found {len(urls_to_try)} urls to try to get the media")


    @optional_typecheck
    def dl_audio_from_url(trial: int, url: str) -> PosixPath:
        file_name = loaders_temp_dir / \
            f"online_media_{uuid6.uuid6()}"  # without extension!
        ydl_opts = {
            'format': 'bestaudio/best',
            # 'force_generic_extractor': True,
            # 'default_search': 'auto',
            # 'match_filter': lambda x: None,
            'hls_prefer_native': True,
            'postprocessors': [{ 'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192',
            }],
            # with extension
            'outtmpl': f'{file_name.absolute().resolve()}.%(ext)s',
            'verbose': is_verbose,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        candidate = []
        for f in loaders_temp_dir.iterdir():
            if file_name.name in f.name:
                candidate.append(f)
        assert len(candidate), f"Audio file of {url} failed to download?"
        assert len(
            candidate) == 1, f"Multiple audio file found for video: '{candidate}'"
        audio_file = candidate[0].absolute()
        return audio_file


    audio_file = None
    good_url = None
    for iurl, url in enumerate(urls_to_try):
        try:
            audio_file = dl_audio_from_url(trial=iurl, url=url)
            good_url = url
            break
        except Exception as err:
            red(f"Failed #{iurl+1}/{len(urls_to_try)} to download a media from url '{url}': '{err}'")

    assert audio_file is not None, f"Failed to find suitable media for url '{path}'"

    audio_hash = file_hasher({"path": str(Path(audio_file).absolute())})
    audio_path = loaders_temp_dir / f"audio_from_video_{uuid6.uuid6()}.mp3"
    assert not audio_path.exists()

    # extract audio from video
    try:
        whi(f"Exporting audio from {audio_file} to {audio_path} (this can take some time)")
        t = time.time()
        ffmpeg.input(
            audio_file,
        ).output(
            str(audio_path.resolve().absolute())
        ).run()
        whi(f"Done extracting audio in {time.time()-t:.2f}s")
    except Exception as err:
        red(
            f"Error when getting audio from video using ffmpeg. Retrying with pydub. Error: '{err}'")

        try:
            Path(audio_path).unlink(missing_ok=True)
            audio = pydub.AudioSegment.from_file(audio_file)
            # extract audio from video
            whi(f"Extracting audio from {audio_file} to {audio_path} (this can take some time)")
            t = time.time()
            audio.export(audio_path, format="mp3")
            whi(f"Done extracting audio in {time.time()-t:.2f}s")
        except Exception as err:
            raise Exception(
                f"Error when getting audio from video using ffmpeg: '{err}'")

    assert Path(audio_path).exists(), f"FileNotFound: {audio_path}"

    # now need the hash from the mp3, not video
    audio_hash = file_hasher({"path": audio_path})

    sub_loaders_temp_dir = loaders_temp_dir / "local_audio"
    sub_loaders_temp_dir.mkdir()
    parsed_audio = load_local_audio(
        path=audio_path,
        loaders_temp_dir=sub_loaders_temp_dir,
        file_hash=audio_hash,
        audio_backend=audio_backend,
        whisper_lang=whisper_lang,
        whisper_prompt=whisper_prompt,
        deepgram_kwargs=deepgram_kwargs,
        audio_unsilence=audio_unsilence,
    )

    for ipa, pa in enumerate(parsed_audio):
        parsed_audio[ipa].metadata["online_media_url"] = str(good_url)

    return parsed_audio
