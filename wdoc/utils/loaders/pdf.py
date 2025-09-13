import signal
import sys
import tempfile
import traceback
from functools import partial
from pathlib import Path

import ftfy
import openparse
import requests
from beartype.typing import List, Optional, Union
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import (
    OnlinePDFLoader,
    PDFMinerLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from loguru import logger
from tqdm import tqdm
from unstructured.cleaners.core import clean_extra_whitespace

from wdoc.utils.env import env, is_linux, is_out_piped
from wdoc.utils.errors import TimeoutPdfLoaderError
from wdoc.utils.loaders.shared import debug_return_empty, signal_timeout
from wdoc.utils.misc import (
    check_docs_tkn_length,
    doc_loaders_cache,
    file_hasher,
    max_token,
    min_lang_prob,
    min_token,
    optional_strip_unexp_args,
)

try:
    import pdftotext
except Exception as err:
    if env.WDOC_VERBOSE:
        logger.warning(f"Failed to import optional package 'pdftotext': '{err}'")
        if is_linux:
            logger.warning(
                "On linux, you can try to install pdftotext with :\nsudo "
                "apt install build-essential libpoppler-cpp-dev pkg-config "
                "python3-dev\nThen:\nuv pip install pdftotext"
            )


class OpenparseDocumentParser:
    def __init__(
        self,
        path: Union[str, Path],
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
        assert nodes, "No nodes found"
        del base_metadata["nodes"]

        docs = []
        for node in nodes:
            meta = base_metadata.copy()
            meta.update(node)
            assert meta["bbox"], "No bbox found"
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
    "pymupdf": PyMuPDFLoader,  # good for metadata
    "pdfplumber": PDFPlumberLoader,  # good for metadata
    "pdfminer": PDFMinerLoader,  # little metadata
    "pypdfloader": PyPDFLoader,  # little metadata
    "pypdfium2": PyPDFium2Loader,  # little metadata
    # "pdftotext": None,  # optional support, see below
    "openparse": OpenparseDocumentParser,  # gets page number too, finds individual elements, kinda slow but good, optional table support
    "unstructured_fast": partial(
        UnstructuredPDFLoader,
        strategy="fast",
    ),
    "unstructured_elements_fast": partial(
        UnstructuredPDFLoader,
        mode="elements",
        strategy="fast",
    ),
    "unstructured_hires": partial(
        UnstructuredPDFLoader,
        strategy="hi_res",
    ),
    "unstructured_elements_hires": partial(
        UnstructuredPDFLoader,
        mode="elements",
        strategy="hi_res",
    ),
    "unstructured_fast_clean_table": partial(
        UnstructuredPDFLoader,
        strategy="fast",
        post_processors=[clean_extra_whitespace],
        infer_table_structure=True,
        # languages=["en"],
    ),
    "unstructured_elements_fast_clean_table": partial(
        UnstructuredPDFLoader,
        mode="elements",
        strategy="fast",
        post_processors=[clean_extra_whitespace],
        infer_table_structure=True,
        # languages=["en"],
    ),
    "unstructured_hires_clean_table": partial(
        UnstructuredPDFLoader,
        strategy="hi_res",
        post_processors=[clean_extra_whitespace],
        infer_table_structure=True,
        # languages=["en"],
    ),
    "unstructured_elements_hires_clean_table": partial(
        UnstructuredPDFLoader,
        mode="elements",
        strategy="hi_res",
        post_processors=[clean_extra_whitespace],
        infer_table_structure=True,
        # languages=["en"],
    ),
}

# pdftotext is kinda weird to install on windows so support it
# only if it's correctly imported
if "pdftotext" in sys.modules:

    class pdftotext_loader_class:
        "simple wrapper for pdftotext to make it load by pdf_loader"

        def __init__(self, path: Union[str, Path]) -> None:
            self.path = path

        def load(self) -> List[Document]:
            with open(self.path, "rb") as f:
                docs = [
                    Document(page_content=d, metadata={"page": idoc})
                    for idoc, d in enumerate(pdftotext.PDF(f))
                ]
                return docs

    pdf_loaders["pdftotext"] = pdftotext_loader_class

pdf_loader_max_timeout = env.WDOC_MAX_PDF_LOADER_TIMEOUT
if env.WDOC_VERBOSE:
    if pdf_loader_max_timeout > 0:
        logger.warning(f"Will use a PDF loader timeout of {pdf_loader_max_timeout}s")
    else:
        logger.warning("Not using a pdf loader timeout")


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
def load_online_pdf(
    path: str,
    text_splitter: TextSplitter,
    file_hash: str,
    pdf_parsers: Union[str, List[str]] = "pymupdf",  # used only if online loading fails
    doccheck_min_lang_prob: float = min_lang_prob,
    doccheck_min_token: int = min_token,
    doccheck_max_token: int = max_token,
) -> List[Document]:
    logger.info(f"Loading online pdf: '{path}'")

    try:
        response = requests.get(path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

        docs = load_pdf(
            path=temp_file.name,
            text_splitter=text_splitter,
            file_hash=file_hasher({"path": temp_file.name}),
            pdf_parsers=pdf_parsers,
            doccheck_min_lang_prob=doccheck_min_lang_prob,
            doccheck_min_token=doccheck_min_token,
            doccheck_max_token=doccheck_max_token,
        )
        return docs

    except Exception as err:
        logger.warning(
            f"Failed parsing online PDF {path} by downloading it and trying to parse because of error '{err}'. Retrying one last time with OnlinePDFLoader."
        )
        loader = OnlinePDFLoader(path)
        if pdf_loader_max_timeout > 0:
            with signal_timeout(
                timeout=pdf_loader_max_timeout,
                exception=TimeoutPdfLoaderError,
            ):
                docs = loader.load()
            try:
                signal.alarm(0)  # disable alarm again just in case
            except Exception:
                pass
        else:
            docs = loader.load()

        return docs


@doc_loaders_cache.cache(ignore=["path"])
def _pdf_loader(loader_name: str, path: str, file_hash: str) -> List[Document]:
    loader = pdf_loaders[loader_name](path)
    docs = loader.load()
    assert isinstance(docs, list), f"Output of {loader_name} is of type {type(docs)}"
    assert all(
        isinstance(d, Document) for d in docs
    ), f"Output of {loader_name} contains elements that are not Documents: {[type(c) for c in docs]}"
    return docs


@debug_return_empty
@optional_strip_unexp_args
def load_pdf(
    path: Union[str, Path],
    text_splitter: TextSplitter,
    file_hash: str,
    pdf_parsers: Union[str, List[str]] = "pymupdf",
    doccheck_min_lang_prob: float = min_lang_prob,
    doccheck_min_token: int = min_token,
    doccheck_max_token: int = max_token,
) -> List[Document]:
    path = Path(path)
    logger.info(f"Loading pdf: '{path}'")
    assert path.exists(), f"file not found: '{path}'"
    name = path.name
    if len(name) > 30:
        name = name[:15] + "..." + name[-15:]

    if isinstance(pdf_parsers, str):
        pdf_parsers = pdf_parsers.strip().split(",")
    assert pdf_parsers, "No pdf_parsers found"
    assert len(pdf_parsers) == len(
        set(pdf_parsers)
    ), f"You pdf_parsers list contains non unique elements. List: {pdf_parsers}"
    for pdfp in pdf_parsers:
        assert (
            pdfp in pdf_loaders
        ), f"The PDF loader '{pdfp}' was not present in the pdf_loaders keys. Your 'pdf_parsers' argument seems wrong."

    loaded_docs = {}
    # using language detection to keep the parsing with the highest lang
    # probability
    probs = {}
    passed_errs = []
    warned_errs = []

    info = "magic not run"
    try:
        import magic

        info = str(magic.from_file(path))
    except Exception as err:
        logger.warning(f"Failed to run python-magic: '{err}'")
    if "pdf" not in info.lower():
        logger.debug(
            f"WARNING: magic says that your PDF is not a PDF:\npath={path}\nMagic info='{info}'"
        )

    pbar = tqdm(
        total=len(pdf_parsers),
        desc=f"Parsing PDF {name}",
        unit="loader",
        disable=is_out_piped,
    )
    for loader_name in pdf_parsers:
        pbar.desc = f"Parsing PDF {name} with {loader_name}"
        try:
            if env.WDOC_DEBUG:
                logger.warning(f"Trying to parse {path} using {loader_name}")

            if pdf_loader_max_timeout > 0:
                with signal_timeout(
                    timeout=pdf_loader_max_timeout,
                    exception=TimeoutPdfLoaderError,
                ):
                    docs = _pdf_loader(loader_name, str(path), file_hash)
                try:
                    signal.alarm(0)  # disable alarm again just in case
                except Exception:
                    pass
            else:
                docs = _pdf_loader(loader_name, path, file_hash)

            pbar.update(1)

            for i, d in enumerate(docs):
                try:
                    pc = ftfy.fix_text(d.page_content)
                    docs[i].page_content = pc
                    # stupid pydantic error
                except Exception as err:
                    if "'dict' object has no attribute 'add'" in str(err):
                        pass
                    else:
                        raise
                if "pdf_loader_name" not in docs[i].metadata:
                    docs[i].metadata["pdf_loader_name"] = loader_name

            prob = check_docs_tkn_length(
                docs=docs,
                identifier=path,
                check_language=True,
                min_lang_prob=doccheck_min_lang_prob,
                min_token=doccheck_min_token,
                max_token=doccheck_max_token,
            )

            if prob >= 0.5:
                # only consider it okay if decent quality
                probs[loader_name] = prob
                loaded_docs[loader_name] = docs
                if prob > 0.95:
                    # select this one as its bound to be okay
                    logger.info(
                        f"Early stopping of PDF parsing because {loader_name} has prob {prob} for {path}"
                    )
                    break
            else:
                logger.info(
                    f"Ignore parsing by {loader_name} of '{path}' as it seems of poor quality: prob={prob}"
                )
                continue

            if len(probs.keys()) >= 3:
                # if more than 3 worked, take the best among them to save
                # time on running all the others
                break
        except Exception as err:
            if pdf_loader_max_timeout > 0:
                try:
                    signal.alarm(0)  # disable alarm again just in case
                except Exception:
                    pass
            if "content" not in locals():
                pbar.update(1)
            logger.debug(
                f"Error when parsing '{path}' with {loader_name}: {err}\nMagic info='{info}'"
            )

            if (
                str(err) in passed_errs
                and str(err) not in warned_errs
                and "token" not in str(err)
            ):
                exc_type, exc_obj, exc_tb = sys.exc_info()
                formatted_tb = "\n".join(
                    [str(li).strip() for li in traceback.format_tb(exc_tb)]
                )
                logger.warning(
                    f"The same error happens to multiple pdf loader, something is fishy.\nFull traceback:\n{formatted_tb}"
                )
                warned_errs.append(str(err))
            passed_errs.append(str(err))

    pbar.close()
    assert probs.keys(), f"No pdf parser succeeded to parse {path}"

    # no loader worked, exiting
    if not loaded_docs:
        raise Exception(f"No pdf parser worked for {path}")

    max_prob = max([v for v in probs.values()])

    if env.WDOC_DEBUG:
        logger.debug(f"Language probability after parsing {path}: {probs}")

    return loaded_docs[[name for name in probs if probs[name] == max_prob][0]]
