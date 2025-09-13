import re

import goose3
from beartype.typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PlaywrightURLLoader,
    SeleniumURLLoader,
    UnstructuredURLLoader,
    WebBaseLoader,
)
from loguru import logger

from wdoc.utils.loaders.shared import (
    debug_return_empty,
    get_url_title,
    markdownimage_regex,
)
from wdoc.utils.misc import (
    check_docs_tkn_length,
    doc_loaders_cache,
    optional_strip_unexp_args,
)

markdownlinkparser_regex = re.compile(r"\[([^\]]+)\]\(http[s]?://[^)]+\)")


def md_shorten_image_name(md_image: re.Match) -> str:
    "turn a markdown image link into just the name"
    name = md_image.group(1)
    if len(name) <= 16:
        return name
    else:
        return name[:8] + "…" + name[-8:]


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
def load_url(path: str, title=None) -> List[Document]:
    logger.info(f"Loading url: '{path}'")

    # even if loading fails the title might be found so trying to keep
    # the first working title across trials
    if title == "Untitled":
        title = None

    loaded_success = False
    if not loaded_success:
        try:
            loader = WebBaseLoader("https://r.jina.ai/" + path, raise_for_status=True)
            text = "\n".join([doc.page_content for doc in loader.load()]).strip()
            assert text, "Empty text"
            if not title:
                if text.splitlines()[0].startswith("Title: "):
                    title = text.splitlines()[0].replace("Title: ", "", 1)
            text = text.split("Markdown Content:", 1)[1]
            text = markdownlinkparser_regex.sub(r"\1", text)  # remove links
            # remove markdown images for now as caption is disabled so it's just base64 or something like that, keep only a shorten image name
            text = markdownimage_regex.sub(md_shorten_image_name, text)
            docs = [
                Document(
                    page_content=text,
                    metadata={
                        "parser": "jinareader",
                    },
                )
            ]
            if title:
                for doc in docs:
                    doc.metadata["title"] = title
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            logger.warning(f"Exception when using jina reader to parse url: '{err}'")

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
            logger.warning(f"Exception when using playwright to parse url: '{err}'")

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
            logger.warning(
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
            logger.warning(
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
                if "title" in docs[0].metadata and docs[0].metadata["title"]:
                    title = docs[0].metadata["title"]
                elif article.title:
                    title = article.title
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            logger.warning(f"Exception when using goose to parse url: '{err}'")

    if not loaded_success:
        try:
            loader = UnstructuredURLLoader([path])
            docs = loader.load()
            assert docs, "Empty docs when using UnstructuredURLLoader"
            if not title and "title" in docs[0].metadata and docs[0].metadata["title"]:
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            logger.warning(
                f"Exception when using UnstructuredURLLoader to parse url: '{err}'"
            )

    if not loaded_success:
        try:
            loader = WebBaseLoader(path, raise_for_status=True)
            docs = loader.load()
            assert docs, "Empty docs when using html"
            if not title and "title" in docs[0].metadata and docs[0].metadata["title"]:
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)
            loaded_success = True
        except Exception as err:
            logger.warning(
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
