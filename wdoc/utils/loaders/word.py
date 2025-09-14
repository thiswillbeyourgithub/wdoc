from pathlib import Path

from beartype.typing import List, Union
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)
from loguru import logger

from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import (
    check_docs_tkn_length,
    doc_loaders_cache,
    optional_strip_unexp_args,
)


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_word(
    path: Union[str, Path],
    file_hash: str,
) -> List[Document]:
    path = Path(path)
    assert path.exists(), f"file not found: '{path}'"
    try:
        loader = Docx2txtLoader(path)
        content = loader.load()
        if isinstance(content, str):
            docs = [Document(page_content=content)]
        else:
            assert isinstance(content, List) and all(
                isinstance(c, Document) for c in content
            ), f"unexpected type of content: {str(content)[:1000]}"
            docs = content
        check_docs_tkn_length(docs, path)
    except Exception as err:
        logger.warning(
            f"Error when loading word document with docx2txt, trying with unstructured: '{err}'"
        )
        loader = UnstructuredWordDocumentLoader(path)
        content2 = loader.load()
        if isinstance(content2, str):
            docs = [Document(page_content=content2)]
        else:
            assert isinstance(content2, List) and all(
                isinstance(c, Document) for c in content2
            ), f"unexpected type of content: {str(content2)[:1000]}"
            docs = content2
        check_docs_tkn_length(docs, path)

    return docs
