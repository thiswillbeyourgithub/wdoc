from beartype.typing import List, Union
from pathlib import Path
from langchain.docstore.document import Document

from langchain_community.document_loaders import UnstructuredPowerPointLoader
from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.loaders.misc import optional_strip_unexp_args, doc_loaders_cache


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_powerpoint(
    path: Union[str, Path],
    file_hash: str,
) -> List[Document]:
    path = Path(path)
    assert path.exists(), f"file not found: '{path}'"
    loader = UnstructuredPowerPointLoader(path)
    content = loader.load()

    docs = [
        Document(
            page_content=content,
            metadata={},
        )
    ]
    return docs
