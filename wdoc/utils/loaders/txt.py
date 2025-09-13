from pathlib import Path

from beartype.typing import List, Union
from langchain.docstore.document import Document
from loguru import logger

from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import optional_strip_unexp_args


@debug_return_empty
@optional_strip_unexp_args
def load_txt(path: Union[str, Path], file_hash: str) -> List[Document]:
    path = Path(path)
    logger.info(f"Loading txt: '{path}'")
    assert path.exists(), f"file not found: '{path}'"
    content = path.read_text()
    docs = [Document(page_content=content, metadata={})]
    return docs
