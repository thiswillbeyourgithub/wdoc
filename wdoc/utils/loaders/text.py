import json

from beartype.typing import List, Optional, Union
from langchain.docstore.document import Document
from loguru import logger

from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import optional_strip_unexp_args


@debug_return_empty
@optional_strip_unexp_args
def load_text(
    path: str,
    file_hash: str,
    metadata: Optional[Union[str, dict]] = None,
) -> List[Document]:
    logger.info(f"Loading text input: '{path}'")
    text = path.strip()
    assert text, "Empty text"
    if metadata is None:
        metadata = {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    docs = [
        Document(
            page_content=text,
            metadata=metadata,
        )
    ]
    return docs
