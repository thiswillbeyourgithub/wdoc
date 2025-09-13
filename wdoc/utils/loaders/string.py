from beartype.typing import List
from langchain.docstore.document import Document
from loguru import logger
from prompt_toolkit import prompt

from wdoc.utils.loaders.shared import debug_return_empty


@debug_return_empty
def load_string() -> List[Document]:
    logger.info("Loading string")
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
