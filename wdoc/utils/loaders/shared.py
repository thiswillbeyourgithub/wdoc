import re
import signal
from contextlib import contextmanager
from functools import cache as memoize
from functools import wraps

from beartype.typing import Callable, Union
from langchain.docstore.document import Document

from wdoc.utils.env import env

markdownimage_regex = re.compile(
    r"!\[([^\]]*)\]\s*(\([^\)]+\)|\[[^\]]+\])", flags=re.MULTILINE
)


def debug_return_empty(func: Callable) -> Callable:
    if env.WDOC_EMPTY_LOADER:
        import uuid6

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


@contextmanager
def signal_timeout(timeout: int, exception: Exception):
    "disabled in some joblib backend"
    assert timeout > 0, f"Invalid timeout: {timeout}"

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


@memoize
def get_url_title(url: str) -> Union[str, type(None)]:
    """if the title of the url is not loaded from the loader, trying as last
    resort with this one"""
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(url, raise_for_status=True)
    docs = loader.load()
    if "title" in docs[0].metadata and docs[0].metadata["title"]:
        return docs[0].metadata["title"]
    else:
        return None
