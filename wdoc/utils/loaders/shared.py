import uuid6
from typing import Callable
import signal
from functools import wraps
from contextlib import contextmanager
from langchain.docstore.document import Document

from wdoc.utils.env import env


def debug_return_empty(func: Callable) -> Callable:
    if env.WDOC_EMPTY_LOADER:

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
