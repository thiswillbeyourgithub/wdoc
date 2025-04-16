import os
from loguru import logger

# sensible default user agent
os.environ["USER_AGENT"] = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
)

if str(os.environ.get("WDOC_APPLY_ASYNCIO_PATCH", None)).lower() in [
    "1",
    "true",
    "yes",
]:
    # Apply asyncio patch if enabled
    # needed to fix ollama 'event loop closed' error
    # thanks to https://github.com/BerriAI/litellm/pull/7625/files
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception as e:
        logger.error(f"Failed to patch asyncio loop using nest_asyncio. Error: '{e}'")

from . import utils
from .wdoc import wdoc

__VERSION__ = wdoc.VERSION


__all__ = ["wdoc", "utils"]

if str(os.environ.get("WDOC_APPLY_ASYNCIO_PATCH", None)).lower() in [
    "1",
    "true",
    "yes",
]:
    assert utils.env.env.WDOC_APPLY_ASYNCIO_PATCH
