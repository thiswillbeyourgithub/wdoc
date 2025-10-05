import os
import logging
from loguru import logger

from beartype.claw import beartype_package
from beartype import BeartypeConf

if os.environ.get("WDOC_TYPECHECKING", "") == "crash":
    beartype_package("wdoc")
elif os.environ.get("WDOC_TYPECHECKING", "") == "warn":
    beartype_package(
        "wdoc",
        conf=BeartypeConf(violation_type=UserWarning),
    )
elif os.environ.get("WDOC_TYPECHECKING", "") == "disabled":
    pass

# Suppress faiss INFO logs
logging.getLogger("faiss").setLevel(logging.WARNING)

# sensible default user agent
os.environ["USER_AGENT"] = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
)

failed = False
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
        failed = str(e)

# set default logging log level to info
logging.getLogger("wdoc").setLevel(logging.INFO)

from wdoc.utils import logger  # make sure to setup the logs first

if failed:
    logger.error(f"Failed to patch asyncio loop using nest_asyncio. Error: '{failed}'")

from wdoc import utils
from wdoc.wdoc import wdoc

__VERSION__ = wdoc.VERSION


__all__ = ["wdoc", "utils"]

if str(os.environ.get("WDOC_APPLY_ASYNCIO_PATCH", None)).lower() in [
    "1",
    "true",
    "yes",
]:
    assert utils.env.env.WDOC_APPLY_ASYNCIO_PATCH
