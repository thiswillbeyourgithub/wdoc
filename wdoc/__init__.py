# needed to fix ollama 'event loop closed' error
# thanks to https://github.com/BerriAI/litellm/pull/7625/files
try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception as e:
    print(f"Failed to patch asyncio loop using nest_asyncio. Error: '{e}'")

from . import utils
from .wdoc import wdoc

__VERSION__ = wdoc.VERSION


__all__ = ["wdoc", "utils"]
