# needed to fix ollama 'event loop closed' error
# thanks to https://github.com/BerriAI/litellm/pull/7625/files
import nest_asyncio

nest_asyncio.apply()

from . import utils
from .wdoc import wdoc

__VERSION__ = wdoc.VERSION


__all__ = ["wdoc", "utils"]
