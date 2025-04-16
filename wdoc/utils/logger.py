"""
Code related to loggings, coloured logs, etc.
"""

import json
import warnings
from pathlib import Path
from textwrap import dedent

import rtoml
from beartype.typing import Callable, Dict, List, Optional, Type, Union
from loguru import logger
from platformdirs import user_cache_dir, user_log_dir
from rich.console import Console
from rich.markdown import Markdown

from .flags import md_printing_disabled, is_debug
from .typechecker import optional_typecheck

# ignore warnings from beautiful soup
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

cache_dir = Path(user_cache_dir(appname="wdoc"))
cache_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path(user_log_dir(appname="wdoc"))
log_dir.mkdir(exist_ok=True, parents=True)
log_file = log_dir / "logs.txt"
log_file.touch(exist_ok=True)

# logger
logger.add(
    log_file,
    rotation="100MB",
    retention=5,
    format="{time:YYYY-MM-DD at HH:mm}|{level}|wdoc|{thread}|{process}|{function}|{line}|{message}",
    level="DEBUG" if is_debug else "INFO",
    enqueue=True,
    colorize=True,
    backtrace=True if is_debug else None,
    diagnose=True if is_debug else None,
)
# delete any additional log file
# (log_dir / "logs.txt.4").unlink(missing_ok=True)

console = Console()


@optional_typecheck
def md_printer(message: str, color: Optional[str] = None) -> str:
    "markdown printing"
    message = dedent(message)
    if not md_printing_disabled:
        md = Markdown(message)
        console.print(md, style=color)
    else:
        logger.info(message)
    return message


@optional_typecheck
def set_help_md_as_docstring(obj: Union[Type, Callable]) -> Union[Type, Callable]:
    "set the docstring of wdoc class to wdoc/docs/help.md's content"
    help_file = Path(__file__).parent.parent / "docs/help.md"
    if not help_file.exists():
        logger.warning(
            f"Couldn't find help.md file as '{help_file}'. You can read it at this URL instead: https://github.com/thiswillbeyourgithub/wdoc/blob/main/wdoc/docs/help.md"
        )
        helpcont = "Help documentation not found. Please refer to online documentation."
    else:
        helpcont = help_file.read_text().strip()
        if not helpcont:
            logger.warning("Help documentation file is empty")
            helpcont = (
                "Help documentation is empty. Please refer to online documentation."
            )
    obj.__doc__ = "# Content of wdoc/docs/help.md\n\n" + helpcont
    if isinstance(obj, type):
        obj.__init__.__doc__ = helpcont
    return obj


@optional_typecheck
def set_parse_file_help_md_as_docstring(
    obj: Union[Type, Callable],
) -> Union[Type, Callable]:
    "set the docstring of wdoc.parse_file to wdoc/docs/parse_file_help.md's content"
    parsefilehelp_file = Path(__file__).parent.parent / "docs/parse_file_help.md"
    if not parsefilehelp_file.exists():
        logger.warning(
            f"Couldn't find parse_file_help.md file as '{parsefilehelp_file}'. You can read it at this URL instead: https://github.com/thiswillbeyourgithub/wdoc/blob/main/wdoc/docs/parse_file_help.md"
        )
        parsefilehelp = "Parse file help documentation not found. Please refer to online documentation."
    else:
        parsefilehelp = parsefilehelp_file.read_text().strip()
        if not parsefilehelp:
            logger.warning("Parse file help documentation is empty")
            parsefilehelp = "Parse file help documentation is empty. Please refer to online documentation."
    obj.__doc__ = "# Content of wdoc/docs/parse_file_help.md\n\n" + parsefilehelp
    return obj
