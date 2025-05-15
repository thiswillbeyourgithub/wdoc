"""
Code related to loggings, coloured logs, etc.
"""

import sys
import warnings
from pathlib import Path
from textwrap import dedent

from beartype.typing import Callable, Optional, Type, Union
from loguru import logger
from platformdirs import user_log_dir
from rich.console import Console
from rich.markdown import Markdown

from wdoc.utils.env import env, is_out_piped, is_input_piped
from wdoc.utils.typechecker import optional_typecheck

# ignore warnings from beautiful soup
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

log_dir = Path(user_log_dir(appname="wdoc"))
log_dir.mkdir(exist_ok=True, parents=True)
log_file = log_dir / "logs.txt"
log_file.touch(exist_ok=True)

log_level = "INFO"
if env.WDOC_VERBOSE:
    log_level = "DEBUG"
if env.WDOC_DEBUG:
    log_level = "DEBUG"
if is_out_piped:
    log_level = "CRITICAL"

# reset the default handler of stderr otherwise the user always see all log levels apparently
handlers = logger._core.handlers
if (
    len(handlers) == 1
    and next(iter(handlers.values())).levelno == 10
    and "<stderr>" in str(next(iter(handlers.values())))
):
    logger.remove()
    logger.add(
        sys.stderr,
        format="{level} {time: HH:mm}({function}:{line}): {message}",
        level="ERROR",
        enqueue=True,
        colorize=True if not is_out_piped else False,
        backtrace=True if env.WDOC_DEBUG else None,
        diagnose=True if env.WDOC_DEBUG else None,
    )

# logger for the log_file
logger.add(
    log_file,
    rotation="100MB",
    retention=5,
    format="{time:YYYY-MM-DD at HH:mm:ss}|{level}|{thread}|{process}|{function}|{line}|{message}",
    level="DEBUG",
    enqueue=True,
    colorize=False,
    backtrace=True,
    diagnose=True,
    serialize=False,
)
# logger for the user stdout
logger.add(
    sys.stdout,
    format="{level} {time: HH:mm}({function}:{line}): {message}",
    level=log_level,
    enqueue=True,
    colorize=True if is_out_piped else False,
    backtrace=True if env.WDOC_DEBUG else None,
    diagnose=True if env.WDOC_DEBUG else None,
)

logger.debug(f"log_file location: {log_file}")

if is_input_piped:
    logger.debug("Detected input pipe")
if is_out_piped:
    logger.debug("Detected output pipe")

console = Console()


@optional_typecheck
def md_printer(message: str, color: Optional[str] = None) -> str:
    "markdown rendering and printing to console, unless we are in a pipe"
    message = dedent(message)
    if not is_out_piped:
        md = Markdown(message)
        logger.debug(message)
        console.print(md, style=color)
    else:
        logger.info(message)  # always shown
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
