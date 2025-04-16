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

from .flags import md_printing_disabled
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
    level="DEBUG",
    enqueue=False,
    colorize=False,
)
# delete any additional log file
# (log_dir / "logs.txt.4").unlink(missing_ok=True)


colors = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "reset": "\033[0m",
    "white": "\033[0m",
    "purple": "\033[95m",
}


@optional_typecheck
def get_coloured_logger(color_asked: str, level: str) -> Callable:
    """used to print color coded logs"""
    col = colors[color_asked]

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs
    @optional_typecheck
    def printer(string: Union[str, Dict, List, Exception], **args) -> str:
        if isinstance(string, Exception):
            string = str(string)
        if isinstance(string, dict):
            try:
                string = rtoml.dumps(string, pretty=True)
            except Exception:
                string = json.dumps(string, indent=2, ensure_ascii=False)
        if isinstance(string, list):
            try:
                string = ",".join(string)
            except Exception:
                pass
        try:
            string = str(string)
        except Exception:
            try:
                string = string.__str__()
            except Exception:
                string = string.__repr__()
        for k, v in colors.items():
            string = string.replace(v, "")
        getattr(logger, level)(string)
        return string

    return printer


deb = get_coloured_logger("white", level="debug")
whi = get_coloured_logger("white", level="info")
yel = get_coloured_logger("yellow", level="warning")
red = get_coloured_logger("red", level="warning")

console = Console()


@optional_typecheck
def md_printer(message: str, color: Optional[str] = None) -> str:
    "markdown printing"
    message = dedent(message)
    if not md_printing_disabled:
        md = Markdown(message)
        console.print(md, style=color)
    else:
        if not color:
            whi(message)
        elif color in "red":
            red(message)
        elif color in "white":
            whi(message)
        elif color in "yellow":
            yel(message)
        else:
            whi(message)
    return message


@optional_typecheck
def set_help_md_as_docstring(obj: Union[Type, Callable]) -> Union[Type, Callable]:
    "set the docstring of wdoc class to wdoc/docs/help.md's content"
    help_file = Path(__file__).parent.parent / "docs/help.md"
    if not help_file.exists():
        red(
            f"Couldn't find help.md file as '{help_file}'. You can read it at this URL instead: https://github.com/thiswillbeyourgithub/wdoc/blob/main/wdoc/docs/help.md"
        )
        helpcont = "Help documentation not found. Please refer to online documentation."
    else:
        helpcont = help_file.read_text().strip()
        if not helpcont:
            red("Help documentation file is empty")
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
        red(
            f"Couldn't find parse_file_help.md file as '{parsefilehelp_file}'. You can read it at this URL instead: https://github.com/thiswillbeyourgithub/wdoc/blob/main/wdoc/docs/parse_file_help.md"
        )
        parsefilehelp = "Parse file help documentation not found. Please refer to online documentation."
    else:
        parsefilehelp = parsefilehelp_file.read_text().strip()
        if not parsefilehelp:
            red("Parse file help documentation is empty")
            parsefilehelp = "Parse file help documentation is empty. Please refer to online documentation."
    obj.__doc__ = "# Content of wdoc/docs/parse_file_help.md\n\n" + parsefilehelp
    return obj
