"""
Code related to loggings, coloured logs, etc.
"""

import rtoml
import json
from textwrap import dedent
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from typing import Type, Callable, Optional, Union, List, Dict
from rich.markdown import Markdown
from rich.console import Console
from platformdirs import user_cache_dir, user_log_dir
import warnings

from .typechecker import optional_typecheck
from .flags import md_printing_disabled, is_silent

# ignore warnings from beautiful soup
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

cache_dir = Path(user_cache_dir(appname="wdoc"))
assert cache_dir.parent.exists() or cache_dir.parent.parent.exists(
), f"Invalid cache dir location: '{cache_dir}'"
cache_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path(user_log_dir(appname="wdoc"))
assert log_dir.parent.exists() or log_dir.parent.parent.exists(
) or log_dir.parent.parent.parent.exists(), f"Invalid log_dir location: '{log_dir}'"
log_dir.mkdir(exist_ok=True, parents=True)
log_file = (log_dir / "logs.txt")
log_file.touch(exist_ok=True)

# logger
try:
    logger.remove()
except Exception as err:
    pass
logger.add(
    log_file,
    rotation="100MB",
    retention=5,
    format='{time} {level} wdoc {thread} {process} {function} {line} {message}',
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
def get_coloured_logger(color_asked: str) -> Callable:
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
        logger.info(string)
        if not is_silent:
            tqdm.write(col + string + colors["reset"], **args)
        return string
    return printer


whi = get_coloured_logger("white")
yel = get_coloured_logger("yellow")
red = get_coloured_logger("red")

console = Console()


@optional_typecheck
def md_printer(message: str, color: Optional[str] = None) -> str:
    "markdown printing"
    message = dedent(message)
    if not md_printing_disabled:
        logger.info(message)
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
def set_USAGE_as_docstring(obj: Union[Type, Callable]) -> Union[Type, Callable]:
    "set the docstring of wdoc class to wdoc/docs/USAGE.md's content"
    usage_file = Path(__file__).parent.parent / "docs/USAGE.md"
    assert usage_file.exists(), f"Couldn't find USAGE.md file as '{usage_file}'. You can read it at this URL instead: https://github.com/thiswillbeyourgithub/wdoc/blob/main/wdoc/docs/USAGE.md"
    usage = usage_file.read_text().strip()
    assert usage
    obj.__doc__ = obj.__doc__ + "\n\n# Content of wdoc/docs/USAGE.md\n\n" + usage
    if isinstance(obj, type):
        obj.__init__.__doc__ = usage
    return obj
