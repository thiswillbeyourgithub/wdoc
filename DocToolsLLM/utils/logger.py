"""
Code related to loggings, coloured logs, etc.
"""

import rtoml
import json
import requests
import time
from tqdm import tqdm
import logging
import logging.handlers
from pathlib import Path
from typing import Type, Callable, Optional, Union
from rich.markdown import Markdown
from rich.console import Console
from platformdirs import user_cache_dir

from .typechecker import optional_typecheck

assert Path(user_cache_dir()).exists(), f"User cache dir not found: '{user_cache_dir()}'"
cache_dir = Path(user_cache_dir()) / "DocToolsLLM"
cache_dir.mkdir(exist_ok=True)
log_dir = cache_dir / "logs"
log_dir.mkdir(exist_ok=True)
(log_dir / "logs.txt").touch(exist_ok=True)

# logger
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "logs.txt",
        mode="a",
        encoding=None,
        delay=0,
        maxBytes=1024*1024*100,  # max 100mb
        backupCount=3,
        )
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(handler)
# delete any additional log file
(log_dir / "logs.txt.4").unlink(missing_ok=True)


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
    def printer(string: str, **args) -> str:
        inp = string
        if isinstance(string, dict):
            try:
                string = rtoml.dumps(string, pretty=True)
            except Exception:
                string = json.dumps(string, indent=2)
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
        log.info(string)
        tqdm.write(col + string + colors["reset"], **args)
        return inp
    return printer


whi = get_coloured_logger("white")
yel = get_coloured_logger("yellow")
red = get_coloured_logger("red")

console = Console()

@optional_typecheck
def md_printer(message: str, color: Optional[str] = None) -> str:
    "markdown printing"
    log.info(message)
    md = Markdown(message)
    console.print(md, style=color)
    return message

@optional_typecheck
def set_docstring(obj: Union[Type, Callable]) -> Union[Type, Callable]:
    "set the docstring of DocToolsLLM class to DocToolsLLM/docs/USAGE.md's content"
    usage_file = Path("DocToolsLLM/docs/USAGE.md")
    assert usage_file.exists()
    usage = usage_file.read_text().strip()
    assert usage
    obj.__doc__ = usage
    if isinstance(obj, type):
        obj.__init__.__doc__ = usage
    return obj
