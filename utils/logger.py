import rtoml
import json
import requests
import time
from tqdm import tqdm
import logging
import logging.handlers
from pathlib import Path
from typing import Callable, Optional
from rich.markdown import Markdown
from rich.console import Console

from .typechecker import optional_typecheck

# adds logger, restrict it to X lines
local_dir = Path.cwd()
(local_dir / "logs.txt").touch(exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
handler = logging.handlers.RotatingFileHandler(
        filename=local_dir / "logs.txt",
        mode="a",
        encoding=None,
        delay=0,
        maxBytes=1024*1024*100,  # max 50mb
        backupCount=3,
        )
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(handler)
# delete any additional log file
(local_dir / "logs.txt.3").unlink(missing_ok=True)


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
            except:
                pass
        try:
            string = str(string)
        except:
            try:
                string = string.__str__()
            except:
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
def md_printer(message: str, color: Optional[str] = None) -> None:
    log.info(message)
    md = Markdown(message)
    console.print(md, style=color)

# phone notification
@optional_typecheck
def create_ntfy_func(url: str) -> Callable:
    @optional_typecheck
    def ntfy_func(text: str) -> str:
        requests.post(
                url=url,
                headers={"Title": "DocTools Summary"},
                data=text.encode("utf-8"),
                )
        return red(text)
    return ntfy_func
