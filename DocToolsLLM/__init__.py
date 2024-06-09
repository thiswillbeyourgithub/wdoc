"""
Default file, used as entry point.
"""

import fire
from typing import List, Tuple
from rich.markdown import Markdown
from rich.console import Console

from .DocToolsLLM import DocToolsLLM_class

import logging
logging.getLogger().setLevel(logging.ERROR)

def fire_wrapper(
    h: bool = False,
    help: bool = False,
    *args,
    **kwargs,
    ) -> Tuple[List, dict]:
    "used to catch --help arg to display it better then fire does on its own"
    assert "h" not in args and "h" not in kwargs
    assert "help" not in args and "help" not in kwargs

    kwargs["help"] = (h in ["h", "help", True] or help in ["h", "help", True])
    if not kwargs["help"]:
        args = [h, help] + list(args)
    return args, kwargs


def cli_launcher() -> None:
    args, kwargs = fire.Fire(fire_wrapper)
    if kwargs["help"]:
        md = Markdown(DocToolsLLM_class.__doc__)
        console = Console()
        console.print(md, style=None)
        raise SystemExit()
    else:
        instance = fire.Fire(DocToolsLLM_class)
