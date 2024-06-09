"""
Default file, used as entry point.
"""

import sys
import fire
from typing import List, Tuple
from rich.markdown import Markdown
from rich.console import Console

from .DocToolsLLM import DocToolsLLM_class


def fire_wrapper(
    h: bool = False,
    help: bool = False,
    *args,
    **kwargs,
    ) -> Tuple[List, dict]:
    "used to catch --help arg to display it better then fire does on its own"
    assert "h" not in args and "h" not in kwargs
    assert "help" not in args and "help" not in kwargs

    if (h in ["h", "help", True] or help in ["h", "help", True]):  # --help or similar intentions
        return [], {"help": True}

    # parse args as if nothing happened
    args = list(args)
    if h:
        args.insert(0, h)
    if help:
        args.insert(1, help)

    # while we're at it, make it so that
    # "DocToolsLLM summary" is parsed like "DocToolsLLM --task=summary"
    if args and isinstance(args[0], str):
        args[0] = args[0].replace("summary", "summarize")
        if args[0] in ["query", "search", "summarize", "summarize_then_query"]:
            assert "task" not in kwargs, f"Tried to give task as arg and kwarg?\nargs: {args}\bnkwargs: {kwargs}"
            kwargs["task"] = args.pop(0)
    return args, kwargs


def cli_launcher() -> None:
    sys_args = sys.argv
    if "--completion" in sys_args:
        return fire.Fire(DocToolsLLM_class)

    args, kwargs = fire.Fire(fire_wrapper)

    if "help" in kwargs:
        md = Markdown(DocToolsLLM_class.__doc__)
        console = Console()
        console.print(md, style=None)
        raise SystemExit()
    else:
        instance = DocToolsLLM_class(*args, **kwargs)
