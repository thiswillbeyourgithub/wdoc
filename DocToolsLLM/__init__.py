"""
Default file, used as entry point.
"""

import os
import sys
import fire
from typing import List, Tuple
from rich.markdown import Markdown
from rich.console import Console

# optional type checking via beartype
if "DOCTOOLS_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_TYPECHECKING"] = "disabled"
if os.environ["DOCTOOLS_TYPECHECKING"] == "disabled":
    pass
else:
    from beartype.claw import beartype_this_package
    if os.environ["DOCTOOLS_TYPECHECKING"] == "warn":
        from beartype import BeartypeConf
        beartype_this_package(conf=BeartypeConf(violation_type=UserWarning))
    elif os.environ["DOCTOOLS_TYPECHECKING"] == "crash":
        beartype_this_package()
    else:
        raise ValueError(f"Unexpected env value for DOCTOOLS_TYPECHECKING")

from .DocToolsLLM import DocToolsLLM_class as DocToolsLLM

__all__ = [
    "DocToolsLLM",
    "cli_launcher",
    "utils",
]

__VERSION__ = DocToolsLLM.VERSION


def fire_wrapper(
    *args,
    **kwargs,
    ) -> dict:
    "used to catch --help arg to display it better than fire would do"

    # --help but not catched by sys.argv
    if "help" in kwargs and kwargs["help"]:
        print("Showing help")
        DocToolsLLM.md_printer(DocToolsLLM.__doc__)
        raise SystemExit()

    # no args given
    if not any([args, kwargs]):
        print("Empty arguments, showing help")
        DocToolsLLM.md_printer(DocToolsLLM.__doc__)
        raise SystemExit()

    # while we're at it, make it so that
    # "DocToolsLLM summary" is parsed like "DocToolsLLM --task=summary"
    args = list(args)
    if args and isinstance(args[0], str):
        if args[0].replace("summary", "summarize") in ["query", "search", "summarize", "summarize_then_query"]:
            assert "task" not in kwargs or not kwargs["task"], f"Tried to give task as arg and kwarg?\n- args: {args}\n- kwargs: {kwargs}"
            kwargs["task"] = args.pop(0).replace("summary", "summarize")

    # prepare the parsing of --query
    if "query" not in kwargs:
        kwargs["query"] = None
    if kwargs["query"] in [True, None, False]:
        kwargs["query"] = ""
    else:
        kwargs["query"] = str(kwargs["query"])

    # any remaining args is put in --query
    if args:
        if not kwargs["query"]:
            kwargs["query"] = " ".join(map(str, args))
        else:
            kwargs["query"] += " " + " ".join(map(str, args))
        args = []

    kwargs["query"] = kwargs["query"].replace("summary", "summarize")

    assert not args
    return kwargs


def cli_launcher() -> None:
    sys_args = sys.argv
    if "--help" in sys_args:
        print("Showing help")
        DocToolsLLM.md_printer(DocToolsLLM.__doc__)
        raise SystemExit()
    if "--completion" in sys_args:
        return fire.Fire(DocToolsLLM)

    kwargs = fire.Fire(fire_wrapper)
    instance = DocToolsLLM(**kwargs)
