"""
Default file, used as entry point.
"""

import sys
import fire

from .WDoc import WDoc

__all__ = [
    "WDoc",
    "cli_launcher",
    "utils",
]

__VERSION__ = WDoc.VERSION


def fire_wrapper(
    *args,
    **kwargs,
) -> dict:
    "used to catch --help arg to display it better than fire would do"

    # --help but not catched by sys.argv
    if "help" in kwargs and kwargs["help"]:
        print("Showing help")
        WDoc.md_printer(WDoc.__doc__)
        raise SystemExit()

    # no args given
    if not any([args, kwargs]):
        print("Empty arguments, showing help")
        WDoc.md_printer(WDoc.__doc__)
        raise SystemExit()

    # while we're at it, make it so that
    # "WDoc summary" is parsed like "WDoc --task=summary"
    args = list(args)
    if args and isinstance(args[0], str):
        if args[0].replace("summary", "summarize") in ["query", "search", "summarize", "summarize_then_query"]:
            assert "task" not in kwargs or not kwargs[
                "task"], f"Tried to give task as arg and kwarg?\n- args: {args}\n- kwargs: {kwargs}"
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
    if "--version" in sys_args:
        return __VERSION__
    if "--help" in sys_args:
        print("Showing help")
        WDoc.md_printer(WDoc.__doc__)
        raise SystemExit()
    if "--" in sys_args and "--completion" in sys_args:
        return fire.Fire(WDoc)

    kwargs = fire.Fire(fire_wrapper)
    instance = WDoc(**kwargs)
