"""
Default file, used as entry point.
"""

import sys
import fire
import json

import lazy_import
lazy_import.lazy_module("litellm")
lazy_import.lazy_module('numpy')
lazy_import.lazy_module('faiss')
lazy_import.lazy_module('zlib')
lazy_import.lazy_module('dill')
lazy_import.lazy_module('sqlite3')
lazy_import.lazy_module('tldextract')
lazy_import.lazy_module('pyfiglet')
lazy_import.lazy_module('youtube_dl')
lazy_import.lazy_module('ankipandas')
lazy_import.lazy_module('pandas')
lazy_import.lazy_module('ftfy')
lazy_import.lazy_module('bs4')
lazy_import.lazy_module('goose3')
lazy_import.lazy_module('LogseqMarkdownParser')
lazy_import.lazy_module("deepgram")
lazy_import.lazy_module("pydub")
lazy_import.lazy_module("ffmpeg")
lazy_import.lazy_module("torchaudio")
lazy_import.lazy_module("playwright")
lazy_import.lazy_module("openparse")
lazy_import.lazy_module('youtube_dl')
lazy_import.lazy_module('ankipandas')
lazy_import.lazy_module('pandas')
lazy_import.lazy_module('ftfy')
lazy_import.lazy_module("deepgram")
lazy_import.lazy_module("pydub")
lazy_import.lazy_module("ffmpeg")
lazy_import.lazy_module("openparse")
lazy_import.lazy_module("scipy")
lazy_import.lazy_module('youtube_dl.utils')
lazy_import.lazy_module('LogseqMarkdownParser')
lazy_import.lazy_module("sklearn.metrics")
lazy_import.lazy_module("sklearn.decomposition")
lazy_import.lazy_module("sklearn.preprocessing")


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
    if "task" in kwargs:
        if "query" not in kwargs:
            kwargs["query"] = None
        if kwargs["query"] in [True, None, False]:
            kwargs["query"] = ""
        else:
            kwargs["query"] = str(kwargs["query"])

        if "path" not in kwargs:
            kwargs["path"] = None
        if kwargs["path"] in [True, None, False]:
            kwargs["path"] = ""
        else:
            kwargs["path"] = str(kwargs["path"])

    # any remaining args is put in --query (or --path if first)
    if args:
        if not kwargs["path"]:
            kwargs["path"] = str(args.pop(0))

        if not kwargs["query"]:
            kwargs["query"] = " ".join(map(str, args))
        else:
            kwargs["query"] += " " + " ".join(map(str, args))
        args = []
    assert not args


    kwargs["query"] = kwargs["query"].replace("summary", "summarize")

    return kwargs


def cli_launcher() -> None:
    sys_args = sys.argv
    if "--version" in sys_args:
        print(f"WDoc version: {__VERSION__}")
        raise SystemExit()
    if "--help" in sys_args:
        print("Showing help")
        WDoc.md_printer(WDoc.__doc__)
        raise SystemExit()
    if "--" in sys_args and "--completion" in sys_args:
        return fire.Fire(WDoc)

    kwargs = fire.Fire(fire_wrapper)
    instance = WDoc(**kwargs)

def cli_parse_file() -> None:
    sys_args = sys.argv
    if "--help" in sys_args:
        print("Showing help")
        fire.Fire(WDoc.parse_file)
        raise SystemExit()

    # parse args manually to allow piping
    args = []
    kwargs = {}
    for val in sys_args:
        if val in args or val in kwargs.values():
            continue
        if val.startswith("--"):
            val = val[2:]
            if "=" in val:
                k, v = val.split("=", 1)
                kwargs[k] = v
                continue
            args.append(val)
        else:
            if args:
                kwargs[args.pop(-1)] = val
    for k, v in kwargs.items():
        if v.startswith("\"") and v.endswith("\""):
            v = v[1:-1]
            kwargs[k] = v
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
            kwargs[k] = v
        if str(v).lower() == "false":
            kwargs[k] = False
        elif str(v).lower() == "true":
            kwargs[k] = True
    for a in args:
        kwargs[a] = True

    parsed = WDoc.parse_file(**kwargs)
    if isinstance(parsed, list):
        d = {
            "documents": {
                "content": d.page_content,
                "metadata": d.metadata
            } for d in parsed
        }
        try:
            out = json.dumps(d, indent=2)
        except Exception as err:
            out = str(d)
    else:
        assert isinstance(parsed, str)
        out = parsed

    sys.stdout.write(out)
    sys.stdout.flush()
