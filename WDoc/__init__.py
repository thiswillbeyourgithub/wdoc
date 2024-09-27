"""
Default file, used as entry point.
"""

import os
import sys
import fire
import json
import copy

import lazy_import
from queue import Queue
import threading


if "WDOC_DISABLE_LAZYLOADING" not in os.environ:

    def import_worker(q: Queue):
        while True:
            module = q.get()
            if module is None:
                return
            # print(f"Importing {module}")
            if "." in module:
                first = ".".join(module.split(".")[:-1])
                last = module.split(".")[-1]
                exec(f"from {first} import {last}")
            else:
                first = module
                exec(f"import {module}")

            assert first in sys.modules, f"Error when importing '{first}'"
            if "Lazily-loaded" in str(sys.modules[first]):
                # print(f"Module is lazy loaded so far: {first}")
                try:
                    dir(sys.modules[first])
                except Exception as e:
                    raise Exception(
                        f"Error when unlazyloading module '{first}'. Error: '{e}'"
                        "\nYou can try setting the env variable "
                        "WDOC_DISABLE_LAZYLOADING to 'true'"
                        "Don't hesitate to open an issue!"
                    ) from e
                # print(f"Unlazyloaded module: {first}")


    q = Queue()
    thread = threading.Thread(target=import_worker, args=(q,), daemon=False)
    thread.start()

    def threaded_loading(module: str) -> None:
        q.put(module)
        lazy_import.lazy_module(module)

    threaded_loading("langchain.text_splitter")
    threaded_loading("litellm")
    threaded_loading('numpy')
    threaded_loading('faiss')
    threaded_loading('zlib')
    threaded_loading('dill')
    threaded_loading('sqlite3')
    threaded_loading('tldextract')
    threaded_loading('pyfiglet')
    threaded_loading('youtube_dl')
    threaded_loading('youtube_dl.utils')
    threaded_loading('pandas')
    threaded_loading('ankipandas')
    threaded_loading('ftfy')
    threaded_loading('bs4')
    threaded_loading('goose3')
    threaded_loading('LogseqMarkdownParser')
    threaded_loading("deepgram")
    threaded_loading("pydub")
    threaded_loading("ffmpeg")
    threaded_loading("torchaudio")
    threaded_loading("playwright.sync_api")
    threaded_loading("openparse")
    threaded_loading("scipy")
    threaded_loading("sklearn.metrics")
    threaded_loading("sklearn.decomposition")
    threaded_loading("sklearn.preprocessing")

    q.put(None)  # kill the import worker
else:
    print("WDoc: disabled lazy loading")

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

    if "path" in kwargs and kwargs["path"] in ["", None, True, False]:
        del kwargs["path"]

    return kwargs


def cli_launcher() -> None:
    sys_args = copy.deepcopy(sys.argv)
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
    _ = WDoc(**kwargs)


def cli_parse_file() -> None:
    sys_args = copy.deepcopy(sys.argv)
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
        except Exception:
            out = str(d)
    else:
        assert isinstance(parsed, str)
        out = parsed

    sys.stdout.write(out)
    sys.stdout.flush()
