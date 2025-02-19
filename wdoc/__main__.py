"""
Entry point file
"""

import json
import os
import re
import sys
from pathlib import Path

import fire

# Just in case: we set the env variable of WDOC_PRIVATE_MODE as early as possible, even before importing wdoc
if re.findall(r"\b--private\b", " ".join(sys.argv)):
    print("Detected --private mode: setting WDOC_PRIVATE_MODE to True")
    os.environ["WDOC_PRIVATE_MODE"] = True

from .wdoc import is_verbose, wdoc, whi


def cli_launcher() -> None:
    """entry point function, modifies arguments on the fly for easier
    shorthands then call wdoc"""
    sysline = " ".join(sys.argv)

    # turn 'wdoc parse' into 'wdoc_parse_file'
    if (
        "wdoc parse " in sysline
        or "wdoc parse_file " in sysline
        or "__main__.py parse " in sysline
        or "__main__.py parse_file  " in sysline
    ):
        if is_verbose:
            whi("Replacing 'wdoc parse' by 'wdoc_parse_file'")
        sys.argv[0] = str(Path(sys.argv[0]).parent / "wdoc_parse_file")
        del sys.argv[1]
        sysline = " ".join(sys.argv)
        cli_parse_file()
        raise SystemExit(0)

    if " --version" in sysline:
        print(f"wdoc version: {wdoc.VERSION}")
        raise SystemExit(0)
    elif " --help" in sysline or " -h" in sysline:
        print("Showing help")
        wdoc.md_printer(wdoc.__doc__)
        raise SystemExit(0)
    elif " --completion" in sysline:
        if " -- --completion" in sysline:
            fire.Fire(wdoc)
            raise SystemExit(0)
        else:
            raise Exception(
                "To create completion scripts, use '-- --completion' as arguments."
            )
    elif len(sys.argv) == 1:
        whi("No args shown. Use '--help' to display the help.")
        raise SystemExit(0)

    # while we're at it, make it so that
    # "wdoc summary" is parsed like "wdoc --task=summary"
    if " --task" not in sysline:
        arg_replacement_rules = {
            "query": "--task=query",
            "search": "--task=search",
            "summarize": "--task=summarize",
            "summarize_then_query": "--task=summarize_then_query",
            "summary": "--task=summarize",
            "summary_then_query": "--task=summarize_then_query",
        }
        if sys.argv[1] in arg_replacement_rules:
            bef = sys.argv[1]
            aft = arg_replacement_rules[bef]
            if is_verbose:
                whi(f"Replaced argument '{bef}' to '{aft}'")
            sys.argv[1] = aft

    # make it so that 'wdoc --task=query THING' becomes 'wdoc --task=query --path=THING'
    if (
        ("--path" not in sysline)
        and (not re.findall(r"\bstring\b", sysline))
        and (not re.findall(r"\banki\b", sysline))
    ):
        # if string is present that can be because of --filetype=string or
        # --filetype=anki, in which case 'path' argument is not needed
        path = sys.argv[2]
        newarg = f"--path={path}"
        sys.argv[2] = newarg
        if is_verbose:
            whi(f"Replaced '{path}' to '{newarg}'")

    fire.Fire(wdoc)


def cli_parse_file() -> None:
    sys_args = sys.argv
    if "--help" in sys_args:
        print("Showing help")
        wdoc.md_printer(wdoc.parse_file.__doc__)
        # help(wdoc.parse_file)
        # print(wdoc.parse_file.__doc__)
        # fire.Fire(wdoc.parse_file)
        raise SystemExit(0)

    if "--pipe" in sys_args or "pipe" in sys_args:
        # parse args manually to allow piping
        args = []
        kwargs = {}
        for val in sys_args:
            if val == "--pipe" or val == "pipe":
                continue
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
            if v == "--pipe" or v == "pipe":
                continue
            if k == "--pipe" or k == "pipe":
                continue
            if v.startswith('"') and v.endswith('"'):
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

        # all loaders need a path arg except anki and string
        if (
            "filetype" in kwargs
            and kwargs["filetype"] in ["anki", "string"]
            and "path" not in kwargs
        ):
            kwargs["path"] = None
        if is_verbose:
            whi(f"Arguments that will be used for parser: '{kwargs}'")
        parsed = wdoc.parse_file(**kwargs)
    else:
        parsed = fire.Fire(wdoc.parse_file)

    if isinstance(parsed, list):
        if all(not isinstance(d, dict) for d in parsed):
            parsed_d = [
                {"page_content": d.page_content, "metadata": d.metadata} for d in parsed
            ]
        else:
            parsed_d = parsed
        try:
            out = json.dumps(parsed_d, indent=2, ensure_ascii=False)
        except Exception:
            out = str(parsed_d)
    else:
        assert isinstance(parsed, str)
        out = parsed

    sys.stdout.write(out)
    sys.stdout.flush()


if __name__ == "__main__":
    cli_launcher()
