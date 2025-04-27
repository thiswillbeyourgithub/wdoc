"""
Entry point file
"""

import json
import os
import re
import sys
from pathlib import Path
import tempfile
import fire
from loguru import logger

# Just in case: we set the env variable of WDOC_PRIVATE_MODE as early as possible, even before importing wdoc
if re.findall(r"\b--private\b", " ".join(sys.argv)):
    logger.warning("Detected --private mode: setting WDOC_PRIVATE_MODE to True")
    os.environ["WDOC_PRIVATE_MODE"] = True

from .utils import logger as importedlogger  # make sure to setup the logs first
from .wdoc import wdoc
from .utils.env import env
from .utils.misc import piped_input

# if __main__ is called, then we are using the cli instead of importing the class from python
wdoc.__import_mode__ = False


def cli_launcher() -> None:
    """entry point function, modifies arguments on the fly for easier
    shorthands then call wdoc"""

    # if a pipe is used we store it in a file then pass this file as argument.
    # Either by replacing "-" by its path or appending it at the end of the
    # command line.
    if piped_input:
        sysline = " ".join(sys.argv)
        if isinstance(piped_input, bytes):
            if "--filetype" not in sysline:
                logger.info(
                    "When piping binary data it is recommended to use a --filetype argument otherwise python-magic<=0.4.27 often crashes. See https://github.com/ahupp/python-magic/issues/261"
                )
            temp_file = tempfile.NamedTemporaryFile(
                prefix="wdoc_piped_input_",
                delete=False,
                mode="wb",
            )
            logger.debug(
                f"Detected binary piped data, storing it in '{temp_file.name}'"
            )
        elif isinstance(piped_input, str):
            temp_file = tempfile.NamedTemporaryFile(
                prefix="wdoc_piped_input_",
                suffix=".txt",
                delete=False,
                mode="w",
            )
            logger.debug(f"Detected text piped data, storing it in '{temp_file.name}'")
            if "--filetype" not in sysline:
                logger.debug("Setting the filetype as 'txt'")
                sys.argv.append("--filetype=txt")
        else:
            raise ValueError(type(piped_input))
        temp_file.write(piped_input)
        temp_file.close()

        # insert the temp_file as path
        if "-" in sys.argv:
            for i, x in enumerate(sys.argv):
                if x == "-":
                    sys.argv[i] = temp_file.name
        elif "--path=-" in sys.argv:
            for i, x in enumerate(sys.argv):
                if x == "--path=-":
                    sys.argv[i] = "--path=" + temp_file.name
        else:
            sys.argv.append("--path=" + temp_file.name)

    # reload sysline var
    sysline = " ".join(sys.argv)

    # turn 'wdoc parse' into 'wdoc_parse_file'
    if (
        "wdoc parse " in sysline
        or "wdoc parse_file " in sysline
        or "__main__.py parse " in sysline
        or "__main__.py parse_file  " in sysline
    ):
        if env.WDOC_VERBOSE:
            logger.info("Replacing 'wdoc parse' by 'wdoc_parse_file'")
        sys.argv[0] = str(Path(sys.argv[0]).parent / "wdoc_parse_file")
        del sys.argv[1]
        sysline = " ".join(sys.argv)
        cli_parse_file()
        sys.exit(0)

    if " --version" in sysline:
        print(f"wdoc version: {wdoc.VERSION}")
        sys.exit(0)
    elif " --help" in sysline or " -h" in sysline:
        print("Showing help")
        importedlogger.md_printer(wdoc.__doc__)
        sys.exit(0)
    elif " --completion" in sysline:
        if " -- --completion" in sysline:
            fire.Fire(wdoc)
            sys.exit(0)
        else:
            raise Exception(
                "To create completion scripts, use '-- --completion' as arguments."
            )
    elif len(sys.argv) == 1:
        logger.info("No args shown. Use '--help' to display the help.")
        sys.exit(0)

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
            if env.WDOC_VERBOSE:
                logger.info(f"Replaced argument '{bef}' to '{aft}'")
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
        if env.WDOC_VERBOSE:
            logger.info(f"Replaced '{path}' to '{newarg}'")

    fire.Fire(wdoc)


def cli_parse_file() -> None:
    sys_args = sys.argv
    if "--help" in sys_args:
        print("Showing help")
        importedlogger.md_printer(wdoc.parse_file.__doc__)
        # help(wdoc.parse_file)
        # print(wdoc.parse_file.__doc__)
        # fire.Fire(wdoc.parse_file)
        sys.exit(0)

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
        if env.WDOC_VERBOSE:
            logger.info(f"Arguments that will be used for parser: '{kwargs}'")
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

    try:
        sys.stdout.write(out)
        sys.stdout.flush()
    except BrokenPipeError as e:
        logger.debug(
            f"Encountered a BrokenPipeError, crashing silently because it indicates that the code after the pipe crashed so we should not flood the output: {e}"
        )


if __name__ == "__main__":
    cli_launcher()
