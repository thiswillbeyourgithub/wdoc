"""
Entry point file
"""

import json
import json
import os
import re
import sys
from pathlib import Path
import tempfile
from typing import Union, Optional
import fire
from loguru import logger
import rtoml

# Just in case: we set the env variable of WDOC_PRIVATE_MODE as early as possible, even before importing wdoc
if re.findall(r"\b--private\b", " ".join(sys.argv)):
    logger.warning("Detected --private mode: setting WDOC_PRIVATE_MODE to True")
    os.environ["WDOC_PRIVATE_MODE"] = True

from .utils import logger as importedlogger  # make sure to setup the logs first
from .wdoc import wdoc
from .utils.env import env, is_out_piped
from .utils.misc import get_piped_input

# if __main__ is called, then we are using the cli instead of importing the class from python
wdoc.__import_mode__ = False


def handle_piped_input(piped_data: Union[str, bytes]) -> str:
    """Processes piped input and returns the appropriate argument string."""
    temp_file = None
    arg_to_add = None

    if isinstance(piped_data, str):
        logger.debug("Processing piped input as string.")
        # Check for URL
        if piped_data.startswith("http") and " " not in piped_data:
            logger.debug("Detected URL in piped input.")
            arg_to_add = piped_data
        # Check for existing path
        elif Path(piped_data).exists():
            logger.debug("Detected existing path in piped input.")
            arg_to_add = piped_data
        else:
            # Try JSON
            try:
                loaded_json = json.loads(piped_data)
                if not isinstance(
                    loaded_json, str
                ):  # Ensure it's not just a JSON string
                    logger.debug(
                        "Detected JSON object in piped input. Writing to temp file."
                    )
                    temp_file = tempfile.NamedTemporaryFile(
                        prefix="wdoc_piped_input_",
                        suffix=".json",
                        delete=False,
                        mode="w",
                        encoding="utf-8",
                    )
                    temp_file.write(piped_data)
                    temp_file.close()
                    arg_to_add = temp_file.name
                else:
                    logger.debug(
                        "Piped input is a JSON string, treating as generic text."
                    )
            except json.JSONDecodeError:
                logger.debug("Piped input is not valid JSON.")
                pass  # Not JSON

            # Try TOML (only if not already handled as JSON or other specific cases)
            if arg_to_add is None:
                try:
                    rtoml.loads(piped_data)  # Just check if it loads
                    logger.debug("Detected TOML in piped input. Writing to temp file.")
                    temp_file = tempfile.NamedTemporaryFile(
                        prefix="wdoc_piped_input_",
                        suffix=".toml",
                        delete=False,
                        mode="w",
                        encoding="utf-8",
                    )
                    temp_file.write(piped_data)
                    temp_file.close()
                    arg_to_add = temp_file.name
                except rtoml.TomlParsingError:
                    logger.debug("Piped input is not valid TOML.")
                    pass  # Not TOML

    # Fallback: Write to generic temp file (handles bytes and non-special strings)
    if arg_to_add is None:
        if isinstance(piped_data, str):
            logger.debug("Writing generic string piped input to temp file.")
            temp_file = tempfile.NamedTemporaryFile(
                prefix="wdoc_piped_input_",
                delete=False,
                mode="w",
                encoding="utf-8",
            )
            temp_file.write(piped_data)
            temp_file.close()
            arg_to_add = temp_file.name
        elif isinstance(piped_data, bytes):
            logger.debug("Writing binary piped input to temp file.")
            temp_file = tempfile.NamedTemporaryFile(
                prefix="wdoc_piped_input_",
                delete=False,
                mode="wb",
            )
            temp_file.write(piped_data)
            temp_file.close()
            arg_to_add = temp_file.name
        else:
            # Should not happen based on get_piped_input's return type hint, but good to be safe
            raise TypeError(f"Unexpected type for piped_data: {type(piped_data)}")

    # Check for filetype argument presence
    sysline = " ".join(sys.argv)
    # Check both --filetype=val and --filetype val patterns
    if "--filetype" not in sysline and not any(
        arg.startswith("filetype=") for arg in sys.argv
    ):
        logger.warning(
            "Piped input detected, but no --filetype argument was provided. "
            "Relying on file content heuristics, which might be unreliable. "
            "Consider adding --filetype for robustness."
        )

    if arg_to_add is None:
        # This case should ideally not be reached if logic is correct
        raise RuntimeError("Failed to determine argument to add for piped input.")

    return arg_to_add


def cli_launcher() -> None:
    """entry point function, modifies arguments on the fly for easier
    shorthands then call wdoc"""

    piped_input = get_piped_input()
    if piped_input:
        logger.debug("Processing piped input...")
        new_arg = handle_piped_input(piped_input)
        logger.debug(f"Adding '{new_arg}' to arguments based on piped input.")
        sys.argv.append(new_arg)  # Append the new argument

    # reload sysline var (important after potentially modifying sys.argv)
    sysline = " ".join(sys.argv)

    # turn 'wdoc parse' into 'wdoc_parse_file'
    if (
        "wdoc parse " in sysline
        or "wdoc parse_file " in sysline
        or "__main__.py parse " in sysline
        or "__main__.py parse_file  " in sysline
    ):
        logger.debug("Replacing 'wdoc parse' by 'wdoc_parse_file'")
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
        if is_out_piped:
            print(wdoc.__doc__)
        else:
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
            logger.debug(f"Replaced argument '{bef}' to '{aft}'")
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
        logger.debug(f"Replaced '{path}' to '{newarg}'")

    fire.Fire(wdoc)


def cli_parse_file() -> None:
    sys_args = sys.argv
    if "--help" in sys_args:
        print("Showing help")
        if is_out_piped:
            print(wdoc.parse_file.__doc__)
        else:
            importedlogger.md_printer(wdoc.parse_file.__doc__)
        # help(wdoc.parse_file)
        # print(wdoc.parse_file.__doc__)
        # fire.Fire(wdoc.parse_file)
        sys.exit(0)

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
