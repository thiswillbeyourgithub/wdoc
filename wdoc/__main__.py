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
from typing import Union
import fire
import logging
from loguru import logger
import rtoml

# Just in case: we set the env variable of WDOC_PRIVATE_MODE as early as possible, even before importing wdoc
if re.findall(r"\b--private\b", " ".join(sys.argv)):
    logger.warning("Detected --private mode: setting WDOC_PRIVATE_MODE to True")
    os.environ["WDOC_PRIVATE_MODE"] = True

from wdoc.utils import logger as importedlogger  # make sure to setup the logs first
from wdoc.wdoc import wdoc
from wdoc.utils.env import env, is_out_piped
from wdoc.utils.typechecker import optional_typecheck
from wdoc.utils.misc import get_piped_input, tasks_list
from typing import Tuple, List, Dict, Any
import io

# if __main__ is called, then we are using the cli instead of importing the class from python
wdoc.__import_mode__ = False


@optional_typecheck
def parse_args_fire() -> Tuple[List[Any], Dict[str, Any]]:
    """
    Parses command-line arguments using fire.Fire without printing its output.

    Temporarily redirects stdout to suppress fire's default output during parsing.

    Returns:
        A tuple containing the list of positional arguments and a dictionary of keyword arguments.
    """
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to suppress fire output
    try:
        args, kwargs = fire.Fire(lambda *args, **kwargs: (list(args), kwargs))
    finally:
        sys.stdout = original_stdout  # Restore original stdout
    return args, kwargs


@optional_typecheck
def handle_piped_input(piped_data: Union[str, bytes]) -> str:
    """Processes piped input and returns the appropriate argument string."""
    temp_file = None
    arg_to_add = None

    if isinstance(piped_data, str):
        logger.debug("Processing piped input as string.")
        nline = len(piped_data.splitlines())

        # Check for URL
        if piped_data.startswith("http") and " " not in piped_data:
            logger.debug("Detected URL in piped input.")
            arg_to_add = piped_data
        # Check if it's a path
        elif nline == 1 and len(piped_data) < 150 and Path(piped_data).exists():
            # we cap the length because otherwise if the line is very long Path crashes
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

            # assume it should be a txt file
            if arg_to_add is None:
                logger.debug("Detected .txt in piped input. Writing to temp file.")
                temp_file = tempfile.NamedTemporaryFile(
                    prefix="wdoc_piped_input_",
                    suffix=".txt",
                    delete=False,
                    mode="w",
                    encoding="utf-8",
                )
                temp_file.write(piped_data)
                temp_file.close()
                arg_to_add = temp_file.name

    # Fallback: Write to generic temp file (handles bytes and non-special strings)
    if arg_to_add is None:
        if isinstance(piped_data, bytes):
            logger.debug("Writing binary piped input to temp file.")
            temp_file = tempfile.NamedTemporaryFile(
                prefix="wdoc_piped_input_",
                delete=False,
                mode="wb",
            )
            temp_file.write(piped_data)
            temp_file.close()
            arg_to_add = temp_file.name
        elif isinstance(piped_data, str):
            raise TypeError(
                f"Unexpected type for piped_data: it should have been handled:\n{piped_data}."
            )
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

    # check the args and kwargs manually to make parsing more intuitive until click is used instead of fire
    args, kwargs = parse_args_fire()

    # first thing: set the appropriate debug level
    if "verbose" in kwargs or "v" in args:
        logging.getLogger("wdoc").setLevel(logging.DEBUG)

    logger.debug(
        f"Started cli wdoc with sys.argv: '{sys.argv}'. Detected args: '{args}'. Detected kwargs: '{kwargs}'"
    )

    # crash if no args
    if len(sys.argv) == 1 or (not (args or kwargs)):
        logger.info("No args shown. Use '--help' to display the help.")
        sys.exit(0)

    if "version" in kwargs:
        print(f"wdoc version: {wdoc.VERSION}")
        sys.exit(0)

    if "help" in kwargs or "h" in kwargs:
        print("Showing help")
        if ("task" in kwargs and kwargs["task"] == "parse") or "parse" in args:
            target = wdoc.parse_file
        else:
            target = wdoc
        doc = getattr(target, "__doc__")
        if is_out_piped:
            print(doc)
        else:
            importedlogger.md_printer(doc)
        sys.exit(0)

    # turn 'summary' into 'summarize' etc
    # We need to re-parse args after modifying sys.argv
    needs_reparse = False
    if "summary" in args:
        sys.argv[sys.argv.index("summary")] = "summarize"
        needs_reparse = True
    if "summary_then_query" in args:
        sys.argv[sys.argv.index("summary_then_query")] = "summarize_then_query"
        needs_reparse = True

    if needs_reparse:
        args, kwargs = parse_args_fire()

    # turn "wdoc query" into "wdoc --task=query", same for the other tasks
    if "task" not in kwargs:
        matching_tasks = [t for t in args if t in tasks_list]
        assert (
            len(matching_tasks) != 0
        ), f"Found no task in the args: '{args}', wdoc needs one of {tasks_list}"
        assert (
            len(matching_tasks) == 1
        ), f"Found multiple potential tasks in args: '{args}', wdoc needs one of {tasks_list}"
        task = matching_tasks[0]
        logger.debug(f"Moving task '{task}' from args to kwargs")
        args.remove(task)
        kwargs["task"] = task
        sys.argv[sys.argv.index(task)] = f"--task={task}"

    # if we are receiving from a pipe, use heuristics to see if we store it to a file or to an arg, or as --path value
    piped_input = get_piped_input()
    if piped_input:
        logger.debug("Processing piped input...")
        new_arg = handle_piped_input(piped_input)
        if "path" in args:
            logger.debug(
                f"Adding '{new_arg}' to --path arguments based on piped input."
            )
            args.remove("path")
            sys.argv[sys.argv.index("path")] = f"--path={new_arg}"
        else:
            logger.debug(f"Adding '{new_arg}' to arguments based on piped input.")
            args.append(new_arg)
            sys.argv.append(new_arg)  # Append the new argument

    # if no --path but an arg: use it as path arg
    if not ("path" in args or "path" in kwargs):
        if len(args) == 1:
            sys.argv.remove(args[0])
            sys.argv.append(f"--path={args[0]}")
            logger.debug(f"Set the argument '{args[0]}' to a --path argument")
            kwargs["path"] = args.pop(0)

    # if --path is empty give it the remaining arg
    if ("path" not in kwargs or isinstance(kwargs["path"], (bool, type(None)))) and len(
        args
    ) == 1:
        sys.argv.remove(args[0])
        sys.argv.append(f"--path={args[0]}")
        logger.debug(f"Set the argument '{args[0]}' to a --path argument")
        kwargs["path"] = args.pop(0)

    # if args is not empty, we have not succesfully parsed everything
    if args:
        logger.warning(
            f"It appears the arguments parsing was not complete: remaining args were not dispatched to kwargs: '{args}'"
        )

    if kwargs["task"] == "parse":
        call_parse_file()
        sys.exit(0)

    elif "completion" in kwargs or "--completion" in sys.argv:
        if " -- --completion" in " ".join(sys.argv):
            fire.Fire(wdoc)
            sys.exit(0)
        else:
            raise Exception(
                "To create completion scripts, use '-- --completion' as arguments."
            )
    else:
        logger.debug(
            f"Launching wdoc after arg transformation: Remaining sys.argv: '{sys.argv}'. Detected args: '{args}'. Detected kwargs: '{kwargs}'"
        )
        fire.Fire(wdoc)


@optional_typecheck
def call_parse_file() -> None:
    if is_out_piped:
        args, kwargs = parse_args_fire()
        parsed = wdoc.parse_file(*args, **kwargs)
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

    print(parsed)
    # try:
    #     sys.stdout.write(out)
    #     sys.stdout.flush()
    # except BrokenPipeError as e:
    #     logger.error(
    #         f"wdoc encountered a BrokenPipeError, crashing because it indicates that the code after the pipe crashed so we should not flood the output: {e}"
    #     )


if __name__ == "__main__":
    cli_launcher()
