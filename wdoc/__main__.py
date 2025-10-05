"""
Entry point file
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Union
import fire
import logging
from loguru import logger

# Just in case: we set the env variable of WDOC_PRIVATE_MODE as early as possible, even before importing wdoc
if re.findall(r"\b--private\b", " ".join(sys.argv)):
    logger.warning("Detected --private mode: setting WDOC_PRIVATE_MODE to True")
    os.environ["WDOC_PRIVATE_MODE"] = True

from wdoc.utils import logger as importedlogger  # make sure to setup the logs first
from wdoc.wdoc import wdoc
from wdoc.utils.env import is_out_piped
from wdoc.utils.misc import get_piped_input, tasks_list
from wdoc.utils.batch_file_loader import infer_filetype, NoInferrableFiletype
from typing import Tuple, List, Dict, Any
import io

# if __main__ is called, then we are using the cli instead of importing the class from python
wdoc.__import_mode__ = False

# fix fire always opening pager
# Source: https://github.com/google/python-fire/issues/188
fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")


def parse_args_fire() -> Tuple[List[Any], Dict[str, Any]]:
    """
    Parses command-line arguments using fire.Fire without printing its output.

    Temporarily redirects stdout to suppress fire's default output during parsing.

    Returns:
        A tuple containing the list of positional arguments and a dictionary of keyword arguments.
    """
    # fix issue when using " -- --completion"
    argline = sys.argv.copy()
    if "--" in argline:
        sys.argv.remove("--")
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to suppress fire output
    try:
        args, kwargs = fire.Fire(lambda *args, **kwargs: (list(args), kwargs))
    finally:
        sys.stdout = original_stdout  # Restore original stdout
    sys.argv = argline
    return args, kwargs


def handle_piped_input(piped_data: Union[str, bytes]) -> str:
    """Processes piped input and returns the appropriate argument string."""
    import tempfile

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
                import rtoml

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
            target = wdoc.parse_doc
        else:
            target = wdoc
        doc = getattr(target, "__doc__")
        assert (
            doc
        ), f"The signature wrapping apparently failed: empty __doc__ for {target}"
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

    if "web" in args:
        logger.debug("Detected 'web' in args, setting 'task' to 'query'")
        args.pop(args.index("web"))
        sys.argv.pop(sys.argv.index("web"))
        kwargs["task"] = "query"
        sys.argv.append("--task='query'")

        if "filetype" not in kwargs:
            logger.debug("Web search: specifying that 'filetype' is 'ddg'")
            kwargs["filetype"] = "ddg"
            sys.argv.append("--filetype=ddg")
        elif kwargs["filetype"] != "ddg":
            logger.warning("Web search: forcing argument 'filetype' to be 'ddg'")
            kwargs["filetype"] = "ddg"
            sys.argv.append("--filetype=ddg")

        if "query" not in kwargs and "path" not in kwargs:
            if len(args) == 1:
                logger.debug(
                    "Web search task without 'query' nor 'path' but with positional arg: using it as query and path"
                )
                temp = args.pop(0)
                sys.argv.pop(sys.argv.index(temp))
                kwargs["path"] = temp
                sys.argv.append(f"--path={temp}")
                kwargs["query"] = temp
                sys.argv.append(f"--query={temp}")
            else:
                logger.warning(
                    "Web search task with no 'query' nor 'path' but several positional arg: expecting only one to treat it as query and path"
                )

    if "completion" in kwargs or "--completion" in sys.argv:
        if " -- --completion" in " ".join(sys.argv):
            if " parse " in " ".join(sys.argv):
                sys.argv.remove("parse")
                fire.Fire(wdoc.parse_doc)
            else:
                fire.Fire(wdoc)
            sys.exit(0)
        else:
            raise Exception(
                "To create completion scripts, use '-- --completion' as arguments."
            )

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

    # replace frequently mystyped argument
    if "ddg_max_result" in kwargs and "ddg_max_results" not in kwargs:
        logger.debug("Replacing wrong arg ddg_max_result by ddg_max_result")
        kwargs["ddg_max_results"] = kwargs["ddg_max_result"]
        del kwargs["ddg_max_result"]
        sys.argv = [
            (
                elem
                if "ddg_max_result" not in elem
                else elem.replace("ddg_max_result", "ddg_max_results")
            )
            for elem in sys.argv
        ]

    # if there are remaining args, use the infer_filetype function to see if they are the missing path or the query
    if args:
        candidates = []
        for arg in args:
            # if we can't infer the filetype then it's probably the implicit --query of the user
            infered = None
            try:
                infered = infer_filetype(arg)
            except NoInferrableFiletype:
                if "query" in kwargs["task"]:
                    infered = "user_query"
            candidates.append(infered)

        if "query" in kwargs["task"] and (
            "query" not in kwargs or not isinstance(kwargs["query"], str)
        ):
            if len([c for c in candidates if c == "user_query"]) == 1:
                query_index = candidates.index("user_query")
                kwargs["query"] = args.pop(query_index)
                sys.argv[sys.argv.index(kwargs["query"])] = (
                    f"--query='{kwargs['query']}'"
                )
                candidates.pop(query_index)

        if (
            args
            and not ("path" in args or "path" in kwargs)
            and len(candidates) == 1
            and candidates[0]
        ):
            kwargs["path"] = args.pop(0)
            sys.argv[sys.argv.index(kwargs["path"])] = f"--path='{kwargs['path']}'"

    # when using web search, make sure to make query and path match the other by default
    if "filetype" in kwargs and kwargs["filetype"] == "ddg":
        if "path" in kwargs and "query" not in kwargs:
            logger.debug(
                "Detected DDG search with 'path' but no 'query' argument, duplicating the 'path' to 'query' then."
            )
            kwargs["query"] = kwargs["path"]
            sys.argv.append(f"--query={kwargs['path']}")
        elif "path" not in kwargs and "query" in kwargs:
            logger.debug(
                "Detected DDG search with 'query' but no 'path' argument, duplicating the 'query' to 'path' then."
            )
            kwargs["path"] = kwargs["query"]
            sys.argv.append(f"--path={kwargs['query']}")

    # if args is not empty, we have not succesfully parsed everything
    if args:
        logger.warning(
            f"It appears the arguments parsing was not complete: remaining args were not dispatched to kwargs: '{args}'"
        )

    if kwargs["task"] == "parse":
        call_parse_doc()
        sys.exit(0)

    else:
        logger.debug(
            f"Launching wdoc after arg transformation: Remaining sys.argv: '{sys.argv}'. Detected args: '{args}'. Detected kwargs: '{kwargs}'"
        )
        # fire.Fire(wdoc)
        # fire can be a bit finicky so let's try instead using the args and kwargs directly:
        _ = wdoc(*args, **kwargs)


def call_parse_doc() -> None:
    if is_out_piped:
        args, kwargs = parse_args_fire()
        parsed = wdoc.parse_doc(*args, **kwargs)
        # Check if out_file was used - if so, don't print to stdout
        if "out_file" in kwargs and kwargs["out_file"]:
            return
    else:
        parsed = fire.Fire(wdoc.parse_doc)
        # For fire.Fire, we need to check sys.argv for out_file
        if "--out_file" in " ".join(sys.argv) or any(
            arg.startswith("out_file=") for arg in sys.argv
        ):
            return

    # Only print to stdout if not writing to file
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

    print(out)


if __name__ == "__main__":
    cli_launcher()
