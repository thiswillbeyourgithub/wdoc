"""
Parse document functionality.
"""

import inspect
import json
from pathlib import Path
from typing import List, Optional, Union

from langchain.docstore.document import Document

from wdoc.utils.batch_file_loader import batch_load_doc
from wdoc.utils.logger import debug_exceptions, set_parse_doc_help_md_as_docstring
from wdoc.utils.misc import DocDict, ModelName


@set_parse_doc_help_md_as_docstring
def parse_doc(
    filetype: str = "auto",
    format: str = "text",
    debug: bool = False,
    verbose: bool = False,
    out_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Union[List[Document], str, List[dict]]:
    """
    This docstring is dynamically updated with the content of wdoc/docs/parse_doc_help.md
    """
    assert format in [
        "text",
        "xml",
        "langchain",
        "langchain_dict",
    ], f"Unexpected --format value: '{format}'"
    default_cli_kwargs = {
        "llm_name": ModelName("cliparser/cliparser"),
        "backend": "loky",  # doesn't matter because n_jobs is 1 anyway
        "n_jobs": 1,
        "loading_failure": "crash",
    }

    if debug:
        debug_exceptions()

    if "task" in kwargs:
        assert (
            kwargs["task"] == "parse"
        ), f"Unexpected task when parsing. Expected 'parse' but got '{kwargs['task']}'"
        del kwargs["task"]
    assert "task" not in kwargs, "Cannot give --task argument if we are only parsing"

    docdict_kwargs = {}
    cli_kwargs = {}
    for k, v in kwargs.items():
        if k in DocDict.allowed_keys:
            docdict_kwargs[k] = v
        else:
            cli_kwargs[k] = v

    # Check if any cli_kwargs arguments are part of wdoc.__init__ signature
    # Import wdoc here to avoid circular imports
    from wdoc.wdoc import wdoc

    wdoc_init_signature = inspect.signature(wdoc.__init__)
    wdoc_init_params = set(wdoc_init_signature.parameters.keys()) - {"self"}

    conflicting_args = set(cli_kwargs.keys()) & wdoc_init_params
    if conflicting_args:
        raise ValueError(
            f"The following arguments are not allowed when using the parser only: {', '.join(sorted(conflicting_args))}. "
            f"These arguments are expected by wdoc.__init__ and can only be used when running the full wdoc workflow. "
            f"Run 'wdoc parse --help' to see available parsing arguments, or 'wdoc --help' for more information."
        )

    for k, v in default_cli_kwargs.items():
        if k not in cli_kwargs:
            cli_kwargs[k] = v

    out = batch_load_doc(
        task="parse",
        filetype=filetype,
        **cli_kwargs,
        **docdict_kwargs,
    )

    # Process format and prepare the result
    if format == "text":
        n = len(out)
        if n > 1:
            result = (
                "Parsed documents:\n"
                + "\n".join(
                    [
                        f"Doc #{i + 1}/{n}\n{d.page_content}\n\n"
                        for i, d in enumerate(out)
                    ]
                ).rstrip()
            )
        else:
            result = f"Parsed document:\n{out[0].page_content.strip()}"
    elif format == "xml":
        result = (
            "<documents>\n"
            + "\n".join(
                [f"<doc id={i}>\n{d.page_content}\n</doc>" for i, d in enumerate(out)]
            )
            + "\n</documents>"
        )
    elif format == "langchain":
        result = out
    elif format == "langchain_dict":
        result = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in out
        ]
    else:
        raise ValueError(format)

    # Handle writing to output file if specified
    if out_file:
        out_file_path = Path(out_file)

        # Check if file exists and is binary
        if out_file_path.exists():
            try:
                # Try to read as text to check if it's binary
                with open(out_file_path, "r", encoding="utf-8") as f:
                    f.read(1)  # Just read one character to test
            except (UnicodeDecodeError, UnicodeError):
                raise ValueError(
                    f"Output file '{out_file_path}' exists and appears to be binary. Cannot append to binary files."
                )

        # Prepare output text for file writing
        if format == "langchain":
            # Convert to JSON for file output
            file_content = json.dumps(
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in result
                ],
                indent=2,
                ensure_ascii=False,
            )
        elif format == "langchain_dict":
            file_content = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            # For "text" and "xml" formats, result is already a string
            file_content = result

        # Append to file
        with open(out_file_path, "a", encoding="utf-8") as f:
            if out_file_path.exists() and out_file_path.stat().st_size > 0:
                f.write("\n")  # Add newline separator if file is not empty
            f.write(file_content)

    return result
