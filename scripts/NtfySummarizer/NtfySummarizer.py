import json
import os
import time
import traceback
from functools import partial
from io import StringIO
from pathlib import Path, PosixPath
from typing import Union

import fire
import requests
from beartype import beartype
from rich.console import Console
from rich.markdown import Markdown

from wdoc import wdoc

VERSION = "0.3"

log_file = Path(__file__).parent / "logs.txt"


@beartype
def log(text: str) -> None:
    "minimalist log file"
    with log_file.open("a") as lf:
        lf.write(f"\n{int(time.time())}: {text}")


@beartype
def _send_notif(
    message: str,
    topic: str,
    title: str = "wdoc Summaries",
) -> str:
    """
    Send a notification to a specified ntfy.sh topic.

    Args:
        message (str): The message content to be sent.
        topic (str): The ntfy.sh topic to send the notification to.
        title (str, optional): The title of the notification. Defaults to "wdoc Summaries".

    Returns:
        str: The message that was sent.
    """
    requests.post(
        url=f"https://ntfy.sh/{topic}",
        data=message.encode(encoding="utf-8"),
        headers={
            "Title": title,
            "Markdown": "yes",
            # "Priority": "urgent",
            # "Tags": "warning,skull"
        },
    )
    return message


@beartype
def _send_file(
    message: str,
    path: Union[str, PosixPath],
    topic: str,
    title: str = "wdoc Summaries",
) -> None:
    """
    Send a file as an attachment to a specified ntfy.sh topic.

    Args:
        message (str): The message content to be sent with the file.
        path (Union[str, PosixPath]): The path to the file to be sent.
        topic (str): The ntfy.sh topic to send the notification to.
        title (str, optional): The title of the notification. Defaults to "wdoc Summaries".

    Raises:
        AssertionError: If the specified file does not exist.
    """
    assert Path(path).exists(), f"file not found: '{path}'"
    requests.post(
        url=f"https://ntfy.sh/{topic}",
        data=open(path, "r"),
        headers={
            "Title": title,
            # "Priority": "urgent",
            # "Tags": "warning,skull"
            "Filename": Path(path).name,
        },
    )


@beartype
def main(
    topic: str,
    message: str = None,
    render_md: bool = False,
) -> None:
    """
    Main function to process a URL or file type and URL, generate a summary, and send it as a notification.

    Args:
        topic (str): The ntfy.sh topic to send the notification to.
        message (str, optional): The message to process. If not provided, will use NTFY_MESSAGE env var.
        render_md (bool, False by default): True to pass the md string into rich for rendering before sending to ntfy

    Raises:
        AssertionError: If neither message argument nor NTFY_MESSAGE env var is available, or if the message format is incorrect.
        Exception: For any errors that occur during processing.
    """
    log(f"Started with topic: '{topic}'. Version: {VERSION}")
    topic = topic.strip()

    # Try env var first, fall back to argument
    if message is None:
        if "NTFY_MESSAGE" not in os.environ:
            raise ValueError(
                "No message provided: need either --message argument or NTFY_MESSAGE environment variable"
            )
        message = os.environ["NTFY_MESSAGE"]
    log(f"Message: {message}")
    sn = partial(
        _send_notif,
        topic=topic,
        title="wdoc Summaries",
    )

    try:
        message = message.strip()
        if message.startswith("http"):
            url = message
            filetype = "auto"
        else:
            filetype, url = message.split(" ", 1)
        log(f"Filetype: {filetype} ; Url: {url}")

        assert topic
        assert message
        assert url.startswith("http"), f"url must start with http, not '{url}'"

        instance = wdoc(
            task="summary",
            path=url,
            # notification_callback=sn,
        )
        results: dict = instance.summary_results
        log("Summary:\n" + json.dumps(results, ensure_ascii=False))

        md = results["summary"]

        full_message = f"""
# Summary
{url}

{md}

- Total cost of those summaries: '{results['doc_total_tokens']}' (${results['doc_total_cost']:.5f})
- Total time saved by those summaries: {results['doc_reading_length']:.1f} minutes
""".strip()

        if render_md:
            log("Using markdown rendering")
            # markdown rendering
            output = StringIO()
            console = Console(file=output, width=80, color_system=None)
            console.print(Markdown(full_message))
            full_message = output.getvalue()

        if len(full_message.encode("utf-8")) <= 4000:
            sn(message=full_message)
        else:
            message_shortened = message.split("://", 1)[1][:30].replace("/", "_")
            (Path(__file__).parent / "summaries").mkdir(exist_ok=True)
            path = (
                Path(__file__).parent
                / "summaries"
                / f"{message_shortened}_{int(time.time())}.md"
            )
            with open(path, "w") as f:
                f.write(full_message)
            try:
                _send_file(
                    title="wdoc Summaries",
                    path=path,
                    topic=topic,
                    message=f"Summary of {url}",
                )
            except Exception as err:
                sn(message=full_message)
                raise
        log("Done\n")
    except Exception as err:

        stack_trace_str = traceback.format_exc()

        err_mess = (
            f"Error for message '{message}':\n{str(err)}\n\nTrace:\n{stack_trace_str}"
        )
        log(err_mess)
        sn(message=err_mess)
        raise


if __name__ == "__main__":
    fire.Fire(main)
