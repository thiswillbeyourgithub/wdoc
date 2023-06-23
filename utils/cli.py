import time
import re
from pathlib import Path
import json
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from .logger import whi


def ask_user(q, top_k, multiline, task):
    """
    Ask the question to the user.

    Accepts multiple prompt commands:
        /top_k=3 to change the top_k value.
        /debug to open a console.
        /multiline to write your question over multiple lines.
    """
    # loading history from files
    prev_questions = []
    pp_file = Path(".cache/previous_questions.json")
    if pp_file.exists():
        pp_list = json.load(pp_file.open("r"))
        assert isinstance(pp_list, list), "Invalid cache type"
        for i, pp in enumerate(pp_list):
            assert isinstance(pp, dict), "Invalid item in cache"
            assert "prompt" in pp, "Invalid item in cache"
        for pp in pp_list:
            if "timestamp" not in pp:
                pp["timestamp"] = 0
            if "task" not in pp:
                pp["task"] = "query"
            if pp not in prev_questions:
                prev_questions.append(pp)
        prev_questions = sorted(
                prev_questions,
                key=lambda x: x["timestamp"],
                )

    prompt_commands = [
            "/multiline",
            "/debug",
            "/top_k=",
            ]
    if task == "query":
        autocomplete = WordCompleter(
                prompt_commands + sorted(
                    [
                    x["prompt"]
                    for x in prev_questions
                    if x["task"] == task
                    ], key=lambda x: x["timestamp"]),
                match_middle=True,
                ignore_case=True)
    else:
        autocomplete = WordCompleter(
                prompt_commands,
                match_middle=True,
                ignore_case=True)

    try:
        try:
            if multiline:
                whi("Multiline mode activated. Use ctrl+D to send.")
            user_question = prompt(q,
                         completer=autocomplete,
                         vi_mode=True,
                         multiline=multiline)
        except (KeyboardInterrupt, EOFError):
            if multiline:
                pass
            else:
                raise

        # quit if needed
        if user_question.strip() in ["quit", "Q", "q"]:
            whi("Quitting.")
            raise SystemExit()

        # auto remove duplicate "slash" (i.e. //) before prompts commands
        for pc in prompt_commands:
            if f"/{pc}" in user_question:
                user_question = user_question.replace(f"/{pc}", f"{pc}")

        # retry if user entered multiple commands
        if len([pc
                for pc in prompt_commands
                if (pc in user_question and 'keywords' not in pc)]) not in [0, 1]:
            whi("You can use at most 1 prompt command in a given query ("
               "excluding keywords).")
            return ask_user(q, top_k, multiline)

        # parse prompt commands
        if "/top_k=" in user_question:
            try:
                prev = top_k
                top_k = int(re.search(r"/top_k=(\d+)", user_question).group(1))
                user_question = re.sub(r"/top_k=(\d+)", "", user_question)
                whi(f"Changed top_k from '{prev}' to '{top_k}'")
            except Exception as err:
                whi(f"Error when changing top_k: '{err}'")
                return ask_user(q, top_k, multiline)

        if "/debug" in user_question:
            whi("Entering debug mode.")
            breakpoint()
            whi("Restarting prompt.")
            return ask_user(q, top_k, multiline)

        if "/multiline" in user_question:
            if multiline is False:
                multiline = True
                whi("Multiline turned on.")
            else:
                multiline = False
                whi("Multiline turned off.")
            return ask_user(q, top_k, multiline)
    except (KeyboardInterrupt, EOFError):
        raise SystemExit()

    # saving new history to file
    if len(
        [x
         for x in prev_questions
         if x["prompt"].strip() == user_question
         ]) == 0:
        prev_questions.append(
                {
                    "prompt": user_question,
                    "timestamp": int(time.time()),
                    "task": task,
                    })
    prev_questions = sorted(
            prev_questions,
            key=lambda x: x["timestamp"],
            )
    json.dump(prev_questions, pp_file.open("w"), indent=4)

    return user_question, top_k, multiline
