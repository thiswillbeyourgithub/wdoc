from .logger import whi
from .lazy_lib_importer import lazy_import_statements, lazy_import
from .typechecker import optional_typecheck

exec(lazy_import_statements("""
import time
import re
from pathlib import Path
import json
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
"""))


@optional_typecheck
def ask_user(q: str, commands: dict):
    """
    Ask the question to the user.

    Accepts multiple prompt commands:
        /top_k=3 to change the top_k value.
        /debug to open a console.
        /multiline to write your question over multiple lines.
        /retriever=X with X:
            'default' to use regular embedding search
            'knn' to use KNN
            'svm' to use SVM
            'hyde' to use Hypothetical Document Embedding search
            'parent' to use parent retriever
            Can use several (i.e 'knn_svm_default')
        /retriever=simple to use regular embedding search
        /relevancy=0.5 to set the relevancy threshold for retrievers that support it
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
            "/retriever",
            "/relevancy=",
            ]
    if commands["task"] == "query":
        autocomplete = WordCompleter(
                prompt_commands + [
                    x["prompt"]
                    for x in prev_questions
                    if x["task"] == commands["task"]
                    ],
                match_middle=True,
                ignore_case=True)
    else:
        autocomplete = WordCompleter(
                prompt_commands,
                match_middle=True,
                ignore_case=True)

    try:
        try:
            if commands["multiline"]:
                whi("Multiline mode activated. Use ctrl+D to send.")
            user_question = prompt(q,
                         completer=autocomplete,
                         vi_mode=True,
                         multiline=commands["multiline"])
        except (KeyboardInterrupt, EOFError):
            if commands["multiline"]:
                pass
            else:
                raise

        # quit if needed
        if user_question.strip() in ["quit", "Q", "q"]:
            whi("Quitting.")
            raise SystemExit()

        # auto remove duplicate "slash" (i.e. //) before prompts commands
        for pc in prompt_commands:
            while f"/{pc}" in user_question:
                user_question = user_question.replace(f"/{pc}", f"{pc}")

        # parse prompt commands
        if "/top_k=" in user_question:
            try:
                prev = commands["top_k"]
                commands["top_k"] = int(re.search(r"/top_k=(\d+)", user_question).group(1))
                user_question = re.sub(r"/top_k=(\d+)", "", user_question)
                whi(f"Changed top_k from '{prev}' to '{commands['top_k']}'")
            except Exception as err:
                whi(f"Error when changing top_k: '{err}'")
                return ask_user(q, commands)

        if "/relevancy=" in user_question:
            try:
                prev = commands["relevancy"]
                commands["relevancy"] = float(re.search(r"/relevancy=([0-9.]+)", user_question).group(1))
                user_question = re.sub(r"/relevancy=([0-9.]+)", "", user_question)
                whi(f"Changed relevancy from '{prev}' to '{commands['relevancy']}'")
            except Exception as err:
                whi(f"Error when changing relevancy: '{err}'")
                return ask_user(q, commands)

        if "/retriever=" in user_question:
            assert user_question.count("/retriever=") == 1, (
                f"multiple retriever commands found: '{user_question}'")
            retr = user_question.split("/retriever=")[1].split(" ")[0]
            commands["retriever"] = retr
            user_question = user_question.replace(f"/retriever={retr}", "").strip()
            whi("Using as retriever: '{retr}'")

        if "/debug" in user_question:
            whi("Entering debug mode.")
            breakpoint()
            whi("Restarting prompt.")
            return ask_user(q, commands)

        if "/multiline" in user_question:
            if not commands["multiline"]:
                commands["multiline"] = True
                whi("Multiline turned on.")
            else:
                commands["multiline"] = False
                whi("Multiline turned off.")
            return ask_user(q, commands)
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
                    "task": commands["task"],
                    })
    prev_questions = sorted(
            prev_questions,
            key=lambda x: x["timestamp"],
            )
    json.dump(prev_questions, pp_file.open("w"), indent=4)

    return user_question, commands
