"""
Chain (logic) used to summarize a document.
"""

from pathlib import Path

from beartype.typing import Any, List, Tuple, Union
from langchain.docstore.document import Document
from tqdm import tqdm
from loguru import logger

from ..misc import thinking_answer_parser, log_and_time_fn
from ..prompts import BASE_SUMMARY_PROMPT, PREV_SUMMARY_TEMPLATE, RECURSION_INSTRUCTION
from ..typechecker import optional_typecheck

HOME = str(Path.home())


@log_and_time_fn
@optional_typecheck
def do_summarize(
    docs: List[Document],
    metadata: str,
    language: str,
    modelbackend: str,
    llm: Any,
    llm_price: List[float],
    verbose: bool,
    n_recursion: int = 0,
) -> Tuple[str, int, Union[float, int], int, int]:
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""

    llm.bind(verbose=verbose)

    total_tokens = [0, 0]
    total_cost = 0

    metadata = metadata.replace(HOME, "~")  # extra privacy just in case a path appears

    assert "[PROGRESS]" in metadata
    for ird, rd in tqdm(enumerate(docs), desc="Summarising splits", total=len(docs)):
        fixed_index = f"{ird + 1}/{len(docs)}"

        messages = BASE_SUMMARY_PROMPT.format_messages(
            language=language,
            metadata=metadata.replace("[PROGRESS]", fixed_index),
            previous_summary=previous_summary,
            recursion_instruction="" if not n_recursion else RECURSION_INSTRUCTION,
            text=rd.page_content,
        )
        if " object at " in llm._get_llm_string():
            logger.warning(
                "Found llm._get_llm_string() value that potentially "
                f"invalidates the cache: '{llm._get_llm_string()}'\n"
                f"Related github issue: 'https://github.com/langchain-ai/langchain/issues/23257'"
            )
        try:
            output = llm._generate_with_cache(messages)
        except Exception as e:
            logger.warning(
                f"Error when generating with cache, trying without cache: '{e}'"
            )
            output = llm._generate(messages)
        if output.generations[0].generation_info is None:
            assert "fake-list-chat-model" in llm._get_llm_string()
            finish = "stop"
        else:
            finish = output.generations[0].generation_info["finish_reason"]
            assert finish == "stop", f"Unexpected finish_reason: '{finish}'"
            assert len(output.generations) == 1
        out = output.generations[0].text
        if output.llm_output:  # only present if not caching
            new_p = output.llm_output["token_usage"]["prompt_tokens"]
            new_c = output.llm_output["token_usage"]["completion_tokens"]
        else:
            new_p = 0
            new_c = 0
        total_tokens[0] += new_p
        total_tokens[1] += new_c
        total_cost += (new_p * llm_price[0] + new_c + llm_price[1]) / 1e6

        # the callback need to be updated manually when _generate is called
        llm.callbacks[0].prompt_tokens += new_p
        llm.callbacks[0].completion_tokens += new_c
        llm.callbacks[0].total_tokens += new_p + new_c

        parsed = thinking_answer_parser(out)
        if verbose and parsed["thinking"]:
            logger.info("Thinking: " + parsed["thinking"])

        output_lines = parsed["answer"].rstrip().splitlines(keepends=True)

        # Remove first line if:
        # - it contains "a deep breath"
        # - it starts with "i'll summarize" (case insensitive)
        # - it's a bullet point containing these phrases
        if output_lines:
            first_line = output_lines[0].lower()
            should_remove = (
                "a deep breath" in first_line
                or first_line.startswith("i'll summarize")
                or (
                    first_line.strip().startswith("- ")
                    and (
                        "a deep breath" in first_line or "i'll summarize" in first_line
                    )
                )
            )
            if should_remove:
                output_lines = output_lines[1:]

        for il, ll in enumerate(output_lines):
            # remove if contains no alphanumeric character
            if not any(char.isalpha() for char in ll.strip()):
                output_lines[il] = None
                continue

            ll = ll.rstrip()

            # replace tabs by 4 spaces
            ll = ll.replace("\t", "    ")
            ll = ll.replace("	", "    ")

            stripped = ll.lstrip()

            ll = ll.replace("- • ", "- ").replace("• ", "- ")  # occasional bad md

            # if a line starts with * instead of -, fix it
            if stripped.startswith("* "):
                ll = ll.replace("*", "-", 1)

            stripped = ll.lstrip()
            # beginning with long dash
            if stripped.startswith("—"):
                ll = ll.replace("—", "-")

            # begin by '-' but not by '- '
            stripped = ll.lstrip()
            if stripped.startswith("-") and not stripped.startswith("- "):
                ll = ll.replace("-", "- ", 1)

            # if a line does not start with - fix it
            stripped = ll.lstrip()
            if not stripped.startswith("- "):
                ll = ll.replace(stripped[0], "- " + stripped[0], 1)

            ll = ll.replace("****", "")

            # if contains uneven number of bold markers
            if ll.count("**") % 2 == 1:
                ll += "**"  # end the bold
            # and italic
            if ll.count("*") % 2 == 1:
                ll += "*"  # end the italic

            output_lines[il] = ll

        good_lines = [li for li in output_lines if (li and li.replace("-", "").strip())]
        output_text = "\n".join(good_lines)

        if verbose:
            logger.info(output_text)

        assert "{previous_summary}" in PREV_SUMMARY_TEMPLATE
        previous_summary = PREV_SUMMARY_TEMPLATE.replace(
            "{previous_summary}",
            "\n".join(good_lines[-10:]),
        )

        summaries.append(output_text)

    # combine summaries as one string separated by markdown separator
    n = len(summaries)
    if n > 1:
        outtext = f"- Chunk 1/{n}\n"
        for i, s in enumerate(summaries):
            outtext += s + "\n"
            if n > 1 and s != summaries[-1]:
                outtext += f"- ---\n- Chunk {i + 2}/{n}\n"
    else:
        outtext = "\n".join(summaries)

    return outtext.rstrip(), n, total_cost, total_tokens[0], total_tokens[1]
