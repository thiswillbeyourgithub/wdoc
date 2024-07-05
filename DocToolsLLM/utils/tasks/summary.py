"""
Chain (logic) used to summarize a document.
"""

from textwrap import indent
from tqdm import tqdm
from typing import List, Tuple, Any, Union

from langchain.docstore.document import Document

from ..prompts import BASE_SUMMARY_PROMPT, RECURSION_INSTRUCTION
from ..logger import whi, red
from ..typechecker import optional_typecheck

@optional_typecheck
def do_summarize(
    docs: List[Document],
    metadata: str,
    language: str,
    modelbackend: str,
    llm: Any,
    llm_price: List[float],
    verbose: bool,
    n_recursion: int=0,
    ) -> Tuple[str, int, int, Union[float, int]]:
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""

    llm.bind(verbose=verbose)

    total_tokens = 0
    total_cost = 0

    assert "[PROGRESS]" in metadata
    for ird, rd in tqdm(enumerate(docs), desc="Summarising splits", total=len(docs)):
        fixed_index = f"{ird + 1}/{len(docs)}"

        messages = BASE_SUMMARY_PROMPT.format_messages(
            text=rd.page_content,
            metadata=metadata.replace("[PROGRESS]", fixed_index),
            language=language,
            previous_summary=previous_summary,
            recursion_instruction="" if not n_recursion else RECURSION_INSTRUCTION
        )
        if " object at " in llm._get_llm_string():
            red(
                "Found llm._get_llm_string() value that potentially "
                f"invalidates the cache: '{llm._get_llm_string()}'\n"
                f"Related github issue: 'https://github.com/langchain-ai/langchain/issues/23257'")
        output = llm._generate_with_cache(messages)
        if hasattr(output.generations[0], "generation_info"):
            finish = output.generations[0].generation_info["finish_reason"]
        else:
            finish = "stop"
            assert "testing" in llm._get_llm_string(), f"Only the testing LLM can lack generation_info in its response"
        assert finish == "stop", f"Unexpected finish_reason: '{finish}'"
        assert len(output.generations) == 1
        out = output.generations[0].text
        if hasattr(output, "llm_output"):
            if output.llm_output:  # only present if not caching
                new_p = output.llm_output["token_usage"]["prompt_tokens"]
                new_c = output.llm_output["token_usage"]["completion_tokens"]
            else:
                new_p = 0
                new_c = 0
        else:
            assert "testing" in llm._get_llm_string(), f"Only the testing LLM can lack llm_output in its response"
            new_p = 0
            new_c = 0
        total_tokens += new_p + new_c
        total_cost += (new_p * llm_price[0] + new_c + llm_price[1]) / 1e6

        # the callback need to be updated manually when _generate is called
        llm.callbacks[0].prompt_tokens += new_p
        llm.callbacks[0].completion_tokens += new_c
        llm.callbacks[0].total_tokens += new_p + new_c

        output_lines = out.rstrip().splitlines()

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

        output_text = "\n".join([s for s in output_lines if s])

        # commented as this is useful for logseq only
        # # if recursive, keep the previous summary and store it as a collapsed
        # # block
        # if n_recursion:
        #     old = [f"- BEFORE RECURSION \#{n_recursion}\n  collapsed:: true"]
        #     old += [indent(o.rstrip(), "    ") for o in rd.page_content.splitlines()]
        #     old = "\n".join(old)
        #     output_text += "\n" + old

        if verbose:
            whi(output_text)

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

    return outtext.rstrip(), n, total_tokens, total_cost
