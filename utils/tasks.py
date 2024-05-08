from textwrap import indent
from tqdm import tqdm

# from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils.prompts import (
        summary_rules,
        system_summary_template,
        human_summary_template,
        system_summary_template_recursive,
        )
from utils.logger import whi, yel, red

# prompts to summarize
summarize_prompt = PromptTemplate(
        template=system_summary_template + "\n\n" + human_summary_template + "\n\nYour summary:\n",
        input_variables=["text", "previous_summary", "metadata", "rules"],
        )
summarize_prompt_recursive = PromptTemplate(
        template=system_summary_template_recursive + "\n\n" + human_summary_template + "\n\nYour summary:\n",
        input_variables=["text", "previous_summary", "metadata", "rules"],
        )

# chat models
chatgpt_summary_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_summary_template),
            HumanMessagePromptTemplate.from_template(human_summary_template),
            ],
        )
chatgpt_summary_messages_recursive = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_summary_template_recursive),
            HumanMessagePromptTemplate.from_template(human_summary_template),
            ],
        )

def do_summarize(
        docs,
        metadata,
        language,
        modelbackend,
        llm,
        callback,
        verbose,
        n_recursion=0,
        logseq_mode=False,
        ):
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""


    if n_recursion:
        prompt=chatgpt_summary_messages_recursive if modelbackend == "openai" else summarize_prompt_recursive
    else:
        prompt=chatgpt_summary_messages if modelbackend == "openai" else summarize_prompt
    summarize_chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=verbose,
            )

    assert "[PROGRESS]" in metadata
    with callback() as cb:
        for ird, rd in tqdm(enumerate(docs), desc="Summarising splits"):
            fixed_index = f"{ird + 1}/{len(docs)}"

            out = summarize_chain(
                    {
                        "input_documents": [rd],
                        "metadata": metadata.replace("[PROGRESS]", fixed_index),
                        "rules": summary_rules.replace("[LANGUAGE]", language),
                        "previous_summary": previous_summary,
                        },
                    return_only_outputs=False,
                    )

            output_lines = out["output_text"].rstrip().splitlines()

            for il, ll in enumerate(output_lines):
                # remove if contains no alphanumeric character
                if not any(char.isalpha() for char in ll.strip()):
                    output_lines[il] = None
                    continue

                ll = ll.rstrip()
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

                # replace leading double spaces by tabs
                cnt_inside = ll.lstrip().count("  ")
                cnt_leading = ll.count("  ") - cnt_inside
                if cnt_leading:
                    ll = ll.replace("  ", "\t", cnt_leading)
                # make sure no space have been forgotten
                ll = ll.replace("\t ", "\t\t")

                output_lines[il] = ll

            output_text = "\n".join([s for s in output_lines if s])

            # if recursive, keep the previous summary and store it as a collapsed
            # block
            if n_recursion and logseq_mode:
                old = [f"- BEFORE RECURSION \#{n_recursion}\n  collapsed:: true"]
                old += [indent(o.rstrip(), "\t") for o in rd.page_content.splitlines()]
                old = "\n".join(old)
                output_text += "\n" + old

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

    return outtext.rstrip(), n, cb.total_tokens, cb.total_cost
