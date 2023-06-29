from tqdm import tqdm

from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils.prompts import (
        summary_rules,
        system_summary_template, human_summary_template,
        checksummary_rules,
        system_checksummary_template, human_checksummary_template,
        )
from utils.logger import whi, yel, red

# prompts to summarize
summarize_prompt = PromptTemplate(
        template=system_summary_template + "\n\n" + human_summary_template,
        input_variables=["text", "previous_summary", "metadata", "rules"],
        )
chatgpt_summary_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_summary_template),
            HumanMessagePromptTemplate.from_template(human_summary_template),
            ],
        )

# prompt to check the summarization quality
checksummary_prompt = PromptTemplate(
        template=system_checksummary_template + "\n\n" + human_checksummary_template,
        input_variables=["summary_to_check", "rules"],
        )
chatgpt_checksummary_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_checksummary_template),
            HumanMessagePromptTemplate.from_template(human_checksummary_template),
            ],
        )

def do_summarize(
        n_to_combine,
        n_summpasscheck,
        docs,
        metadata,
        model,
        llm,
        callback,
        verbose,
        ):
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""

    summarize_chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=chatgpt_summary_messages if model == "openai" else summarize_prompt,
            verbose=verbose,
            )
    checksumm_chain = LLMChain(
            llm=llm,
            prompt=chatgpt_checksummary_messages if model == "openai" else checksummary_prompt,
            verbose=verbose,
            )

    with callback() as cb:
        for ird, rd in tqdm(enumerate(docs), desc="Summarising splits"):
            # when ird == n_to_combine, the first n_to_combine summaries
            # will be checked n_summpasscheck times for compactness. So the
            # progression number have to be reset to avoid giving
            # false impressions to the LLM.
            if ird <= n_to_combine:
                fixed_index = f"{ird + 1 - min(n_to_combine, ird)}/{len(docs) - min(n_to_combine, ird)}"
            else:
                fixed_index = f"{ird + 1}/{len(docs)}"

            out = summarize_chain(
                    {
                        "input_documents": [rd],
                        "metadata": metadata.replace("[PROGRESS]", fixed_index),
                        "rules": summary_rules,
                        "previous_summary": previous_summary,
                        },
                    return_only_outputs=False,
                    )

            summaries.append(out["output_text"])

            # given the influence of the first few summaries, make sure it's compact
            # and follows the rules
            if ird <= n_to_combine:  # combine the first n summaries and make it more compact
                summaries = ["\n".join(summaries)]

            # run the check also on each individual paragraph without
            # combining
            if verbose and n_summpasscheck:
                red(f"Chunk summary {ird} before check:\n{summaries[-1]}")
            for trial in range(n_summpasscheck):
                summaries[-1] = checksumm_chain(
                        {
                            "summary_to_check": summaries[-1],
                            "rules": checksummary_rules,
                            }
                        )["text"]
            if verbose and n_summpasscheck:
                red(f"Chunk summary {ird} after check:\n{summaries[-1]}")

            previous_summary = f"For reference, here's the summary of the previous chunk of text (you can directly continue this summary and use the same indentation as there is an overlap between the two chunks):\n'''\n{summaries[-1]}\n'''"
            if metadata:
                previous_summary = "\n" + previous_summary

    # combine summaries as one string
    n = len(summaries)
    outtext = f"-Chunk 1/{n}\n"
    for i, s in enumerate(summaries):
        outtext += s + "\n"
        if s != summaries[-1]:
            outtext += f"- ---\n- Chunk {i + 2}/{n}\n"

    return outtext, cb.total_tokens, cb.total_cost
