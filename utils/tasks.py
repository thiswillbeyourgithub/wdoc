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

n_to_combine = 1  # careful, indices start at 0. So setting n_to_combine at 3 means that the first 4 paragraphs will get combined into one. This will certainly lose meaningful information.
n_check = 3  # number of check to do

def do_summarize(docs, metadata, model, llm, callback, verbose):
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""

    with callback() as cb:
        for ird, rd in tqdm(enumerate(docs), desc="Summarising splits"):
            # when ird == n_to_combine, the first n_to_combine summaries
            # will be checked n_check times for compactness. So the
            # progression number have to be reset to avoid giving
            # false impressions to the LLM.
            if ird > n_to_combine:
                fixed_index = f"{ird + 1 - n_to_combine}/{len(docs) - n_to_combine}"
            else:
                fixed_index = f"{ird + 1}/{len(docs)}"

            summarize_chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=chatgpt_summary_messages if model == "openai" else summarize_prompt,
                    verbose=verbose,
                    )
            out = summarize_chain(
                    {
                        "input_documents": [rd],
                        "metadata": metadata.replace("[PROGRESS]", fixed_index),
                        "rules": summary_rules,
                        "previous_summary": previous_summary,
                        },
                    return_only_outputs=False,
                    )

            summ = out["output_text"]

            summaries.append(summ)

            # given the influence of the first few summaries, make sure it's compact
            # and follows the rules
            if ird == n_to_combine:  # combine the first n summaries and make it more compact
                summ = "\n".join([s for s in summaries])

                red(f"Checking '{n_check}' times the first '{n_to_combine}' summaries.")
                red(f"Summary before correction:\n{summ}")

                for trial in range(n_check):
                    checksumm_chain = LLMChain(
                            llm=llm,
                            prompt=chatgpt_checksummary_messages if model == "openai" else checksummary_prompt,
                            verbose=verbose,
                            )
                    summ = checksumm_chain(
                            {
                                "summary_to_check": summ,
                                "rules": checksummary_rules,
                                }
                            )["text"]

                summaries = [summ]
                red(f"Summary after correction:\n{summ}")


            previous_summary = f"For context, here's the summary of the previous section of the text:\n'''\n{summaries[-1]}\n'''"
            if metadata:
                previous_summary = "\n" + previous_summary

    outtext = "\n- ---\n".join(summaries)

    return outtext, cb.total_tokens, cb.total_cost
