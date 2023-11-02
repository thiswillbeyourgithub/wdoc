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
        # checksummary_rules,
        # system_checksummary_template, human_checksummary_template,
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

# # prompt to check the summarization quality
# checksummary_prompt = PromptTemplate(
#         template=system_checksummary_template + "\n\n" + human_checksummary_template,
#         input_variables=["summary_to_check", "rules"],
#         )
# chatgpt_checksummary_messages = ChatPromptTemplate.from_messages(
#         [
#             SystemMessagePromptTemplate.from_template(system_checksummary_template),
#             HumanMessagePromptTemplate.from_template(human_checksummary_template),
#             ],
#         )

def do_summarize(
        docs,
        metadata,
        language,
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
    # checksumm_chain = LLMChain(
    #         llm=llm,
    #         prompt=chatgpt_checksummary_messages if model == "openai" else checksummary_prompt,
    #         verbose=verbose,
    #         )

    assert "[PROGRESS]" in metadata
    with callback() as cb:
        for ird, rd in tqdm(enumerate(docs), desc="Summarising splits"):
            fixed_index = f"{ird + 1}/{len(docs)}"

            out = summarize_chain(
                    {
                        "input_documents": [rd],
                        "metadata": metadata.replace("[PROGRESS]", fixed_index),
                        "rules": summary_rules.replace("LANGUAGE", language),
                        "previous_summary": previous_summary,
                        },
                    return_only_outputs=False,
                    )

            summaries.append(out["output_text"].strip())

            previous_summary = f"End of the summary of the previous section (avoid overlap by directly starting a new bullet point):\n'''\n{summaries[-1]}\n'''"
            if metadata:
                previous_summary = "\n\n" + previous_summary

    # combine summaries as one string
    n = len(summaries)
    if n > 1:
        outtext = f"- Chunk 1/{n}\n"
        for i, s in enumerate(summaries):
            outtext += s + "\n"
            if n > 1 and s != summaries[-1]:
                outtext += f"- ---\n- Chunk {i + 2}/{n}\n"
    else:
        outtext = "\n".join(summaries)

    # check if some line is only made of repeated character
    for i, s in enumerate(outtext.split("\n")):
        if s == "- ---":
            continue
        if not s.replace("-", " ").strip():
            continue
        if not any(char.isalpha() for char in s):
            raise Exception(f"One line was only made of repeated characters?:\n'''\n{s}\n'''\nsummary:\n'''\n{outtext}\n'''")

    return outtext, n, cb.total_tokens, cb.total_cost
