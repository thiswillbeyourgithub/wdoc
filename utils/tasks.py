from tqdm import tqdm

from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils.prompts import summary_rules, system_summary_template, human_summary_template

summarize_prompt = PromptTemplate(
        template=system_summary_template + "\n\n" + human_summary_template,
        input_variables=["text", "previous_summary", "metadata", "rules"])


chatgpt_summary_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_summary_template),
            HumanMessagePromptTemplate.from_template(human_summary_template),
            ],
        )

def do_summarize(docs, metadata, model, llm, callback, verbose):
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""

    with callback() as cb:
        for ird, rd in tqdm(enumerate(docs), desc="Summarising splits"):

            # use either chat messages or one prompt
            if model == "openai":
                # use chat messages
                call_dict = {
                        "input_documents": [rd],
                        "metadata": metadata.replace("[PROGRESS]", f"{ird+1}/{len(docs)}"),
                        "rules": summary_rules,
                        "previous_summary": previous_summary,
                        }
                prompt = chatgpt_summary_messages
            else:
                # use one prompt
                call_dict = {"input_documents": [rd]}
                prompt = summarize_prompt.partial(
                        metadata=metadata.replace("[PROGRESS]", f"{ird+1}/{len(docs)}"),
                        rules=summary_rules,
                        previous_summary=previous_summary,
                        )

            summarize_chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt,
                    verbose=verbose,
                    )
            out = summarize_chain(
                    call_dict,
                    return_only_outputs=False,
                    )

            summaries.append(out["output_text"])

            previous_summary = f"For context, here's the summary of the previous section of the text:\n'''\n{summaries[-1]}\n'''"
            if metadata:
                previous_summary = "\n" + previous_summary

    outtext = "\n- ---\n".join(summaries)

    return outtext, cb.total_tokens, cb.total_cost
