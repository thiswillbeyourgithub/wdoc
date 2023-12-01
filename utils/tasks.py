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

            output_lines = out["output_text"].rstrip().splitlines()

            for il, ll in enumerate(output_lines):
                # remove if contains no alphanumeric character
                if not any(char.isalpha() for char in ll.strip()):
                    output_lines[il] = None
                    continue

                ll = ll.rstrip()

                # if a line does not start with - or *, fix it
                stripped = ll.lstrip()
                if not stripped.startswith("- ") and not stripped.startswith("* "):
                    ll = ll.replace(stripped[0], "- " + stripped[0], 1)

                # if contains uneven number of bold markers
                if ll.count("**") % 2 == 1:
                    ll += "**"  # and the bold
                if ll.count("*") % 2 == 1 and not ll.lstrip().startswith("* "):
                    ll += "*"  # end the italic

                output_lines[il] = ll

            output_text = "\n".join([s for s in output_lines if s])

            if verbose:
                whi(output_text)

            summaries.append(output_text)

            # finding the end of the summary to give as context to the next one
            lines = "\n".join(summaries).splitlines()
            end_of_latest_summary = []
            # add the lines of the previous summary in reverse order
            # and stop when there is no indentation
            for line in lines[::-1]:
                end_of_latest_summary.insert(0, line.rstrip())
                if not line.startswith("\t"):
                    break
            end_of_latest_summary = "\n".join(end_of_latest_summary)
            previous_summary = f"Here's the end of the summary of the previous section. Take this into consideration to avoid repeating information (there is a huge overlap between both sections). If relevant, you can start with the same indentation.\n'''\{end_of_latest_summary}\n'''"
            if metadata:
                previous_summary = "\n\n" + previous_summary

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
