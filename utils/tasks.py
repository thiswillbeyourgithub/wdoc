from tqdm import tqdm

from langchain.chains.summarize import load_summarize_chain

from utils.prompts import summarize_prompt, summary_rules


def do_summarize(docs, metadata, llm, callback, verbose):
    "summarize each chunk of a long document"
    summaries = []
    previous_summary = ""

    with callback() as cb:
        for ird, rd in tqdm(enumerate(docs), desc="Summarising splits"):
            summarize_chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=summarize_prompt.partial(
                        metadata=metadata.replace("[PROGRESS]", f"{ird+1}/{len(docs)}"),
                        rules=summary_rules,
                        previous_summary=previous_summary,
                        ),
                    verbose=verbose,
                    )
            out = summarize_chain(
                    {"input_documents": [rd]},
                    return_only_outputs=False,
                    )

            summaries.append(out["output_text"])

            previous_summary = f"For context, here's the summary of the previous section of the text:\n'''\n{summaries[-1]}\n'''\n"

    outtext = "- ---".join(summaries)

    return outtext, cb.total_tokens, cb.total_cost
