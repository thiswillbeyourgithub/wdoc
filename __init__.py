from pathlib import Path
import fire
import os
from tqdm import tqdm
from datetime import datetime

from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA


from utils.prompts import refine_prompt, PROMPT
from utils.llm import load_llm
from utils.file_loader import load_documents
from utils.misc import check_kwargs
from utils.logger import whi, yel, red
from utils.cli import ask_user

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

def process_task(llm, callback, **kwargs):
    red("\nProcessing task")

    if kwargs["task"] == "query":
        db = kwargs["loaded_embeddings"]

        # set default ask_user argument
        multiline = False
        top_k = 3

        while True:
            try:
                with callback() as cb:
                    query, top_k, multiline = ask_user(
                            "\n\nWhat is your question? (Q to quit)\n",
                            top_k=top_k,
                            multiline=multiline,
                            )
                    retriever = db.as_retriever(search_kwargs={"k": top_k})
                    qa = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                            verbose=True,
                            )

                    ans = qa(
                            inputs={"query": query},
                            return_only_outputs=False,
                            include_run_info=True,
                            )

                red(ans["result"])

                whi("\n\nSources:")
                for doc in ans["source_documents"]:
                    for toprint in [
                            "filetype", "nid", "anki_deck", "ntags"]:
                        if toprint in doc.metadata:
                            val = doc.metadata[toprint]
                            yel(f"    * {toprint}: {val}")
                    whi("    * Head:")
                    whi(f'{doc.metadata["head"].strip():>30}')

                red(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost})")

            except Exception as err:
                whi(f"Error: '{err}'")
                if "debug" in kwargs:
                    raise
                breakpoint()

    elif kwargs["task"] == "summary":
        docs = kwargs["loaded_docs"]
        with callback() as cb:
            chain = load_summarize_chain(
                    llm,
                    chain_type="refine",
                    return_intermediate_steps=True,
                    question_prompt=PROMPT,
                    refine_prompt=refine_prompt,
                    verbose=True,
                    )
            out = chain(
                    {"input_documents": docs},
                    return_only_outputs=True,
                    )
        whi(cb.total_tokens)
        whi(cb.total_cost)

        t = out["output_text"]
        for bulletpoint in t.split("\n"):
            whi(bulletpoint)

    else:
        raise ValueError(kwargs["task"])


if __name__ == "__main__":
    kwargs = fire.Fire(check_kwargs)

    llm, callback = load_llm(**kwargs)
    kwargs = load_documents(**kwargs)

    out = process_task(llm, callback, **kwargs)

    whi("Done.\nOpenning debugger.")
    breakpoint()
