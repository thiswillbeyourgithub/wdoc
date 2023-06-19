import pickle
from pathlib import Path
from pprint import pprint
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

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

def process_task(**kwargs):
    whi("Processing task")
    docs = kwargs["loaded_docs"]

    if kwargs["task"] == "query":
        with open(str(kwargs["savetopickle"]), "wb") as f:
            pickle.dump(
                    [kwargs["loaded_docs"], kwargs["loaded_embeddings"]],
                    f)
        db = kwargs["loaded_embeddings"]
        retriever = db.as_retriever(search_kwargs={"k": 5})
        qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True,
                )

        while True:
            try:
                with callback as cb:
                    query = input("\n\nEnter a question:\n>")
                    ans = qa(
                            inputs={"query": query},
                            return_only_outputs=False,
                            include_run_info=True,
                            )
                    whi(cb.total_tokens)
                    whi(cb.total_cost)
                whi(ans["result"])
                whi("\n\nSources:")
                for doc in ans["source_documents"]:
                    pprint(f"{doc.metadata.items()}")
            except Exception as err:
                whi(f"Error: '{err}'")
                breakpoint()

    elif kwargs["task"] == "summary":
        with callback as cb:
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
    if "loadfrompickle" in kwargs:
        red("Loading documents and embeddings from pickle file")
        path = Path(kwargs["loadfrompickle"])
        assert path.exists(), f"pickle file not found at '{path}'"
        with open(str(path), "rb") as f:
            loaded = pickle.load(f)
            kwargs["loaded_docs"] = loaded[0]
            kwargs["loaded_embeddings"] = loaded[1]
    else:
        kwargs = load_documents(**kwargs)
    whi(f"\n\nLoaded '{len(kwargs['loaded_docs'])}' documents")

    out = process_task(**kwargs)

    whi("Done.\nOpenning debugger.")
    breakpoint()
