from pprint import pprint
import fire
import os
from tqdm import tqdm
from datetime import datetime

from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


from utils.prompts import refine_prompt, PROMPT
from utils.llm import load_llm
from utils.file_loader import load_doc
from utils.misc import get_kwargs, hasher, docstore_cache

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

def process_task(**kwargs):
    print("Processing task")
    if kwargs["task"] == "query":
        if "sbert_model" not in kwargs:
            kwargs["sbert_model"]="paraphrase-multilingual-mpnet-base-v2"
        embeddings = SentenceTransformerEmbeddings(model_name=kwargs["sbert_model"])
        model_hash = hasher(kwargs["sbert_model"])

        done_list = set()
        db = None
        for doc in tqdm(docs, desc="embedding documents"):
            hashcheck = f'{doc.metadata["hash"]}_{model_hash}'
            if (docstore_cache / hashcheck).exists():
                tqdm.write("Loaded from cache")
                temp = FAISS.load_local(str(docstore_cache / hashcheck), embeddings)
            else:
                tqdm.write("Computing embeddings")
                temp = FAISS.from_documents([doc], embeddings)
                temp.save_local(str(docstore_cache / hashcheck))
            if db is None:
                db = temp
            else:
                if not hashcheck in done_list:
                    db.merge_from(temp)
                    done_list.add(hashcheck)
                else:
                    tqdm.write(f"File '{doc.metadata['path']}' was already added, skipping.")
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
                    print(cb.total_tokens)
                    print(cb.total_cost)
                print(ans["result"])
                print("\n\nSources:")
                for doc in ans["source_documents"]:
                    pprint(f"{doc.metadata.items()}")
            except Exception as err:
                print(f"Error: '{err}'")
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
            print(cb.total_tokens)
            print(cb.total_cost)

        t = out["output_text"]
        for bulletpoint in t.split("\n"):
            print(bulletpoint)

    else:
        raise ValueError(kwargs["task"])


if __name__ == "__main__":
    kwargs = fire.Fire(get_kwargs)

    llm, callback = load_llm(**kwargs)
    docs = load_doc(**kwargs)
    print(f"\n\nLoaded '{len(docs)}' documents")

    out = process_task(**kwargs)

    print("Done.\nOpenning debugger.")
    breakpoint()
