# source https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html

import json
from bs4 import BeautifulSoup
import shutil
import ankipandas as akp
from pprint import pprint
import fire
from pathlib import Path
import os
from pprint import pprint
from tqdm import tqdm
import hashlib
from datetime import datetime
from joblib import Memory

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, FakeListLLM
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import DataFrameLoader
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"


Path(".cache").mkdir(exist_ok=True)
Path(".cache/docstore_cache").mkdir(exist_ok=True)
Path(".cache/split_cache").mkdir(exist_ok=True)

docstore_cache = Path(".cache/docstore_cache/")
split_cache = Memory(".cache/split_cache/")

def hasher(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

def html_to_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

class fakecallback:
    total_tokens = 0
    total_cost = 0

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

def load_llm(model="gpt4all", local_path="./ggml-wizardLM-7B.q4_2.bin", **kwargs):
    if model.lower() == "openai":
        print("Loading openai models")
        assert Path("API_KEY.txt").exists(), "No api key found"
        os.environ["OPENAI_API_KEY"] = str(Path("API_KEY.txt").read_text()).strip()

        llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                verbose=True,
                streaming=True,
                )
        callback = get_openai_callback()
    elif model.lower() == "gpt4all":
        print(f"loading gpt4all: '{local_path}'")
        local_path = Path(local_path)
        assert local_path.exists(), "local model not found"
        callbacks = [StreamingStdOutCallbackHandler()]
        # Verbose is required to pass to the callback manager
        llm = GPT4All(
                model=str(local_path.absolute()),
                n_ctx=512,
                n_threads=4,
                callbacks=callbacks,
                verbose=True,
                streaming=True,
                )
        callback = fakecallback()
    elif model.lower() == "fake":
        llm = FakeListLLM(verbose=True, responses=["fake answer"]*10)
        callback = fakecallback()
    else:
        raise ValueError(model)
    print("done loading model.\n")
    return llm, callback


text_splitter = CharacterTextSplitter()

def load_doc(**kwargs):
    filetype = kwargs["filetype"]

    if filetype in "path_list"  or "recursive" in filetype:
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        print(filetype)
        if "recursive" in filetype:
            assert filetype.count("recursive") == 1, "can't use more than 1 recursive word"
            assert filetype.count("recursive") + filetype.count("path_list"), "can't use recusrive and path_list in the same line"
            assert "pattern" in kwargs, "missing 'pattern' key in args"
            pattern = kwargs["pattern"]
            doclist = [p for p in Path(path).rglob(pattern)]
            doclist = [str(p) for p in doclist if p.is_file()]
            if "exclude" in kwargs:
                for exc in kwargs["exclude"]:
                    doclist = [p for p in doclist if exc not in p]
                del kwargs["exclude"]
            assert doclist, "empty recursive search!"
            assert " " in filetype, "missing space in recursive filetype"

        elif filetype == "path_list":
            doclist = str(Path(path).read_text()).splitlines()
        else:
            raise ValueError(filetype)

        docs = []
        for item in tqdm(doclist, desc="loading list of documents"):
            item = item.strip()
            if not item:
                continue
            if item.startswith("#"):
                continue
            if filetype == "path_list":
                meta = json.loads(item.strip())
                assert isinstance(meta, dict), f"meta from line '{item}' is not dict but '{type(meta)}'"
                assert "filetype" in meta, "no key 'filetype' in meta"
            elif "recursive" in filetype:
                meta = kwargs.copy()
                meta["filetype"] = filetype.replace("recursive", "").strip()
                meta["path"] = item
                assert Path(meta["path"]).exists(), f"file '{item}' does not exist"
                del meta["pattern"]
            else:
                raise ValueError(filetype)
            docs.extend(load_doc(**meta))
        return docs

    if filetype == "youtube":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        print(f"Loading youtube: '{path}'")
        loader = YoutubeLoader.from_youtube_url(
                path,
                add_video_info=True,
                language=[kwargs["language"]],
                translation=kwargs["translation"],
                )
        loader.load()
        docs = loader.load_and_split()

    elif filetype == "pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        print(f"Loading pdf: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        loader = PyPDFLoader(path)
        # try:
        docs = split_cache.eval(loader.load_and_split)
        # except Exception as err:
        #     print(f"Error when using cache to load '{path}': '{err}'")
        #     docs = loader.load_and_split()

    elif filetype == "anki":
        needed_keys = ["anki_deck", "anki_notetype", "anki_profile"]
        for nk in needed_keys:
            assert nk in kwargs, f"Missing '{nk}' in arguments from load_doc"
        profile = kwargs["anki_profile"]
        print(f"Loading anki profile: '{profile}'")
        original_db = akp.find_db(user=profile)
        name = f"{profile}".replace(" ", "_")
        temp_db = shutil.copy(original_db, f"./.cache/anki_collection_{name.replace('/', '_')}")
        col = akp.Collection(path=temp_db)
        cards = col.cards.merge_notes()
        cards["cdeck"] = cards["cdeck"].apply(lambda x: x.replace("\x1f", "::"))
        cards = cards[cards["cdeck"].str.startswith(kwargs["anki_deck"])]
        cards = cards[cards["nmodel"].str.startswith(kwargs["anki_notetype"])]
        cards["fields"] = cards["nflds"].apply(lambda x: "\n\n".join(x)[:500])
        cards["fields"] = cards["fields"].apply(lambda x: html_to_text(x))
        loader = DataFrameLoader(
            cards,
            page_content_column="fields",
            )
        docs = loader.load()
        for i in range(len(docs)):
            docs[i].metadata["anki_profile"] = profile
            docs[i].metadata["anki_deck"] = kwargs["anki_deck"]
            docs[i].metadata["anki_notetype"] = kwargs["anki_notetype"]
            docs[i].metadata["path"] = f"Anki profile '{profile}'"

    else:
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        print(f"Loading txt: '{path}'")
        print(path)
        assert Path(path).exists(), f"file not found: '{path}'"
        with open(path) as f:
            content = f.read()
        texts = split_cache.eval(text_splitter.split_text, content)
        docs = [Document(page_content=t) for t in texts]
    for i in range(len(docs)):
        docs[i].metadata["hash"] = hasher(docs[i].page_content)
        docs[i].metadata["head"] = str(docs[i].page_content)[:100]
        if "path" not in docs[i].metadata and "path" in locals():
            docs[i].metadata["path"] = path
    return docs


prompt_template = """Write in the same language of the input an easy to read summary of the author's reasonning paragraph by paragraph as logically indented markdown bullet points:

'''
{text}
'''

CONCISE SUMMARY AS LOGICALLY INDENTED MARKDOWN BULLET POINTS:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
    """Your job is to continue a summary of a long text as logically indented markdown bullet points of the author's reasonning.
    We have provided an existing summary up to this point:
    '''
    {existing_answer}
    '''

    You have to continue the summary by adding the bullet points of the following part of the article (only if relevant, stay concise, avoid expliciting what is implied by the previous bullet points):
    '''
    {text}
    '''
    Given this new section of the document, refine the summary as logically indented markdown bullet points. If the new section is not worth it, simply return the original summary."""
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)

def get_kwargs(**kwargs):
    return kwargs

if __name__ == "__main__":
    kwargs = fire.Fire(get_kwargs)

    llm, callback = load_llm(**kwargs)
    docs = load_doc(**kwargs)

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
                continue

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

    else:
        raise ValueError(kwargs["task"])

    t = out["output_text"]
    for bulletpoint in t.split("\n"):
        print(bulletpoint)

    print("Openning console.")
    import code ; code.interact(local=locals())
