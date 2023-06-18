# source https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html

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

def load_doc(path, filetype, **kwargs):
    if filetype == "path_list":
        doclist = str(Path(path).read_text()).splitlines()
        docs = []
        for line in tqdm(doclist, desc="loading list of documents"):
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.count(" ; ") == 0:
                raise Exception(f"Missing ' ; ' in {line}")
            elif line.count(" ; ") == 1:
                linepath, filetype = line.split(" ; ")
            elif line.count(" ; ") > 1:
                splits = line.split(" ; ")
                filetype = splits[-1]
                linepath = " ; ".join(splits[:-1])
            docs.extend(load_doc(linepath, filetype))
        return docs

    if filetype == "youtube":
        if "youtube.com" in path:
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
        print(f"Loading pdf: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        loader = PyPDFLoader(path)
        docs = split_cache.eval(loader.load_and_split)

    else:
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
        if "path" not in docs[i].metadata:
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
            hashcheck = doc.metadata["hash"] + model_hash
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
