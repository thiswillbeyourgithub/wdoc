# source https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html

import fire
from pathlib import Path
import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.callbacks import get_openai_callback
from pprint import pprint

assert Path("API_KEY.txt").exists(), "No api key found"
os.environ["OPENAI_API_KEY"] = str(Path("API_KEY.txt").read_text()).strip()

llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        verbose=True,
        )

text_splitter = CharacterTextSplitter()

def load_doc(path, filetype, *args, **kwargs):
    if filetype == "youtube":
        if "youtube.com" in path:
            print("Loading youtube")
            loader = YoutubeLoader.from_youtube_url(
                    path,
                    add_video_info=True,
                    language=[kwargs["language"]],
                    translation=kwargs["translation"],
                    )
            loader.load()
            docs = loader.load_and_split()
    elif filetype == "pdf":
        print("Loading pdf")
        assert Path(path).exists(), f"file not found: '{path}'"
        loader = PyPDFLoader(path)
        docs = loader.load_and_split()[:2]
        breakpoint()
    else:
        print("Loading txt")
        assert Path(path).exists(), f"file not found: '{path}'"
        with open(path) as f:
            content = f.read()
        if len(content) > 1000:
            print("Long content, openning console")
            breakpoint()
        texts = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in texts]
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

if __name__ == "__main__":
    import sys
    docs = fire.Fire(load_doc)
    with get_openai_callback() as cb:
        chain = load_summarize_chain(llm, chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt, verbose=True)
        out = chain({"input_documents": docs}, return_only_outputs=True)
        print(cb.total_tokens)
        print(cb.total_cost)

    t = out["output_text"]
    for bulletpoint in t.split("\n"):
        print(bulletpoint)

    print("Openning console.")
    import code ; code.interact(local=locals())
