# source https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html

from pathlib import Path
import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from pprint import pprint

assert Path("API_KEY.txt").exists(), "No api key found"
os.environ["OPENAI_API_KEY"] = str(Path("API_KEY.txt").read_text()).strip()

llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        verbose=True,
        )

text_splitter = CharacterTextSplitter()

def load_doc(path):
    assert Path(path).exists(), f"file not found: '{path}'"
    with open(path) as f:
        content = f.read()[:1000]
    texts = text_splitter.split_text(content)
    if len(texts) > 5:
        ans = input(f"Number of texts splits: '{len(texts)}'. Continue? (y/n)\n>")
        if ans != "y":
            raise SystemExit("Quitting")
    docs = [Document(page_content=t) for t in texts]
    return docs


prompt_template = """Write a very concise summary of the author's reasonning paragraph by paragraph as logically indented markdown bullet points:

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
    docs = load_doc(sys.argv[-1])
    chain = load_summarize_chain(llm, chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt)
    out = chain({"input_documents": docs}, return_only_outputs=True)

    t = out["output_text"]
    for bulletpoint in t.split("\n"):
        print(bulletpoint)

    print("Openning console.")
    import code ; code.interact(local=locals())
