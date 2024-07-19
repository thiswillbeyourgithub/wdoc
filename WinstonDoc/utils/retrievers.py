"""
Retrievers used to retrieve the appropriate embeddings for a given query.
"""

from shutil import rmtree
from typing import Optional, Any, Callable, List

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore

from .misc import cache_dir, get_splitter
from .typechecker import optional_typecheck


@optional_typecheck
def create_hyde_retriever(
    query: str,

    llm: Any,
    top_k: int,
    relevancy: float,

    embeddings: Any,
    loaded_embeddings: Any,
) -> Any:
    """
    create a retriever only for the subset of documents from the
    loaded_embeddings that were found using HyDE technique (i.e. asking
    the llm to create a hypothetical answer and use the embedding of this
    answer to search similar content)

    The code is a little strange because it actually reloads only a portion
    of the embeddings from cache if possible.

    https://python.langchain.com/docs/use_cases/question_answering/how_to/hyde
    """

    HyDE_template = """Please imagine the answer to the user's question about a document:
User question: {question}
Answer:"""
    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template=HyDE_template,
    )

    hyde_chain = LLMChain(
        llm=llm,
        prompt=hyde_prompt,
    )

    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=hyde_chain,
        base_embeddings=embeddings,
    )
    loaded_embeddings.save_local("temp")
    db = FAISS.load_local("temp", hyde_embeddings,
                          allow_dangerous_deserialization=True)
    rmtree("temp")

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "distance_metric": "cos",
            "score_threshold": relevancy,
        }
    )
    return retriever


@optional_typecheck
def create_parent_retriever(
    task: str,
    loaded_embeddings: Any,
    loaded_docs: List[Document],
    top_k: int,
    relevancy: float,
) -> Any:
    "https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever"
    csp = get_splitter(task)
    psp = get_splitter(task)
    psp._chunk_size *= 4
    parent = ParentDocumentRetriever(
        vectorstore=loaded_embeddings,
        docstore=LocalFileStore(cache_dir / "parent_retriever"),
        child_splitter=csp,
        parent_splitter=psp,
        search_type="similarity",
        search_kwargs={
            "k": top_k,
            "distance_metric": "cos",
            "score_threshold": relevancy,
        }
    )
    parent.add_documents(loaded_docs)
    return parent
