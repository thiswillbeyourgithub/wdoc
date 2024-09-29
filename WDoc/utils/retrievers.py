"""
Retrievers used to retrieve the appropriate embeddings for a given query.
"""

from typing import Any, List
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from .misc import cache_dir, get_splitter
from .typechecker import optional_typecheck

@optional_typecheck
def create_multiquery_retriever(
    llm,
    retriever: BaseRetriever,
    ) -> MultiQueryRetriever:
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )
    return retriever_from_llm

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
            "score_threshold": relevancy,
        }
    )
    parent.add_documents(loaded_docs)
    return parent
