"""
Retrievers used to retrieve the appropriate embeddings for a given query.
"""

from beartype.typing import Any, List, Union
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from .misc import cache_dir, get_splitter
from .prompts import multiquery_parser, prompts
from .typechecker import optional_typecheck


@optional_typecheck
def create_multiquery_retriever(
    llm: Union[ChatLiteLLM, ChatOpenAI],
    retriever: BaseRetriever,
) -> MultiQueryRetriever:
    # advanced mode using pydantic parsers
    llm_chain = prompts.multiquery | llm | multiquery_parser
    mqr = MultiQueryRetriever(
        retriever=retriever,
        llm_chain=llm_chain,
    )

    # TODO: fix the fallback: the llm_chain has to have a callback instead
    # # as pydantic parsing can be complicated for some model
    # # we keep the default multi query retriever as a fallback
    # default = MultiQueryRetriever.from_llm(
    #     retriever=retriever,
    #     llm=llm,
    # )
    # resilient = mqr.with_fallbacks(fallbacks=[default])
    # return resilient
    return mqr


@optional_typecheck
def create_parent_retriever(
    task: str,
    loaded_embeddings: Any,
    loaded_docs: List[Document],
    top_k: int,
    relevancy: float,
) -> BaseRetriever:
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
        },
    )
    parent.add_documents(loaded_docs)
    return parent
