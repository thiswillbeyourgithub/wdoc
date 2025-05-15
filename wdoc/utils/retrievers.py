"""
Retrievers used to retrieve the appropriate embeddings for a given query.
"""

from beartype.typing import Any, List, Union
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.retrievers import BaseRetriever

from wdoc.utils.misc import cache_dir, get_splitter
from wdoc.utils.prompts import multiquery_parser, prompts
from wdoc.utils.typechecker import optional_typecheck
from wdoc.utils.customs.compressed_embeddings_cacher import LocalFileStore


@optional_typecheck
def create_multiquery_retriever(
    llm: Union[ChatLiteLLM],
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
    lfs = LocalFileStore(
        database_path=cache_dir / "parent_retriever",
        verbose=is_verbose,
        name="parent_retriever",
    )
    parent = ParentDocumentRetriever(
        vectorstore=loaded_embeddings,
        docstore=lfs,
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
