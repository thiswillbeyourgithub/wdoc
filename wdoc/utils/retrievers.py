"""
Retrievers used to retrieve the appropriate embeddings for a given query.
"""

from beartype.typing import Any, List, Optional
from langchain.docstore.document import Document
from loguru import logger

# from langchain.storage import LocalFileStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

from wdoc.utils.env import env
from wdoc.utils.misc import cache_dir, get_splitter
from wdoc.utils.prompts import multiquery_parser, prompts
from wdoc.utils.customs.compressed_embeddings_cacher import LocalFileStore


def create_multiquery_retriever(
    llm: "langchain_litellm.ChatLiteLLM",
    retriever: BaseRetriever,
) -> BaseRetriever:
    # advanced mode using pydantic parsers
    llm_chain = prompts.multiquery | llm | multiquery_parser
    from langchain.retrievers.multi_query import MultiQueryRetriever

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
        verbose=env.WDOC_VERBOSE,
        name="parent_retriever",
    )
    from langchain.retrievers import ParentDocumentRetriever

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


def get_all_texts(loaded_embeddings: Embeddings) -> List[str]:
    return [v.page_content for k, v in loaded_embeddings.docstore._dict.items()]


def create_retrievers(
    query_retrievers: str,
    loaded_embeddings,
    embedding_engine,
    llm,
    top_k: int,
    relevancy: float,
    task: str,
    loaded_docs: Optional[List[Document]],
) -> BaseRetriever:
    """Create and return list of retrievers based on query_retrievers setting."""
    retrievers = []
    all_texts = None
    if "multiquery" in query_retrievers.lower():
        retrievers.append(
            create_multiquery_retriever(
                llm=llm,
                retriever=loaded_embeddings.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": top_k,
                        "score_threshold": relevancy,
                    },
                ),
            )
        )

    if "knn" in query_retrievers.lower():
        if not all_texts:
            all_texts = get_all_texts(loaded_embeddings)
        from langchain_community.retrievers import KNNRetriever

        retrievers.append(
            KNNRetriever.from_texts(
                all_texts,
                embedding_engine,
                relevancy_threshold=relevancy,
                k=top_k,
            )
        )
    if "svm" in query_retrievers:
        if not all_texts:
            all_texts = get_all_texts(loaded_embeddings)
        from langchain_community.retrievers import SVMRetriever

        retrievers.append(
            SVMRetriever.from_texts(
                all_texts,
                embedding_engine,
                relevancy_threshold=relevancy,
                k=top_k,
            )
        )
    if "parent" in query_retrievers.lower():
        if not loaded_docs:
            logger.warning(
                "To use the 'parent' retriever, we have have loaded documents but we haven't. This might be because you are loading from an index directly instead of creating embeddings during this run. As an experimental workaround, we load the documents from the loaded embeddings."
            )
            loaded_docs = list(loaded_embeddings.docstore._dict.values())
        retrievers.append(
            create_parent_retriever(
                task=task,
                loaded_embeddings=loaded_embeddings,
                loaded_docs=loaded_docs,
                top_k=top_k,
                relevancy=relevancy,
            )
        )

    if "basic" in query_retrievers.lower():
        retrievers.append(
            loaded_embeddings.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": top_k,
                    "score_threshold": relevancy,
                },
            )
        )

    assert (
        retrievers
    ), "No retriever selected. Probably cause by a wrong cli_command or query_retrievers arg."

    if len(retrievers) == 1:
        retriever = retrievers[0]
    else:
        from langchain.retrievers.merger_retriever import MergerRetriever

        merge_retriever = MergerRetriever(retrievers=retrievers)

        # remove redundant results from the merged retrievers:
        from langchain_community.document_transformers import EmbeddingsRedundantFilter
        from langchain.retrievers.document_compressors import DocumentCompressorPipeline
        from langchain.retrievers import ContextualCompressionRetriever

        filtered = EmbeddingsRedundantFilter(
            embeddings=embedding_engine,
            similarity_threshold=0.999,
        )
        filter_pipeline = DocumentCompressorPipeline(transformers=[filtered])
        retriever = ContextualCompressionRetriever(
            base_compressor=filter_pipeline, base_retriever=merge_retriever
        )
    return retriever
