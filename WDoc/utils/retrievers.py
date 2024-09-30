"""
Retrievers used to retrieve the appropriate embeddings for a given query.
"""

from typing import Any, List, Union
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatLiteLLM
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator


from .misc import cache_dir, get_splitter
from .typechecker import optional_typecheck
from .prompts import prompts


# https://python.langchain.com/docs/how_to/output_parser_structured/
class ExpandedQuery(BaseModel):
    thoughts: str = Field(description="Reasonning to expand the query")
    output_queries: List[str] = Field(description="List containing each output query")

    @model_validator(mode="before")
    @classmethod
    def nonempty_queries(cls, values: dict) -> dict:
        oq = values["output_queries"]
        if not isinstance(oq, list):
            raise ValueError("output_queries has to be a list")
        if not oq:
            raise ValueError("output_queries can't be empty")
        if not all(isinstance(q, str) for q in oq):
            raise ValueError("output_queries has to be a list of str")
        oq = [q.strip() for q in oq if q.strip()]
        if not oq:
            raise ValueError("output_queries can't be empty after removing empty strings")
        return values

parser = PydanticOutputParser(pydantic_object=ExpandedQuery)

@optional_typecheck
def create_multiquery_retriever(
    llm: Union[ChatLiteLLM, ChatOpenAI],
    retriever: BaseRetriever,
    ) -> MultiQueryRetriever:
    # advanced mode using pydantic parsers
    llm_chain = prompts.multiquery | llm | parser
    mqr = MultiQueryRetriever(
        retriever=retriever,
        llm_chain=llm_chain,
        parser_key="output_queries",
    )

    # as pydantic parsing can be complicated for some model
    # we keep the default multi query retriever as a fallback
    default = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )
    resilient = mqr.with_fallbacks(fallbacks=[default])

    return resilient


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
        }
    )
    parent.add_documents(loaded_docs)
    return parent
