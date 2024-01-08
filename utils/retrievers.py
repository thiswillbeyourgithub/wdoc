from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore

from .file_loader import get_splitter


def create_hyde_retriever(
        query,
        filetype,

        llm,
        top_k,
        relevancy,

        embeddings_engine,
        embeddings,
        ):
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
    Document type: [[filetype]]
    User question: {question}
    Answer:""".replace("[[filetype]]", filetype)
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
        base_embeddings=embeddings_engine,
        )
    db = FAISS.from_documents(
            documents=[Document(page_content="")],
            embedding=hyde_embeddings,
            )
    # get id of the dummy doc
    dummy = list(db.docstore._dict.keys())
    db.delete(dummy)
    db.merge_from(embeddings)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "distance_metric": "cos",
            "score_threshold": relevancy,
            }
        )
    return retriever


def create_parent_retriever(
        task,
        loaded_embeddings,
        loaded_docs,
        top_k,
        relevancy,
        ):
    "https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever"
    csp = get_splitter(task)
    psp = get_splitter(task)
    psp._chunk_size *= 4
    parent = ParentDocumentRetriever(
            vectorstore=loaded_embeddings,
            docstore=LocalFileStore(".cache/parent_retriever"),
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
