"""
Chain (logic) used for search tasks.
"""

from langchain_core.runnables import chain

from wdoc.utils.misc import log_and_time_fn


@log_and_time_fn
def retrieve_documents_for_search(retriever):
    """
    Create a retrieve documents chain for search tasks.

    Parameters
    ----------
    retriever : object
        The retriever object to use for document retrieval.

    Returns
    -------
    RunnableLambda
        A chain that retrieves documents using the provided retriever.
    """

    def _retrieve_documents(inputs):
        return {
            "unfiltered_docs": retriever.invoke(inputs["question_for_embedding"]),
            "question_to_answer": inputs["question_to_answer"],
        }

    _retrieve_documents = chain(_retrieve_documents)
    return _retrieve_documents
