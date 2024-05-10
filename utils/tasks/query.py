import re
from typing import Tuple, List, Union
from langchain.docstore.document import Document
from langchain_core.runnables import chain

from utils.typechecker import optional_typecheck
from utils.errors import NoDocumentsRetrieved, NoDocumentsAfterWeakLLMFiltering
from utils.logger import red


@optional_typecheck
def format_chat_history(chat_history: List[Tuple]) -> str:
    "to load the chat history into the RAG chain"
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

@optional_typecheck
def check_intermediate_answer(ans: str) -> bool:
    "filters out the intermediate answers that are deemed irrelevant."
    if (
        ((not re.search(r"\bIRRELEVANT\b", ans)) and len(ans) < len("IRRELEVANT") * 2)
        or
        len(ans) >= len("IRRELEVANT") * 2
        ):
        return True
    return False


@chain
@optional_typecheck
def refilter_docs(inputs: dict) -> List[Document]:
    "filter documents find via RAG based on if the weak model answered 0 or 1"
    unfiltered_docs = inputs["unfiltered_docs"]
    evaluations = inputs["evaluations"]
    assert isinstance(unfiltered_docs, list), f"unfiltered_docs should be a list, not {type(unfiltered_docs)}"
    assert isinstance(evaluations, list), f"evaluations should be a list, not {type(evaluations)}"
    assert len(unfiltered_docs) == len(evaluations), f"len of unfiltered_docs is {len(unfiltered_docs)} but len of evaluations is {len(evaluations)}"
    if not unfiltered_docs:
        raise NoDocumentsRetrieved("No document corresponding to the query")
    filtered_docs = []
    for ie, evals in enumerate(evaluations):
        if not isinstance(evals, list):
            evals = [evals]
        if all(list(map(str.isdigit, evals))):
            evals = list(map(int, evals))
            if sum(evals) != 0:
                filtered_docs.append(unfiltered_docs[ie])
        else:
            red(f"Evals contained strings so keeping the doc: '{evals}'")
            filtered_docs.append(unfiltered_docs[ie])
    if not filtered_docs:
        raise NoDocumentsAfterWeakLLMFiltering(
            "No document remained after filtering with the query")
    return filtered_docs

