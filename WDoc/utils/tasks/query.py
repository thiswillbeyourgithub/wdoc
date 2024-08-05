"""
Chain (logic) used to query a document.
"""

import re
from typing import Tuple, List, Any, Union
from langchain.docstore.document import Document
from langchain_core.runnables import chain
from langchain_core.runnables.base import RunnableLambda
from joblib import Memory
from tqdm import tqdm
import numpy as np

from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.chat_models import ChatLiteLLM
from langchain_openai import ChatOpenAI

from ..typechecker import optional_typecheck
from ..errors import NoDocumentsRetrieved, NoDocumentsAfterLLMEvalFiltering, InvalidDocEvaluationByLLMEval
from ..logger import red
from ..misc import cache_dir

import lazy_import
pd = lazy_import.lazy_module('pandas')
metrics = lazy_import.lazy_module("sklearn.metrics")
PCA = lazy_import.lazy_class("sklearn.decomposition.PCA")
StandardScaler = lazy_import.lazy_class("sklearn.preprocessing.StandardScaler")
scipy = lazy_import.lazy_module("scipy")

(cache_dir / "query_eval_llm").mkdir(exist_ok=True)
query_eval_cache = Memory(cache_dir / "query_eval_llm", verbose=0)
irrelevant_regex = re.compile(r"\bIRRELEVANT\b")


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
        ((not irrelevant_regex.search(ans)) and len(ans) < len("IRRELEVANT") * 2)
        or
        len(ans) >= len("IRRELEVANT") * 2
    ):
        return True
    return False


@chain
@optional_typecheck
def refilter_docs(inputs: dict) -> List[Document]:
    "filter documents find via RAG based on if the eval llm answered 0 or 1"
    unfiltered_docs = inputs["unfiltered_docs"]
    evaluations = inputs["evaluations"]
    assert isinstance(
        unfiltered_docs, list), f"unfiltered_docs should be a list, not {type(unfiltered_docs)}"
    assert isinstance(
        evaluations, list), f"evaluations should be a list, not {type(evaluations)}"
    assert len(unfiltered_docs) == len(
        evaluations), f"len of unfiltered_docs is {len(unfiltered_docs)} but len of evaluations is {len(evaluations)}"
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
        raise NoDocumentsAfterLLMEvalFiltering(
            "No document remained after filtering with the query")
    return filtered_docs


@optional_typecheck
def parse_eval_output(output: str) -> str:
    mess = f"The eval LLM returned an output that can't be parsed as 0 or 1: '{output}'"
    # empty
    if not output.strip():
        raise InvalidDocEvaluationByLLMEval(mess)

    if "-" in output:
        raise InvalidDocEvaluationByLLMEval(mess)

    digits = [d for d in list(output) if d.isdigit()]

    # contain no digits
    if not digits:
        raise InvalidDocEvaluationByLLMEval(mess)

    # good
    elif len(digits) == 1:
        if digits[0] == "0":
            return "0"
        elif digits[0] == "1":
            return "1"
        else:
            raise InvalidDocEvaluationByLLMEval(mess)

    # ambiguous
    elif "0" in digits and "1" in digits:
        raise InvalidDocEvaluationByLLMEval(mess)
    elif "0" not in digits and "1" not in digits:
        raise InvalidDocEvaluationByLLMEval(mess)

    raise Exception(
        f"Unexpected output when parsing eval llm evaluation of a doc: '{mess}'")


@optional_typecheck
def collate_intermediate_answers(
    list_ia: List[str],
    embedding_engine: CacheBackedEmbeddings,
    ) -> str:
    """write the intermediate answers in a single string to be
    combined by the LLM"""
    # remove answers deemed irrelevant
    list_ia = [ia for ia in list_ia if check_intermediate_answer(ia)]

    try:
        list_ia = semantic_sorting(
            texts=list_ia,
            embedding_engine=embedding_engine,
        )
    except Exception as err:
        red(f"Failed to do semantic sorting of intermediate answers: {err}")

    out = "Intermediate answers:"
    for iia, ia in enumerate(list_ia):
        out += f"[{iia + 1}]:\n{ia}\n---\n"
    return out

@optional_typecheck
def semantic_sorting(
    texts: List[str],
    embedding_engine: CacheBackedEmbeddings,
    ) -> List[str]:
    """
    Given a list of text, embed them, do a hierarchical clutering then
    sort the list according to the leaf order. This probably helps the LLM
    to combine the intermediate answers into one.
    """
    assert texts, "No input text received"

    # deduplicate texts
    temp = []
    [temp.append(t) for t in texts if t not in temp]
    texts = temp

    if len(texts) < 3:
        return texts
    elif len(texts) > 1000:
        red(
            f"Found {len(texts)} (>1000), this can be too much for the "
            "clustering so returning the list of text as is")
        return texts

    # get embeddings
    embeds = np.array([embedding_engine.embed_query(t) for t in texts]).squeeze()
    n_dim = [d for d in embeds.shape if d != len(texts)][0]
    assert n_dim > 2, f"Unexpected number of dimension: {n_dim}, shape was {embeds.shape}"

    if n_dim > 100 and len(texts) > 10:
        scaler = StandardScaler()
        embed_scaled = scaler.fit_transform(embeds)
        pca = PCA(n_components=100)
        embeds_reduced = pca.fit_transform(embed_scaled)
        assert embeds_reduced.shape[0] == embeds.shape[0]
        vr = np.cumsum(pca.explained_variance_ratio_)
        if vr <= 0.95:
            red(f"Found lower than exepcted PCA explained variance ratio: {vr:.4f}")
        embeddings = pd.DataFrame(
            columns=[f"v_{i}" for i in range(n_dim)],
            index=[i for i in range(len(texts))],
            data=embeds_reduced,
        )
    else:
        embeddings = pd.DataFrame(
            columns=[f"v_{i}" for i in range(n_dim)],
            index=[i for i in range(len(texts))],
            data=embeds,
        )

    # get the pairwise distance matrix
    pairwise_distances = metrics.pairwise_distances
    dist = pd.DataFrame(
        columns=embeddings.index,
        index=embeddings.index,
        data=pairwise_distances(
            embeddings.values,
            n_jobs=-1,
            metric="euclidean",
            )
        )
    # make sure the intersection is 0 and not a very small float
    for ind in dist.index:
        dist.at[ind, ind] = 0
    # make sure it's symetric
    dist = dist.add(dist.T).div(2)

    # get the hierarchichal semantic sorting order
    dist = scipy.spatial.distance.squareform(dist.values)  # convert to condensed format
    Z = scipy.cluster.hierarchy.linkage(dist, method='ward', optimal_ordering=True)
    order = scipy.cluster.hierarchy.leaves_list(Z)
    out_texts = [texts[o] for o in order]
    assert len(set(out_texts)) == len(out_texts), "duplicates"
    assert len(out_texts) == len(texts), "extra out_texts"
    assert not any(o for o in out_texts if o not in texts)
    assert not any(t for t in texts if t not in out_texts)
    # whi(f"Done in {int(time.time()-start)}s")
    assert len(texts) == len(out_texts)

    return out_texts


@optional_typecheck
def pbar_chain(
    llm: Union[ChatLiteLLM, ChatOpenAI, FakeListChatModel],
    len_func: str,
    **tqdm_kwargs,
    ) -> RunnableLambda:
    "create a chain that just sets a tqdm progress bar"

    @chain
    def actual_pbar_chain(
        inputs: Union[dict, List],
        llm: Union[ChatLiteLLM, ChatOpenAI, FakeListChatModel] = llm,
        ) -> Union[dict, List]:

        llm.callbacks[0].pbar.append(
            tqdm(
                total=eval(len_func),
                **tqdm_kwargs,
            )
        )
        if not llm.callbacks[0].pbar[-1].total:
            red(f"Empty total for pbar: {llm.callbacks[0].pbar[-1]}")

        return inputs

    return actual_pbar_chain

@optional_typecheck
def pbar_closer(
    llm: Union[ChatLiteLLM, ChatOpenAI, FakeListChatModel],
    ) -> RunnableLambda:
    "close a pbar created by pbar_chain"

    @chain
    def actual_pbar_closer(
        inputs: Union[dict, List],
        llm: Union[ChatLiteLLM, ChatOpenAI, FakeListChatModel] = llm,
        ) -> Union[dict, List]:
        pbar = llm.callbacks[0].pbar[-1]
        pbar.update(pbar.total - pbar.n)
        pbar.close()

        return inputs
    return actual_pbar_closer
