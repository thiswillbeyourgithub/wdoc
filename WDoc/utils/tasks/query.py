"""
Chain (logic) used to query a document.
"""

import re
from typing import Tuple, List, Union
from langchain.docstore.document import Document
from langchain_core.runnables import chain
from langchain_core.runnables.base import RunnableLambda
from tqdm import tqdm
import numpy as np

from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.chat_models import ChatLiteLLM
from langchain_openai import ChatOpenAI
import pandas as pd
import sklearn.metrics as metrics
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import scipy

from ..typechecker import optional_typecheck
from ..errors import NoDocumentsRetrieved, NoDocumentsAfterLLMEvalFiltering, InvalidDocEvaluationByLLMEval
from ..logger import red, whi
from ..misc import thinking_answer_parser
from ..flags import is_verbose

irrelevant_regex = re.compile(r"\bIRRELEVANT\b")


@optional_typecheck
def check_intermediate_answer(ans: str) -> bool:
    "filters out the intermediate answers that are deemed irrelevant."
    if "<answer>IRRELEVANT</answer>" in ans:
        return False
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
    "filter documents find via RAG based on the digit answered by the eval llm"
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
    for ie, evals in enumerate(evaluations):  # iterating over each document
        if not isinstance(evals, list):
            evals = [evals]
        answers = [thinking_answer_parser(ev)["answer"] for ev in evals]
        for ia, a in enumerate(answers):
            try:
                a = int(a)
            except Exception as err:
                red(f"Document was not evaluated with a number: '{err}' for answer '{a}'\nKeeping the document anyway.")
                a = 5
            answers[ia] = a

        if sum(answers) != 0:
            filtered_docs.append(unfiltered_docs[ie])

    if not filtered_docs:
        raise NoDocumentsAfterLLMEvalFiltering(
            "No document remained after filtering with the query")
    return filtered_docs


@optional_typecheck
def parse_eval_output(output: str) -> str:
    mess = f"The eval LLM returned an output that can't be parsed as expected: '{output}'"
    # empty
    if not output.strip():
        raise InvalidDocEvaluationByLLMEval(mess)

    parsed = thinking_answer_parser(output)

    if is_verbose:
        whi(f"Eval LLM output: '{output}'")

    answer = parsed["answer"]
    try:
        answer = int(answer)
        return str(answer)
    except Exception as err:
        red(f"Document was not evaluated with a number: '{err}' for answer '{answer}'\nKeeping the document anyway.")
        return str(5)

    if "-" in parsed["answer"]:
        raise InvalidDocEvaluationByLLMEval(mess)
    digits = [d for d in list(parsed["answer"]) if d.isdigit()]

    # contain no digits
    if not digits:
        raise InvalidDocEvaluationByLLMEval(mess)

    # good
    elif len(digits) == 1:
        if digits[0] == "0":
            return "0"
        elif digits[0] == "1":
            return "1"
        elif digits[0] == "2":
            return "1"
        else:
            raise InvalidDocEvaluationByLLMEval(mess)
    else:
        # ambiguous
        raise InvalidDocEvaluationByLLMEval(mess)



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
    Return the text directly if less than 5 texts.
    """
    assert texts, "No input text received"

    # deduplicate texts
    temp = []
    [temp.append(t) for t in texts if t not in temp]
    texts = temp

    if len(texts) < 5:
        return texts

    # get embeddings
    embeds = np.array([embedding_engine.embed_query(t) for t in texts]).squeeze()
    n_dim = embeds.shape[1]
    assert n_dim > 2, f"Unexpected number of dimension: {n_dim}, shape was {embeds.shape}"

    max_n_dim = min(100, len(texts))

    if n_dim > max_n_dim:
        scaler = preprocessing.StandardScaler()
        embed_scaled = scaler.fit_transform(embeds)
        pca = decomposition.PCA(n_components=max_n_dim)
        embeds_reduced = pca.fit_transform(embed_scaled)
        assert embeds_reduced.shape[0] == embeds.shape[0]
        vr = np.cumsum(pca.explained_variance_ratio_)[-1]
        if vr <= 0.95:
            red(f"Found lower than exepcted PCA explained variance ratio: {vr:.4f}")
        embeddings = pd.DataFrame(
            columns=[f"v_{i}" for i in range(embeds_reduced.shape[1])],
            index=[i for i in range(len(texts))],
            data=embeds_reduced,
        )
    else:
        embeddings = pd.DataFrame(
            columns=[f"v_{i}" for i in range(embeds.shape[1])],
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
        pbar.n = pbar.total
        pbar.close()

        return inputs
    return actual_pbar_closer
