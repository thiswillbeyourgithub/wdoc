"""
Chain (logic) used to query a document.
"""

import re
import time

import numpy as np
import pandas as pd
import scipy
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
from beartype.typing import List, Literal, Tuple, Union
from langchain.docstore.document import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.runnables import chain
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI
from numpy.typing import NDArray
from tqdm import tqdm

from ..env import WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE
from ..errors import (
    InvalidDocEvaluationByLLMEval,
    NoDocumentsAfterLLMEvalFiltering,
    NoDocumentsRetrieved,
)
from ..flags import is_verbose
from ..logger import red, whi
from ..misc import get_tkn_length, thinking_answer_parser
from ..typechecker import optional_typecheck

irrelevant_regex = re.compile(r"\bIRRELEVANT\b")


@optional_typecheck
def check_intermediate_answer(ans: str) -> bool:
    "filters out the intermediate answers that are deemed irrelevant."
    if "<answer>IRRELEVANT</answer>" in ans:
        return False
    if ((not irrelevant_regex.search(ans)) and len(ans) < len("IRRELEVANT") * 2) or len(
        ans
    ) >= len("IRRELEVANT") * 2:
        return True
    return False


@optional_typecheck
def sieve_documents(instance) -> RunnableLambda:
    """cap the number of retrieved documents as if multiple retrievers are used
    we can end up with a lot more document!
    """

    @chain
    @optional_typecheck
    def _sieve(inputs: dict) -> dict:
        assert "question_to_answer" in inputs, inputs.keys()
        assert "unfiltered_docs" in inputs, inputs.keys()
        # we have to pass an instance otherwise we can't know if the top_k got updated
        assert hasattr(instance, "top_k")
        assert hasattr(instance, "max_top_k")
        if instance.max_top_k:
            assert instance.max_top_k >= instance.top_k
        if len(inputs) > instance.top_k:
            red(
                "Number of documents found via embeddings was "
                f"'{inputs['unfiltered_docs']}' which is > top_k ({instance.top_k}) "
                "so we crop"
            )
        inputs["unfiltered_docs"] = inputs["unfiltered_docs"][: instance.top_k]
        return inputs

    return _sieve


@chain
@optional_typecheck
def refilter_docs(inputs: dict) -> List[Document]:
    "filter documents fond via RAG based on the digit answered by the eval llm"
    unfiltered_docs = inputs["unfiltered_docs"]
    evaluations = inputs["evaluations"]
    assert isinstance(
        unfiltered_docs, list
    ), f"unfiltered_docs should be a list, not {type(unfiltered_docs)}"
    assert isinstance(
        evaluations, list
    ), f"evaluations should be a list, not {type(evaluations)}"
    assert len(unfiltered_docs) == len(
        evaluations
    ), f"len of unfiltered_docs is {len(unfiltered_docs)} but len of evaluations is {len(evaluations)}"
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
                red(
                    f"Document was not evaluated with a number: '{err}' for answer '{a}'\nKeeping the document anyway."
                )
                a = 5
            answers[ia] = a

        if sum(answers) / len(answers) >= 3:
            filtered_docs.append(unfiltered_docs[ie])

    if not filtered_docs:
        raise NoDocumentsAfterLLMEvalFiltering(
            "No document remained after filtering with the query"
        )
    return filtered_docs


@optional_typecheck
def parse_eval_output(output: str) -> str:
    mess = (
        f"The eval LLM returned an output that can't be parsed as expected: '{output}'"
    )
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
        red(
            f"Document was not evaluated with a number: '{err}' for answer '{answer}'\nKeeping the document anyway."
        )
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
) -> str:
    """write the intermediate answers in a single string to be
    combined by the LLM"""
    # remove answers deemed irrelevant
    list_ia = [ia for ia in list_ia if check_intermediate_answer(ia)]
    assert (
        len(list_ia) >= 2
    ), f"Cannot collate a single intermediate answer!\n{list_ia[0]}"

    out = "Intermediate answers:"
    for iia, ia in enumerate(list_ia):
        ia = ia.replace("- • ", "- ").replace("• ", "- ")  # occasional bad md
        out += f"""
<ia source_id={iia + 1}>
{ia}
</ia>\n""".lstrip()
    return out


@optional_typecheck
def semantic_batching(
    texts: List[str],
    embedding_engine: CacheBackedEmbeddings,
) -> List[List[str]]:
    """
    Given a list of text, embed them, do a hierarchical clutering then
    sort the list according to the leaf order, then create buckets that best
    contain each subtopic while keeping a reasonnable number of tokens.
    This probably helps the LLM to combine the intermediate answers
    into one.
    Note that the documents are also sorted inside each batch, so that iterating
    over each document of each batch in order will follow the optimal leaf order.
    """
    max_token = WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE

    assert texts, "No input text received"
    assert len(texts) > 1, f"received only one text: {texts}"

    # deduplicate texts
    temp = []
    [temp.append(t) for t in texts if t not in temp]
    texts = temp

    if len(texts) <= 3:
        return [texts]

    text_sizes = {t: get_tkn_length(t) for t in texts}

    # get embeddings
    n_trial = 3
    for trial in range(n_trial):
        try:
            embeds = np.array(embedding_engine.embed_documents(texts)).squeeze()
            break
        except Exception as e:
            red(
                f"Error at trial {trial+1}/{n_trial} when trying to embed texts for semantic batching: '{e}'"
            )
            if trial + 1 >= n_trial:
                red("Too many errors so crashing")
                raise
            else:
                time.sleep(2)

    n_dim = embeds.shape[1]
    assert (
        n_dim > 2
    ), f"Unexpected number of dimension: {n_dim}, shape was {embeds.shape}"

    max_n_dim = min(100, len(texts))

    # optional dimension reduction to gain time
    try:
        if n_dim > max_n_dim:
            scaler = preprocessing.StandardScaler()
            embed_scaled = scaler.fit_transform(embeds)
            pca = decomposition.PCA(n_components=max_n_dim)
            embeds_reduced = pca.fit_transform(embed_scaled)
            assert embeds_reduced.shape[0] == embeds.shape[0]
            vr = np.cumsum(pca.explained_variance_ratio_)[-1]
            if vr <= 0.90:
                red(f"Found lower than exepcted PCA explained variance ratio: {vr:.4f}")
            assert (
                vr >= 0.75
            ), f"Found substancially low explained variance ratio afer pca at {vr:.4f} so not using dimension reduction"
            embeddings = pd.DataFrame(
                columns=[f"v_{i}" for i in range(embeds_reduced.shape[1])],
                index=[i for i in range(len(texts))],
                data=embeds_reduced,
            )
    except Exception as err:
        red(
            f"Error when doing dimension reduction for semantic batching. Original shape: {embeds.shape}. Error: '{err}'\nContinuing anyway."
        )

    if "embeddings" not in locals():
        embeddings = pd.DataFrame(
            columns=[f"v_{i}" for i in range(embeds.shape[1])],
            index=[i for i in range(len(texts))],
            data=embeds,
        )

    # get the pairwise distance matrix
    pairwise_distances = metrics.pairwise_distances
    pd_dist = pd.DataFrame(
        columns=embeddings.index,
        index=embeddings.index,
        data=pairwise_distances(
            embeddings.values,
            n_jobs=-1,
            metric="euclidean",
        ),
    )
    # make sure the intersection is 0 and not a very small float
    for ind in pd_dist.index:
        pd_dist.at[ind, ind] = 0
    # make sure it's symetric
    pd_dist = pd_dist.add(pd_dist.T).div(2)

    # get the hierarchichal semantic sorting order
    dist: NDArray[int] = scipy.spatial.distance.squareform(
        pd_dist.values
    )  # convert to condensed format
    Z: NDArray[Tuple[int, Literal[4]]] = scipy.cluster.hierarchy.linkage(
        dist, method="ward", optimal_ordering=True
    )

    order: NDArray[int] = scipy.cluster.hierarchy.leaves_list(Z)

    # TODO:; if <= 6 texts we should make 2 or 3 batch just using the order

    # # this would just return the list of strings in the best order
    # out_texts = [texts[o] for o in order]
    # assert len(set(out_texts)) == len(out_texts), "duplicates"
    # assert len(out_texts) == len(texts), "extra out_texts"
    # assert not any(o for o in out_texts if o not in texts)
    # assert not any(t for t in texts if t not in out_texts)
    # # whi(f"Done in {int(time.time()-start)}s")
    # assert len(texts) == len(out_texts)

    # get each bucket if we were only looking at the number of texts
    cluster_trials = {}
    cluster_mean_tkn = {}
    for divider in [3, 4, 5, 6]:
        cluster_labels = scipy.cluster.hierarchy.fcluster(
            Z, len(pd_dist.index) // divider, criterion="maxclust"
        )
        labels = np.unique(cluster_labels)
        labels.sort()
        if len(labels) == 1:  # re cluster if only one label found
            continue

        # use heuristics to find the best number of dividers by looking
        # at the average number of token in each clusters
        total_mean = 0
        for lab in labels:
            lt = [texts[int(ind)] for ind in np.argwhere(cluster_labels == lab)]
            lsize = sum([text_sizes[t] for t in lt])
            lmean = lsize / len(lt)
            total_mean += lmean
        total_mean /= len(labels)
        cluster_mean_tkn[divider] = total_mean
        cluster_trials[divider] = cluster_labels

    best_clusters = None
    for d, ct in cluster_mean_tkn.items():
        if ct < max_token and ct >= max_token / 2:
            best_clusters = cluster_trials[d]
            break
    if best_clusters is None:
        best_tkns = min(list(cluster_mean_tkn.values()))
        for d, ct in cluster_mean_tkn.items():
            if ct == best_tkns:
                best_clusters = cluster_trials[d]
                break
    assert best_clusters is not None
    cluster_labels = best_clusters

    labels = np.unique(cluster_labels)
    labels.sort()

    assert len(labels) > 1, cluster_labels

    # make sure no cluster contains only one text
    while not all((cluster_labels == lab).sum() > 1 for lab in labels):
        if is_verbose:
            whi("Remapping clusters.")
        for lab in labels:
            if (cluster_labels == lab).sum() == 1:
                t = texts[np.argmax(cluster_labels == lab)]
                # the closest is always itself so checking the 2nd closest
                t_close = (pd_dist.loc[texts.index(t), :]).nsmallest(2).index.tolist()
                assert texts.index(t) == t_close[0]
                t_closest = t_close[1]
                l_closest = cluster_labels[t_closest]
                if (cluster_labels == l_closest).sum() + 1 == len(texts):
                    # merging small to big would result in only one cluster:
                    # better to even them out
                    assert len(labels) == 2, labels
                    cluster_labels[t_closest] = lab
                    if is_verbose:
                        whi(f"Remapped one item from cluster {l_closest} to {lab}")
                else:  # good to go
                    cluster_labels[cluster_labels == lab] = l_closest
                    if is_verbose:
                        whi(f"Remapped single item of cluster {lab} to {l_closest}")
                break
        labels = np.unique(cluster_labels)
        labels.sort()
    assert len(labels) > 1, cluster_labels
    assert all((cluster_labels == lab).sum() > 1 for lab in labels), cluster_labels

    # Create buckets
    buckets = []
    current_bucket = []
    current_tokens = 0

    # fill each bucket until reaching max_token
    for lab in labels:
        lab_ind = np.argwhere(cluster_labels == lab)
        assert len(lab_ind) > 1, f"{lab_ind}\n{cluster_labels}"
        assert len(lab_ind) < len(texts), f"{lab_ind}\n{cluster_labels}"
        for clustid in lab_ind:
            text = texts[int(clustid)]
            size = text_sizes[text]
            if (current_tokens + size > max_token) and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [text]
                current_tokens = 0
            else:
                current_bucket.append(text)
                current_tokens += size

        assert current_bucket
        buckets.append(current_bucket)
        current_bucket = []
        current_tokens = 0
    assert all(bucket for bucket in buckets), "Empty buckets"

    # sort each bucket based on the optimal order
    for ib, b in enumerate(buckets):
        buckets[ib] = sorted(b, key=lambda t: order[texts.index(t)])

    # now if any bucket contains only one text, that means it has too many
    # tokens itself, so we reequilibrate from the previous buckets
    while not all(len(b) >= 2 for b in buckets):
        if is_verbose:
            whi(f"Merging sub buckets. Current len: {len(buckets)}")
        for ib, b in enumerate(buckets):
            assert b
            if len(b) == 1:
                # figure out which bucket to merge with
                if ib == 0:  # first , merge with next
                    next_id = ib + 1
                elif ib + 1 == len(buckets):  # last, take the penultimate
                    next_id = ib - 1
                elif ib != len(
                    buckets
                ):  # not first nor last, take the neighbour with least minimal distance
                    t_cur = b[0]
                    prev = min(
                        [
                            pd_dist.loc[texts.index(t_cur), texts.index(t)]
                            for t in buckets[ib - 1]
                        ]
                    )
                    next = min(
                        [
                            pd_dist.loc[texts.index(t_cur), texts.index(t)]
                            for t in buckets[ib + 1]
                        ]
                    )
                    assert prev > 0 and next > 0
                    if prev < next:
                        next_id = ib - 1
                    else:
                        next_id = ib + 1
                assert buckets[next_id], buckets[next_id]
                if is_verbose:
                    whi(f"Next_id is {next_id}")

                if len(buckets[next_id]) == 1:  # both texts are big, merge them anyway
                    if next_id > ib:
                        buckets[next_id].insert(0, b.pop())
                    else:
                        buckets[next_id].append(b.pop())
                    assert not b, b
                elif (
                    len(buckets[next_id]) == 2
                ):  # merging 2:1 -> 1:2 would create a loop
                    if next_id > ib:
                        buckets[next_id].insert(0, b.pop())
                    else:
                        buckets[next_id].append(b.pop())
                    assert not b, b
                else:
                    # send text to the next bucket, at the correct position
                    if next_id > ib:
                        b.append(buckets[next_id].pop(0))
                    else:
                        b.append(buckets[next_id].pop(-1))
                assert id(b) == id(buckets[ib])
                break
        buckets = [b for b in buckets if b]
    assert all(
        len(b) >= 2 for b in buckets
    ), f"Invalid size of buckets: '{[len(b) for b in buckets]}'"

    unchained = []
    [unchained.extend(b) for b in buckets]
    assert len(unchained) == len(
        set(unchained)
    ), "There were duplicate texts in buckets!"
    assert all(t in texts for t in unchained), "Some text of buckets were added!"

    return buckets


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
