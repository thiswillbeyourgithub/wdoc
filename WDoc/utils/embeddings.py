"""
* Class used to create the embeddings.
* Loads and store embeddings for each document.
"""

from typing import List, Union, Optional, Any, Tuple, Callable
# import math
import hashlib
import os
import random
import time
from pathlib import Path, PosixPath
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import wraps

import numpy as np
from pydantic import Extra
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.storage import LocalFileStore
from .customs.compressed_embeddings_cache import LocalFileStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
import litellm

from .misc import cache_dir, get_tkn_length
from .logger import whi, red
from .typechecker import optional_typecheck
from .flags import is_verbose
from .env import WDOC_EXPIRE_CACHE_DAYS, WDOC_MOD_FAISS_SCORE_FN

def status(message: str):
    if is_verbose:
        whi(f"STATUS: {message}")

(cache_dir / "faiss_embeddings").mkdir(exist_ok=True)

# Source: https://api.python.langchain.com/en/latest/_modules/langchain_community/embeddings/huggingface.html#HuggingFaceEmbeddings
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "


if WDOC_MOD_FAISS_SCORE_FN:
    def score_function(distance: float) -> float:
        """
        Scoring function for faiss to make sure it's positive.
        Related issue: https://github.com/langchain-ai/langchain/issues/17333

        In langchain the default value is the euclidean relevance score:
        return 1.0 - distance / math.sqrt(2)

        The output is a similarity score: it must be [0,1] such that
        0 is the most dissimilar, 1 is the most similar document.
        """
        # To disable it but simply check: uncomment this and add "import math"
        # assert distance >= 0, distance
        # return 1.0 - distance / math.sqrt(2)
        new = 1 - ((1 + distance) / 2)
        return new
else:
    score_function = None


@optional_typecheck
def load_embeddings(
    embed_model: str,
    embed_kwargs: dict,
    load_embeds_from: Optional[Union[str, PosixPath]],
    save_embeds_as: Union[str, PosixPath],
    loaded_docs: Any,
    dollar_limit: Union[int, float],
    private: bool,
    use_rolling: bool,
    cli_kwargs: dict,
) -> Tuple[FAISS, CacheBackedEmbeddings]:
    """loads embeddings for each document"""
    backend = embed_model.split("/", 1)[0]
    embed_model = embed_model.replace(backend + "/", "")
    embed_model_str = embed_model.replace("/", "_")
    if "embed_instruct" in cli_kwargs and cli_kwargs["embed_instruct"]:
        instruct = True
    else:
        instruct = False

    if is_verbose:
        whi(f"Selected embedding model '{embed_model}' of backend {backend}")
    if backend == "openai":
        assert not private, f"Set private but tried to use openai embeddings"
        assert "OPENAI_API_KEY" in os.environ and os.environ[
            "OPENAI_API_KEY"] and "REDACTED" not in os.environ["OPENAI_API_KEY"], "Missing OPENAI_API_KEY"

        embeddings = OpenAIEmbeddings(
            model=embed_model,
            # model="text-embedding-ada-002",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            **embed_kwargs,
        )

    elif backend == "huggingface":
        assert not private, f"Set private but tried to use huggingface embeddings, which might not be as private as using sentencetransformers"
        model_kwargs = {
            "device": "cpu",
            # "device": "cuda",
        }
        model_kwargs.update(embed_kwargs)
        if "google" in embed_model and "gemma" in embed_model.lower():
            assert "HUGGINGFACE_API_KEY" in os.environ and os.environ[
                "HUGGINGFACE_API_KEY"] and "REDACTED" not in os.environ["HUGGINGFACE_API_KEY"], "Missing HUGGINGFACE_API_KEY"
            hftkn = os.environ["HUGGINGFACE_API_KEY"]
            # your token to use the models
            model_kwargs['use_auth_token'] = hftkn
        if instruct:
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=embed_model,
                model_kwargs=model_kwargs,
                embed_instruction=DEFAULT_EMBED_INSTRUCTION,
                query_instruction=DEFAULT_QUERY_INSTRUCTION,
            )
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs=model_kwargs,
            )

        if "google" in embed_model and "gemma" in embed_model.lower():
            # please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`
            # or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[pad]'})
            embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token

    elif backend == "sentencetransformers":
        if private:
            red(f"Private is set and will use sentencetransformers backend")
        if use_rolling:
            embed_kwargs.update(
                {
                    "batch_size": 1,
                    "pooling": "meanpool",
                    "device": None,
                }
            )
            embeddings = RollingWindowEmbeddings(
                model_name=embed_model,
                encode_kwargs=embed_kwargs,
            )
        else:
            embed_kwargs.update(
                {
                    "batch_size": 1,
                    "device": None,
                }
            )
            embeddings = SentenceTransformerEmbeddings(
                model_name=embed_model,
                encode_kwargs=embed_kwargs,
            )

    else:
        raise ValueError(f"Invalid embedding backend: {backend}")

    if "/" in embed_model:
        try:
            if Path(embed_model).exists():
                with open(Path(embed_model).resolve().absolute().__str__(), "rb") as f:
                    h = hashlib.sha256(
                        f.read() + str(instruct)
                    ).hexdigest()[:15]
                embed_model_str = Path(embed_model).name + "_" + h
        except Exception:
            pass
    assert "/" not in embed_model_str
    if private:
        embed_model_str = "private_" + embed_model_str

    lfs = LocalFileStore(
        root_path=cache_dir / "CacheEmbeddings" / embed_model_str,
        update_atime=True,
        compress=True
    )
    cache_content = list(lfs.yield_keys())
    whi(f"Found {len(cache_content)} embeddings in local cache")

    # cached_embeddings = embeddings
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        lfs,
        namespace=embed_model_str,
    )

    # reload passed embeddings
    if load_embeds_from:
        red("Reloading documents and embeddings from file")
        path = Path(load_embeds_from)
        assert path.exists(), f"file not found at '{path}'"
        db = FAISS.load_local(
            str(path),
            cached_embeddings,
            relevance_score_fn=score_function,
            allow_dangerous_deserialization=True)
        n_doc = len(db.index_to_docstore_id.keys())
        red(f"Loaded {n_doc} documents")
        return db, cached_embeddings

    whi("\nLoading embeddings.")

    db = None
    ti = time.time()
    docs = loaded_docs
    whi(f"Docs to embed: {len(docs)}")

    # check price of embedding
    full_tkn = sum([get_tkn_length(doc.page_content) for doc in docs])
    whi(
        f"Total number of tokens in documents: '{full_tkn}'")
    if private:
        whi("Not checking token price because private is set")
        price = 0
    elif backend != "openai":
        whi(
            f"Not checking token price because using a private backend: {backend}")
        price = 0
    elif f"{backend}/{embed_model}" in litellm.model_cost:
        price = litellm.model_cost[f"{backend}/{embed_model}"]["input_cost_per_token"]
        assert litellm.model_cost[f"{backend}/{embed_model}"]["output_cost_per_token"] == 0
    elif embed_model in litellm.model_cost:
        price = litellm.model_cost[embed_model]["input_cost_per_token"]
        assert litellm.model_cost[embed_model]["output_cost_per_token"] == 0
    else:
        raise Exception(
            red(f"Couldn't find the price of embedding model {embed_model}"))

    dol_price = full_tkn * price
    red(f"Total cost to embed all tokens is ${dol_price:.6f}")
    if dol_price > dollar_limit:
        ans = input("Do you confirm you are okay to pay this? (y/n)\n>")
        if ans.lower() not in ["y", "yes"]:
            red("Quitting.")
            raise SystemExit()

    # create a faiss index for batch of documents
    ts = time.time()
    batch_size = 1000
    batches = [
        [i * batch_size, (i + 1) * batch_size]
        for i in range(len(docs) // batch_size + 1)
    ]

    def embedand_one_batch(
        batch: List,
        ib: int,
    ):
        n_trial = 3
        for trial in range(n_trial):
            # whi(f"Embedding batch #{ib + 1}")
            try:
                temp = FAISS.from_documents(
                    docs[batch[0]:batch[1]],
                    cached_embeddings,
                    normalize_L2=True,
                    relevance_score_fn=score_function,
                )
                break
            except Exception as e:
                red(f"Thread #{ib + 1} Error at trial {trial+1}/{n_trial} when trying to embed documents: {e}")
                if trial + 1 >= n_trial:
                    red("Too many errors: bypassing the cache:")
                    temp = FAISS.from_documents(
                        docs[batch[0]:batch[1]],
                        cached_embeddings.underlying_embeddings,
                        normalize_L2=True,
                        relevance_score_fn=score_function,
                    )
                    break
                else:
                    time.sleep(1)
        return temp
    temp_dbs = Parallel(
        backend="threading",
        n_jobs=10,
        verbose=0 if not is_verbose else 51,
    )(
        delayed(embedand_one_batch)(
            batch=batch,
            ib=ib,
        )
        for ib, batch in tqdm(
            enumerate(batches),
            total=len(batches),
            desc="Embedding by batch",
            # disable=not is_verbose,
        )
    )
    for temp in temp_dbs:
        if not db:
            db = temp
        else:
            db.merge_from(temp)

    whi(f"Done creating index (total time: {time.time()-ti:.2f}s)")

    # saving embeddings
    db.save_local(save_embeds_as)

    return db, cached_embeddings

class RollingWindowEmbeddings(SentenceTransformerEmbeddings, extra=Extra.allow):
    @optional_typecheck
    def __init__(self, *args, **kwargs):
        assert "encode_kwargs" in kwargs
        if "normalize_embeddings" in kwargs["encode_kwargs"]:
            assert kwargs["encode_kwargs"]["normalize_embeddings"] is False, (
                "Not supposed to normalize embeddings using RollingWindowEmbeddings")
        assert kwargs["encode_kwargs"]["pooling"] in ["maxpool", "meanpool"]
        pooltech = kwargs["encode_kwargs"]["pooling"]
        del kwargs["encode_kwargs"]["pooling"]

        super().__init__(*args, **kwargs)
        self.__pool_technique = pooltech

    @optional_typecheck
    def embed_documents(self, texts, *args, **kwargs):
        """sbert silently crops any token above the max_seq_length,
        so we do a windowing embedding then pool (maxpool or meanpool)
        No normalization is done because the faiss index does it for us
        """
        model = self.client
        sentences = texts
        max_len = model.get_max_seq_length()

        if not isinstance(max_len, int):
            # the clip model has a different way to use the encoder
            # sources : https://github.com/UKPLab/sentence-transformers/issues/1269
            assert "clip" in str(model).lower(), (
                f"sbert model with no 'max_seq_length' attribute and not clip: '{model}'")
            max_len = 77
            encode = model._first_module().processor.tokenizer.encode
        else:
            if hasattr(model.tokenizer, "encode"):
                # most models
                encode = model.tokenizer.encode
            else:
                # word embeddings models like glove
                encode = model.tokenizer.tokenize

        assert isinstance(max_len, int), "n must be int"
        n23 = (max_len * 2) // 3
        add_sent = []  # additional sentences
        add_sent_idx = []  # indices to keep track of sub sentences

        for i, s in enumerate(sentences):
            # skip if the sentence is short
            length = len(encode(s))
            if length <= max_len:
                continue

            # otherwise, split the sentence at regular interval
            # then do the embedding of each
            # and finally pool those sub embeddings together
            sub_sentences = []
            words = s.split(" ")
            avg_tkn = length / len(words)
            # start at 90% of the supposed max_len
            j = int(max_len / avg_tkn * 0.8)
            while len(encode(" ".join(words))) > max_len:

                # if reached max length, use that minus one word
                until_j = len(encode(" ".join(words[:j])))
                if until_j >= max_len:
                    jjj = 1
                    while len(encode(" ".join(words[:j-jjj]))) >= max_len:
                        jjj += 1
                    sub_sentences.append(" ".join(words[:j-jjj]))

                    # remove first word until 1/3 of the max_token was removed
                    # this way we have a rolling window
                    jj = max(1, int((max_len // 3) / avg_tkn * 0.8))
                    while len(encode(" ".join(words[jj:j-jjj]))) > n23:
                        jj += 1
                    words = words[jj:]

                    j = int(max_len / avg_tkn * 0.8)
                else:
                    diff = abs(max_len - until_j)
                    if diff > 10:
                        j += max(1, int(10 / avg_tkn))
                    else:
                        j += 1

            sub_sentences.append(" ".join(words))

            sentences[i] = " "  # discard this sentence as we will keep only
            # the sub sentences pooled

            # remove empty text just in case
            if "" in sub_sentences:
                while "" in sub_sentences:
                    sub_sentences.remove("")
            assert sum([len(encode(ss)) > max_len for ss in sub_sentences]) == 0, (
                f"error when splitting long sentences: {sub_sentences}")
            add_sent.extend(sub_sentences)
            add_sent_idx.extend([i] * len(sub_sentences))

        if add_sent:
            sent_check = [
                len(encode(s)) > max_len
                for s in sentences
            ]
            addsent_check = [
                len(encode(s)) > max_len
                for s in add_sent
            ]
            assert sum(sent_check + addsent_check) == 0, (
                f"The rolling average failed apparently:\n{sent_check}\n{addsent_check}")

        vectors = super().embed_documents(sentences + add_sent)
        t = type(vectors)

        if isinstance(vectors, list):
            vectors = np.array(vectors)

        if add_sent:
            # at the position of the original sentence (not split)
            # add the vectors of the corresponding sub_sentence
            # then return only the 'pooled' section
            assert len(add_sent) == len(add_sent_idx), (
                "Invalid add_sent length")
            offset = len(sentences)
            for sid in list(set(add_sent_idx)):
                id_range = [i for i, j in enumerate(add_sent_idx) if j == sid]
                add_sent_vec = vectors[
                    offset + min(id_range): offset + max(id_range), :]
                if self.__pool_technique == "maxpool":
                    vectors[sid] = np.amax(add_sent_vec, axis=0)
                elif self.__pool_technique == "meanpool":
                    vectors[sid] = np.sum(add_sent_vec, axis=0)
                else:
                    raise ValueError(self.__pool_technique)
            vectors = vectors[:offset]

        if not isinstance(vectors, t):
            vectors = vectors.tolist()
        assert isinstance(vectors, t), "wrong type?"
        return vectors

