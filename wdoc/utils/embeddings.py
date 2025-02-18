"""
* Class used to create the embeddings.
* Loads and store embeddings for each document.
"""

# import math
import hashlib
import os
import random
import time
from functools import wraps
from pathlib import Path

import litellm
import numpy as np
from beartype.typing import Any, Callable, List, Optional, Tuple, Union
from joblib import Parallel, delayed
from langchain_core.embeddings import Embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# from langchain.storage import LocalFileStore
from .customs.compressed_embeddings_cacher import LocalFileStore
from .env import WDOC_EXPIRE_CACHE_DAYS, WDOC_MOD_FAISS_SCORE_FN, WDOC_DEFAULT_EMBED_DIMENSION
from .flags import is_verbose
from .logger import red, whi
from .misc import cache_dir, get_tkn_length
from .typechecker import optional_typecheck


def status(message: str):
    if is_verbose:
        whi(f"STATUS: {message}")


(cache_dir / "faiss_embeddings").mkdir(exist_ok=True)

# Source: https://api.python.langchain.com/en/latest/_modules/langchain_community/embeddings/huggingface.html#HuggingFaceEmbeddings
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)


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
    load_embeds_from: Optional[Union[str, Path]],
    api_base: Optional[str],
    save_embeds_as: Union[str, Path],
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
        if private:
            assert api_base, "If private is set, api_base must be set too"
        else:
            assert (
                "OPENAI_API_KEY" in os.environ
                and os.environ["OPENAI_API_KEY"]
                and "REDACTED" not in os.environ["OPENAI_API_KEY"]
            ), "Missing OPENAI_API_KEY"

        embeddings = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            api_base=api_base,
            dimensions=WDOC_DEFAULT_EMBED_DIMENSION,  # defaults to None
            **embed_kwargs,
        )

    elif backend == "huggingface":
        assert (
            not private
        ), f"Set private but tried to use huggingface embeddings, which might not be as private as using sentencetransformers"
        model_kwargs = {
            "device": "cpu",
            # "device": "cuda",
        }
        model_kwargs.update(embed_kwargs)
        if "google" in embed_model and "gemma" in embed_model.lower():
            assert (
                "HUGGINGFACE_API_KEY" in os.environ
                and os.environ["HUGGINGFACE_API_KEY"]
                and "REDACTED" not in os.environ["HUGGINGFACE_API_KEY"]
            ), "Missing HUGGINGFACE_API_KEY"
            hftkn = os.environ["HUGGINGFACE_API_KEY"]
            # your token to use the models
            model_kwargs["use_auth_token"] = hftkn
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
            embeddings.client.tokenizer.pad_token = (
                embeddings.client.tokenizer.eos_token
            )

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

    try:
        test_embeddings(embeddings)
    except Exception as e:
        red(f"Error when testing embeddings, something is probably wrong with the backend. Error is '{e}'. Please open a github issue to help the developper")

    if "/" in embed_model:
        try:
            if Path(embed_model).exists():
                with open(Path(embed_model).resolve().absolute().__str__(), "rb") as f:
                    h = hashlib.sha256(f.read() + str(instruct)).hexdigest()[:15]
                embed_model_str = Path(embed_model).name + "_" + h
        except Exception:
            pass
    assert "/" not in embed_model_str
    if private:
        embed_model_str = "private_" + embed_model_str

    # lfs = LocalFileStore(
    #     root_path=cache_dir / "CacheEmbeddings" / embed_model_str,
    #     update_atime=True,
    # )
    lfs = LocalFileStore(
        database_path=cache_dir / "CacheEmbeddings" / embed_model_str,
        expiration_days=WDOC_EXPIRE_CACHE_DAYS,
        verbose=is_verbose,
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
            allow_dangerous_deserialization=True,
        )
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
    whi(f"Total number of tokens in documents: '{full_tkn}'")
    if private:
        whi("Not checking token price because private is set")
        price = 0
    elif backend != "openai":
        whi(f"Not checking token price because using a private backend: {backend}")
        price = 0
    elif f"{backend}/{embed_model}" in litellm.model_cost:
        price = litellm.model_cost[f"{backend}/{embed_model}"]["input_cost_per_token"]
        assert (
            litellm.model_cost[f"{backend}/{embed_model}"]["output_cost_per_token"] == 0
        )
    elif embed_model in litellm.model_cost:
        price = litellm.model_cost[embed_model]["input_cost_per_token"]
        assert litellm.model_cost[embed_model]["output_cost_per_token"] == 0
    else:
        raise Exception(
            red(f"Couldn't find the price of embedding model {embed_model}")
        )

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
                    batch,
                    cached_embeddings,
                    normalize_L2=True,
                    relevance_score_fn=score_function,
                )
                break
            except Exception as e:
                red(
                    f"Thread #{ib + 1} Error at trial {trial+1}/{n_trial} when trying to embed documents: {e}"
                )
                if trial + 1 >= n_trial:
                    red("Too many errors: bypassing the cache:")
                    temp = FAISS.from_documents(
                        batch,
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
            batch=docs[batch[0]: batch[1]],
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

def test_embeddings(embeddings_engine: Embeddings) -> None:
    "Simple testing of embeddings to know early if something seems wrong"
    vec1 = embeddings.embed_query("This is a test")
    vec2 = embeddings.embed_documents(["This is another test"])
    shape1 = np.array(vec1).shape
    shape2 = np.array(vec2[0]).shape
    assert shape1==shape2, f"Test vectors 1 has shape {shape1} but vector 2 has shape {shape2}"
    assert not (vec1 == vec2).all(), f"Test vectors 1 and 2 are identical despite different inputs"
    assert not ((vec1 == 0).all() or (vec2 == 0).all()), "Test vectors 1 or 2 or both is only zeroes"
