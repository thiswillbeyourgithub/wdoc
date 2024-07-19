"""
* Class used to create the embeddings.
* Loads and store embeddings for each document.
"""

from typing import List, Union, Optional, Any, Tuple
import hashlib
import os
import queue
import faiss
import random
import time
from pathlib import Path, PosixPath
from tqdm import tqdm
import threading

import numpy as np
from pydantic import Extra
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings

from .misc import cache_dir, get_tkn_length
from .logger import whi, red
from .typechecker import optional_typecheck
from .flags import is_verbose

import lazy_import
litellm = lazy_import.lazy_module("litellm")


(cache_dir / "faiss_embeddings").mkdir(exist_ok=True)

# Source: https://api.python.langchain.com/en/latest/_modules/langchain_community/embeddings/huggingface.html#HuggingFaceEmbeddings
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "


class InstructLlamaCPPEmbeddings(LlamaCppEmbeddings, extra=Extra.allow):
    """wrapper around the class LlamaCppEmbeddings to add an instruction
    before the text to embed."""
    @optional_typecheck
    def __init__(self, *args, **kwargs):
        embed_instruction = DEFAULT_EMBED_INSTRUCTION
        query_instruction = DEFAULT_QUERY_INSTRUCTION
        if "embed_instruction" in kwargs:
            embed_instruction = kwargs["embed_instruction"]
            del kwargs["embed_instruction"]
        if "query_instruction" in kwargs:
            query_instruction = kwargs["query_instruction"]
            del kwargs["query_instruction"]

        super().__init__(*args, **kwargs)
        self.embed_instruction = embed_instruction
        self.query_instruction = query_instruction

    @optional_typecheck
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [self.embed_instruction + t for t in texts]
        embeddings = [self.client.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    @optional_typecheck
    def embed_query(self, text: str) -> List[float]:
        text = self.query_instruction + text
        embedding = self.client.embed(text)
        return list(map(float, embedding))


@optional_typecheck
def load_embeddings(
    embed_model: str,
    embed_kwargs: dict,
    load_embeds_from: Optional[str],
    save_embeds_as: str,
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
        assert not private, f"Set private but tried to use huggingface embeddings, which might not be as private as using sentencetransformers or llamacppembeddings"
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

    elif backend == "llamacppembeddings":
        if private:
            red(f"Private is set and will use llamacppembeddings backend")
        llamacppkwargs = {
            "f16_kv": False,
            "logits_all": False,
            "n_batch": 8,
            "n_ctx": 8192,
            "n_gpu_layers": 0,
            "n_parts": -1,
            "n_threads": 4,
            "seed": 42,
            "use_mlock": False,
            "verbose": False,
            "vocab_only": False,
        }
        llamacppkwargs.update(embed_kwargs)
        assert Path(embed_model).exists(), f"File not found {embed_model}"

        assert "model_path" not in llamacppkwargs, "llamacppembeddings model_path must be supplied via --embed_model arg"

        red(
            f"Loading llamacppembeddings at path {embed_model} with arguments {llamacppkwargs}")
        # method overloading to make it an instruct model
        if instruct:
            embeddings = InstructLlamaCPPEmbeddings(
                model_path=embed_model,
                **llamacppkwargs,
            )
        else:
            embeddings = LlamaCppEmbeddings(
                model_path=embed_model,
                **llamacppkwargs,
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

    lfs = LocalFileStore(cache_dir / "embeddings" / embed_model_str)
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
        db = FAISS.load_local(str(path), cached_embeddings,
                              allow_dangerous_deserialization=True)
        n_doc = len(db.index_to_docstore_id.keys())
        red(f"Loaded {n_doc} documents")
        return db, cached_embeddings

    whi("\nLoading embeddings.")

    docs = loaded_docs
    if len(docs) >= 50:
        docs = sorted(docs, key=lambda x: random.random())

    embeddings_cache = cache_dir / "faiss_embeddings" / embed_model_str
    embeddings_cache.mkdir(exist_ok=True)
    ti = time.time()
    whi(f"Creating FAISS index for {len(docs)} documents")

    in_cache = [p for p in embeddings_cache.iterdir()]
    whi(f"Found {len(in_cache)} embeddings in cache")
    to_embed = []

    # load previous faiss index from cache
    n_loader = 10
    loader_queues = [(queue.Queue(maxsize=10), queue.Queue())
                     for i in range(n_loader)]
    loader_workers = [
        threading.Thread(
            target=faiss_loader,
            args=(cached_embeddings, qin, qout),
            daemon=False,
        ) for qin, qout in loader_queues]
    [t.start() for t in loader_workers]
    timeout = 10
    list_of_files = {f.stem for f in embeddings_cache.iterdir()
                     if "faiss_index" in f.suffix}
    for doc in tqdm(docs, desc="Loading embeddings from cache"):
        if doc.metadata["content_hash"] in list_of_files:
            fi = embeddings_cache / \
                str(doc.metadata["content_hash"] + ".faiss_index")
            assert fi.exists()
            # select 2 workers at random and choose the one with the smallest queue
            queue_candidates = random.sample(loader_queues, k=2)
            queue_sizes = [q[0].qsize() for q in queue_candidates]
            lq = queue_candidates[queue_sizes.index(min(queue_sizes))][0]
            lq.put((fi, doc.metadata))
        else:
            to_embed.append(doc)

    # ask workers to stop and return their db then get the merged dbs
    [q[0].put((False, None)) for q in loader_queues]
    merged_dbs = [q[1].get(timeout=timeout) for q in loader_queues]
    merged_dbs = [m for m in merged_dbs if m is not None]
    assert all(q[1].get(timeout=timeout) == "Stopped" for q in loader_queues)
    whi("Asking loader workers to shutdown")
    [t.join(timeout=timeout) for t in loader_workers]
    assert all([not t.is_alive() for t in loader_workers]
               ), "Faiss loader workers failed to stop"

    # merge dbs as one
    db = None
    if merged_dbs:
        assert db is None
        db = merged_dbs.pop(0)
    if merged_dbs:
        [db.merge_from(m) for m in merged_dbs]
        in_db = len(db.docstore._dict.keys())
        assert in_db == len(docs) - len(to_embed), (
            f"Invalid number of loaded documents: found {in_db} but "
            f"expected {len(docs)-len(to_embed)}"
        )

    whi(f"Docs left to embed: {len(to_embed)}")

    # check price of embedding
    full_tkn = sum([get_tkn_length(doc.page_content) for doc in to_embed])
    whi(
        f"Total number of tokens in documents (not checking if already present in cache): '{full_tkn}'")
    if private:
        whi(f"Not checking token price because private is set")
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
    if to_embed:
        batch_size = 1000
        batches = [
            [i * batch_size, (i + 1) * batch_size]
            for i in range(len(to_embed) // batch_size + 1)
        ]
        n_saver = 10
        saver_queues = [(queue.Queue(maxsize=10), queue.Queue())
                        for i in range(n_saver)]
        saver_workers = [
            threading.Thread(
                target=faiss_saver,
                args=(embeddings_cache, cached_embeddings, qin, qout),
                daemon=False,
            ) for qin, qout in saver_queues]
        [t.start() for t in saver_workers]
        assert all([t.is_alive() for t in saver_workers]
                   ), "Saver workers failed to load"

        for ib, batch in tqdm(enumerate(batches), total=len(batches), desc="Embedding by batch"):
            whi(f"Embedding batch #{ib + 1}")
            temp = FAISS.from_documents(
                to_embed[batch[0]:batch[1]],
                cached_embeddings,
                normalize_L2=True
            )

            whi(f"Saving batch #{ib + 1}")
            # save the faiss index as 1 embedding for 1 document
            # get the id of each document
            doc_ids = list(temp.docstore._dict.keys())
            # get the embedding of each document
            vecs = faiss.rev_swig_ptr(temp.index.get_xb(), len(
                doc_ids) * temp.index.d).reshape(len(doc_ids), temp.index.d)
            vecs = np.vsplit(vecs, vecs.shape[0])
            for docuid, embe in zip(temp.docstore._dict.keys(), vecs):
                docu = temp.docstore._dict[docuid]
                assert all([t.is_alive()
                           for t in saver_workers]), "Some saving thread died"

                # select 2 workers at random and choose the one with the smallest queue
                queue_candidates = random.sample(saver_queues, k=2)
                queue_sizes = [q[0].qsize() for q in queue_candidates]
                sq = queue_candidates[queue_sizes.index(min(queue_sizes))][0]
                sq.put((True, docuid, docu, embe.squeeze()))

            if not db:
                db = temp
            else:
                db.merge_from(temp)

        whi("Waiting for saver workers to finish.")
        stop_counter = 0
        while any(t.is_alive() for t in saver_workers):
            stop_counter += 1
            [q[0].put((False, None, None, None)) for i, q in enumerate(
                saver_queues) if saver_workers[i].is_alive()]
            exit_code = [q[1].get(timeout=timeout) for i, q in enumerate(
                saver_queues) if saver_workers[i].is_alive()]
            if not all(e.startswith("Stopped") for e in exit_code):
                whi(
                    f"Not all faiss worker stopped at tr #{stop_counter}: {exit_code}")
        [t.join(timeout=timeout) for t in saver_workers]
        assert all([not t.is_alive() for t in saver_workers]
                   ), "Faiss saver workers failed to stop"
    whi(f"Saving indexes took {time.time()-ts:.2f}s")

    whi(f"Done creating index (total time: {time.time()-ti:.2f}s)")

    # saving embeddings
    db.save_local(save_embeds_as)

    return db, cached_embeddings


@optional_typecheck
def faiss_loader(
        cached_embeddings: CacheBackedEmbeddings,
        qin: queue.Queue,
        qout: queue.Queue) -> None:
    """load a faiss index. Merge many other index to it. Then return the
    merged index. This makes it way fast to load a very large number of index
    """
    db = None
    while True:
        fi, metadata = qin.get()
        if fi is False:
            assert metadata is None
            qout.put(db)
            qout.put("Stopped")
            break
        assert metadata is not None
        temp = FAISS.load_local(fi, cached_embeddings,
                                allow_dangerous_deserialization=True)
        temp.docstore._dict[list(temp.docstore._dict.keys())[
            0]].metadata = metadata
        if not db:
            db = temp
        else:
            try:
                db.merge_from(temp)
            except Exception as err:
                red(f"Error when loading cache from {fi}: {err}\nDeleting {fi}")
                [p.unlink() for p in fi.iterdir()]
                fi.rmdir()
    return


@optional_typecheck
def faiss_saver(
        path: Union[str, PosixPath],
        cached_embeddings: CacheBackedEmbeddings,
        qin: queue.Queue,
        qout: queue.Queue) -> None:
    """create a faiss index containing only a single document then save it"""
    while True:
        message, docid, document, embedding = qin.get()
        if message is False:
            assert docid is None and document is None and embedding is None
            qout.put("Stopped")
            break

        file = (path / str(document.metadata["content_hash"] + ".faiss_index"))
        db = FAISS.from_embeddings(
            text_embeddings=[[document.page_content, embedding]],
            embedding=cached_embeddings,
            metadatas=[document.metadata],
            ids=[docid],
            normalize_L2=True)
        db.save_local(file)
    return


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
