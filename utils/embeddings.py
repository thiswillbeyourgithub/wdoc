import random
import time
import copy
from pathlib import Path
from tqdm import tqdm
import threading

import numpy as np
from sklearn.preprocessing import Normalizer
from pydantic import Extra

from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings

from .logger import whi, red
from .file_loader import get_tkn_length


def load_embeddings(embed_model, loadfrom, saveas, debug, loaded_docs, dollar_limit, kwargs):
    """loads embeddings for each document"""

    if embed_model == "openai":
        red("Using openai embedding model")
        assert Path("API_KEY.txt").exists(), "No API_KEY.txt found"

        embeddings = OpenAIEmbeddings(
                openai_api_key=str(Path("API_KEY.txt").read_text()).strip()
                )

    else:
        embeddings = RollingWindowEmbeddings(
                model_name=embed_model,
                encode_kwargs={
                    "batch_size": 1,
                    "show_progress_bar": True,
                    "normalize_embeddings": True,
                    },
                )

    lfs = LocalFileStore(f".cache/embeddings/{embed_model}")
    cache_content = list(lfs.yield_keys())
    red(f"Found {len(cache_content)} embeddings in local cache")

    # cached_embeddings = embeddings
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            lfs,
            namespace=embed_model,
            )

    # reload passed embeddings
    if loadfrom:
        red("Reloading documents and embeddings from file")
        path = Path(loadfrom)
        assert path.exists(), f"file not found at '{path}'"
        db = FAISS.load_local(str(path), cached_embeddings)
        return db, cached_embeddings

    red("\nLoading embeddings.")

    docs = loaded_docs
    if len(docs) >= 50:
        docs = sorted(docs, key=lambda x: random.random())

    Path(".cache").mkdir(exist_ok=True)
    Path(".cache/faiss_embeddings").mkdir(exist_ok=True)
    embeddings_cache = Path(f".cache/faiss_embeddings/{embed_model}")
    embeddings_cache.mkdir(exist_ok=True)
    t = time.time()
    whi(f"Creating FAISS index for {len(docs)} documents")

    in_cache = [p for p in embeddings_cache.iterdir()]
    whi(f"Found {len(in_cache)} embeddings in cache")
    db = None
    to_embed = []

    # load previous faiss index from cache
    for doc in tqdm(docs, desc="Loading embeddings from cache"):
        fi = embeddings_cache / str(doc.metadata["hash"] + ".faiss_index")
        if fi.exists():
            temp = FAISS.load_local(fi, cached_embeddings)
            if not db and temp:
                db = temp
            else:
                try:
                    db.merge_from(temp)
                except Exception as err:
                    red(f"Error when loading cache from {fi}: {err}\nDeleting {fi}")
                    [p.unlink() for p in fi.iterdir()]
                    fi.rmdir()
        else:
            to_embed.append(doc)

    whi(f"Docs left to embed: {len(to_embed)}")

    # check price of embedding
    full_tkn = sum([get_tkn_length(doc.page_content) for doc in to_embed])
    red(f"Total number of tokens in documents (not checking if already present in cache): '{full_tkn}'")
    if embed_model == "openai":
        dol_price = full_tkn * 0.0001 / 1000
        red(f"With OpenAI embeddings, the total cost for all tokens is ${dol_price:.4f}")
        if dol_price > dollar_limit:
            ans = input("Do you confirm you are okay to pay this? (y/n)\n>")
            if ans.lower() not in ["y", "yes"]:
                red("Quitting.")
                raise SystemExit()

    # create a faiss index for batch of documents, then save them
    # as 1 document faiss index to cache
    if to_embed:
        batch_size = 1000
        batches = [
                [i * batch_size, (i + 1) * batch_size]
                for i in range(len(to_embed) // batch_size + 1)
                ]
        pbar = tqdm(total=len(to_embed), desc="Saving to cache")
        for batch in tqdm(batches, desc="Embedding by batch"):
            temp = FAISS.from_documents(
                    to_embed[batch[0]:batch[1]],
                    cached_embeddings,
                    normalize_L2=True
                    )

            recursive_faiss_saver(temp, to_embed[batch[0]:batch[1]], embeddings_cache, 0, pbar)

            if not db:
                db = temp
            else:
                db.merge_from(temp)

    # to get vectors from a faiss index
    # vecs = faiss.rev_swig_ptr(temp.index.get_xb(), len(to_embed) * temp.index.d).reshape(len(to_embed), temp.index.d)

    whi(f"Done creating index in {time.time()-t:.2f}s")

    # saving embeddings
    if saveas:
        db.save_local(saveas)

    return db, cached_embeddings

def recursive_faiss_saver(index, documents, path, depth, pbar):
    """split the faiss index by hand into 1 docstore index and save
    it to cache. To split it, as the copy.deepcopy is long we
    use a recursive call to only copy fewer times the full index"""
    doc_ids = [k for k in index.docstore._dict.keys()]
    assert doc_ids, "unexpected empty doc_ids"
    n = 10
    threads = []
    le = len(doc_ids)
    nn = len(doc_ids) // n
    if depth:
        spacer = " " * depth * 2
    else:
        spacer = ""
    info = f"(n={n}, nn={nn}, le={le}, d={depth})"
    if nn > n:  # more than 1 order of magnitude
        for i in range(len(doc_ids) // nn + 1):
            whi(f"{spacer}Creating larger subindex #{i} {info}")
            sub_index = copy.deepcopy(index)
            sub_docids = doc_ids[i * nn: (i + 1) * nn]
            to_del = [d for d in doc_ids if d not in sub_docids]
            if not to_del or not sub_docids:
                continue
            sub_index.delete(to_del)
            threads.extend(
                    recursive_faiss_saver(
                        sub_index, documents[i * nn:(i + 1) * nn], path, depth + 1, pbar)
                    )

    elif len(doc_ids) > n:
        for i in range(len(doc_ids) // n + 1):
            whi(f"{spacer}Creating subindex #{i} {info}")
            sub_index = copy.deepcopy(index)
            sub_docids = doc_ids[i * n: (i + 1) * n]
            to_del = [d for d in doc_ids if d not in sub_docids]
            if not to_del or not sub_docids:
                continue
            sub_index.delete(to_del)
            threads.extend(
                    recursive_faiss_saver(
                        sub_index, documents[i * n:(i + 1) * n], path, depth + 1, pbar)
                    )
            while sum([t.is_alive() for t in threads]) > 3 * n:
                time.sleep(0.1)
    else:
        for i, did in enumerate(doc_ids):
            whi(f"{spacer}Saving {documents[i].metadata['hash']}.faiss_index {info}")
            to_del = [d for d in doc_ids if d != did]
            if not to_del:
                continue
            file = (path / str(documents[i].metadata["hash"] + ".faiss_index"))
            assert not file.exists(), "cache file already exists!"
            thread = threading.Thread(
                    target=save_one_index,
                    args=(copy.deepcopy(index), to_del, file, pbar),
                    )
            thread.start()
            threads.append(thread)
        return threads
    [t.join() for t in threads]
    return []

def save_one_index(index, to_del, file, pbar):
    index.delete(to_del)
    index.save_local(file)
    pbar.update(1)

class RollingWindowEmbeddings(SentenceTransformerEmbeddings, extra=Extra.allow):
    def __init__(self, *args, **kwargs):
        if "encode_kwargs" not in kwargs:
            kwargs["encode_kwargs"] = {}
        if "normalize_embeddings" not in kwargs["encode_kwargs"]:
            kwargs["encode_kwargs"]["normalize_embeddings"] = False
        # kwargs["encode_kwargs"]["show_progress_bar"] = True

        super().__init__(*args, **kwargs)
        self.__do_normalize = kwargs["encode_kwargs"]["normalize_embeddings"]

    def embed_documents(self, texts, *args, **kwargs):
        """sbert silently crops any token above the max_seq_length,
        so we do a windowing embedding then maxpool then normalization.
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
            # and finally maxpool those sub embeddings together
            # the renormalization happens later in the code
            sub_sentences = []
            words = s.split(" ")
            avg_tkn = length / len(words)
            j = int(max_len / avg_tkn * 0.8)  # start at 90% of the supposed max_len
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
                    jj = int((max_len // 3) / avg_tkn * 0.8)
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
            # the sub sentences maxpooled

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
            # then return only the 'maxpooled' section
            assert len(add_sent) == len(add_sent_idx), (
                "Invalid add_sent length")
            offset = len(sentences)
            for sid in list(set(add_sent_idx)):
                id_range = [i for i, j in enumerate(add_sent_idx) if j == sid]
                add_sent_vec = vectors[
                        offset + min(id_range): offset + max(id_range), :]
                vectors[sid] = np.amax(add_sent_vec, axis=0)
            vectors = vectors[:offset]

        # normalize
        if self.__do_normalize:
            normalizer = Normalizer(norm="l2")
            vectors = normalizer.transform(vectors)

        if not isinstance(vectors, t):
            vectors = vectors.tolist()
        assert isinstance(vectors, t), "wrong type?"
        return vectors
