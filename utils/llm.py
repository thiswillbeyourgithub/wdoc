import os
from pathlib import Path
from typing import Dict, Any
from pydantic import Extra

import numpy as np
from sklearn.preprocessing import Normalizer

import langchain
from langchain.llms import GPT4All, FakeListLLM, LlamaCpp
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.cache import SQLiteCache
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import SentenceTransformerEmbeddings

from .logger import whi, yel, red

langchain.llm_cache = SQLiteCache(database_path=".cache/langchain.db")

class AnswerConversationBufferMemory(ConversationBufferMemory):
    """
    quick fix from https://github.com/hwchase17/langchain/issues/5630
    """
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})


def load_llm(model, local_llm_path):
    """load language model"""
    if model.lower() == "openai":
        whi("Loading openai models")
        assert Path("API_KEY.txt").exists(), "No api key found"
        os.environ["OPENAI_API_KEY"] = str(Path("API_KEY.txt").read_text()).strip()

        llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                verbose=True,
                )
        callback = get_openai_callback

    elif model.lower() == "llama":
        whi("Loading llama models")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
                model_path=local_llm_path,
                callback_manager=callback_manager,
                verbose=True,
                n_threads=4,
                )
        callback = fakecallback

    elif model.lower() == "gpt4all":
        whi(f"loading gpt4all: '{local_llm_path}'")
        local_llm_path = Path(local_llm_path)
        assert local_llm_path.exists(), "local model not found"
        callbacks = [StreamingStdOutCallbackHandler()]
        # Verbose is required to pass to the callback manager
        llm = GPT4All(
                model=str(local_llm_path.absolute()),
                n_ctx=512,
                n_threads=4,
                callbacks=callbacks,
                verbose=True,
                streaming=True,
                )
        callback = fakecallback
    elif model.lower() in ["fake", "test", "testing"]:
        llm = FakeListLLM(verbose=True, responses=[f"Fake answer n°{i}" for i in range(1, 100)])
        callback = fakecallback
    else:
        raise ValueError(model)

    whi("done loading model.\n")
    return llm, callback


class fakecallback:
    """used by gpt4all to avoid bugs"""
    total_tokens = 0
    total_cost = 0
    args = None
    kwds = None
    func = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __str__(self):
        pass


class RollingWindowEmbeddings(SentenceTransformerEmbeddings, extra=Extra.allow):
    def __init__(self, *args, **kwargs):
        if "encode_kwargs" not in kwargs:
            kwargs["encode_kwargs"] = {}
        if "normalize_embeddings" not in kwargs["encode_kwargs"]:
            kwargs["encode_kwargs"]["normalize_embeddings"] = False

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
                        j += int(10 / avg_tkn)
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
