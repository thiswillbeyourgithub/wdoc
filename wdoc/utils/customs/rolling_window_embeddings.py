import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pydantic import Extra


class RollingWindowEmbeddings(SentenceTransformerEmbeddings, extra=Extra.allow):
    @optional_typecheck
    def __init__(self, *args, **kwargs):
        assert "encode_kwargs" in kwargs
        if "normalize_embeddings" in kwargs["encode_kwargs"]:
            assert (
                kwargs["encode_kwargs"]["normalize_embeddings"] is False
            ), "Not supposed to normalize embeddings using RollingWindowEmbeddings"
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
            assert (
                "clip" in str(model).lower()
            ), f"sbert model with no 'max_seq_length' attribute and not clip: '{model}'"
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
                    while len(encode(" ".join(words[: j - jjj]))) >= max_len:
                        jjj += 1
                    sub_sentences.append(" ".join(words[: j - jjj]))

                    # remove first word until 1/3 of the max_token was removed
                    # this way we have a rolling window
                    jj = max(1, int((max_len // 3) / avg_tkn * 0.8))
                    while len(encode(" ".join(words[jj: j - jjj]))) > n23:
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
            assert (
                sum([len(encode(ss)) > max_len for ss in sub_sentences]) == 0
            ), f"error when splitting long sentences: {sub_sentences}"
            add_sent.extend(sub_sentences)
            add_sent_idx.extend([i] * len(sub_sentences))

        if add_sent:
            sent_check = [len(encode(s)) > max_len for s in sentences]
            addsent_check = [len(encode(s)) > max_len for s in add_sent]
            assert (
                sum(sent_check + addsent_check) == 0
            ), f"The rolling average failed apparently:\n{sent_check}\n{addsent_check}"

        vectors = super().embed_documents(sentences + add_sent)
        t = type(vectors)

        if isinstance(vectors, list):
            vectors = np.array(vectors)

        if add_sent:
            # at the position of the original sentence (not split)
            # add the vectors of the corresponding sub_sentence
            # then return only the 'pooled' section
            assert len(add_sent) == len(add_sent_idx), "Invalid add_sent length"
            offset = len(sentences)
            for sid in list(set(add_sent_idx)):
                id_range = [i for i, j in enumerate(add_sent_idx) if j == sid]
                add_sent_vec = vectors[
                    offset + min(id_range): offset + max(id_range), :
                ]
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
