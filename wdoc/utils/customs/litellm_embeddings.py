"""
Custom embeddings to use litellm. This allows using for example
"ollama/bge-m3" as a model name.
Source: https://python.langchain.com/docs/how_to/custom_embeddings/
"""

from typing import List, Optional

import litellm
from langchain_core.embeddings import Embeddings


class LiteLLMEmbeddings(Embeddings):
    """Litellm embedding model integration."""

    def __init__(
        self,
        model: str,
        dimensions: Optional[int],
        api_base: Optional[str],
        private: bool,
        **embed_kwargs,
    ):
        assert (
            "/" in model
        ), "model must contain a /, for example 'ollama/bge-m3' or 'openai/text-embedding-ada-002'"
        if private:
            if not api_base:
                assert any(
                    provider in model for provider in ["ollama", "huggingface"]
                ), "--private argument is set and api_base not overridden BUT the model does not contain ollama nor huggingface, this can be a mistake so crashing out of abundance of caution. If you think this is a bug please open an issue on github."
        self.model = model
        self.dimensions = dimensions
        self.private = private
        self.api_base = api_base
        self.embed_kwargs = embed_kwargs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # https://docs.litellm.ai/docs/embedding/supported_embedding
        vecs = litellm.embedding(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
            timeout=600,
            api_base=self.api_base,
            user="wdoc_embeddings",
            drop_params=True,
            # 'sentence-similarity', 'feature-extraction', 'rerank', 'embed', 'similarity'
            # input_type="feature-extraction",  # seems to crash for openai
            **self.embed_kwargs,
        )
        if hasattr(
            vecs, "data"
        ):  # must an EmbeddingsResponse format, for example ollama
            data = vecs.data
            if isinstance(data, list) and isinstance(data[0], dict):
                vecs = [v["embedding"] for v in data]
            elif isinstance(data, list) and hasattr(data[0], "embedding"):
                vecs = [v.embedding for v in data]
            else:
                raise Exception(
                    f"Failed to parsed output of litellm embedding for model '{self.model}'. String rendering is '{vecs}'. Please open a github issue to get that fixed."
                )
        return vecs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
