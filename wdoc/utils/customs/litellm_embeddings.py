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
            tiemout=600,
            api_base=self.api_base,
            user="wdoc_embeddings",
            **self.embed_kwargs,
        )
        return vecs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
