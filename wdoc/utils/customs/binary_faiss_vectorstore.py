"""
Inherit from FAISS vectorstore from langchain but using binary embeddings from faiss.


Source:
https://python.langchain.com/api_reference/_modules/langchain_community/vectorstores/faiss.html#FAISS
https://github.com/facebookresearch/faiss/wiki/Binary-indexes
"""

from __future__ import annotations
from beartype import beartype

import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import (
    dependable_faiss_import,
    _len_check_if_sized,
)

logger = logging.getLogger(__name__)


class BinaryFAISS(FAISS):
    """FAISS vector store integration for binary embeddings.

    This subclass is specifically designed for binary embeddings that use
    Hamming distance for similarity calculations. Binary embeddings are
    represented as uint8 arrays where each bit represents a feature.

    Key differences from FAISS:
    - Uses binary FAISS indices (IndexBinaryFlat)
    - Only supports Hamming distance strategy
    - Does not support L2 normalization (incompatible with binary)
    - Uses Hamming distance for relevance scoring
    - Embeddings must be binary (uint8 arrays)

    Setup:
        Install ``langchain_community`` and ``faiss-cpu`` python packages.

        .. code-block:: bash

            pip install -qU langchain_community faiss-cpu

    Key init args — indexing params:
        embedding_function: Embeddings
            Embedding function that produces binary embeddings.

    Key init args — client params:
        index: Any
            Binary FAISS index to use (e.g., IndexBinaryFlat).
        docstore: Docstore
            Docstore to use.
        index_to_docstore_id: Dict[int, str]
            Mapping of index to docstore id.

    Instantiate:
        .. code-block:: python

            import faiss
            from langchain_community.vectorstores import BinaryFAISS
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_openai import OpenAIEmbeddings

            # Assuming binary embeddings of 128 bits (16 bytes)
            index = faiss.IndexBinaryFlat(128)

            vector_store = BinaryFAISS(
                embedding_function=binary_embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
    """

    def __init__(
        self,
        embedding_function: Union[
            Callable[[str], List[int]],  # Binary embeddings return List[int] or bytes
            Embeddings,
        ],
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        """Initialize BinaryFAISS with necessary components.

        Args:
            embedding_function: Function or Embeddings object that produces binary embeddings
            index: Binary FAISS index (must be a binary index type)
            docstore: Document storage
            index_to_docstore_id: Mapping from index to docstore IDs
            relevance_score_fn: Optional custom relevance scoring function
            normalize_L2: Must be False for binary embeddings
            distance_strategy: Must be compatible with binary embeddings

        Raises:
            ValueError: If incompatible options are specified for binary embeddings
        """
        # Validate binary-incompatible options
        if normalize_L2:
            raise ValueError(
                "L2 normalization is not compatible with binary embeddings. "
                "Set normalize_L2=False."
            )

        if distance_strategy not in [DistanceStrategy.EUCLIDEAN_DISTANCE]:
            # For binary embeddings, we interpret EUCLIDEAN_DISTANCE as Hamming distance
            # since that's the most natural distance metric for binary vectors
            raise ValueError(
                f"Distance strategy {distance_strategy} is not supported for binary embeddings. "
                f"Only EUCLIDEAN_DISTANCE (interpreted as Hamming distance) is supported."
            )

        # Validate that the index is a binary index
        if not hasattr(index, "add") or not str(type(index)).find("Binary") != -1:
            # This is a heuristic check - binary indices typically have 'Binary' in their class name
            logger.warning(
                "The provided index may not be a binary FAISS index. "
                "Binary embeddings require binary indices like IndexBinaryFlat."
            )

        super().__init__(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            relevance_score_fn=relevance_score_fn,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
        )
        self._original_embedding_function = embedding_function

    def embedding_function(self, texts):
        """Override to convert embeddings to binary"""
        # Get original embeddings
        embeddings = self._original_embedding_function(texts)

        # make sure we have a properly formatted array
        embeddings = np.array(embeddings).squeeze()

        return self._vec_to_binary(embeddings)

    async def aembedding_function(self, texts):
        """Override to convert embeddings to binary for async operations"""
        # Get original embeddings asynchronously
        if isinstance(self._original_embedding_function, Embeddings):
            embeddings = await self._original_embedding_function.aembed_documents(texts)
        else:
            raise Exception(
                "`embedding_function` is expected to be an Embeddings object for async operations"
            )

        # make sure we have a properly formatted array
        embeddings = np.array(embeddings).squeeze()

        return self._vec_to_binary(embeddings)

    @classmethod
    @beartype
    def _vec_to_binary(
        self, vectors: Union[np.ndarray, List[float], List[List[float]]]
    ) -> np.ndarray:
        """Convert vectors to binary format"""
        vectors = np.array(vectors)
        binary_vectors = vectors > 0
        if len(binary_vectors.shape) == 1:
            d = binary_vectors.shape[0]
        elif len(binary_vectors.shape) == 2:
            d = binary_vectors.shape[1]
        else:
            raise Exception(
                f"Unexpected dimension of embeddings to turn to binary: {binary_vectors.shape}"
            )

        # faiss only supports dimensions multiple of 8 so we add if necessary
        # source: https://github.com/facebookresearch/faiss/wiki/Binary-indexes
        if d % 8 != 0:
            padding = 8 - (d % 8)
            binary_vectors = np.pad(
                binary_vectors, ((0, 0), (0, padding)), mode="constant"
            )
        return np.packbits(binary_vectors, axis=1)

    def _embed_documents(self, texts: List[str]) -> List[List[int]]:
        """Embed documents and ensure they are in binary format."""
        return self.embedding_function(texts)

    async def _aembed_documents(self, texts: List[str]) -> List[List[int]]:
        """Embed documents asynchronously and ensure they are in binary format."""
        embeddings = await self.aembedding_function(texts)

        # Validate that embeddings are binary
        for i, embedding in enumerate(embeddings):
            if not all(
                isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding
            ):
                raise ValueError(
                    f"Embedding {i} contains non-binary values. "
                    f"Binary embeddings must contain only integers in range [0, 255] "
                    f"representing packed bits."
                )

        return embeddings

    def _embed_query(self, text: str) -> List[int]:
        """Embed query and ensure it is in binary format."""
        embedding = self.embedding_function([text])[0]

        # Validate that embedding is binary
        if not all(
            isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding
        ):
            raise ValueError(
                "Query embedding contains non-binary values. "
                "Binary embeddings must contain only integers in range [0, 255] "
                "representing packed bits."
            )

        return embedding

    async def _aembed_query(self, text: str) -> List[int]:
        """Embed query asynchronously and ensure it is in binary format."""
        embeddings = await self.aembedding_function([text])
        embedding = embeddings[0]

        # Validate that embedding is binary
        if not all(
            isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding
        ):
            raise ValueError(
                "Query embedding contains non-binary values. "
                "Binary embeddings must contain only integers in range [0, 255] "
                "representing packed bits."
            )

        return embedding

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[int]],
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add binary embeddings to the index."""
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )

        # Binary embeddings don't support L2 normalization
        if self._normalize_L2:
            raise ValueError("L2 normalization is not supported for binary embeddings.")

        _len_check_if_sized(texts, metadatas, "texts", "metadatas")

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        _len_check_if_sized(texts, ids, "texts", "ids")

        _metadatas = metadatas or ({} for _ in texts)
        documents = [
            Document(id=id_, page_content=t, metadata=m)
            for id_, t, m in zip(ids, texts, _metadatas)
        ]

        _len_check_if_sized(documents, embeddings, "documents", "embeddings")

        if ids and len(ids) != len(set(ids)):
            raise ValueError("Duplicate ids found in the ids list.")

        # Convert to binary format for FAISS
        vector = self._vec_to_binary(embeddings)

        self.index.add(vector)

        # Add information to docstore and index.
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        self.index_to_docstore_id.update(index_to_id)
        return ids

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[int],
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to binary embedding vector.

        Args:
            embedding: Binary embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            fetch_k: Number of Documents to fetch before filtering. Defaults to 20.
            **kwargs: Additional arguments including score_threshold.

        Returns:
            List of documents most similar to the query and Hamming distance
            in float for each. Lower score represents more similarity.
        """
        # Validate binary embedding
        if not all(
            isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding
        ):
            raise ValueError(
                "Query embedding contains non-binary values. "
                "Binary embeddings must contain only integers in range [0, 255]."
            )

        vector = self._vec_to_binary(embeddings)

        # Binary embeddings don't support L2 normalization
        if self._normalize_L2:
            raise ValueError("L2 normalization is not supported for binary embeddings.")

        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []

        if filter is not None:
            filter_func = self._create_filter_func(filter)

        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                if filter_func(doc.metadata):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            # For Hamming distance (binary), lower scores are better (more similar)
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if similarity <= score_threshold
            ]
        return docs[:k]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[int],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using maximal marginal relevance for binary embeddings.

        Note: MMR for binary embeddings uses Hamming distance calculations.
        """
        # Validate binary embedding
        if not all(
            isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding
        ):
            raise ValueError(
                "Query embedding contains non-binary values. "
                "Binary embeddings must contain only integers in range [0, 255]."
            )

        scores, indices = self.index.search(
            np.array([embedding], dtype=np.uint8),
            fetch_k if filter is None else fetch_k * 2,
        )

        if filter is not None:
            filter_func = self._create_filter_func(filter)
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if filter_func(doc.metadata):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])

        # For binary indices, we need to reconstruct as binary vectors
        embeddings = []
        for i in indices[0]:
            if i != -1:
                reconstructed = self.index.reconstruct(int(i))
                embeddings.append(reconstructed)

        if not embeddings:
            return []

        # Convert to float for MMR calculation, then back to binary for distance calculations
        embeddings_float = [embedding.astype(np.float32) for embedding in embeddings]
        query_embedding_float = np.array(embedding, dtype=np.float32)

        mmr_selected = maximal_marginal_relevance(
            np.array([query_embedding_float]),
            embeddings_float,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs_and_scores = []
        for i in mmr_selected:
            if indices[0][i] == -1:
                continue
            _id = self.index_to_docstore_id[indices[0][i]]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs_and_scores.append((doc, scores[0][i]))

        return docs_and_scores

    @classmethod
    def __from(
        cls,
        texts: Iterable[str],
        embeddings: List[List[int]],
        embedding: Embeddings,
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        **kwargs: Any,
    ) -> "BinaryFAISS":
        """Create BinaryFAISS from binary embeddings."""
        if normalize_L2:
            raise ValueError("L2 normalization is not supported for binary embeddings.")

        if distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE:
            raise ValueError(
                f"Distance strategy {distance_strategy} is not supported for binary embeddings. "
                f"Only EUCLIDEAN_DISTANCE (interpreted as Hamming distance) is supported."
            )

        faiss = dependable_faiss_import()

        # Create binary index - for binary embeddings, we use IndexBinaryFlat
        # The dimension should be in bits, so multiply by 8 if embeddings are in bytes
        embedding_dim_bits = len(embeddings[0]) * 8  # Assuming embeddings are bytes
        index = faiss.IndexBinaryFlat(embedding_dim_bits)

        docstore = kwargs.pop("docstore", InMemoryDocstore())
        index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})
        vecstore = cls(
            embedding,
            index,
            docstore,
            index_to_docstore_id,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )
        vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
        return vecstore

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "BinaryFAISS":
        """Construct BinaryFAISS from raw documents with binary embeddings."""
        embeddings = embedding.embed_documents(texts)

        binary_embeddings = BinaryFAISS._vec_to_binary(embeddings)

        return cls.__from(
            texts,
            binary_embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "BinaryFAISS":
        """Construct BinaryFAISS from raw documents with binary embeddings asynchronously."""
        embeddings = await embedding.aembed_documents(texts)

        binary_embeddings = BinaryFAISS._vec_to_binary(embeddings)

        return cls.__from(
            texts,
            binary_embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function for binary embeddings (Hamming distance)."""
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # For binary embeddings, we use Hamming distance
        # Lower Hamming distance means higher similarity
        return self._hamming_relevance_score_fn

    def _hamming_relevance_score_fn(self, distance: float) -> float:
        """Convert Hamming distance to a relevance score between 0 and 1.

        Args:
            distance: Hamming distance (number of differing bits)

        Returns:
            Relevance score where 1.0 is most relevant (distance=0) and
            0.0 is least relevant (maximum possible distance)
        """
        # Get the dimension in bits from the binary index
        max_distance = self.index.d

        # Normalize the distance: 0 distance = 1.0 relevance, max distance = 0.0 relevance
        # Clamp to ensure we don't get negative scores if distance > max_distance
        normalized_distance = min(distance / max_distance, 1.0)
        return 1.0 - normalized_distance
