import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from wdoc.utils.misc import ModelName
from wdoc.utils.embeddings import load_embeddings_engine
from wdoc.utils.customs.binary_faiss_vectorstore import CompressedFAISS, BinaryFAISS

# Test constants - these should match the ones from the main test file
WDOC_TEST_OPENAI_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OPENAI_EMBED_MODEL", "text-embedding-3-small"
)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_compressed_faiss_functionality():
    """Test that CompressedFAISS works as well as native FAISS with compression."""
    from wdoc.utils.customs.binary_faiss_vectorstore import CompressedFAISS
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from wdoc.utils.misc import ModelName
    from wdoc.utils.embeddings import load_embeddings_engine
    import numpy as np

    # Create test documents
    test_docs = [
        Document(page_content="The cat sat on the mat.", metadata={"source": "test1"}),
        Document(
            page_content="Python is a programming language.",
            metadata={"source": "test2"},
        ),
        Document(
            page_content="Machine learning uses algorithms.",
            metadata={"source": "test3"},
        ),
        Document(
            page_content="Natural language processing analyzes text.",
            metadata={"source": "test4"},
        ),
        Document(
            page_content="Artificial intelligence mimics human behavior.",
            metadata={"source": "test5"},
        ),
    ]

    # Use OpenAI embeddings for a more reliable test
    openai_embedding = load_embeddings_engine(
        modelname=ModelName(f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    # Create temporary directories for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        regular_faiss_path = os.path.join(temp_dir, "regular_faiss")
        compressed_faiss_path = os.path.join(temp_dir, "compressed_faiss")

        # Create regular FAISS vectorstore
        regular_faiss = FAISS.from_documents(test_docs, openai_embedding)
        regular_faiss.save_local(regular_faiss_path)

        # Create compressed FAISS vectorstore
        compressed_faiss = CompressedFAISS.from_documents(test_docs, openai_embedding)
        compressed_faiss.save_local(compressed_faiss_path)

        # Load both vectorstores
        loaded_regular = FAISS.load_local(
            regular_faiss_path, openai_embedding, allow_dangerous_deserialization=True
        )
        loaded_compressed = CompressedFAISS.load_local(
            compressed_faiss_path,
            openai_embedding,
            allow_dangerous_deserialization=True,
        )

        # Test that both have the same number of documents
        assert len(loaded_regular.index_to_docstore_id) == len(test_docs)
        assert len(loaded_compressed.index_to_docstore_id) == len(test_docs)
        assert len(loaded_regular.index_to_docstore_id) == len(
            loaded_compressed.index_to_docstore_id
        )

        # Test similarity search on both
        query = "programming and algorithms"

        regular_results = loaded_regular.similarity_search(query, k=3)
        compressed_results = loaded_compressed.similarity_search(query, k=3)

        # Both should return the same number of results
        assert len(regular_results) == len(compressed_results)
        assert len(regular_results) == 3

        # Results should contain the same documents (though order might vary slightly)
        regular_contents = {doc.page_content for doc in regular_results}
        compressed_contents = {doc.page_content for doc in compressed_results}
        assert regular_contents == compressed_contents

        # Test similarity search with scores
        regular_results_with_scores = loaded_regular.similarity_search_with_score(
            query, k=2
        )
        compressed_results_with_scores = loaded_compressed.similarity_search_with_score(
            query, k=2
        )

        assert len(regular_results_with_scores) == 2
        assert len(compressed_results_with_scores) == 2

        # Verify that scores are reasonable (between 0 and some reasonable upper bound)
        for doc, score in regular_results_with_scores:
            assert isinstance(score, np.float32)
            assert score >= 0

        for doc, score in compressed_results_with_scores:
            assert isinstance(score, np.float32)
            assert score >= 0

        # Test that compressed files exist and are valid
        assert os.path.exists(os.path.join(compressed_faiss_path, "index.faiss"))
        assert os.path.exists(os.path.join(compressed_faiss_path, "index.pkl"))

        # Check that the compressed pickle file is potentially smaller due to compression
        # (This is hard to guarantee with small test data, but we can at least verify it loads)
        with open(os.path.join(compressed_faiss_path, "index.pkl"), "rb") as f:
            compressed_data = f.read()
        assert len(compressed_data) > 0  # Should have some content

        # Test that we can add documents to the loaded compressed FAISS
        new_doc = Document(
            page_content="New document about vectors.", metadata={"source": "test6"}
        )
        original_count = len(loaded_compressed.index_to_docstore_id)
        loaded_compressed.add_documents([new_doc])
        assert len(loaded_compressed.index_to_docstore_id) == original_count + 1

        # Test search still works after adding documents
        search_results = loaded_compressed.similarity_search("vectors", k=1)
        assert len(search_results) == 1


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_binary_faiss_functionality():
    """Test that BinaryFAISS preserves semantic relationships with binary embeddings."""
    from wdoc.utils.customs.binary_faiss_vectorstore import BinaryFAISS
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from wdoc.utils.misc import ModelName
    from wdoc.utils.embeddings import load_embeddings_engine
    import numpy as np

    # Create test words: 4 related programming words + 1 completely unrelated word
    related_words = ["python", "programming", "algorithm", "software"]
    outlier_word = "banana"
    all_words = related_words + [outlier_word]

    # Create test documents
    test_docs = [
        Document(page_content=word, metadata={"source": f"test_{word}"})
        for word in all_words
    ]

    # Use OpenAI embeddings for a more reliable test
    openai_embedding = load_embeddings_engine(
        modelname=ModelName(f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    def test_semantic_relationships(vectorstore, store_name):
        """Helper function to test semantic relationships in any vectorstore."""
        # Calculate all pairwise distances within the related group
        related_distances = []
        for i, word1 in enumerate(related_words):
            for j, word2 in enumerate(related_words):
                if i < j:  # Only compute each pair once
                    results1 = vectorstore.similarity_search_with_score(word1, k=5)
                    # Find the distance to word2
                    for doc, distance in results1:
                        if doc.page_content == word2:
                            related_distances.append(distance)
                            break

        # Calculate distances from outlier to each related word
        outlier_distances = []
        outlier_results = vectorstore.similarity_search_with_score(outlier_word, k=5)
        for doc, distance in outlier_results:
            if doc.page_content in related_words:
                outlier_distances.append(distance)

        # Verify we got the expected number of distances
        assert (
            len(related_distances) == 6
        ), f"Expected 6 related distances for {store_name}, got {len(related_distances)}"  # C(4,2) = 6 pairs
        assert (
            len(outlier_distances) == 4
        ), f"Expected 4 outlier distances for {store_name}, got {len(outlier_distances)}"  # 4 related words

        # The key test: minimum distance from outlier should be greater than maximum distance within related group
        max_related_distance = max(related_distances)
        min_outlier_distance = min(outlier_distances)

        assert min_outlier_distance > max_related_distance, (
            f"{store_name} failed to preserve semantic relationships: "
            f"minimum outlier distance ({min_outlier_distance}) should be greater than "
            f"maximum related distance ({max_related_distance}). "
            f"Related distances: {related_distances}, "
            f"Outlier distances: {outlier_distances}"
        )

        return (
            max_related_distance,
            min_outlier_distance,
            related_distances,
            outlier_distances,
        )

    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        regular_faiss_path = os.path.join(temp_dir, "regular_faiss")
        binary_faiss_path = os.path.join(temp_dir, "binary_faiss")

        # SANITY CHECK: First test with regular FAISS to ensure embeddings preserve semantic relationships
        regular_faiss = FAISS.from_documents(test_docs, openai_embedding)
        regular_faiss.save_local(regular_faiss_path)
        loaded_regular = FAISS.load_local(
            regular_faiss_path, openai_embedding, allow_dangerous_deserialization=True
        )

        # Test semantic relationships on regular FAISS
        (
            regular_max_related,
            regular_min_outlier,
            regular_related_distances,
            regular_outlier_distances,
        ) = test_semantic_relationships(loaded_regular, "Regular FAISS")

        # Now test with BinaryFAISS - it should preserve the same semantic relationships
        binary_faiss = BinaryFAISS.from_documents(test_docs, openai_embedding)
        binary_faiss.save_local(binary_faiss_path)

        # Load the vectorstore
        loaded_binary = BinaryFAISS.load_local(
            binary_faiss_path,
            openai_embedding,
            allow_dangerous_deserialization=True,
        )

        # Test that we have the correct number of documents
        assert len(loaded_binary.index_to_docstore_id) == len(test_docs)

        # Test semantic relationships on BinaryFAISS
        (
            binary_max_related,
            binary_min_outlier,
            binary_related_distances,
            binary_outlier_distances,
        ) = test_semantic_relationships(loaded_binary, "BinaryFAISS")

        # Test that we can still do similarity search properly
        search_results = loaded_binary.similarity_search("programming", k=3)
        assert len(search_results) == 3

        # The first result should be the exact match
        assert search_results[0].page_content == "programming"

        # Other results should be related programming terms, not the outlier
        result_contents = [doc.page_content for doc in search_results]
        assert (
            outlier_word not in result_contents[:3]
        ), f"Outlier '{outlier_word}' appeared in top 3 results for 'programming': {result_contents}"

        # also test for regular embeddings
        search_results = loaded_regular.similarity_search("programming", k=3)
        assert len(search_results) == 3
        assert search_results[0].page_content == "programming"
        result_contents = [doc.page_content for doc in search_results]
        assert (
            outlier_word not in result_contents[:3]
        ), f"Outlier '{outlier_word}' appeared in top 3 results for 'programming': {result_contents}"

        # Test similarity search with scores
        search_with_scores = loaded_binary.similarity_search_with_score(
            "algorithm", k=2
        )
        assert len(search_with_scores) == 2

        # Verify that scores are reasonable for binary embeddings (Hamming distances)
        for doc, score in search_with_scores:
            assert isinstance(score, (float, np.float32, np.int32))
            assert score >= 0, f"Distance should be non-negative, got {score}"
            # For Hamming distance, the maximum possible distance is the number of bits
            # which should be reasonable (not astronomically large)
            assert (
                score <= 10000
            ), f"Distance seems unreasonably large for Hamming distance: {score}"

        # EDGE CASE TESTS

        # Test 1: Edge case with k larger than available documents
        large_k_results = loaded_binary.similarity_search("python", k=10)
        assert len(large_k_results) == len(
            test_docs
        ), f"Expected {len(test_docs)} results when k > num_docs, got {len(large_k_results)}"

        # Test 2: Edge case with k=0 (should crash)
        with pytest.raises(AssertionError):
            zero_k_results = loaded_binary.similarity_search("python", k=0)

        # Test 3: Test with single document vectorstore
        single_doc = [Document(page_content="single", metadata={"source": "single"})]
        single_faiss = BinaryFAISS.from_documents(single_doc, openai_embedding)
        single_results = single_faiss.similarity_search("single", k=1)
        assert len(single_results) == 1
        assert single_results[0].page_content == "single"

        # Test 4: Test with empty query (should still work)
        with pytest.raises(AssertionError):
            empty_query_results = loaded_binary.similarity_search("", k=2)

        # Test 5: Test with duplicate documents
        duplicate_docs = [
            Document(page_content="duplicate", metadata={"source": "dup1"}),
            Document(page_content="duplicate", metadata={"source": "dup2"}),
            Document(page_content="unique", metadata={"source": "unique"}),
        ]
        dup_faiss = BinaryFAISS.from_documents(duplicate_docs, openai_embedding)
        dup_results = dup_faiss.similarity_search_with_score("duplicate", k=3)
        assert len(dup_results) == 3
        # The duplicate documents should have very similar (ideally identical) scores
        duplicate_scores = [
            score for doc, score in dup_results if doc.page_content == "duplicate"
        ]
        assert len(duplicate_scores) == 2, "Should find both duplicate documents"
        # Allow for small floating point differences
        assert (
            abs(duplicate_scores[0] - duplicate_scores[1]) < 1e-6
        ), f"Duplicate documents should have nearly identical scores: {duplicate_scores}"

        # Test 6: Test with very short and very long content
        extreme_docs = [
            Document(page_content="a", metadata={"source": "short"}),  # Very short
            Document(page_content="x" * 1000, metadata={"source": "long"}),  # Very long
            Document(
                page_content="medium length content here", metadata={"source": "medium"}
            ),
        ]
        extreme_faiss = BinaryFAISS.from_documents(extreme_docs, openai_embedding)
        short_results = extreme_faiss.similarity_search("a", k=1)
        long_results = extreme_faiss.similarity_search(
            "x" * 500, k=1
        )  # Query with long text
        assert len(short_results) == 1
        assert len(long_results) == 1

        # Test 7: Test score_threshold parameter
        threshold_results = loaded_binary.similarity_search_with_score(
            "programming", k=5, score_threshold=binary_max_related
        )
        # All results should have scores <= threshold
        for doc, score in threshold_results:
            assert (
                score <= binary_max_related
            ), f"Score {score} exceeds threshold {binary_max_related}"

        # Test 8: Test that distances are consistent (same query should give same results)
        results1 = loaded_binary.similarity_search_with_score("python", k=3)
        results2 = loaded_binary.similarity_search_with_score("python", k=3)
        assert len(results1) == len(results2)
        for (doc1, score1), (doc2, score2) in zip(results1, results2):
            assert doc1.page_content == doc2.page_content
            # For binary embeddings, allow for small numerical differences in integer scores
            assert (
                abs(score1 - score2) <= 1
            ), f"Scores should be nearly identical for same query: {score1} vs {score2}"

        # Test 9: Test that all returned documents are actually from our original set
        all_search_results = loaded_binary.similarity_search(
            "test query", k=len(test_docs)
        )
        returned_contents = {doc.page_content for doc in all_search_results}
        original_contents = {doc.page_content for doc in test_docs}
        assert (
            returned_contents == original_contents
        ), "All returned documents should be from original set"

        # Test 10: Verify that binary and regular FAISS produce different distances
        # This confirms that binary conversion actually changes the distance calculations
        regular_python_results = loaded_regular.similarity_search_with_score(
            "python", k=5
        )
        binary_python_results = loaded_binary.similarity_search_with_score(
            "python", k=5
        )

        # Find the distance to "programming" in both vectorstores
        regular_python_to_programming_distance = None
        binary_python_to_programming_distance = None

        for doc, distance in regular_python_results:
            if doc.page_content == "programming":
                regular_python_to_programming_distance = distance
                break

        for doc, distance in binary_python_results:
            if doc.page_content == "programming":
                binary_python_to_programming_distance = distance
                break

        # Both distances should be found
        assert (
            regular_python_to_programming_distance is not None
        ), "Could not find 'programming' in regular FAISS results for 'python' query"
        assert (
            binary_python_to_programming_distance is not None
        ), "Could not find 'programming' in binary FAISS results for 'python' query"

        # The distances should be different, confirming binary conversion affects calculations
        assert (
            regular_python_to_programming_distance
            != binary_python_to_programming_distance
        ), (
            f"Regular FAISS and BinaryFAISS should produce different distances between 'python' and 'programming'. "
            f"Regular: {regular_python_to_programming_distance}, Binary: {binary_python_to_programming_distance}. "
            f"If they are the same, the binary conversion may not be working correctly."
        )

        # Test 11: Verify binary embedding properties
        # Get raw embeddings to check they're actually binary
        test_embeddings = loaded_binary._embed_documents(["test"])
        assert len(test_embeddings) == 1
        embedding = test_embeddings[0]
        # Should be uint8 values (0-255)
        assert all(
            isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding
        ), "Binary embeddings should be uint8 values"


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_binary_faiss_edge_cases_and_errors():
    """Test BinaryFAISS error conditions and edge cases."""
    from wdoc.utils.customs.binary_faiss_vectorstore import BinaryFAISS
    from langchain_core.documents import Document
    from wdoc.utils.misc import ModelName
    from wdoc.utils.embeddings import load_embeddings_engine
    import numpy as np

    # Use OpenAI embeddings for testing
    openai_embedding = load_embeddings_engine(
        modelname=ModelName(f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    # Test 1: Error when trying to use normalize_L2=True
    with pytest.raises(
        ValueError, match="L2 normalization is not supported for binary embeddings.*"
    ):
        BinaryFAISS.from_documents(
            [Document(page_content="test", metadata={})],
            openai_embedding,
            normalize_L2=True,
        )

    # Test 2: Error when trying to use unsupported distance strategy
    from langchain_community.vectorstores.utils import DistanceStrategy

    with pytest.raises(
        ValueError, match="Distance strategy .* is not supported for binary embeddings"
    ):
        BinaryFAISS.from_documents(
            [Document(page_content="test", metadata={})],
            openai_embedding,
            distance_strategy=DistanceStrategy.COSINE,
        )

    # Test 3: Test with documents that have no content (empty strings)
    empty_content_docs = [
        Document(page_content="", metadata={"source": "empty1"}),
        Document(
            page_content="   ", metadata={"source": "whitespace"}
        ),  # Just whitespace
        Document(page_content="actual content", metadata={"source": "content"}),
    ]

    # This should crash
    with pytest.raises(AssertionError):
        empty_faiss = BinaryFAISS.from_documents(empty_content_docs, openai_embedding)

    # Test 4: Test with special characters and unicode
    special_docs = [
        Document(page_content="café résumé naïve", metadata={"source": "unicode"}),
        Document(page_content="🚀🎉💻", metadata={"source": "emoji"}),
        Document(page_content="@#$%^&*()", metadata={"source": "symbols"}),
        Document(page_content="\n\tò\r", metadata={"source": "whitespace_chars"}),
    ]

    special_faiss = BinaryFAISS.from_documents(special_docs, openai_embedding)
    special_results = special_faiss.similarity_search("café", k=1)
    assert len(special_results) == 1

    # Test 5: Test with extremely repetitive content
    repetitive_docs = [
        Document(page_content="a" * 10, metadata={"source": "short_repeat"}),
        Document(page_content="b" * 100, metadata={"source": "medium_repeat"}),
        Document(page_content="c" * 1000, metadata={"source": "long_repeat"}),
    ]

    rep_faiss = BinaryFAISS.from_documents(repetitive_docs, openai_embedding)
    rep_results = rep_faiss.similarity_search("aaa", k=1)
    assert len(rep_results) == 1

    # Test 6: Test that we can handle documents with identical metadata
    identical_meta_docs = [
        Document(page_content="content1", metadata={"type": "test", "id": 1}),
        Document(
            page_content="content2", metadata={"type": "test", "id": 1}
        ),  # Same metadata
        Document(
            page_content="content3", metadata={"type": "test", "id": 1}
        ),  # Same metadata
    ]

    meta_faiss = BinaryFAISS.from_documents(identical_meta_docs, openai_embedding)
    meta_results = meta_faiss.similarity_search("content", k=3)
    assert len(meta_results) == 3

    # Test 7: Test maximum marginal relevance with edge cases
    mmr_docs = [
        Document(page_content=f"document {i}", metadata={"id": i}) for i in range(5)
    ]
    mmr_faiss = BinaryFAISS.from_documents(mmr_docs, openai_embedding)

    # Test MMR with k larger than fetch_k
    mmr_results = mmr_faiss.max_marginal_relevance_search("document", k=3, fetch_k=2)
    assert len(mmr_results) <= 2  # Should be limited by fetch_k

    # Test MMR with lambda_mult edge values
    mmr_results_0 = mmr_faiss.max_marginal_relevance_search(
        "document", k=2, lambda_mult=0.0
    )
    mmr_results_1 = mmr_faiss.max_marginal_relevance_search(
        "document", k=2, lambda_mult=1.0
    )
    assert len(mmr_results_0) == 2
    assert len(mmr_results_1) == 2

    # Test 8: Test with numeric strings and mixed content
    numeric_docs = [
        Document(page_content="123", metadata={"source": "numeric"}),
        Document(page_content="12.34", metadata={"source": "decimal"}),
        Document(page_content="word123", metadata={"source": "mixed"}),
    ]

    numeric_faiss = BinaryFAISS.from_documents(numeric_docs, openai_embedding)
    numeric_results = numeric_faiss.similarity_search("1 2 3", k=2)
    assert len(numeric_results) == 2
