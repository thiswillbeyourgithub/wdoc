import os
import sys
import subprocess
import tempfile
import hashlib
from pathlib import Path
from copy import copy

import pytest
import requests
from langchain_core.documents.base import Document

# add an unexpected env variable to make sure nothing crashes
os.environ["WDOC_TEST_UNEXPECTED_VARIABLE_1"] = "testing"

os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"

# test binary embeddings
os.environ["WDOC_MOD_FAISS_BINARY"] = "true"
os.environ["WDOC_MOD_FAISS_SCORE_FN"] = (
    "false"  # needs to be disabled for BINARY to work
)

from wdoc.wdoc import wdoc
from wdoc.utils.misc import ModelName
from wdoc.utils.embeddings import load_embeddings_engine
from wdoc.utils.embeddings import test_embeddings as _test_embeddings
from wdoc.utils.tasks.query import semantic_batching
from wdoc.utils.embeddings import load_embeddings_engine
from wdoc.utils.misc import ModelName, get_piped_input
from wdoc.utils.env import env

os.environ["WDOC_TYPECHECKING"] = "crash"

# Default model names if not specified in environment
# we are testing different providers just in case there are unexpected backend issues
WDOC_TEST_OPENAI_MODEL = os.getenv("WDOC_TEST_OPENAI_MODEL", "gpt-4o")
WDOC_TEST_OPENAI_EVAL_MODEL = os.getenv("WDOC_TEST_OPENAI_EVAL_MODEL", "gpt-4o-mini")
WDOC_TEST_OPENAI_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OPENAI_EMBED_MODEL", "text-embedding-3-small"
)

WDOC_TEST_OPENROUTER_MODEL = os.getenv(
    "WDOC_TEST_OPENROUTER_MODEL", "openrouter/openai/gpt-4o"
)
WDOC_TEST_OPENROUTER_EVAL_MODEL = os.getenv(
    "WDOC_TEST_OPENROUTER_EVAL_MODEL", "openrouter/openai/gpt-4o-mini"
)

WDOC_TEST_OLLAMA_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2:Q4_K_M"
)

# also make sure the default models work
WDOC_TEST_DEFAULT_MODEL = os.getenv("WDOC_TEST_DEFAULT_MODEL", env.WDOC_DEFAULT_MODEL)
WDOC_TEST_DEFAULT_EVAL_MODEL = os.getenv(
    "WDOC_TEST_DEFAULT_EVAL_MODEL", env.WDOC_DEFAULT_QUERY_EVAL_MODEL
)
WDOC_TEST_DEFAULT_EMBED_MODEL = os.getenv(
    "WDOC_TEST_DEFAULT_EMBED_MODEL", env.WDOC_DEFAULT_EMBED_MODEL
)

os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "true"

# the unexpected env var should be tested both before import and before run:
os.environ["WDOC_TEST_UNEXPECTED_VARIABLE_2"] = "testing"

ANALOGY_QUESTION = "What is the analogy used by the speaker"


@pytest.mark.basic
def test_wdoc_version():
    """Test that wdoc has a valid version string."""
    assert isinstance(wdoc.VERSION, str)
    assert len(wdoc.VERSION.split(".")) == 3


@pytest.mark.basic
def test_fail_parse_small_file_text(sample_text_file):
    """Test that a too small text file parsing fails."""
    # should fail because the file is too small
    with pytest.raises(Exception):
        wdoc.parse_doc(
            path=str(sample_text_file),
            filetype="txt",
        )


@pytest.mark.basic
def test_parse_doc_text(sample_text_file):
    """Test basic text file parsing."""
    # make a bigger text file
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain",
    )
    assert len(docs) > 0
    assert docs[0].page_content.startswith("This is a test document")
    assert "multiple lines" in docs[0].page_content


@pytest.mark.basic
def test_parse_doc_formats(sample_text_file):
    """Test text-only output from parse_doc."""
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain",
    )
    assert isinstance(docs, list), type(docs)
    assert len(docs) == 1, len(docs)
    assert all(isinstance(d, Document) for d in docs), ",".join(type(d) for d in docs)
    doc = docs[0]
    assert isinstance(doc, Document), type(doc)
    assert doc.page_content.startswith("This is a test document"), doc
    assert "multiple lines" in doc.page_content, doc.page_content

    ld = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain_dict",
    )
    assert isinstance(ld, list), type(ld)
    for ldd in ld:
        assert isinstance(ldd, dict), ldd
        assert "page_content" in ldd, ldd
        assert "metadata" in ldd, ldd

    text = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="text",
    )
    assert isinstance(text, str), type(text)

    xml = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="xml",
    )
    assert isinstance(xml, str), type(xml)

    assert xml != text


@pytest.mark.basic
def test_invalid_filetype():
    """Test that invalid filetype raises an error."""
    with pytest.raises(Exception):
        wdoc.parse_doc(
            path="dummy.txt",
            filetype="invalid_type",
        )


@pytest.mark.basic
def test_parse_online_pdf():
    """Test parsing an online PDF about situational awareness."""
    docs = wdoc.parse_doc(
        path="https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf",
        filetype="online_pdf",
        format="langchain",
    )
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    assert any("alphago" in d.page_content.lower() for d in docs)


@pytest.mark.basic
def test_help_output_shell():
    """Test that --help output contains expected docstring."""
    result = subprocess.run(
        ["wdoc", "--help"],
        capture_output=True,
        text=True,
        check=False,
        # stderr=subprocess.STDOUT,
    )
    output = result.stdout + result.stderr
    print(output)
    assert (
        "This docstring is dynamically updated with the content of wdoc/docs/help.md"
        not in output
    )
    assert "Content of wdoc/docs/help.md" in output


@pytest.mark.basic
def test_help_output_python():
    """Test that --help output contains expected docstring."""
    result = subprocess.run(
        ["python", "-m", "wdoc", "--help"], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert (
        "This docstring is dynamically updated with the content of wdoc/docs/help.md"
        not in output
    )
    assert "Content of wdoc/docs/help.md" in output


@pytest.mark.basic
def test_parse_doc_help_output_shell():
    """Test that --help output contains expected docstring."""
    result = subprocess.run(
        ["wdoc", "parse", "--help"], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert (
        "This docstring is dynamically updated with the content of wdoc/docs/parse_doc_help.md"
        not in output
    )
    assert "Content of wdoc/docs/parse_doc_help.md" in output


@pytest.mark.basic
def test_parse_doc_help_output_python():
    """Test that --help output contains expected docstring."""
    result = subprocess.run(
        ["python", "-m", "wdoc", "parse", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert (
        "This docstring is dynamically updated with the content of wdoc/docs/parse_doc_help.md"
        not in output
    )
    assert "Content of wdoc/docs/parse_doc_help.md" in output


@pytest.mark.basic
def test_parse_docx():
    """Test parsing a DOCX file."""
    # Create temporary file and download sample DOCX file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    url = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_500KB_DOCX.docx"

    # Download file using requests with proper headers to avoid 403 error
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Write content to temporary file
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    # Verify SHA512 checksum
    expected_hash = "64b73b409688cc5b5675c07d9df4b83d353fa85026a9d686d6725e50f388930e1d57c56cc6cfebd5f2cecc06d7ef89ae7495bd5411ca0eac4b0df63a7d6c82dc"
    with open(tmp_path, "rb") as f:
        file_hash = hashlib.sha512(f.read()).hexdigest()
    assert (
        file_hash == expected_hash
    ), f"File hash {file_hash} does not match expected {expected_hash}"

    try:
        # Parse the file
        docs = wdoc.parse_doc(
            path=tmp_path,
            # filetype="word",
            format="langchain",
        )

        # Verify output
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # Check that we got some actual content
        assert any(len(d.page_content.strip()) > 0 for d in docs)

    finally:
        # Cleanup
        os.unlink(tmp_path)


@pytest.mark.basic
def test_parse_nytimes():
    """Test parsing the NYTimes homepage."""
    docs = wdoc.parse_doc(
        path="https://www.nytimes.com/",
        filetype="auto",
        format="langchain",
    )
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    # Check that we got some actual content
    assert any(len(d.page_content.strip()) > 0 for d in docs)


@pytest.mark.basic
def test_parse_nytimes_shell():
    """Test parsing the NYTimes homepage via command line."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "wdoc",
            "parse",
            "https://www.nytimes.com/",
            "--format",
            "text",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout

    # Check for common news-related terms in the output
    assert "Times" in output, "Output should contain 'Times'"
    assert "news" in output.lower(), "Output should contain 'news'"
    assert (
        "journal" in output.lower() or "article" in output.lower()
    ), "Output should contain journalism-related terms"

    # Verify we got substantial content
    assert len(output) > 1000, "Expected significant text content from NYTimes"


@pytest.mark.basic
def test_error_message_shell_debug():
    """check if we get the error message we expect in debug mode"""
    command = "wdoc --task=summarize --path https://lemonde.fr/ --filetype=test --debug"
    expected_substring = "(Pdb) "
    try:
        subprocess.run(
            command,
            shell=True,  # Execute command through the shell
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode stdout/stderr as text (str)
            check=False,  # Raise CalledProcessError if command returns non-zero exit code
            timeout=20,
        )
        raise Exception("Should not reach this")

    except subprocess.TimeoutExpired as e:

        # Get the standard output
        stdout_output = str(e.stdout)

        # Assert that the expected substring is present in the standard output
        assert (
            expected_substring in stdout_output
        ), f"Expected '{expected_substring}' not found in command output:\n{stdout_output}"


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_semantic_batching():
    """Test that semantic_batching properly groups related texts."""
    texts = [
        "The cat chased the mouse around the house",
        "Python is a popular programming language",
        "JavaScript is used for web development",
        "The dog barked at the mailman yesterday",
    ]

    embeddings = load_embeddings_engine(
        modelname=ModelName(env.WDOC_DEFAULT_EMBED_MODEL),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    batches = semantic_batching(texts, embeddings)

    # Basic validation
    assert isinstance(batches, list)
    assert len(batches) >= 1
    assert all(isinstance(batch, list) for batch in batches)

    # Check all texts are present
    all_texts = []
    for batch in batches:
        all_texts.extend(batch)
    assert sorted(all_texts) == sorted(texts)

    # Check semantic grouping (programming languages should be together)
    for batch in batches:
        if "Python" in batch[0]:
            assert any("JavaScript" in text for text in batch)
        elif "JavaScript" in batch[0]:
            assert any("Python" in text for text in batch)

    assert texts != all_texts and texts != all_texts[::-1], all_texts


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban():
    """Test summarization of Tim Urban's procrastination video. Three times to make sure the caching and caching disabling works."""
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "false"
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "urban" in out["summary"].lower() or "procrastinat" in out["summary"].lower()
    assert out["doc_total_cost"] > 0

    inst2 = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        filetype="auto",
    )
    out2 = inst2.summary_task()
    assert "monkey" in out2["summary"].lower()
    assert (
        out2["doc_total_cost"] == 0
    ), "Normally we should be reusing the cache so cost should be 0"

    inst3 = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
    )
    out3 = inst3.summary_task()
    assert "monkey" in out3["summary"].lower()
    assert (
        out3["doc_total_cost"] > 0
    ), "Normally we disabled the cache so cost should be higher than 0"
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "true"


@pytest.mark.basic
def test_summary_tim_urban_testing_model():
    """Test summarization of Tim Urban's procrastination video with testing model."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model="testing",  # Use the special testing model
        filetype="auto",
    )
    out = inst.summary_task()
    # The 'testing' model should return a fixed string
    assert "Lorem ipsum dolor sit amet" in out["summary"]


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_debug():
    """Test summarization of Tim Urban's procrastination video."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
        debug=True,
    )
    out = inst.summary_task()
    assert "monkey" in out["summary"].lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_with_out_file():
    """Test that summary is properly written to output file."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        output_path = tmp.name

    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        filetype="auto",
        out_file=output_path,
    )
    assert inst.__import_mode__
    inst.summary_task()

    # Verify the output file
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()

    # Check for expected content in the summary
    assert len(content) > 0
    assert "arj7oStGLkU" in content
    assert (
        "Inside the mind of a master procrastinator" in content
        or "monkey" in content.lower()
    )
    assert "wdoc version" in content
    if os.path.exists(output_path):
        os.unlink(output_path)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban():
    """Test query task on Tim Urban's procrastination video. Three times to test the caching."""
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "false"
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()
    assert out["total_cost"] > 0

    inst2 = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out2 = inst2.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer2 = out2["final_answer"]
    assert "monkey" in final_answer2.lower()
    assert out2["total_cost"] == 0

    inst3 = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out3 = inst3.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer3 = out3["final_answer"]
    assert "monkey" in final_answer3.lower()
    assert out3["total_cost"] > 0
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "true"


@pytest.mark.basic
def test_query_tim_urban_testing_model():
    """Test query task on Tim Urban's procrastination video with testing model."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model="testing",  # Use the special testing model
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",  # Keep eval model for now, might need adjustment if testing model affects eval
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",  # Keep embed model
        disable_llm_cache=True,
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    # The 'testing' model should return a fixed string
    assert "Lorem ipsum dolor sit amet" in final_answer


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_whisper_tim_urban():
    """Test summarization of Tim Urban's video using whisper transcription."""
    out = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        youtube_audio_backend="whisper",
        whisper_lang="en",
    )


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_ollama_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName(f"ollama/{WDOC_TEST_OLLAMA_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    _test_embeddings(emb)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_openai_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName(f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    _test_embeddings(emb)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_mistral_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName("mistral/mistral-embed"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    _test_embeddings(emb)


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

    # Use mistral embeddings for a more realistic test
    mistral_embedding = load_embeddings_engine(
        modelname=ModelName("mistral/mistral-embed"),
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
        regular_faiss = FAISS.from_documents(test_docs, mistral_embedding)
        regular_faiss.save_local(regular_faiss_path)

        # Create compressed FAISS vectorstore
        compressed_faiss = CompressedFAISS.from_documents(test_docs, mistral_embedding)
        compressed_faiss.save_local(compressed_faiss_path)

        # Load both vectorstores
        loaded_regular = FAISS.load_local(
            regular_faiss_path, mistral_embedding, allow_dangerous_deserialization=True
        )
        loaded_compressed = CompressedFAISS.load_local(
            compressed_faiss_path,
            mistral_embedding,
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

    # Use mistral embeddings for a more realistic test
    mistral_embedding = load_embeddings_engine(
        modelname=ModelName("mistral/mistral-embed"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        binary_faiss_path = os.path.join(temp_dir, "binary_faiss")

        # Create binary FAISS vectorstore
        binary_faiss = BinaryFAISS.from_documents(test_docs, mistral_embedding)
        binary_faiss.save_local(binary_faiss_path)

        # Load the vectorstore
        loaded_binary = BinaryFAISS.load_local(
            binary_faiss_path,
            mistral_embedding,
            allow_dangerous_deserialization=True,
        )

        # Test that we have the correct number of documents
        assert len(loaded_binary.index_to_docstore_id) == len(test_docs)

        # Calculate all pairwise distances within the related group
        related_distances = []
        for i, word1 in enumerate(related_words):
            for j, word2 in enumerate(related_words):
                if i < j:  # Only compute each pair once
                    results1 = loaded_binary.similarity_search_with_score(word1, k=5)
                    # Find the distance to word2
                    for doc, distance in results1:
                        if doc.page_content == word2:
                            related_distances.append(distance)
                            break

        # Calculate distances from outlier to each related word
        outlier_distances = []
        outlier_results = loaded_binary.similarity_search_with_score(outlier_word, k=5)
        for doc, distance in outlier_results:
            if doc.page_content in related_words:
                outlier_distances.append(distance)

        # Verify we got the expected number of distances
        assert (
            len(related_distances) == 6
        ), f"Expected 6 related distances, got {len(related_distances)}"  # C(4,2) = 6 pairs
        assert (
            len(outlier_distances) == 4
        ), f"Expected 4 outlier distances, got {len(outlier_distances)}"  # 4 related words

        # The key test: minimum distance from outlier should be greater than maximum distance within related group
        max_related_distance = max(related_distances)
        min_outlier_distance = min(outlier_distances)

        assert min_outlier_distance > max_related_distance, (
            f"Binary embeddings failed to preserve semantic relationships: "
            f"minimum outlier distance ({min_outlier_distance}) should be greater than "
            f"maximum related distance ({max_related_distance}). "
            f"Related distances: {related_distances}, "
            f"Outlier distances: {outlier_distances}"
        )

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

        # Test similarity search with scores
        search_with_scores = loaded_binary.similarity_search_with_score(
            "algorithm", k=2
        )
        assert len(search_with_scores) == 2

        # Verify that scores are reasonable for binary embeddings (Hamming distances)
        for doc, score in search_with_scores:
            assert score == 0 or isinstance(score, (float, np.float32))
            assert score >= 0, f"Distance should be non-negative, got {score}"
            # For Hamming distance, the maximum possible distance is the number of bits
            # which should be reasonable (not astronomically large)
            assert (
                score <= 10000
            ), f"Distance seems unreasonably large for Hamming distance: {score}"


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

    # Use mistral embeddings for testing
    mistral_embedding = load_embeddings_engine(
        modelname=ModelName("mistral/mistral-embed"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    # Test 1: Error when trying to use normalize_L2=True
    with pytest.raises(ValueError, match="L2 normalization is not compatible with binary embeddings"):
        BinaryFAISS.from_documents(
            [Document(page_content="test", metadata={})],
            mistral_embedding,
            normalize_L2=True
        )

    # Test 2: Error when trying to use unsupported distance strategy
    from langchain_community.vectorstores.utils import DistanceStrategy
    with pytest.raises(ValueError, match="Distance strategy .* is not supported for binary embeddings"):
        BinaryFAISS.from_documents(
            [Document(page_content="test", metadata={})],
            mistral_embedding,
            distance_strategy=DistanceStrategy.COSINE
        )

    # Test 3: Test with documents that have no content (empty strings)
    empty_content_docs = [
        Document(page_content="", metadata={"source": "empty1"}),
        Document(page_content="   ", metadata={"source": "whitespace"}),  # Just whitespace
        Document(page_content="actual content", metadata={"source": "content"}),
    ]
    
    # This should work without crashing
    empty_faiss = BinaryFAISS.from_documents(empty_content_docs, mistral_embedding)
    empty_results = empty_faiss.similarity_search("test", k=2)
    assert len(empty_results) <= 3  # Should not crash

    # Test 4: Test with special characters and unicode
    special_docs = [
        Document(page_content="café résumé naïve", metadata={"source": "unicode"}),
        Document(page_content="🚀🎉💻", metadata={"source": "emoji"}),  
        Document(page_content="@#$%^&*()", metadata={"source": "symbols"}),
        Document(page_content="\n\t\r", metadata={"source": "whitespace_chars"}),
    ]
    
    special_faiss = BinaryFAISS.from_documents(special_docs, mistral_embedding)
    special_results = special_faiss.similarity_search("café", k=1)
    assert len(special_results) == 1

    # Test 5: Test with extremely repetitive content
    repetitive_docs = [
        Document(page_content="a" * 10, metadata={"source": "short_repeat"}),
        Document(page_content="b" * 100, metadata={"source": "medium_repeat"}),
        Document(page_content="c" * 1000, metadata={"source": "long_repeat"}),
    ]
    
    rep_faiss = BinaryFAISS.from_documents(repetitive_docs, mistral_embedding)
    rep_results = rep_faiss.similarity_search("aaa", k=1)
    assert len(rep_results) == 1

    # Test 6: Test that we can handle documents with identical metadata
    identical_meta_docs = [
        Document(page_content="content1", metadata={"type": "test", "id": 1}),
        Document(page_content="content2", metadata={"type": "test", "id": 1}),  # Same metadata
        Document(page_content="content3", metadata={"type": "test", "id": 1}),  # Same metadata
    ]
    
    meta_faiss = BinaryFAISS.from_documents(identical_meta_docs, mistral_embedding)
    meta_results = meta_faiss.similarity_search("content", k=3)
    assert len(meta_results) == 3

    # Test 7: Test maximum marginal relevance with edge cases
    mmr_docs = [Document(page_content=f"document {i}", metadata={"id": i}) for i in range(5)]
    mmr_faiss = BinaryFAISS.from_documents(mmr_docs, mistral_embedding)
    
    # Test MMR with k larger than fetch_k
    mmr_results = mmr_faiss.max_marginal_relevance_search("document", k=3, fetch_k=2)
    assert len(mmr_results) <= 2  # Should be limited by fetch_k
    
    # Test MMR with lambda_mult edge values
    mmr_results_0 = mmr_faiss.max_marginal_relevance_search("document", k=2, lambda_mult=0.0)
    mmr_results_1 = mmr_faiss.max_marginal_relevance_search("document", k=2, lambda_mult=1.0)
    assert len(mmr_results_0) == 2
    assert len(mmr_results_1) == 2

    # Test 8: Test with numeric strings and mixed content
    numeric_docs = [
        Document(page_content="123", metadata={"source": "numeric"}),
        Document(page_content="12.34", metadata={"source": "decimal"}),
        Document(page_content="word123", metadata={"source": "mixed"}),
        Document(page_content="", metadata={"source": "empty"}),
    ]
    
    numeric_faiss = BinaryFAISS.from_documents(numeric_docs, mistral_embedding)
    numeric_results = numeric_faiss.similarity_search("123", k=2)
    assert len(numeric_results) == 2


        # EDGE CASE TESTS

        # Test 1: Edge case with k larger than available documents
        large_k_results = loaded_binary.similarity_search("python", k=10)
        assert len(large_k_results) == len(test_docs), f"Expected {len(test_docs)} results when k > num_docs, got {len(large_k_results)}"

        # Test 2: Edge case with k=0 (should return empty list)
        zero_k_results = loaded_binary.similarity_search("python", k=0)
        assert len(zero_k_results) == 0, f"Expected 0 results when k=0, got {len(zero_k_results)}"

        # Test 3: Test with single document vectorstore
        single_doc = [Document(page_content="single", metadata={"source": "single"})]
        single_faiss = BinaryFAISS.from_documents(single_doc, mistral_embedding)
        single_results = single_faiss.similarity_search("single", k=1)
        assert len(single_results) == 1
        assert single_results[0].page_content == "single"

        # Test 4: Test with empty query (should still work)
        empty_query_results = loaded_binary.similarity_search("", k=2)
        assert len(empty_query_results) == 2, "Empty query should still return results"

        # Test 5: Test with duplicate documents
        duplicate_docs = [
            Document(page_content="duplicate", metadata={"source": "dup1"}),
            Document(page_content="duplicate", metadata={"source": "dup2"}),
            Document(page_content="unique", metadata={"source": "unique"}),
        ]
        dup_faiss = BinaryFAISS.from_documents(duplicate_docs, mistral_embedding)
        dup_results = dup_faiss.similarity_search_with_score("duplicate", k=3)
        assert len(dup_results) == 3
        # The duplicate documents should have very similar (ideally identical) scores
        duplicate_scores = [score for doc, score in dup_results if doc.page_content == "duplicate"]
        assert len(duplicate_scores) == 2, "Should find both duplicate documents"
        # Allow for small floating point differences
        assert abs(duplicate_scores[0] - duplicate_scores[1]) < 1e-6, f"Duplicate documents should have nearly identical scores: {duplicate_scores}"

        # Test 6: Test with very short and very long content
        extreme_docs = [
            Document(page_content="a", metadata={"source": "short"}),  # Very short
            Document(page_content="x" * 1000, metadata={"source": "long"}),  # Very long
            Document(page_content="medium length content here", metadata={"source": "medium"}),
        ]
        extreme_faiss = BinaryFAISS.from_documents(extreme_docs, mistral_embedding)
        short_results = extreme_faiss.similarity_search("a", k=1)
        long_results = extreme_faiss.similarity_search("x" * 500, k=1)  # Query with long text
        assert len(short_results) == 1
        assert len(long_results) == 1

        # Test 7: Test score_threshold parameter
        threshold_results = loaded_binary.similarity_search_with_score(
            "programming", k=5, score_threshold=max_related_distance
        )
        # All results should have scores <= threshold
        for doc, score in threshold_results:
            assert score <= max_related_distance, f"Score {score} exceeds threshold {max_related_distance}"

        # Test 8: Test that distances are consistent (same query should give same results)
        results1 = loaded_binary.similarity_search_with_score("python", k=3)
        results2 = loaded_binary.similarity_search_with_score("python", k=3)
        assert len(results1) == len(results2)
        for (doc1, score1), (doc2, score2) in zip(results1, results2):
            assert doc1.page_content == doc2.page_content
            assert abs(score1 - score2) < 1e-10, f"Scores should be identical for same query: {score1} vs {score2}"

        # Test 9: Test that all returned documents are actually from our original set
        all_search_results = loaded_binary.similarity_search("test query", k=len(test_docs))
        returned_contents = {doc.page_content for doc in all_search_results}
        original_contents = {doc.page_content for doc in test_docs}
        assert returned_contents == original_contents, "All returned documents should be from original set"

        # Test 10: Verify binary embedding properties
        # Get raw embeddings to check they're actually binary
        test_embeddings = loaded_binary._embed_documents(["test"])
        assert len(test_embeddings) == 1
        embedding = test_embeddings[0]
        # Should be uint8 values (0-255)
        assert all(isinstance(x, (int, np.integer)) and 0 <= x <= 255 for x in embedding), "Binary embeddings should be uint8 values"


@pytest.mark.basic
def test_get_piped_input_detection():
    """Test that get_piped_input correctly detects piped text and bytes."""
    # Test text piping
    input_text = "This is test text.\nWith multiple lines."
    cmd_text = [
        sys.executable,
        "-c",
        "import sys; from wdoc.utils.misc import get_piped_input; data = get_piped_input(); sys.stdout.write(data)",
    ]
    result_text = subprocess.run(
        cmd_text, input=input_text, text=True, capture_output=True, check=True
    )
    assert result_text.stdout == input_text

    # Test binary piping
    input_bytes = b"\x01\x02\x03\xff\xfe\x00binary data"
    cmd_bytes = [
        sys.executable,
        "-c",
        "import sys; from wdoc.utils.misc import get_piped_input; data = get_piped_input(); sys.stdout.buffer.write(data)",
    ]
    result_bytes = subprocess.run(
        cmd_bytes, input=input_bytes, capture_output=True, check=True
    )
    assert result_bytes.stdout == input_bytes


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_openrouter():
    """Test summarization of Tim Urban's procrastination video using openrouter model."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_OPENROUTER_MODEL,
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "monkey" in out["summary"].lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban_openrouter():
    """Test query task on Tim Urban's procrastination video using openrouter."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_OPENROUTER_MODEL,
        query_eval_model=WDOC_TEST_OPENROUTER_EVAL_MODEL,
        disable_llm_cache=True,
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_default_model():
    """Test summarization of Tim Urban's procrastination video using the default model."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_DEFAULT_MODEL,
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "monkey" in out["summary"].lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban_default_model():
    """Test query task on Tim Urban's procrastination video using the default model."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_DEFAULT_MODEL,
        query_eval_model=WDOC_TEST_DEFAULT_EVAL_MODEL,
        embed_model=WDOC_TEST_DEFAULT_EMBED_MODEL,
        disable_llm_cache=True,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()


# The pipe query and summaries test are broken. I think the issue is deep within
# pytest as it works fine in the shell. They are kept below for legacy documentation
# @pytest.mark.basic
# def test_cli_pipe_query(capsys):
#     """Test piping wdoc --help output into a wdoc query command using subprocess."""
#     # Command to get help output
#     help_cmd = ["wdoc", "--help"]
#     # Command to query, taking input from pipe
#     query_cmd = [
#         "wdoc",
#         "query",
#         "--query",
#         "does wdoc have a local html file filetype?",
#         "--model=testing/testing",
#         "--oneoff",
#     ]
#
#     with capsys.disabled():
#         # Start the help process, redirecting stdout and stderr
#         help_process = subprocess.Popen(
#             help_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
#         )
#
#         # Start the query process, taking stdin from help_process's stdout
#         # Redirect query's stdout and stderr
#         query_process = subprocess.Popen(
#             query_cmd,
#             stdin=help_process.stdout,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )
#
#         # Allow help_process stdout to be read by query_process
#         help_process.stdout.close()
#
#         # Get the output and error from the query process
#         stdout, stderr = query_process.communicate(timeout=120)
#         return_code = query_process.wait()
#
#         # Combine output for assertion
#         output = stdout + stderr
#
#         assert (
#             "Lorem ipsum dolor sit amet" in output
#         ), f"Output did not contain expected testing string:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\nReturn Code: {return_code}"
#
#
# @pytest.mark.basic
# def test_cli_pipe_summarize(sample_text_file, capsys):
#     """Test piping wdoc parse output into a wdoc summarize command using subprocess."""
#     # Ensure the sample file has enough content for parsing/summarization
#     f = Path(sample_text_file)
#     content = f.read_text()
#     f.write_text(50 * (content + "\n"))
#
#     # Command to parse the file
#     parse_cmd = ["wdoc", "parse", str(sample_text_file), "--format", "text"]
#     # Command to summarize, taking input from pipe
#     summarize_cmd = ["wdoc", "summarize", "--model=testing/testing", "--oneoff"]
#
#     with capsys.disabled():
#         # Start the parse process, redirecting stdout and stderr
#         parse_process = subprocess.Popen(
#             parse_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
#         )
#
#         # Start the summarize process, taking stdin from parse_process's stdout
#         # Redirect summarize's stdout and stderr
#         summarize_process = subprocess.Popen(
#             summarize_cmd,
#             stdin=parse_process.stdout,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )
#
#         # Allow parse_process stdout to be read by summarize_process
#         parse_process.stdout.close()
#
#         # Get the output and error from the summarize process
#         stdout, stderr = summarize_process.communicate(timeout=120)
#         return_code = summarize_process.wait()
#
#         # Combine output for assertion
#         output = stdout + stderr
#
#         assert (
#             "Lorem ipsum dolor sit amet" in output
#         ), f"Output did not contain expected testing string:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\nReturn Code: {return_code}"
