import os
import sys
import subprocess
import tempfile
from pathlib import Path
from copy import copy

import pytest
from langchain_core.documents.base import Document

os.environ["WDOC_TYPECHECKING"] = "crash"

# Default model names if not specified in environment
WDOC_TEST_OLLAMA_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2"
)
WDOC_TEST_OPENAI_MODEL = os.getenv("WDOC_TEST_OPENAI_MODEL", "gpt-4o")
WDOC_TEST_OPENAI_EVAL_MODEL = os.getenv("WDOC_TEST_OPENAI_EVAL_MODEL", "gpt-4o-mini")
WDOC_TEST_OPENAI_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OPENAI_EMBED_MODEL", "text-embedding-3-small"
)

from wdoc.wdoc import wdoc
from wdoc.utils.misc import ModelName
from wdoc.utils.embeddings import load_embeddings_engine
from wdoc.utils.embeddings import test_embeddings as _test_embeddings
from wdoc.utils.tasks.query import semantic_batching
from wdoc.utils.embeddings import load_embeddings_engine
from wdoc.utils.misc import ModelName
from wdoc.utils.env import env


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
        wdoc.parse_file(
            path=str(sample_text_file),
            filetype="txt",
        )


@pytest.mark.basic
def test_parse_file_text(sample_text_file):
    """Test basic text file parsing."""
    # make a bigger text file
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain",
    )
    assert len(docs) > 0
    assert docs[0].page_content.startswith("This is a test document")
    assert "multiple lines" in docs[0].page_content


@pytest.mark.basic
def test_parse_file_formats(sample_text_file):
    """Test text-only output from parse_file."""
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_file(
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

    ld = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain_dict",
    )
    assert isinstance(ld, list), type(ld)
    for ldd in ld:
        assert isinstance(ldd, dict), ldd
        assert "page_content" in ldd, ldd
        assert "metadata" in ldd, ldd

    text = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="text",
    )
    assert isinstance(text, str), type(text)

    xml = wdoc.parse_file(
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
        wdoc.parse_file(
            path="dummy.txt",
            filetype="invalid_type",
        )


@pytest.mark.basic
def test_parse_online_pdf():
    """Test parsing an online PDF about situational awareness."""
    docs = wdoc.parse_file(
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
def test_parse_file_help_output_shell():
    """Test that --help output contains expected docstring."""
    result = subprocess.run(
        ["wdoc", "parse", "--help"], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert (
        "This docstring is dynamically updated with the content of wdoc/docs/parse_file_help.md"
        not in output
    )
    assert "Content of wdoc/docs/parse_file_help.md" in output


@pytest.mark.basic
def test_parse_file_help_output_python():
    """Test that --help output contains expected docstring."""
    result = subprocess.run(
        ["python", "-m", "wdoc", "parse", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert (
        "This docstring is dynamically updated with the content of wdoc/docs/parse_file_help.md"
        not in output
    )
    assert "Content of wdoc/docs/parse_file_help.md" in output


@pytest.mark.basic
def test_parse_docx():
    """Test parsing a DOCX file."""
    # Create temporary file and download sample DOCX file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    url = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_500KB_DOCX.docx"
    subprocess.run(["wget", "-q", url, "-O", tmp_path], check=True)

    try:
        # Parse the file
        docs = wdoc.parse_file(
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
    docs = wdoc.parse_file(
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
    """Test summarization of Tim Urban's procrastination video."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "tim urban" in out["summary"].lower()


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
        # filetype="youtube",
        filetype="auto",
        debug=True,
    )
    out = inst.summary_task()
    assert "tim urban" in out["summary"].lower()


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
        or "tim urban" in content.lower()
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
    """Test query task on Tim Urban's procrastination video."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query="What is the allegory used by the speaker",
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()


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
