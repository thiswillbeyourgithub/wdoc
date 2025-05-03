import os
import sys
import subprocess
import tempfile
from pathlib import Path
from copy import copy

import pytest
from langchain_core.documents.base import Document

# add an unexpected env variable to make sure nothing crashes
os.environ["WDOC_TEST_UNEXPECTED_VARIABLE_1"] = "testing"

os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"

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
    "WDOC_TEST_OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2"
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
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "tim urban" in out["summary"].lower()
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
    assert "tim urban" in out2["summary"].lower()
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
    assert "tim urban" in out3["summary"].lower()
    assert (
        out3["doc_total_cost"] > 0
    ), "Normally we disabled the cache so cost should be higher than 0"


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
    """Test query task on Tim Urban's procrastination video. Three times to test the caching."""
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
        query="What is the allegory used by the speaker",
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
        query="What is the allegory used by the speaker",
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
        query="What is the allegory used by the speaker",
    )
    final_answer3 = out3["final_answer"]
    assert "monkey" in final_answer3.lower()
    assert out3["total_cost"] > 0


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
        query="What is the allegory used by the speaker",
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
    assert "tim urban" in out["summary"].lower()


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
        query="What is the allegory used by the speaker",
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
    assert "tim urban" in out["summary"].lower()


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
        query="What is the allegory used by the speaker",
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
