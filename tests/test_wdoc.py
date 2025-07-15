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
