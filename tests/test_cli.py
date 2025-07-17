import os
import sys
import subprocess

import pytest

# Environment setup for CLI tests
os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"
os.environ["WDOC_TYPECHECKING"] = "crash"


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
def test_ddg_search_nvidia():
    """Test DuckDuckGo search functionality with NVIDIA query."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "wdoc",
            "--task=query",
            "--path=How is NVidia doing this month?",
            "--query=How is NVidia doing this month?",
            "--filetype=ddg",
            "--ddg_max_result=3",
            "--model=testing/testing",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    output = result.stdout + result.stderr

    # Check that we got some output and no major errors
    assert (
        len(output) > 100
    ), f"Expected substantial output from DDG search, got: {output}"

    # Should contain the testing model's standard response
    assert (
        "Lorem ipsum dolor sit amet" in output
    ), f"Output did not contain expected testing string: {output}"

    # Should not contain error messages about DDG functionality
    assert (
        "Error" not in output or "error" not in output.lower()
    ), f"Unexpected error in DDG search: {output}"


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
