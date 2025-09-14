#!/bin/zsh

set -e
set -o pipefail
set -u

# Environment setup for CLI tests (same as in pytest file)
export WDOC_TYPECHECKING="crash"

# Store the venv python path to ensure we use it consistently
PYTHON_EXEC=$(which python)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test result tracking
PASSED_TESTS=0
FAILED_TESTS=0
TOTAL_TESTS=0

# Helper function to run a test and track results
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo "${YELLOW}[TEST $TOTAL_TESTS]${NC} Running: $test_name"
    
    if $test_function; then
        echo "${GREEN}[PASS]${NC} $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "${RED}[FAIL]${NC} $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        # Don't exit immediately, continue with other tests
        return 1
    fi
}

# Test functions (converted from pytest)
test_help_output_shell() {
    local output
    if ! output=$(wdoc --help); then
        echo "FAIL: wdoc --help command failed"
        return 1
    fi
    
    # Check that dynamic docstring placeholder is not present
    if echo "$output" | grep -q "This docstring is dynamically updated with the content of wdoc/docs/help.md"; then
        echo "FAIL: Found dynamic docstring placeholder in help output"
        return 1
    fi
    
    # Check that actual content is present
    if ! echo "$output" | grep -q "Content of wdoc/docs/help.md"; then
        echo "FAIL: Did not find expected content in help output"
        return 1
    fi
    
    return 0
}

test_help_output_python() {
    local output
    if ! output=$($PYTHON_EXEC -m wdoc --help); then
        echo "FAIL: python -m wdoc --help command failed"
        return 1
    fi
    
    # Check that dynamic docstring placeholder is not present
    if echo "$output" | grep -q "This docstring is dynamically updated with the content of wdoc/docs/help.md"; then
        echo "FAIL: Found dynamic docstring placeholder in help output"
        return 1
    fi
    
    # Check that actual content is present
    if ! echo "$output" | grep -q "Content of wdoc/docs/help.md"; then
        echo "FAIL: Did not find expected content in help output"
        return 1
    fi
    
    return 0
}

test_parse_doc_help_output_shell() {
    local output
    if ! output=$(wdoc parse --help); then
        echo "FAIL: wdoc parse --help command failed"
        return 1
    fi
    
    # Check that dynamic docstring placeholder is not present
    if echo "$output" | grep -q "This docstring is dynamically updated with the content of wdoc/docs/parse_doc_help.md"; then
        echo "FAIL: Found dynamic docstring placeholder in parse help output"
        return 1
    fi
    
    # Check that actual content is present
    if ! echo "$output" | grep -q "Content of wdoc/docs/parse_doc_help.md"; then
        echo "FAIL: Did not find expected content in parse help output"
        return 1
    fi
    
    return 0
}

test_parse_doc_help_output_python() {
    local output
    if ! output=$($PYTHON_EXEC -m wdoc parse --help); then
        echo "FAIL: python -m wdoc parse --help command failed"
        return 1
    fi
    
    # Check that dynamic docstring placeholder is not present
    if echo "$output" | grep -q "This docstring is dynamically updated with the content of wdoc/docs/parse_doc_help.md"; then
        echo "FAIL: Found dynamic docstring placeholder in parse help output"
        return 1
    fi
    
    # Check that actual content is present
    if ! echo "$output" | grep -q "Content of wdoc/docs/parse_doc_help.md"; then
        echo "FAIL: Did not find expected content in parse help output"
        return 1
    fi
    
    return 0
}

test_error_message_shell_debug() {
    local command="wdoc --task=summarize --path https://lemonde.fr/ --filetype=test --debug"
    local expected_substring="(Pdb) "
    local output
    
    # Use timeout and expect it to timeout since debug mode waits for input
    # Capture both stdout and stderr, and handle the timeout exit code
    if output=$(timeout 20s zsh -c "$command"); then
        echo "FAIL: Command should have timed out but completed successfully"
        return 1
    fi
    
    # timeout returns 124 on timeout, which is what we expect because --debug on error creates a prompt
    local exit_code=$?
    if [[ $exit_code -ne 124 ]]; then
        echo "FAIL: Expected timeout (exit code 124), got exit code $exit_code"
        return 1
    fi
    
    # Check if we got the expected substring in the output
    if ! echo "$output" | grep -q "$expected_substring"; then
        echo "FAIL: Expected '$expected_substring' not found in command output"
        echo "Output was: $output"
        return 1
    fi
    
    return 0
}

test_get_piped_input_detection() {
    # Test text piping
    local input_text="This is test text.\nWith multiple lines."
    local cmd_text='import sys; from wdoc.utils.misc import get_piped_input; data = get_piped_input(); sys.stdout.write(data)'
    local result_text
    
    if ! result_text=$(echo -e "$input_text" | $PYTHON_EXEC -c "$cmd_text"); then
        echo "FAIL: Text piping command failed"
        return 1
    fi
    
    if ! echo "$result_text" | grep -q "This is test text"; then
        echo "FAIL: Text piping did not work correctly"
        echo "Expected to find input text in output, got: $result_text"
        return 1
    fi
    
    # Test binary piping (using printf to create binary-like data)
    local cmd_bytes='import sys; from wdoc.utils.misc import get_piped_input; data = get_piped_input(); sys.stdout.buffer.write(data)'
    local result_bytes
    
    if ! result_bytes=$(printf '\x01\x02\x03\xffbinary data' | $PYTHON_EXEC -c "$cmd_bytes"); then
        echo "FAIL: Binary piping command failed"
        return 1
    fi
    
    if ! echo "$result_bytes" | grep -q "binary data"; then
        echo "FAIL: Binary piping did not work correctly"
        echo "Expected to find binary data in output, got: $result_bytes"
        return 1
    fi
    
    return 0
}

test_parse_nytimes_shell() {
    local output
    if ! output=$($PYTHON_EXEC -m wdoc parse "https://www.nytimes.com/" --format text); then
        echo "FAIL: NYTimes parsing command failed"
        return 1
    fi
    
    # Verify we got substantial content
    local output_length=${#output}
    if [[ $output_length -le 100 ]]; then
        echo "FAIL: Expected significant text content from NYTimes, got only $output_length characters"
        echo "Output was: $output"
        return 1
    fi
    
    echo "INFO: Got $output_length characters from NYTimes"
    return 0
}

test_ddg_search_nvidia() {
    local output
    local cmd="$PYTHON_EXEC -m wdoc --task=query --path='How is Nvidia doing this month?' --query='How is Nvidia doing this month?' --filetype=ddg --ddg_max_result=3 --ddg_region=us-US --model=testing/testing --loading_failure=warn --oneoff --file_loader_parallel_backend=threading"
    
    if ! output=$(timeout 120s zsh -c "$cmd"); then
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            echo "FAIL: DDG search command timed out"
        else
            echo "FAIL: DDG search command failed with exit code $exit_code"
        fi
        echo "Output was: $output"
        return 1
    fi
    
    # Check that we got some output
    local output_length=${#output}
    if [[ $output_length -le 100 ]]; then
        echo "FAIL: Expected substantial output from DDG search, got only $output_length characters"
        echo "Output was: $output"
        return 1
    fi
    
    # Should contain the testing model's standard response
    if ! echo "$output" | grep -q "Lorem ipsum dolor sit amet"; then
        echo "FAIL: Output did not contain expected testing string"
        echo "Output was: $output"
        return 1
    fi
    
    echo "INFO: Got $output_length characters from DDG search"
    return 0
}

# Main execution
echo "${YELLOW}Starting CLI tests...${NC}"
echo "Using Python executable: $PYTHON_EXEC"
echo

# Run basic tests
echo "${YELLOW}=== BASIC TESTS ===${NC}"
run_test "Help output (shell)" test_help_output_shell
run_test "Help output (python)" test_help_output_python  
run_test "Parse doc help output (shell)" test_parse_doc_help_output_shell
run_test "Parse doc help output (python)" test_parse_doc_help_output_python
run_test "Error message in debug mode" test_error_message_shell_debug
run_test "Piped input detection" test_get_piped_input_detection
run_test "Parse NYTimes homepage" test_parse_nytimes_shell

echo
echo "${YELLOW}=== API TESTS ===${NC}"
echo "Note: API tests require external network access and may take longer"

# Check if we should run API tests (similar to pytest's -m api marker)
if [[ "${1:-}" == "api" ]] || [[ "$*" == *"api"* ]]; then
    run_test "DuckDuckGo search functionality" test_ddg_search_nvidia
else
    echo "${YELLOW}Skipping API tests. Use 'api' argument to run them.${NC}"
fi

echo
echo "${YELLOW}=== TEST SUMMARY ===${NC}"
echo "Total tests: $TOTAL_TESTS"
echo "${GREEN}Passed: $PASSED_TESTS${NC}"
echo "${RED}Failed: $FAILED_TESTS${NC}"

if [[ $FAILED_TESTS -gt 0 ]]; then
    echo "${RED}Some tests failed!${NC}"
    exit 1
else
    echo "${GREEN}All tests passed!${NC}"
    exit 0
fi
