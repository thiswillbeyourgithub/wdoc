#!/bin/zsh

set -e
set -o pipefail
set -u

# Crash early if a model that will be used requires an API key we don't have.
# Defaults mirror those in test_wdoc.py and wdoc/utils/env.py.
_ALL_TEST_MODELS=(
    "${WDOC_TEST_OPENAI_MODEL:-gpt-4o}"
    "${WDOC_TEST_OPENAI_EVAL_MODEL:-gpt-4o-mini}"
    "${WDOC_TEST_OPENAI_EMBED_MODEL:-text-embedding-3-small}"
    "${WDOC_TEST_OPENROUTER_MODEL:-openrouter/mistralai/mistral-small-3.2-24b-instruct}"
    "${WDOC_TEST_OPENROUTER_EVAL_MODEL:-openrouter/mistralai/mistral-small-3.2-24b-instruct}"
    "${WDOC_TEST_DEFAULT_MODEL:-openrouter/deepseek/deepseek-v4-pro}"
    "${WDOC_TEST_DEFAULT_EVAL_MODEL:-openrouter/deepseek/deepseek-v4-flash}"
    "${WDOC_TEST_DEFAULT_EMBED_MODEL:-openai/text-embedding-3-small}"
    "${WDOC_TEST_MISTRAL_EMBED_MODEL:-mistral/mistral-embed}"
)
_check_provider_key() {
    local prefix="$1" keyvar="$2" m
    for m in "${_ALL_TEST_MODELS[@]}"; do
        if [[ "$m" == ${prefix}* ]]; then
            if [[ -z "${(P)keyvar:-}" ]]; then
                echo "ERROR: $keyvar env var is not set but a test model starts with '$prefix'. Set $keyvar or override the relevant WDOC_TEST_* model env vars." >&2
                exit 1
            fi
            return 0
        fi
    done
}
_check_provider_key "openrouter/" OPENROUTER_API_KEY
_check_provider_key "openai/" OPENAI_API_KEY
_check_provider_key "mistral/" MISTRAL_API_KEY
if [[ -z "${WDOC_WHISPER_API_KEY:-}" ]]; then
    echo "ERROR: WDOC_WHISPER_API_KEY env var is not set but is required to test whisper transcription. Set WDOC_WHISPER_API_KEY before running tests." >&2
    exit 1
fi
# Zotero loader: the api test needs real credentials (or a reachable local
# Zotero) plus a small selector to fan out. Mirror the whisper check so a
# missing setup crashes early instead of silently skipping the test.
if [[ -z "${ZOTERO_API_KEY:-}" ]]; then
    echo "ERROR: ZOTERO_API_KEY env var is not set but is required to test the zotero loader. Set ZOTERO_API_KEY (and ZOTERO_LIBRARY_ID) before running tests." >&2
    exit 1
fi
if [[ -z "${ZOTERO_COLLECTION_NAME:-}" ]]; then
    echo "ERROR: ZOTERO_COLLECTION_NAME env var is not set but is required to exercise the zotero loader api test. Set it to the name of a (preferably small) collection in your library." >&2
    exit 1
fi
# Karakeep loader: the api test creates its own temporary bookmark and deletes
# it, so it only needs real credentials (endpoint + api key). Mirror the zotero
# check so a missing setup crashes early instead of silently skipping the test.
if [[ -z "${KARAKEEP_PYTHON_API_KEY:-}" ]]; then
    echo "ERROR: KARAKEEP_PYTHON_API_KEY env var is not set but is required to test the karakeep loader. Set KARAKEEP_PYTHON_API_KEY (and KARAKEEP_PYTHON_API_ENDPOINT) before running tests." >&2
    exit 1
fi
if [[ -z "${KARAKEEP_PYTHON_API_ENDPOINT:-}" ]]; then
    echo "ERROR: KARAKEEP_PYTHON_API_ENDPOINT env var is not set but is required to test the karakeep loader. Set it to your instance URL including /api/v1/." >&2
    exit 1
fi

# cleanup previous
[[ "$(type deactivate)" == "deactivate is a shell function"* ]] && deactivate
[ -e "temp" ] && rm -r temp
[ -e "test_env" ] && rm -r test_env
[ -e "wdoc_user_cache_dir" ] && rm -r wdoc_user_cache_dir
[ -e "__pycache__" ] && rm -r __pycache__

# setup venv
uv venv test_venv --python 3.13.5
source test_venv/bin/activate
sleep 1

# install wdoc
uv pip install -e "..[full,dev,fasttext,pdftotext]"

# Store the venv python path to ensure we use it consistently
PYTHON_EXEC=$(which python)

mkdir temp
cd temp

# start tests
echo "\nTesting CLI using a shell script"
../test_cli.sh api
echo "Done with CLI using a shell script"

echo "\nTesting parsing (basic)"
$PYTHON_EXEC -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_parsing.py
echo "Done with parsing (basic)"

echo "\nTesting wdoc (basic)"
$PYTHON_EXEC -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_wdoc.py
echo "Done with wdoc (basic)"

echo "\nTesting zotero (basic)"
$PYTHON_EXEC -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_zotero.py
echo "Done with zotero (basic)"

echo "\nTesting zotero (api)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ../test_zotero.py
echo "Done with zotero (api)"

echo "\nTesting karakeep (basic)"
$PYTHON_EXEC -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_karakeep.py
echo "Done with karakeep (basic)"

echo "\nTesting karakeep (api)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ../test_karakeep.py
echo "Done with karakeep (api)"

echo "\nTesting vectorstores (api)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ../test_vectorstores.py
echo "Done with vectorstores (api)"

echo "\nTesting wdoc (api)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ../test_wdoc.py
echo "Done with wdoc (api)"

echo "\nDone with first round of pytest!"

cd ..

# cleanup
deactivate
# trash test_venv temp if present
[ -e "temp" ] && rm -r temp
[ -e "tests/temp" ] && rm -r temp
[ -e "test_env" ] && rm -r test_env
[ -e "tests/test_env" ] && rm -r test_env
[ -e "wdoc_user_cache_dir" ] && rm -r wdoc_user_cache_dir
[ -e "tests/wdoc_user_cache_dir" ] && rm -r wdoc_user_cache_dir
[ -e "__pycache__" ] && rm -r __pycache__
[ -e "tests/__pycache__" ] && rm -r __pycache__

cd ..
echo "Succesfuly ran all tests"
