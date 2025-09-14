#!/bin/zsh

set -e
set -o pipefail
set -u

# cleanup previous
[[ "$(type deactivate)" == "deactivate is a shell function"* ]] && deactivate
[ -e "temp" ] && rm -rv temp
[ -e "test_env" ] && rm -rv test_env
[ -e "__pycache__" ] && rm -rv __pycache__

# setup venv
uv venv test_venv --python 3.12.11
source test_venv/bin/activate
sleep 1

# install wdoc
uv pip install -e ".."

# install test suite
uv pip install pytest pytest-xdist
sleep 1

# Store the venv python path to ensure we use it consistently
PYTHON_EXEC=$(which python)

mkdir temp
cd temp

# start tests
echo "\nTesting CLI using a shell script"
../test_cli.sh
echo "Done with CLI using a shell script"

echo "\nTesting parsing (basic)"
$PYTHON_EXEC -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_parsing.py
echo "Done with parsing (basic)"

echo "\nTesting wdoc (basic)"
$PYTHON_EXEC -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_wdoc.py
echo "Done with wdoc (basic)"

echo "\nTesting vectorstores (api)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ../test_vectorstores.py
echo "Done with vectorstores (api)"

echo "\nTesting wdoc (api)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ../test_wdoc.py
echo "Done with wdoc (api)"

echo "\nDone with first round of pytest!"

cd ..

# also check if we can install those then redo some of the tests
uv pip install -e "..[fasttext]"
uv pip install -e "..[pdftotext]"
cd temp

echo "\nTesting wdoc (basic)"
$PYTHON_EXEC -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ../test_wdoc.py
echo "Done with wdoc (basic)"

# check if we can install the dev test
cd ..
uv pip install -e "..[dev]"

# check if we can install the full wdoc
uv pip install -e "..[full]"

# cleanup
deactivate
trash test_venv temp
cd ..
echo "Succesfuly ran all tests"
