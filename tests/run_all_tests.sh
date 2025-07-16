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
mkdir temp

# start tests
cd temp
python -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic ..
python -m pytest -n auto --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m api ..

# also check if we can install those then redo some of the tests
cd ..
uv pip install -e "..[with-fasttext]"
uv pip install -e "..[with-pdftotext]"
cd temp
python -m pytest --disable-warnings --show-capture=no --code-highlight=yes --tb=short -m basic

# check if we can install the dev test
cd ..
uv pip install -e "..[dev]"

# cleanup
deactivate
trash test_venv temp
cd ..
echo "Succesfuly ran all tests"
