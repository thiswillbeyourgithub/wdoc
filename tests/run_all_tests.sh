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
python -m pytest ../test_wdoc.py -n auto -v && \
python -m pytest ../test_wdoc.py -m api -v -x

# also check if we can install those then redo some of the tests
cd ..
uv pip install -e "..[fasttext]"
uv pip install -e "..[pdftotextt]"
cd temp
python -m pytest ../test_wdoc.py -n auto -v && \

# check if we can install the dev test
cd ..
uv pip install -e "..[dev]"

# cleanup
deactivate
trash test_venv temp
cd ..
echo "Succesfuly ran all tests"
