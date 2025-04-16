#!/bin/zsh

deactivate
trash test_venv temp

uv venv test_venv &&\
source test_venv/bin/activate &&\
sleep 1 &&\
uv pip install -e ".." &&\
uv pip install pytest pytest-xdist &&\
sleep 1 &&\
mkdir temp &&\
cd temp &&\
python -m pytest ../test_wdoc.py -n auto -v && \
python -m pytest ../test_wdoc.py -m api -v &&\
# also check if we can install those then redo some of the tests
uv pip install -e "..[fasttext]" &&\
uv pip install -e "..[pdftotextt]" &&\
python -m pytest ../test_wdoc.py -n auto -v && \
# check if we can install the dev test
uv pip install -e "..[dev]" &&\
cd .. &&\
deactivate &&\
trash test_venv temp &&\
echo "Succesfuly ran all tests"
