#!/bin/zsh

deactivate
trash test_venv temp

uv venv test_venv &&\
source test_venv/bin/activate &&\
sleep 1 &&\
uv pip install -e ".." &&\
uv pip install -e "..[fasttext]" &&\
uv pip install -e "..[pdftotextt]" &&\
uv pip install pytest &&\
sleep 1 &&\
mkdir temp &&\
cd temp &&\
python -m pytest ../test_wdoc.py -v && \
python -m pytest ../test_wdoc.py -v -m api &&\
cd .. &&\
deactivate &&\
trash test_venv temp &&\
echo "Succesfuly ran all tests"
