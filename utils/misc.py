import re
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
from joblib import Memory

Path(".cache").mkdir(exist_ok=True)
Path(".cache/docstore_cache").mkdir(exist_ok=True)
Path(".cache/split_cache").mkdir(exist_ok=True)

docstore_cache = Path(".cache/docstore_cache/")
split_cache = Memory(".cache/split_cache/")


def hasher(text):
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def html_to_text(html, issoup):
    """used to strip any html present in the text files"""
    if not issoup:
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        if "<img" in text:
            text = re.sub("<img src=.*?>", "[IMAGE]", text, flags=re.M|re.DOTALL)
        return text
    else:
        return html.get_text()


def check_kwargs(**kwargs):
    """
    Parameters
    ----------
    --task str, default query
        either query or summary. query means to load the input files then wait
        for user question. summary means the input will be passed through a
        summarization prompt to get the idea.

    --filetype str
        the type of input. Depending on the value, different other parameters
        are needed. If path_list is used, the line of the input file can contain
        any of those parameters as long as they are as json. You can find
        an example of path_list file in utils/file_list.txt

        Supported values => relevant parameters
            * youtube => --path must be a link to youtube --language=fr to use french transcripts --translation=en to use the transcripts after translation to english
            * pdf => --path is path to pdf
            * txt => --path is path to txt
            * anki => --anki_profile is the name of the profile --anki_deck the beginning of the deckname --anki_notetype the beginning of the notetype to keep --anki_fields list of fields to keep
            * string => no other parameters needed, will ask to provide a string
            * path_list => --path is path to a txt file that contains a json for each line containing at least a filetype and a path key/value but can contain any parameters described here
            * recursive => --path is the starting path --pattern is the globbing patterns to append --exclude can be a list of regex that excludes some paths --recursed_filetype is the filetype to use for each of the found path

    --model str
        either gpt4all, llama, openai or fake/test/testing to use a fake answer.

    --local_llm_path str
        if model is not openai, this needs to point to a compatible model

    --sbert_model str, default "distiluse-base-multilingual-cased-v1"
        sentence_transformer embedding model to use. If you change this,
        the embedding cache will be populated with new elements (the hash
        used to check for previous values includes the name of the sbert model)

    --saveas str, default .cache/latest_docs_and_embeddings
        only used if task is query
        save the latest 'inputs' to a file. Can be loaded again with
        --loadfrom to speed up loading time. This loads both the
        split documents and embeddings but will not update itself if the
        original files have changed.

    --loadfrom str, default None
        if not filetype argument is given, loadfrom will be set to the
        same default value as saveas
        For more, see --saveas

    --top_k int, default 3
        retrieval argument

    --debug
        if present as argument, sometimes will open a debugger instead before crashing
    """
    assert "loaded_docs" not in kwargs, "'loaded_docs' cannot be an argument as it is used internally"
    assert "loaded_embeddings" not in kwargs, "'loaded_embeddings' cannot be an argument as it is used internally"
    if "sbert_model" not in kwargs:
        kwargs["sbert_model"] = "distiluse-base-multilingual-cased-v1"
    if "task" not in kwargs:
        kwargs["task"] = "query"
    assert kwargs["task"] in ["query", "summary"], "invalid task value"
    if kwargs["task"] == "summary":
        assert not "loadfrom" in kwargs, "can't use loadfrom if task is summary"
    if "saveas" not in kwargs:
        kwargs["saveas"] = str(docstore_cache.parent / "latest_docs_and_embeddings")
    if "filetype" not in kwargs and "loadfrom" not in kwargs:
        kwargs["filetype"] = None
        kwargs["loadfrom"] = str(docstore_cache.parent / "latest_docs_and_embeddings")
    if "top_k" not in kwargs:
        kwargs["top_k"] = 3

    return kwargs
