from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
from joblib import Memory


def hasher(text):
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:10]

def html_to_text(html, issoup):
    """used to strip any html present in the text files"""
    if not issoup:
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    else:
        return html.get_text()

def get_kwargs(**kwargs):
    """
    Parameters
    ----------
    --task str
        either query or summary. query means to load the input files then wait
        for user question. summary means the input will be passed through a
        summarization prompt to get the idea.

    --filetype str
        the type of input. Depending on the value, different other parameters
        are needed. If path_list is used, the line of the input file can contain
        any of those parameters as long as they are as json.

        Supported values => relevant parameters
            * youtube => --path must be a link to youtube --language=fr to use french transcripts --translation=en to use the transcripts after translation to english
            * pdf => --path is path to pdf
            * txt => --path is path to txt
            * path_list => --path is path to a txt file that contains a json for each line containing at least a filetype and a path key/value but can contain any parameters described here
            * anki => --anki_profile is the name of the profile --anki_deck the beginning of the deckname --anki_notetype the beginning of the notetype to keep
            * recursive => --path is the starting path --pattern is the globbing patterns to append --exclude can be a list of regex that excludes some paths --recursed_filetype is the filetype to use for each of the found path

    --model str, default gpt4all
        either gpt4all or openai or fake/test/testing to use a fake answer.

    --gpt4all_model_path str
        if model is gpt4all, this needs to point to a compatible model
    """
    return kwargs

Path(".cache").mkdir(exist_ok=True)
Path(".cache/docstore_cache").mkdir(exist_ok=True)
Path(".cache/split_cache").mkdir(exist_ok=True)

docstore_cache = Path(".cache/docstore_cache/")
split_cache = Memory(".cache/split_cache/")
