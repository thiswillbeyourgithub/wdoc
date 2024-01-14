import urllib
import json
from datetime import timedelta
import re
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
from joblib import Memory

from .logger import red

Path(".cache").mkdir(exist_ok=True)
Path(".cache/loaddoc_cache").mkdir(exist_ok=True)

loaddoc_cache = Memory(".cache/loaddoc_cache/", verbose=1)

# remove cache files older than 90 days
try:
    loaddoc_cache.reduce_size(age_limit=timedelta(90))
except Exception as err:
    red(f"Error when reducing cache size: '{err}'")



def hasher(text):
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def html_to_text(html, issoup):
    """used to strip any html present in the text files"""
    if not issoup:
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        if "<img" in text:
            text = re.sub("<img src=.*?>", "[IMAGE]", text, flags=re.M|re.DOTALL)
            if "<img" in text:
                red("Failed to remove <img from anki card")
        return text
    else:
        return html.get_text()

def ankiconnect(action, **params):
    "talk to anki via ankiconnect addon"
    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)
                             ).encode('utf-8')

    try:
        response = json.load(urllib.request.urlopen(
            urllib.request.Request(
                'http://localhost:8765',
                requestJson)))
    except (ConnectionRefusedError, urllib.error.URLError) as e:
        raise Exception(f"{str(e)}: is Anki open and 'ankiconnect "
                        "addon' enabled? Firewall issue?")

    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']
