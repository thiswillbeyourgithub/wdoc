from typing import List, Union

from .logger import red
from .lazy_lib_importer import lazy_import_statements, lazy_import
from .typechecker import optional_typecheck

exec(lazy_import_statements("""
import urllib
import json
from datetime import timedelta
import re
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
from joblib import Memory

from langchain.docstore.document import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain
"""))



Path(".cache").mkdir(exist_ok=True)
Path(".cache/loaddoc_cache").mkdir(exist_ok=True)

loaddoc_cache = Memory(".cache/loaddoc_cache/", verbose=1)


@optional_typecheck
def hasher(text: str) -> str:
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@optional_typecheck
def html_to_text(html: str, issoup: bool) -> str:
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

@optional_typecheck
def _request_wrapper(action: str, **params) -> dict:
    return {'action': action, 'params': params, 'version': 6}

@optional_typecheck
def ankiconnect(action: str, **params) -> Union[List, str]:
    "talk to anki via ankiconnect addon"

    requestJson = json.dumps(_request_wrapper(action, **params)
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


@chain
@optional_typecheck
def debug_chain(inputs: dict) -> dict:
    "use it between | pipes | in a chain to open the debugger"
    try:
        red(inputs.keys())
    except Exception as err:
        red(f"Failed to print inputs: {err}")
    breakpoint()
    return inputs
