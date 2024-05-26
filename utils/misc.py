from typing import List, Union, Any
from joblib import Memory
import os
import urllib
import json
import re
from pathlib import Path
from platformdirs import user_cache_dir

from .logger import red
from .lazy_lib_importer import lazy_import_statements, lazy_import
from .typechecker import optional_typecheck

exec(lazy_import_statements("""
from bs4 import BeautifulSoup
import hashlib
from difflib import get_close_matches

import litellm
from langchain.docstore.document import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain
"""))


assert Path(user_cache_dir()).exists(), f"User cache dir not found: '{user_cache_dir()}'"
cache_dir = Path(user_cache_dir()) / "DocToolsLLM"
cache_dir.mkdir(exist_ok=True)
loaddoc_cache_dir = (cache_dir / "loaddoc_cache")
loaddoc_cache_dir.mkdir(exist_ok=True)
loaddoc_cache = Memory(loaddoc_cache_dir, verbose=0)


@optional_typecheck
def hasher(text: str) -> str:
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]

@optional_typecheck
def file_hasher(doc: dict) -> str:
    """used to hash a file's content, as describe by a dict"""
    if "path" in doc and Path(doc["path"]).exists():
        with open(doc["path"], "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:20]
    return None

@optional_typecheck
def html_to_text(html: str) -> str:
    """used to strip any html present in the text files"""
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    if "<img" in text:
        text = re.sub("<img src=.*?>", "[IMAGE]", text, flags=re.M|re.DOTALL)
        if "<img" in text:
            red("Failed to remove <img from anki card")
    return text

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
def debug_chain(inputs: Union[dict, List]) -> Union[dict, List]:
    "use it between | pipes | in a chain to open the debugger"
    if hasattr(inputs, "keys"):
        red(inputs.keys())
    breakpoint()
    return inputs


@optional_typecheck
def model_name_matcher(model: str) -> str:
    "find the best match for a modelname"
    assert "testing" not in model

    if model in list(litellm.model_cost.keys()):
        return model

    # some openai models are identified by their name directly without
    # the usual 'openai/modelname' syntax
    if "/" in model and model.split("/", 1)[1] in list(litellm.model_cost.keys()):
        return model

    # find the currently set api keys to avoid matching models from
    # unset providers
    backends = []
    for k, v in dict(os.environ).items():
        if k.endswith("_API_KEY"):
            backend = k.split("_API_KEY")[0]
            if backend not in backends:
                backends.append(backend.lower())
    for file in Path(".").iterdir():
        if file.name.endswith("_API_KEY.txt"):
            backend = file.name.split("_API_KEY.txt")[0]
            if backend not in backends:
                backends.append(backend.lower())
    assert backends, "No API keys found in environnment nor local files"

    models = []
    for m in list(litellm.model_cost.keys()):
        if "/" in m and get_close_matches(m.split("/", 1)[0].lower(), backends, n=1):
            models.append(m)
        elif "." in m and get_close_matches(m.split(".", 1)[0].lower(), backends, n=1):
            models.append(m)
        elif not ("/" in m or "." in m):
            models.append(m)
    assert models, "No models found after filtering for backends"

    match = []

    # trying with heuristics

    # match a turbo model in priority
    if not match and "turbo" not in model:
        match = get_close_matches(f"{model}-turbo", models, n=1)

    # fuzzy search
    if not match:
        match = get_close_matches(model, models, n=1)

    # keep trying without the backend
    if not match and "/" in model and "turbo" not in model:
        match = get_close_matches(model.split("/", 1)[1] + "-turbo", models, n=1)

    if not match and "/" in model:
        match = get_close_matches(model.split("/", 1)[1], models, n=1)

    if not match:
        raise Exception(f"Couldn't find a model to match {model} after filtering for backends {','.join(backends)}")

    best_match = match[0]

    # warn if ambiguous
    if len(match) > 1:
        red(f"Several match found for model named '{model}': '{','.join(match)}'\nWill use {best_match}")
    red(f"Maching name for model with heuristics: {model}->{best_match}")
    return best_match
