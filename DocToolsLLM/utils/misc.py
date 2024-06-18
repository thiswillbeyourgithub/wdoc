"""
Miscellanous functions etc.
"""

import sys
from typing import List, Union, Any
from joblib import Memory
import socket
import os
import urllib
import json
import re
from pathlib import Path
from platformdirs import user_cache_dir
from difflib import get_close_matches
from bs4 import BeautifulSoup
import hashlib
import lazy_import
import tiktoken
from functools import partial

from langchain.docstore.document import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain

from .logger import red
from .typechecker import optional_typecheck
from .flags import is_verbose

litellm = lazy_import.lazy_module("litellm")
Document = lazy_import.lazy_class('langchain.docstore.document.Document')
TextSplitter = lazy_import.lazy_class('langchain.text_splitter.TextSplitter')
RecursiveCharacterTextSplitter = lazy_import.lazy_class('langchain.text_splitter.RecursiveCharacterTextSplitter')

try:
    import ftlangdetect
except Exception as err:
    if is_verbose:
        print(f"Couldn't import optional package 'ftlangdetect', trying to import langdetect (but it's much slower): '{err}'")
    try:
        import langdetect
    except Exception as err:
        if is_verbose:
            print(f"Couldn't import optional package 'langdetect': '{err}'")

if "ftlangdetect" in globals():
    @optional_typecheck
    def language_detector(text: str) -> float:
        return ftlangdetect.detect(text)["score"]
elif "language_detect" in globals():
    @optional_typecheck
    def language_detector(text: str) -> float:
        return langdetect.detect_langs(text)[0].prob
else:
    def language_detector(text: str) -> None:
        return None

assert Path(user_cache_dir()).exists(), f"User cache dir not found: '{user_cache_dir()}'"
cache_dir = (Path(user_cache_dir()) / "DocToolsLLM").resolve()
cache_dir.mkdir(exist_ok=True)
loaddoc_cache_dir = (cache_dir / "loaddoc_cache")
loaddoc_cache_dir.mkdir(exist_ok=True)
loaddoc_cache = Memory(loaddoc_cache_dir, verbose=0)
hashdoc_cache_dir = (cache_dir / "hashdoc_cache")
hashdoc_cache_dir.mkdir(exist_ok=True)
hashdoc_cache = Memory(hashdoc_cache_dir, verbose=0)

# for reading length estimation
wpm = 250
average_word_length = 6

# separators used for the text splitter
recur_separator = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "...", ".", " ", ""]

# used to get token length estimation
tokenizers = {
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
}

min_token = 50
max_token = 1_000_000
max_lines = 100_000
min_lang_prob = 0.50

# loader specific arguments
loader_specific_keys = {
    "anki_deck": str,
    "anki_fields": str,
    "anki_mode": str,
    "anki_notetype": str,
    "anki_profile": str,

    "audio_backend": str,

    "whisper_lang": str,
    "whisper_prompt": str,

    "deepgram_kwargs": dict,

    "youtube_language": str,
    "youtube_translation": str,
    "youtube_audio_backend": str,

    "load_functions": List[str],
}

# extra arguments supported when instanciating doctools
extra_args_keys = {
    "embed_instruct": str,
    "exclude": str,
    "file_loader_n_jobs": int,
    "filter_content": Union[List[str], str],
    "filter_metadata": Union[List[str], str],
    "include": str,
    "out_file": str,
    "path": str,
    "source_tag": str,
    "loading_failure": str,
}
extra_args_keys.update(loader_specific_keys)

# keys that can legally be part of a doc_kwarg
doc_kwargs_keys = [
    "path",
    "filetype",
    "file_hash",
    "source_tag",
] + list(loader_specific_keys.keys())

def hasher(text: str) -> str:
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]

def file_hasher(doc: dict) -> str:
    """used to hash a file's content, as describe by a dict
    A caching mechanism is used to avoid recomputing hash of file that
    have the same path and metadata.
    If the doc dict does not contain a path, the hash of the dict will be
    returned.
    """
    if "path" in doc and Path(doc["path"]).exists():
        file = Path(doc["path"])
        stats = file.stat()
        return _file_hasher(
            abs_path=str(file.resolve().absolute()),
            stats=[stats.st_mtime, stats.st_ctime, stats.st_ino, stats.st_size]
        )
    else:
        return hasher(json.dumps(doc))

@hashdoc_cache.cache
def _file_hasher(abs_path: str, stats: List[int]) -> str:
    with open(abs_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:20]

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
    red(f"Matching name for model with heuristics: {model}->{best_match}")
    return best_match


@optional_typecheck
def get_tkn_length(tosplit: str, modelname: str = "gpt-3.5-turbo") -> int:
    if modelname in tokenizers:
        return len(tokenizers[modelname](tosplit))
    else:
        try:
            tokenizers[modelname] = tiktoken.encoding_for_model(modelname.split("/")[-1]).encode
        except Exception:
            modelname="gpt-3.5-turbo"
        return get_tkn_length(tosplit=tosplit, modelname=modelname)

@optional_typecheck
def get_splitter(
    task: str,
    modelname="gpt-3.5-turbo",
    ) -> TextSplitter:
    "we don't use the same text splitter depending on the task"

    max_tokens = 4096
    try:
        max_tokens = litellm.get_model_info(modelname)["max_input_tokens"]

        # don't use overly large chunks anyway
        max_tokens = min(max_tokens, int(2.5 * litellm.get_model_info(modelname)["max_tokens"]))
    except Exception as err:
        red(f"Failed to get max_token limit for model {modelname}: '{err}'")

    model_tkn_length = partial(get_tkn_length, modelname=modelname)

    if task in ["query", "search"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=int(3 / 4 * max_tokens),  # default 4000
            chunk_overlap=500,  # default 200
            length_function=model_tkn_length,
        )
    elif task in ["summarize_then_query", "summarize"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=int(1 / 2 * max_tokens),
            chunk_overlap=500,
            length_function=model_tkn_length,
        )
    elif task == "recursive_summary":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=int(1 / 4 * max_tokens),
            chunk_overlap=300,
            length_function=model_tkn_length,
        )
    else:
        raise Exception(task)
    return text_splitter


@optional_typecheck
def check_docs_tkn_length(docs: List[Document], name: str) -> float:
    """checks that the number of tokens in the document is high enough,
    not too low, and has a high enough language probability,
    otherwise something probably went wrong."""
    size = sum([get_tkn_length(d.page_content) for d in docs])
    nline = len("\n".join([d.page_content for d in docs]).splitlines())
    if nline > max_lines:
        red(
            f"Example of page from document with too many lines : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of lines from '{name}' is {nline} > {max_lines}, probably something went wrong?"
        )
    if size <= min_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{name}' is {size} <= {min_token}, probably something went wrong?"
        )
    if size >= max_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{name}' is {size} >= {max_token}, probably something went wrong?"
        )

    # check if language check is above a threshold
    probs = [
        language_detector(d.page_content.replace("\n", "<br>"))
        for d in docs
    ]
    if probs[0] is None:
        # bypass if language_detector not defined
        return 1
    prob = sum(probs) / len(probs)
    if prob <= min_lang_prob:
        red(
            f"Low language probability for {name}: prob={prob:.3f}<{min_lang_prob}.\nExample page: {docs[len(docs)//2]}"
        )
        raise Exception(
            f"Low language probability for {name}: prob={prob:.3f}.\nExample page: {docs[len(docs)//2]}"
        )
    return prob


def unlazyload_modules():
    """make sure no modules are lazy loaded. Useful when we wan't to make
    sure not to loose time and that everything works smoothly. For example
    who knows what happens when multiprocessing with lazy loaded modules."""
    while True:
        found_one = False
        for k, v in sys.modules.items():
            if "Lazily-loaded" in str(v):
                dir(v)  # this is enough to trigger the loading
                found_one = True
                break  # otherwise dict size change during iteration
            assert "Lazily-loaded" not in str(v)
        if found_one:
            continue
        else:
            break

def disable_internet(allowed: dict) -> None:
    """
    To be extra sure that no connection goes out of the computer when
    --private is used, we overload the socket module to make it only able to
    reach local connection.
    """
    red(
        "Disabling outgoing internet because private mode is on. "
        "The only allowed IPs from now on are the ones from the "
        "argument llm_api_bases")

    # unlazy load all modules as otherwise the overloading can happen too late
    unlazyload_modules()

    # list of certainly allowed IPs
    allowed_IPs = set([
        "localhost",
        "127.0.0.1",
    ])
    vals = list(allowed.values())
    vals = [
        v.split("//")[1].split(":")[0]
        if "//" in v
        else v.split(":")[0]
        for v in vals
    ]
    [allowed_IPs.add(v) for v in vals]

    # list of probably allowed IPs
    private_ranges = [
        ('10.0.0.0', '10.255.255.255'),
        ('172.16.0', '172.31.255.255'),
        ('192.168.0.0', '192.168.255.255'),
        ('127.0.0.0', '127.255.255.255')
    ]

    def is_private(ip) -> bool:
        "detect if the connection would go to our computer or to a remote server"
        if ip in allowed_IPs:
            return True
        ip = int.from_bytes(socket.inet_aton(ip), 'big')
        if ip in allowed_IPs:
            return True
        for start, end in private_ranges:
            if int.from_bytes(socket.inet_aton(start), 'big') <= ip <= int.from_bytes(socket.inet_aton(end), 'big'):
                return True
        return False

    def create_connection(address, *args, **kwargs):
        "overload socket.create_connection to forbid outgoing connections"
        ip = socket.gethostbyname(address[0])
        if not is_private(ip):
            raise RuntimeError("Network connections to the open internet are blocked")
        return socket._original_create_connection(address, *args, **kwargs)

    socket.socket = lambda *args, **kwargs: None
    socket._original_create_connection = socket.create_connection
    socket.create_connection = create_connection
