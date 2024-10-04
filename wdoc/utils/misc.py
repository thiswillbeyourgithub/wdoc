"""
Miscellanous functions etc.
"""

import sys
from typing import List, Union, Callable, get_type_hints, Literal
from joblib import Memory
from joblib import hash as jhash
import socket
import os
import json
from datetime import timedelta
from pathlib import Path, PosixPath
from difflib import get_close_matches
import bs4
from bs4 import GuessedAtParserWarning
import hashlib
from functools import partial, wraps
from functools import cache as memoize
from py_ankiconnect import PyAnkiconnect
import inspect
import litellm
from beartype.door import is_bearable
import warnings

from langchain.docstore.document import Document
from langchain_core.runnables import chain
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter

from .logger import whi, red, yel, cache_dir
from .typechecker import optional_typecheck
from .flags import is_verbose, is_debug
from .errors import UnexpectedDocDictArgument
from .env import WDOC_NO_MODELNAME_MATCHING, WDOC_STRICT_DOCDICT, WDOC_EXPIRE_CACHE_DAYS, WDOC_IMPORT_TYPE

ankiconnect = optional_typecheck(PyAnkiconnect())

# will be replaced when load_one_doc is called, by the path to the file where the loaders can store temporary file
loaders_temp_dir_file = cache_dir / "loaders_temp_dir.txt"

# ignore warnings from beautiful soup that can happen because anki is not exactly html
warnings.filterwarnings("ignore", category=UserWarning, module='bs4', message=".*The input looks more like a filename than markup.*")

# additional warnings to ignore
warnings.filterwarnings("ignore", module='litellm', message=".*Counting tokens for OpenAI model=.*")

try:
    import ftlangdetect

    @optional_typecheck
    def language_detector(text: str) -> float:
        return ftlangdetect.detect(text.lower())["score"]
    assert isinstance(language_detector("This is a test"), float)
except Exception as err:
    if is_verbose:
        red(f"Couldn't import optional package 'ftlangdetect', trying to import langdetect (but it's much slower): '{err}'")
    if "ftlangdetect" in sys.modules:
        del sys.modules["ftlangdetect"]

    try:
        import langdetect

        @optional_typecheck
        def language_detector(text: str) -> float:
            return langdetect.detect_langs(text.lower())[0].prob
        assert isinstance(language_detector("This is a test"), float)
    except Exception as err:
        if is_verbose:
            red(f"Couldn't import optional package 'langdetect' either: '{err}'")
        @optional_typecheck
        def language_detector(text: str) -> None:
            return None

doc_loaders_cache_dir = (cache_dir / "doc_loaders")
doc_loaders_cache_dir.mkdir(exist_ok=True)
doc_loaders_cache = Memory(doc_loaders_cache_dir, verbose=0)
hashdoc_cache_dir = (cache_dir / "doc_hashing")
hashdoc_cache_dir.mkdir(exist_ok=True)
hashdoc_cache = Memory(hashdoc_cache_dir, verbose=0)
(cache_dir / "query_eval_llm").mkdir(exist_ok=True)
query_eval_cache = Memory(cache_dir / "query_eval_llm", verbose=0)

# remove cache files older than X days
if WDOC_EXPIRE_CACHE_DAYS:
    doc_loaders_cache.reduce_size(age_limit=timedelta(days=WDOC_EXPIRE_CACHE_DAYS))
    hashdoc_cache.reduce_size(age_limit=timedelta(days=WDOC_EXPIRE_CACHE_DAYS))
    query_eval_cache.reduce_size(age_limit=timedelta(days=WDOC_EXPIRE_CACHE_DAYS))

# for reading length estimation
wpm = 250
average_word_length = 6

# separators used for the text splitter
recur_separator = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "...", ".", " ", ""]

min_token = 20
max_token = 10_000_000
min_lang_prob = 0.50

printed_unexpected_api_keys = [False]  # to print it only once

# loader specific arguments
filetype_arg_types = {
    "pdf_parsers": Union[str, List[str]],
    "anki_deck": str,
    "anki_notetype": str,
    "anki_profile": str,
    "anki_template": str,
    "anki_tag_filter": str,
    "anki_tag_render_filter": str,

    "json_dict_template": str,
    "json_dict_exclude_keys": List,

    "audio_backend": Literal["whisper", "deepgram"],
    "audio_unsilence": bool,

    "whisper_lang": str,
    "whisper_prompt": str,

    "deepgram_kwargs": dict,

    "youtube_language": str,
    "youtube_translation": str,
    "youtube_audio_backend": Literal["youtube", "whisper", "deepgram"],

    "load_functions": List,

    "doccheck_min_token": int,
    "doccheck_max_token": int,
    "doccheck_min_lang_prob": float,

    "online_media_url_regex": str,
    "online_media_resourcetype_regex": str,

    "loading_failure": str,
}

# extra arguments supported when instanciating wdoc
extra_args_types = {
    "path": Union[str, PosixPath],
    "embed_instruct": str,
    "out_file": Union[str, PosixPath],
    "include": str,
    "exclude": str,
    "filter_content": Union[List[str], str],
    "filter_metadata": Union[List[str], str],
    "source_tag": str,
}
extra_args_types.update(filetype_arg_types)

class DocDict(dict):
    """like dictionnaries but only allows keys that can be used when loading
    a document. Also checks the value type.

    The environnment variable 'WDOC_STRICT_DOCDICT' is a default value
    at instanciation time.
    Depending on WDOC_STRICT_DOCDICT (if not passed manually):
        if True: crash if unexpected arg
        if False: print in red if unexpected arg but add anyway
        if "strip": print in red but don't add
    """
    allowed_keys: set = set(
        sorted(
            ["path", "filetype", "file_hash", "source_tag", "recur_parent_id",
            ] + list(filetype_arg_types.keys())
        )
    )
    allowed_types: dict = filetype_arg_types
    __strict__ = WDOC_STRICT_DOCDICT

    def __hash__(self):
        "make it hashable, to check for duplicates"
        keys = sorted(self.keys())

        as_string = ""
        for k in keys:
            as_string += "\n"
            as_string += str(k)
            try:
                as_string += jhash(self[k])
            except Exception:
                as_string += str(self[k])
        return hash(as_string)

    def __check_values__(self, key, value, strict) -> bool:
        if key not in self.allowed_keys:
            mess  = (f"Cannot set key '{key}' in a DocDict. Allowed keys are "
                f"'{','.join(self.allowed_keys)}'\nYou can use the env "
                "variable WDOC_STRICT_DOCDICT to avoid this issue.")
            if strict is True:
                raise UnexpectedDocDictArgument(mess)
            elif strict is False:
                red(mess)
                return True
            elif strict == "strip":
                red(mess)
                return False
            else:
                raise ValueError(strict)

        elif (key in self.allowed_types) and (value is not None) and (not is_bearable(value, self.allowed_types[key])):
            mess = (f"Type of key {key} should be {self.allowed_types[key]},"
                f"not {type(value)}."
                "\nYou can use the env "
                "variable WDOC_STRICT_DOCDICT to avoid this issue.")

            if strict is True:
                raise UnexpectedDocDictArgument(mess)
            elif strict is False:
                red(mess)
                return True
            elif strict == "strip":
                red(mess)
                return False
            else:
                raise ValueError(strict)
        return True

    def __init__(self, docdict: dict, strict=WDOC_STRICT_DOCDICT) -> None:
        assert docdict, "Can't give an empty docdict as argument"
        assert strict in [True, False, "strip"], "Unexpected strict value"
        ignore_kwargs = []
        for k, v in docdict.items():
            if not self.__check_values__(k, v, strict):
                ignore_kwargs.append(k)

        for ik in ignore_kwargs:
            if ik in docdict:
                del docdict[ik]

        if strict != "strip":
            assert docdict, "Can't create DocDict: no args nor kwargs after filtering!"
        super().__init__(docdict)

        self.__strict__ = strict

    def __setitem__(self, key, value) -> None:
        assert self.__strict__ in [True, False, "strip"], "Unexpected strict value"
        self.__check_values__(key, value, self.__strict__)
        super().__setitem__(key, value)

@optional_typecheck
def optional_strip_unexp_args(func: Callable) -> Callable:
    """if the environment variable WDOC_STRICT_DOCDICT is set to 'true'
    then this automatically removes any unexpected argument before calling a
    loader function for a specific filetype."""
    if not WDOC_STRICT_DOCDICT:
        return optional_typecheck(func)
    else:
        # find the true function, otherwise func can be a decorated truefunc and might forget the annotations.
        if hasattr(func, "func"):
            truefunc = func.func
        else:
            truefunc = func
        while hasattr(truefunc, "func"):
            truefunc = truefunc.func

        @optional_typecheck
        @wraps(truefunc)
        def wrapper(*args, **kwargs):
            assert not args, f"We are not expecting args here, only kwargs. Received {args}"
            sig = inspect.signature(truefunc)
            bound_args = sig.bind_partial(**kwargs)

            # Remove unexpected positional arguments
            bound_args.arguments = {k: v for k, v in bound_args.arguments.items() if k in sig.parameters}

            # Remove unexpected keyword arguments
            kwargs2 = {k: v for k, v in kwargs.items() if k in sig.parameters}

            diffkwargs = {k: v for k, v in kwargs.items() if k not in kwargs2}
            if diffkwargs:
                mess = f"Unexpected args or kwargs in func {func}:"
                for kwarg in diffkwargs:
                    mess += f"\n-KWARG: {kwarg}"
                red(mess)
            assert kwargs2, f"No kwargs2 found for func {func}. There's probably an issue with the decorator"

            return func(**kwargs2)

        return wrapper



@optional_typecheck
def hasher(text: str) -> str:
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@optional_typecheck
def file_hasher(doc: dict) -> str:
    """used to hash a file's content, as describe by a dict
    A caching mechanism is used to avoid recomputing hash of file that
    have the same path and metadata.
    If the doc dict does not contain a path, the hash of the dict will be
    returned.
    """
    if "path" not in doc:
        return hasher(json.dumps(doc, ensure_ascii=False))
    hashable = False
    if "path" in doc and doc["path"] and Path(doc["path"]).exists():
        hashable = True
    if isinstance(doc["path"], str):
        if doc["path"] == "" or (not doc["path"].strip()):
            hashable = False
    if not doc["path"]:
        hashable = False
    if not isinstance(doc["path"], (str, PosixPath)):
        hashable = False

    if hashable:
        file = Path(doc["path"])
        stats = file.stat()
        return _file_hasher(
            abs_path=str(file.resolve().absolute()),
            stats=[stats.st_mtime, stats.st_ctime, stats.st_ino, stats.st_size]
        )
    else:
        return hasher(json.dumps(doc, ensure_ascii=False))


@optional_typecheck
@hashdoc_cache.cache
def _file_hasher(abs_path: str, stats: List[Union[int, float]]) -> str:
    with open(abs_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:20]


@optional_typecheck
def html_to_text(html: str, remove_image: bool = False) -> str:
    """used to strip any html present in the text files"""
    html = html.replace("</li><li>", "<br>")  # otherwise they might get joined
    html = html.replace("</ul><ul>", "<br>")  # otherwise they might get joined
    html = html.replace("<br>", "\n").replace("</br>", "\n")  # otherwise newlines are lost

    soup = bs4.BeautifulSoup(html, 'html.parser')
    content = []
    for element in soup.descendants:
        if element.name == 'img' and (not remove_image):
            element = str(element)
            if element in html:
                content.append(element)
            elif element[:-2] + ">" in html:
                content.append(element[:-2] + ">")
            else:
                if is_verbose:
                    temptext = ' '.join(filter(None, content))
                    red(f"Image not properly parsed from bs4:\n{element}\n{temptext}")
        elif isinstance(element, bs4.NavigableString):
            content.append(str(element).strip())
    text = ' '.join(filter(None, content))
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    if "<img" in text and remove_image:
        red(f"Failed to remove <img from anki card: {text}")
    return text


@chain
@optional_typecheck
def debug_chain(inputs: Union[dict, List]) -> Union[dict, List]:
    "use it between | pipes | in a chain to open the debugger"
    if hasattr(inputs, "keys"):
        red(str(inputs.keys()))
    breakpoint()
    return inputs


@optional_typecheck
def wrapped_model_name_matcher(model: str) -> str:
    "find the best match for a modelname (wrapped to make some check)"
    # find the currently set api keys to avoid matching models from
    # unset providers
    all_backends = list(litellm.models_by_provider.keys())
    backends = []
    for k, v in dict(os.environ).items():
        if k.endswith("_API_KEY"):
            backend = k.split("_API_KEY")[0].lower()
            if backend not in all_backends and is_verbose and not printed_unexpected_api_keys[0]:
                yel(
                    f"Found API_KEY for backend {backend} that is not a known backend for litellm.")
            else:
                backends.append(backend)
    if is_verbose:
        printed_unexpected_api_keys[0] = True
    assert backends, "No API keys found in environnment"

    # filter by providers
    backend, modelname = model.split("/", 1)
    if backend not in all_backends:
        raise Exception(
            f"Model {model} with backend {backend}: backend not found in "
            "litellm.\nList of litellm providers/backend:\n"
            f"{all_backends}"
        )
    if backend not in backends:
        raise Exception(
            f"Trying to use backend {backend} but no API KEY was found for it in the environnment.")
    candidates = litellm.models_by_provider[backend]
    if modelname in candidates:
        return model
    subcandidates = [m for m in candidates if m.startswith(modelname)]
    if len(subcandidates) == 1:
        good = f"{backend}/{subcandidates[0]}"
        return good
    match = get_close_matches(modelname, candidates, n=1)
    if match:
        return match[0]
    else:
        red(f"Couldn't match the modelname {model} to any known model. "
            "Continuing but this will probably crash wdoc further "
            "down the code.")
        return model


@optional_typecheck
def model_name_matcher(model: str) -> str:
    """find the best match for a modelname (wrapper that checks if the matched
    model has a known cost and print the matched name)
    Bypassed if env variable WDOC_NO_MODELNAME_MATCHING is 'true'
    """
    assert "testing" not in model
    assert "/" in model, f"expected / in model '{model}'"
    if WDOC_NO_MODELNAME_MATCHING:
        whi(f"Bypassing model name matching for model '{model}'")
        return model

    out = wrapped_model_name_matcher(model)
    if out != model and is_verbose:
        yel(f"Matched model name {model} to {out}")
    assert out in litellm.model_cost or out.split(
        "/", 1)[1] in litellm.model_cost, f"Neither {out} nor {out.split('/', 1)[1]} found in litellm.model_cost"
    return out

@optional_typecheck
def get_tkn_length(
    tosplit: str,
    modelname: str = "gpt-3.5-turbo",
    ) -> int:
    modelname = modelname.replace("openrouter/", "")
    return litellm.token_counter(model=modelname, text=tosplit)

text_splitters = {}

@optional_typecheck
def get_splitter(
    task: str,
    modelname="gpt-3.5-turbo",
) -> TextSplitter:
    "we don't use the same text splitter depending on the task"
    # avoid creating many times this object
    if task not in text_splitters:
        text_splitters[task] = {}
    if modelname in text_splitters[task]:
        return text_splitters[task]

    if modelname == "testing/testing":
        return get_splitter(task=task, modelname="gpt-3.5-turbo")

    try:
        max_tokens = litellm.get_model_info(modelname)["max_input_tokens"]

        # don't use overly large chunks anyway
        max_tokens = min(max_tokens, 16_000)
    except Exception as err:
        max_tokens = 4096
        red(f"Failed to get max_tokens limit for model {modelname}: '{err}'")

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

    text_splitters[task][modelname] = text_splitter
    return text_splitter


@optional_typecheck
def check_docs_tkn_length(
    docs: List[Document],
    identifier: str,
    min_token: int = min_token,
    max_token: int = max_token,
    min_lang_prob: float = min_lang_prob,
    check_language: bool = False,
) -> float:
    """checks that the number of tokens in the document is high enough,
    not too low, and has a high enough language probability,
    otherwise something probably went wrong."""
    size = sum([get_tkn_length(d.page_content) for d in docs])
    nline = len("\n".join([d.page_content for d in docs]).splitlines())
    if size <= min_token:
        red(
            f"Example of page from document with too few tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{identifier}' is {size} <= {min_token}, probably something went wrong?"
        )
    if size >= max_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{identifier}' is {size} >= {max_token}, probably something went wrong?"
        )
    if check_language is False:
        return 1.0

    # check if language check is above a threshold and cast as lowercase as it's apparently what it was trained on
    try:
        probs = [
            language_detector(d.page_content.replace("\n", "<br>"))
            for d in docs
        ]
        if probs[0] is None or not probs:
            # bypass if language_detector not defined
            return 1.0
        prob = sum(probs) / len(probs)
        if prob <= min_lang_prob:
            raise Exception(
                f"Low language probability for {identifier}: prob={prob:.3f}<{min_lang_prob}.\nExample page: {docs[len(docs)//2]}"
            )
    except Exception as err:
        if str(err).startswith("Low language probability"):
            raise
        else:
            red(f"Error when using language_detector on '{identifier}': {err}. Treating it as valid document.")
            return 1.0
    return prob


@optional_typecheck
def unlazyload_modules():
    """make sure no modules are lazy loaded. Useful when we wan't to make
    sure not to loose time and that everything works smoothly. For example
    who knows what happens when multiprocessing with lazy loaded modules."""
    if WDOC_IMPORT_TYPE not in ["both", "lazy"]:
        red("Lazyloading is disabled so not unlazyloading modules.")
        return

    while True:
        found_one = False
        for k, v in sys.modules.items():
            if "Lazily-loaded" in str(v):
                try:
                    dir(v)  # this is enough to trigger the loading
                    found_one = True
                except Exception as err:
                    raise Exception(
                        f"Error when unlazyloading module '{k}'. Error: '{err}'"
                        "\nThis can be caused by beartype's typechecking"
                        "\nYou can also try setting the env variable "
                        "WDOC_DISABLE_LAZYLOADING to 'true'"
                    ) from err
                break  # otherwise dict size change during iteration
            assert "Lazily-loaded" not in str(v)
        if found_one:
            continue
        else:
            break


@optional_typecheck
def disable_internet(allowed: dict) -> None:
    """
    To be extra sure that no connection goes out of the computer when
    --private is used, we overload the socket module to make it only able to
    reach local connection.
    """
    red(
        "Disabling outgoing internet because private mode is on. "
        "The only allowed IPs from now on are the ones from the "
        "argument llm_api_bases. Note that this permanently filters "
        "outgoing python connections so might interfere with other "
        "python programs is you are importing wdoc instead "
        "of calling it from the shell"
    )

    # unlazyload all modules as otherwise the overloading can happen too late
    unlazyload_modules()

    # list of certainly allowed IPs
    allowed_IPs = set([
        "localhost",
        "127.0.0.1",
    ])
    vals = [
        v.split("//")[1].split(":")[0]
        if "//" in v
        else v.split(":")[0]
        for v in list(allowed.values())
    ]
    [allowed_IPs.add(v) for v in vals]

    # list of probably allowed IPs
    private_ranges = [
        ('10.0.0.0', '10.255.255.255'),
        ('172.16.0', '172.31.255.255'),
        ('192.168.0.0', '192.168.255.255'),
        ('127.0.0.0', '127.255.255.255')
    ]

    @memoize
    @optional_typecheck
    def is_private(ip: str) -> bool:
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

    @optional_typecheck
    def create_connection(address, *args, **kwargs):
        "overload socket.create_connection to forbid outgoing connections"
        ip = socket.gethostbyname(address[0])
        if not is_private(ip):
            raise RuntimeError(
                "Network connections to the open internet are blocked")
        return socket._original_create_connection(address, *args, **kwargs)

    socket.socket = lambda *args, **kwargs: None
    socket._original_create_connection = socket.create_connection
    socket.create_connection = create_connection

    # sanity check
    assert is_private("localhost")
    assert is_private("10.0.1.32")
    assert is_private("192.168.2.35")
    assert is_private("127.12.13.15")

    # checking allowed ips are okay
    for v in vals:
        assert is_private(v), f"An address failed to be set as private: '{v}'"
    for al in list(allowed.values()):
        ip = socket.gethostbyname(al)
        assert is_private(ip), f"An address failed to be set as private: '{al}'"
    try:
        ip = socket.gethostbyname("www.google.com")
        skip = False
    except Exception as err:
        red("Failed to get IP address of www.google.com to check if it is "
            "indeed blocked. You probably did this on purpose so not "
            f"crashing. Error: '{err}'")
        skip = True
    if not skip:
        assert not is_private(ip), f"Failed to set www.google.com as unreachable: IP is '{ip}'"

@optional_typecheck
def set_func_signature(func: Callable) -> Callable:
    """dynamically set the extra args of wdoc.__init__ so that
    instead of **cli_kwargs the signature indicates all allowed arguments.
    Needed to get correct behavior from fire.Fire '--completion' """
    original_sig = inspect.signature(func)
    assert list(original_sig.parameters.values())[-1].kind == inspect.Parameter.VAR_KEYWORD
    new_params = list(original_sig.parameters.values())[:-1]  # Remove **cli_kwargs
    new_params.extend(
                [
                    inspect.Parameter(
                        name=arg,
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        annotation=hint,
                        default=None,
                    )
                    for arg, hint in extra_args_types.items()
            ]
    )
    new_sig = original_sig.replace(parameters=new_params)

    @wraps(func)
    def new_func(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    new_func.__signature__ = new_sig
    new_func.__annotations__ = get_type_hints(func) | extra_args_types

    return new_func

THIN = "<thinking>"
THINE = "</thinking>"
ANSW = "<answer>"
ANSWE = "</answer>"
def thinking_answer_parser(output: str, strict: bool = False) -> dict:
    """separate the <thinking> and <answer> tags in an answer"""
    try:
        # fix </answer> instead of <answer>
        if ANSW not in output and output.count(ANSWE) == 2:
            output = output.replace(ANSWE, ANSW, 1)
        if THIN not in output and output.count(THINE) == 2:
            output = output.replace(THINE, THIN, 1)

        if (THIN not in output) and (ANSW not in output):
            assert THINE not in output, f"Output contains unexpected {THINE}:\n'''\n{output}\n'''"
            assert ANSWE not in output, f"Output contains unexpected {ANSWE}:\n'''\n{output}\n'''"

            return {"thinking": "", "answer": output}

        thinking = ""
        if THIN in output and THINE in output:
            thinking = output.split(THIN, 1)[1].split(THINE, 1)[0].strip()

        answer = ""
        if ANSW in output and ANSWE in output:
            answer = output.replace(thinking, "").split(ANSW, 1)[1].split(ANSWE, 1)[0].strip()

        assert THIN not in answer, f"Parsed answer contained unexpected {THIN}:\n'''\n{output}\n'''"
        assert THINE not in answer, f"Parsed answer contained unexpected {THIN}:\n'''\n{output}\n'''"
        assert ANSW not in answer, f"Parsed answer contained unexpected {ANSW}:\n'''\n{output}\n'''"
        assert ANSWE not in answer, f"Parsed answer contained unexpected {ANSW}:\n'''\n{output}\n'''"

        assert answer, f"No answer could be parsed from LLM output: '{output}'"

        return {"thinking": thinking, "answer": answer}
    except Exception as err:
        if strict:  # otherwise combining answers could snowball into losing lots of text
            raise
        red(f"Error when parsing LLM output to get thinking and answer part.\nError: '{err}'\nOriginal output: '{output}'\nWill continue if not using --debug")
        if is_debug:
            raise
        else:
            assert output.strip(), "LLM output was empty"
            return {
                "thinking": "",
                "answer": f"""
<note>
The following LLM answer might have had a problem during parsing
</note>
<output>
{output}
</output>
""".strip(),
            }
