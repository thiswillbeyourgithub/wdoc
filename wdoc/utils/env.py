"""
Sets the default value for environment variables, parse the actual values,
check their types and finally make them easier to access by other parts of
wdoc.
"""

import sys
import os
from typing import Optional, Union, Literal
from beartype import beartype, BeartypeConf
from beartype.door import is_bearable

# must create it because we can't import it from typechecker.py
warn_typecheck = beartype(conf=BeartypeConf(violation_type=UserWarning))

@warn_typecheck
def parse(val: str) -> Optional[Union[bool, int, str]]:
    if val == "true":
        return True
    elif val == "false":
        return False
    elif val.isdigit():
        return int(val)
    elif val == "none" or val == "":
        return None
    return val

WDOC_TYPECHECKING = "warn"
WDOC_NO_MODELNAME_MATCHING = False
WDOC_ALLOW_NO_PRICE = False
WDOC_OPEN_ANKI = False
WDOC_STRICT_DOCDICT = False
WDOC_MAX_LOADER_TIMEOUT = -1
WDOC_MAX_PDF_LOADER_TIMEOUT = -1  # disabled as it can make the parsing slower
WDOC_PRIVATE_MODE = False
WDOC_DEBUGGER = False
WDOC_EXPIRE_CACHE_DAYS = 0
WDOC_EMPTY_LOADER = False
WDOC_BEHAVIOR_EXCL_INCL_USELESS = "warn"
WDOC_IMPORT_TYPE = "lazy"
WDOC_MOD_FAISS_SCORE_FN = False
WDOC_LLM_MAX_CONCURRENCY = 10
WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE = 500

WDOC_DEFAULT_MODELNAME = "openai/gpt-4o"
WDOC_DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
WDOC_DEFAULT_QUERY_EVAL_MODELNAME = "openai/gpt-4o-mini"

# by default use lazy loading if using --help argument
if " --help" in " ".join(sys.argv):
    print("--help so using lazy loading")
    WDOC_IMPORT_TYPE = "lazy"

valid_types = {
    'WDOC_TYPECHECKING': Literal["disabled", "warn", "crash"],
    'WDOC_NO_MODELNAME_MATCHING': bool,
    'WDOC_ALLOW_NO_PRICE': bool,
    'WDOC_OPEN_ANKI': bool,
    'WDOC_STRICT_DOCDICT': bool,
    'WDOC_MAX_LOADER_TIMEOUT': int,
    'WDOC_MAX_PDF_LOADER_TIMEOUT': int,
    'WDOC_PRIVATE_MODE': bool,
    'WDOC_DEBUGGER': bool,
    'WDOC_EXPIRE_CACHE_DAYS': int,
    'WDOC_EMPTY_LOADER': bool,
    'WDOC_BEHAVIOR_EXCL_INCL_USELESS': Literal["warn", "crash"],
    'WDOC_IMPORT_TYPE': Literal["native", "lazy", "thread", "both"],
    'WDOC_MOD_FAISS_SCORE_FN': bool,
    'WDOC_LLM_MAX_CONCURRENCY': int,
    'WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE': int,
    'WDOC_DEFAULT_MODELNAME': str,
    'WDOC_DEFAULT_EMBED_MODEL': str,
    'WDOC_DEFAULT_QUERY_EVAL_MODELNAME': str,
}

# sanity check for the default values
for k, v in locals().copy().items():
    if not k.startswith("WDOC_"):
        continue
    assert k in valid_types, k
    assert is_bearable(v, valid_types[k]), v

# store the env variable instead of the default values but check their types
for k in os.environ.keys():
    if not k.lower().startswith("wdoc_"):
        continue
    v = parse(os.environ[k])
    assert k in locals().keys(), f"Unexpected key env variable starting by 'wdoc_': {k}"
    assert is_bearable(v, valid_types[k]), f"Unexpected type of env variable '{k}': '{type(v)}' but expected '{valid_types['k']}'"
    locals()[k] = v
