import os
from typing import Any

def parse(val: str) -> Any:
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
WDOC_MAX_PDF_LOADER_TIMEOUT = 5 * 60
WDOC_PRIVATE_MODE = False
WDOC_DEBUGGER = False
WDOC_EXPIRE_CACHE_DAYS = 0
WDOC_EMPTY_LOADER = False
WDOC_BEHAVIOR_EXCL_INCL_USELESS = "warn"

valid_types = {
    'WDOC_TYPECHECKING': str,
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
    'WDOC_BEHAVIOR_EXCL_INCL_USELESS': str,
}

for k in os.environ.keys():
    if not k.startswith("WDOC_"):
        continue
    v = parse(os.environ[k])
    assert k in locals().keys(), f"Unexpected key for WDOC env variable: {k}"
    assert isinstance(v, valid_types[k]), f"Unexpected type of env variable '{k}': '{type(v)}' but expected '{valid_types['k']}'"
    locals()[k] = v
