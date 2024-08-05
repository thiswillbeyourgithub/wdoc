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
WDOC_NO_MODELNAME_MATCHING = None
WDOC_ALLOW_NO_PRICE = None
WDOC_OPEN_ANKI = False
WDOC_STRICT_DOCDICT = None
WDOC_MAX_LOADER_TIMEOUT = 30 * 60
WDOC_MAX_PDF_LOADER_TIMEOUT = 5 * 60
WDOC_PRIVATE_MODE = False
WDOC_DEBUGGER = False

for k in os.environ.keys():
    if not k.startswith("WDOC_"):
        continue
    v = parse(os.environ[k])
    assert k in locals().keys(), f"Unexpected key for WDOC env variable: {k}"
    locals()[k] = v
