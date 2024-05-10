import os
from typeguard import typechecked, check_type
from typing import Callable

if "DOCTOOLS_NO_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_NO_TYPECHECKING"] = "false"
assert os.environ["DOCTOOLS_NO_TYPECHECKING"] in ["true", "false"]


@typechecked
def optional_typecheck(func: Callable) -> Callable:
    if os.environ["DOCTOOLS_NO_TYPECHECKING"] == "false":
        def wrapper(*args, **kwargs):
            return typechecked(func)(*args, **kwargs)
        return wrapper
    else:
        return func
