"""
Decorator used in many places. It does runtime typechecking. By default
it's disabled but the flag WINSTONDOC_TYPECHECKING can be set to "crash" or
to "warn" to just print the error.
"""

import os
from typing import Callable

from beartype import beartype, BeartypeConf


if "WINSTONDOC_TYPECHECKING" not in os.environ:
    os.environ["WINSTONDOC_TYPECHECKING"] = "disabled"

if os.environ["WINSTONDOC_TYPECHECKING"] == "crash":
    optional_typecheck = beartype
elif os.environ["WINSTONDOC_TYPECHECKING"] == "warn":
    optional_typecheck = beartype(
        conf=BeartypeConf(violation_type=UserWarning))
elif os.environ["WINSTONDOC_TYPECHECKING"] == "disabled":
    def optional_typecheck(func: Callable) -> Callable:
        return func
else:
    raise ValueError("Unexpected WINSTONDOC_TYPECHECKING env value")
