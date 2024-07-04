"""
Decorator used in many places. It does runtime typechecking. By default
it's disabled but the flag DOCTOOLS_TYPECHECKING can be set to "crash" or
to "warn" to just print the error.
"""

import os
from typing import Callable

from beartype import beartype, BeartypeConf


if "DOCTOOLS_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_TYPECHECKING"] = "warn"

if os.environ["DOCTOOLS_TYPECHECKING"] == "crash":
    optional_typecheck = beartype
elif os.environ["DOCTOOLS_TYPECHECKING"] == "warn":
    optional_typecheck = beartype(conf=BeartypeConf(violation_type=UserWarning))
elif os.environ["DOCTOOLS_TYPECHECKING"] == "disabled":
    def optional_typecheck(func: Callable) -> Callable:
        return func
else:
    raise ValueError("Unexpected DOCTOOLS_TYPECHECKING env value")
