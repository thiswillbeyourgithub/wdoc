"""
Decorator used in many places. It does runtime typechecking. By default
it's disabled but the flag WDOC_TYPECHECKING can be set to "crash" or
to "warn" to just print the error.
"""

from typing import Callable
from beartype import beartype, BeartypeConf

from .env import WDOC_TYPECHECKING

if WDOC_TYPECHECKING == "crash":
    optional_typecheck = beartype
elif WDOC_TYPECHECKING == "warn":
    optional_typecheck = beartype(
        conf=BeartypeConf(violation_type=UserWarning))
elif WDOC_TYPECHECKING == "disabled":
    def optional_typecheck(func: Callable) -> Callable:
        return func
else:
    raise ValueError("Unexpected WDOC_TYPECHECKING env value")
