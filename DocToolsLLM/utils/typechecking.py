"typechecking via beartype depending on an environment flag"

import os
from typing import Callable

# optional type checking via beartype
if "DOCTOOLS_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_TYPECHECKING"] = "disabled"
if os.environ["DOCTOOLS_TYPECHECKING"] == "disabled":
    def optional_typechecker(func: Callable) -> Callable:
        return func
else:
    from beartype import beartype
    if os.environ["DOCTOOLS_TYPECHECKING"] == "warn":
        from beartype import BeartypeConf
        optional_typechecker = beartype(conf=BeartypeConf(violation_type=UserWarning))
    elif os.environ["DOCTOOLS_TYPECHECKING"] == "crash":
        optional_typechecker = beartype()
    else:
        raise ValueError("Unexpected env value for DOCTOOLS_TYPECHECKING")
