import os
from functools import wraps
from textwrap import indent
from typeguard import typechecked, TypeCheckError
from typing import Callable, Any

if "DOCTOOLS_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_TYPECHECKING"] = "disabled"

def redprint(message: str) -> str:
    "print in red"
    message = "\033[91m" + message + "\033[0m"
    print(message)
    return message

def optional_typecheck(func: Callable) -> Callable:
    if os.environ["DOCTOOLS_TYPECHECKING"] == "crash":
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return typechecked(func)(*args, **kwargs)
            except TypeCheckError as err:
                mess = (
                    f"TypeCheckError in function '{func}'\n"
                    "To disable global "
                    "typechecking, set the runtime flag like so:\n"
                    'DOCTOOLS_TYPECHECKING="disabled" python DocToolsLLM.py \n'
                    f"Original error:\n{indent(str(err), '    ')}\n")
                raise TypeCheckError(redprint(mess)) from err
        return wrapper

    elif os.environ["DOCTOOLS_TYPECHECKING"] == "warn":
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                out_vals = typechecked(func)(*args, **kwargs)
            except TypeCheckError as err:
                redprint(
                    f"TypeCheckError in function '{func}'\n"
                    "To disable global "
                    "typechecking, set the runtime flag like so:\n"
                    'DOCTOOLS_TYPECHECKING="disabled" python DocToolsLLM.py \n'
                    f"Original error:\n{indent(str(err), '    ')}\n")
                out_vals = func(*args, **kwargs)
            return out_vals
        return wrapper

    elif os.environ["DOCTOOLS_TYPECHECKING"] == "disabled":
        return func

    else:
        raise ValueError(
            f"Unexpected value for os.environ['DOCTOOLS_TYPECHECKING']: '{os.environ['DOCTOOLS_TYPECHECKING']}'")
