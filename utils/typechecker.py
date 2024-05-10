import os
from typeguard import typechecked, TypeCheckError
from typing import Callable, Any

if "DOCTOOLS_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_TYPECHECKING"] = "warn"

@typechecked
def redprint(message: str) -> str:
    "print in red"
    message = "\033[91m" + message + "\033[0m"
    print(message)
    return message

@typechecked
def optional_typecheck(func: Callable) -> Callable:
    if os.environ["DOCTOOLS_TYPECHECKING"] == "crash":
        @typechecked
        def wrapper(*args, **kwargs) -> Any:
            try:
                return typechecked(
                    func,
                )(*args, **kwargs)
            except TypeCheckError as err:
                redprint(
                    f"TypeCheckError in function '{func}'\n"
                    "To disable global "
                    "typechecking, set the runtime flag like so:\n"
                    'DOCTOOLS_TYPECHECKING="true" python DocToolsLLM.py \n'
                    f"Original error:\n'''\n{err}\n'''\n")
                raise
        return wrapper
    elif os.environ["DOCTOOLS_TYPECHECKING"] == "warn":
        @typechecked
        def wrapper(*args, **kwargs) -> Any:
            try:
                return typechecked(
                    func,
                )(*args, **kwargs)
            except TypeCheckError as err:
                redprint(
                    f"TypeCheckError in function '{func}'\n"
                    "To disable global "
                    "typechecking, set the runtime flag like so:\n"
                    'DOCTOOLS_TYPECHECKING="true" python DocToolsLLM.py \n'
                    f"Original error:\n'''\n{err}\n'''\n")
                return func(*args, **kwargs)
        return wrapper
    elif os.environ["DOCTOOLS_TYPECHECKING"] == "disabled":
        @typechecked
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
    else:
        raise ValueError(
            f"Unexpected value for os.environ['DOCTOOLS_TYPECHECKING']: '{os.environ['DOCTOOLS_TYPECHECKING']}'")
