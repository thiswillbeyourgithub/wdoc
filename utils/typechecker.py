import os
from typeguard import typechecked, TypeCheckError
from typing import Callable, Any

if "DOCTOOLS_NO_TYPECHECKING" not in os.environ:
    os.environ["DOCTOOLS_NO_TYPECHECKING"] = "false"
assert os.environ["DOCTOOLS_NO_TYPECHECKING"] in ["true", "false"]

@typechecked
def redprint(message: str) -> str:
    message = "\033[91m" + message + "\033[0m"
    print(message)
    return message

@typechecked
def optional_typecheck(func: Callable) -> Callable:
    if os.environ["DOCTOOLS_NO_TYPECHECKING"] == "false":

        @typechecked
        def wrapper(*args, **kwargs) -> Any:
            # redprint(f"CALLING {func}")
            try:
                return typechecked(
                    func,
                )(*args, **kwargs)
            except TypeCheckError as err:
                redprint(
                    f"TypeCheckError in function '{func}'\n"
                    "To disable global "
                    "typechecking, set the runtime flag like so:\n"
                    'DOCTOOLS_NO_TYPECHECKING="true" python DocToolsLLM.py \n'
                    f"Original error:\n'''\n{err}\n'''\n")
                return func(*args, **kwargs)
        return wrapper
    else:
        return func
