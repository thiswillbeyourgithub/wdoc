"""
Sets the default value for environment variables, parse the actual values,
check their types and finally make them easier to access by other parts of
wdoc.
"""

import os
import sys
from functools import wraps

from beartype import BeartypeConf, beartype
from beartype.door import is_bearable
from beartype.typing import Literal, Optional, Union

# must create it because we can't import it from typechecker.py
warn_typecheck = beartype(conf=BeartypeConf(violation_type=UserWarning))


class EnvVar:
    """
    **EXPERIMENTAL**
    This class stores individual environment variables in a way that therre
    values are refreshes if the user modified them. A use case example is to
    have up to date environment values when having multiple users use the
    open-webui wdocTool.
    WDOC_PRIVATE_MODE is excluded from this mechanism, meaning that if it ever is not "False" we never update it for safety reason.
    This should work for ints, floats, str, bool and None but will not work for "isinstance" check.
    """

    def __init__(self, env_name, default_value=None, type_converter=None):
        self.env_name = env_name
        self.default_value = default_value
        self._value = None
        self._type_converter = type_converter

        # Initialize with current env value
        self.__refresh__()

    def __refresh__(self):
        """Get the current value from environment"""
        if self.env_name == "WDOC_PRIVATE_MODE" and self._value != False:
            # if WDOC_PRIVATE_MODE is ever activated, we don't allow deactivating
            return self._value

        env_value = parse(str(os.environ.get(self.env_name)))
        if env_value is None:
            self._value = self.default_value
            return self._value

        if is_bearable(env_value, self._type_converter):
            self._value = env_value
            return self._value

        failed = True
        if isinstance(self._type_converter, type(str)):
            try:
                self._value = self._type_converter(env_value)
            except (ValueError, TypeError):
                pass
        elif isinstance(
            self._type_converter, (type(Optional[str]), type(Literal["a"]))
        ):
            failed = True
            to_try = [self._type_converter]
            if hasattr(self._type_converter, "__args__"):
                to_try.extend(self._type_converter.__args__)
            for t in to_try:
                if t is type(None):
                    continue
                if not callable(t):
                    t = type(t)
                try:
                    assert callable(
                        t
                    ), f"Type '{t}' is not callable but we expect it to be"
                    self._value = t(env_value)
                    failed = False
                    break
                except (ValueError, TypeError):
                    pass
        else:
            raise Exception(
                f"Couldn't handle the type '{self._type_converter}' for env variable '{self.env_name}' with new value '{env_value}'"
            )

        if failed:
            raise TypeError(
                f"Couldn't set env variable value '{env_value}' of '{self.env_name}' to type '{self._type_converter}'"
            )
        else:
            self._value = env_value
            return self._value

    def __getattr__(self, name):
        # Check if env has changed
        current_value = self.__refresh__()
        if name == "__refresh__":
            return current_value
        else:
            print(name)
            return getattr(self._value, name)(*args, **kwargs)

    def __eq__(self, other):
        self.__refresh__()
        return self._value.__eq__(other)

    def __ne__(self, other):
        self.__refresh__()
        return self._value.__ne__(other)

    def __lt__(self, other):
        self.__refresh__()
        return self._value.__lt__(other)

    def __le__(self, other):
        self.__refresh__()
        return self._value.__le__(other)

    def __gt__(self, other):
        self.__refresh__()
        return self._value.__gt__(other)

    def __ge__(self, other):
        self.__refresh__()
        return self._value.__ge__(other)

    def __str__(self):
        self.__refresh__()
        return self._value.__str__()

    def __repr__(self):
        self.__refresh__()
        return self._value.__repr__()

    def __int__(self):
        self.__refresh__()
        return self._value.__int__()

    def __float__(self, other):
        self.__refresh__()
        return self._value.__float__(other)

    def __add__(self, other):
        self.__refresh__()
        return self._value.__add__(other)

    def __isinstance__(self, cls):
        return isinstance(self._value, cls)

    def __instancecheck__(self, instance):
        return isinstance(self._value, instance)


@warn_typecheck
def parse(val: str) -> Optional[Union[bool, int, str]]:
    if val.lower() == "true":
        return True
    elif val.lower() == "false":
        return False
    elif val.isdigit():
        return int(val)
    elif val.lower() == "none" or val == "":
        return None
    else:
        return val


WDOC_TYPECHECKING = "warn"
WDOC_NO_MODELNAME_MATCHING = True
WDOC_ALLOW_NO_PRICE = False
WDOC_OPEN_ANKI = False
WDOC_STRICT_DOCDICT = False
WDOC_MAX_LOADER_TIMEOUT = -1
WDOC_MAX_PDF_LOADER_TIMEOUT = -1  # disabled as it can make the parsing slower
WDOC_PRIVATE_MODE = False
WDOC_DEBUGGER = False
WDOC_EXPIRE_CACHE_DAYS = 0
WDOC_EMPTY_LOADER = False
WDOC_BEHAVIOR_EXCL_INCL_USELESS = "warn"
WDOC_IMPORT_TYPE = "thread"
WDOC_MOD_FAISS_SCORE_FN = False
WDOC_LLM_MAX_CONCURRENCY = 15
WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE = 1000
WDOC_MAX_CHUNK_SIZE = 16_000

WDOC_DEFAULT_MODEL = "anthropic/claude-3-7-sonnet-20250219"
WDOC_DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
WDOC_DEFAULT_EMBED_DIMENSION = None
WDOC_EMBED_TESTING = True
WDOC_DEFAULT_QUERY_EVAL_MODEL = "anthropic/claude-3-5-haiku-20241022"

WDOC_LANGFUSE_PUBLIC_KEY = None
WDOC_LANGFUSE_SECRET_KEY = None
WDOC_LANGFUSE_HOST = None

WDOC_LITELLM_TAGS = None
WDOC_LITELLM_USER = "wdoc_llm"
WDOC_ENABLE_EXPERIMENTAL_ENV = False

# by default use lazy loading if using --help argument
if " --help" in " ".join(sys.argv):
    print("--help so using lazy loading")
    WDOC_IMPORT_TYPE = "lazy"

valid_types = {
    "WDOC_TYPECHECKING": Literal["disabled", "warn", "crash"],
    "WDOC_NO_MODELNAME_MATCHING": bool,
    "WDOC_ALLOW_NO_PRICE": bool,
    "WDOC_OPEN_ANKI": bool,
    "WDOC_STRICT_DOCDICT": Union[bool, Literal["strip"]],
    "WDOC_MAX_LOADER_TIMEOUT": int,
    "WDOC_MAX_PDF_LOADER_TIMEOUT": int,
    "WDOC_PRIVATE_MODE": bool,
    "WDOC_DEBUGGER": bool,
    "WDOC_EXPIRE_CACHE_DAYS": int,
    "WDOC_EMPTY_LOADER": bool,
    "WDOC_BEHAVIOR_EXCL_INCL_USELESS": Literal["warn", "crash"],
    "WDOC_IMPORT_TYPE": Literal["native", "lazy", "thread", "both"],
    "WDOC_MOD_FAISS_SCORE_FN": bool,
    "WDOC_LLM_MAX_CONCURRENCY": int,
    "WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE": int,
    "WDOC_MAX_CHUNK_SIZE": int,
    "WDOC_DEFAULT_MODEL": str,
    "WDOC_DEFAULT_EMBED_MODEL": str,
    "WDOC_DEFAULT_EMBED_DIMENSION": Optional[int],
    "WDOC_EMBED_TESTING": bool,
    "WDOC_DEFAULT_QUERY_EVAL_MODEL": str,
    "WDOC_LANGFUSE_PUBLIC_KEY": Optional[str],
    "WDOC_LANGFUSE_SECRET_KEY": Optional[str],
    "WDOC_LANGFUSE_HOST": Optional[str],
    "WDOC_LITELLM_TAGS": Optional[str],
    "WDOC_LITELLM_USER": str,
    "WDOC_ENABLE_EXPERIMENTAL_ENV": bool,
}

# sanity check for the default values
for k, v in locals().copy().items():
    if not k.startswith("WDOC_"):
        continue
    assert k in valid_types, k
    assert is_bearable(v, valid_types[k]), v

# store the env variable instead of the default values but check their types
for k in os.environ.keys():
    if not k.lower().startswith("wdoc_"):
        continue
    v = parse(os.environ[k])
    # assert k in locals().keys(), f"Unexpected key env variable starting by 'wdoc_': {k}."
    if k not in locals().keys():
        print(
            f"Unexpected key env variable starting by 'wdoc_': {k}. This might be a typo in your configuration!"
        )
    else:
        assert is_bearable(
            v, valid_types[k]
        ), f"Unexpected type of env variable '{k}': '{type(v)}' but expected '{valid_types['k']}'"
        locals()[k] = v


# set them as EnvVar
if WDOC_ENABLE_EXPERIMENTAL_ENV:
    for k, v in locals().copy().items():
        if not k.startswith("WDOC_"):
            continue
        assert k in valid_types, k
        assert is_bearable(v, valid_types[k]), v
        locals()[k] = EnvVar(k, v, valid_types[k])
