"""
Sets the default value for environment variables, parse the actual values,
check their types and finally make them easier to access by other parts of
wdoc.
"""

import os
import sys
from dataclasses import MISSING, dataclass, asdict

from loguru import logger

from beartype import BeartypeConf, beartype
from beartype.door import is_bearable
from beartype.typing import Literal, Optional, Union

from .errors import FrozenAttributeCantBeSet

# must create it because we can't import it from typechecker.py
warn_typecheck = beartype(conf=BeartypeConf(violation_type=UserWarning))


@dataclass
class EnvDataclass:
    __frozen__: bool = False
    WDOC_DUMMY_ENV_VAR: bool = False
    WDOC_TYPECHECKING: Literal["disabled", "warn", "crash"] = "warn"
    WDOC_NO_MODELNAME_MATCHING: bool = True
    WDOC_ALLOW_NO_PRICE: bool = False
    WDOC_OPEN_ANKI: bool = False
    WDOC_STRICT_DOCDICT: Union[bool, Literal["strip"]] = False
    WDOC_MAX_LOADER_TIMEOUT: int = -1
    WDOC_MAX_PDF_LOADER_TIMEOUT: int = -1  # disabled as it can make the parsing slower
    WDOC_PRIVATE_MODE: bool = False
    WDOC_DEBUGGER: bool = False
    WDOC_EXPIRE_CACHE_DAYS: int = 0
    WDOC_EMPTY_LOADER: bool = False
    WDOC_BEHAVIOR_EXCL_INCL_USELESS: Literal["warn", "crash"] = "warn"
    # by default use lazy loading if using --help argument
    WDOC_IMPORT_TYPE: Literal["native", "lazy", "thread", "both"] = (
        "native" if " --help" not in " ".join(sys.argv) else "lazy"
    )
    WDOC_MOD_FAISS_SCORE_FN: bool = False
    WDOC_LLM_MAX_CONCURRENCY: int = 1
    WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE: int = 2000
    WDOC_MAX_CHUNK_SIZE: int = 16_000
    WDOC_INTERMEDIATE_ANSWER_MAX_TOKENS: int = 4000
    WDOC_DEFAULT_MODEL: str = "anthropic/claude-3-7-sonnet-20250219"
    WDOC_DEFAULT_EMBED_MODEL: str = "openai/text-embedding-3-small"
    WDOC_DEFAULT_EMBED_DIMENSION: Optional[int] = None
    WDOC_EMBED_TESTING: bool = True
    WDOC_DISABLE_EMBEDDINGS_CACHE: bool = False
    WDOC_DEFAULT_QUERY_EVAL_MODEL: str = "anthropic/claude-3-5-haiku-20241022"
    WDOC_LANGFUSE_PUBLIC_KEY: Optional[str] = None
    WDOC_LANGFUSE_SECRET_KEY: Optional[str] = None
    WDOC_LANGFUSE_HOST: Optional[str] = None
    WDOC_LITELLM_TAGS: Optional[str] = None
    WDOC_LITELLM_USER: str = "wdoc_llm"
    WDOC_APPLY_ASYNCIO_PATCH: bool = False
    WDOC_CONTINUE_ON_INVALID_EVAL: bool = True

    @warn_typecheck
    def __parse__(self, val: str) -> Optional[Union[bool, int, str]]:
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

    def __setattr__(self, name, value):
        # dont allow unfreezing
        if name == "__frozen__" and self.__frozen__ is True and value is False:
            raise Exception(f"Cannot unfreeze the frozen EnvDataclass instance")

        # allow setting variable values only until frozen
        if self.__frozen__ is not True:
            return super().__setattr__(name, value)

        raise FrozenAttributeCantBeSet(name, value)

    def __getattribute__(self, name):
        # non WDOC env can be gotten right away
        if name in ["__dataclass_fields__", "__frozen__", "__parse__"]:
            return super().__getattribute__(name)

        # get the current value stored in the dataclass
        try:
            cur_val = super().__getattribute__(name)
        except Exception as e:
            raise Exception(
                f"Error when getting attribute {name} from EnvClass: {e}"
            ) from e

        # check that the stored value is of appropriate type
        assert is_bearable(cur_val, self.__dataclass_fields__[name].type), cur_val

        if self.__frozen__ is not True:
            return cur_val

        # get the value from the env
        env_val = os.environ.get(name, MISSING)

        # if missing from env, if the value is not the default that means
        # it has been deleted so we should return the default value. But if
        # it's the default we can return it safely. Special case if the attribute
        # contains 'private', don't allow modifying it at runtime out of paranoia.
        if env_val is MISSING:
            default = self.__dataclass_fields__[name].default
            if cur_val == default:
                return default

            if "private" in name.lower():
                raise AttributeError(
                    f"Error when accessible env variable '{name}': its env variable counterpart is missing but its name contains 'private' so out of an abundance of caution we crash."
                )

            else:
                logger.warning(
                    f"Env variable '{name}' is missing but the stored value ('{cur_val}' ) is different than the default ('{default}'). We are then setting the wdoc attribute to its default value."
                )
                os.environ[name] = default
                return default

        env_val = self.__parse__(env_val)

        # if unchanged, we can return it
        if cur_val == env_val:
            return cur_val

        # the env variable has changed
        logger.warning(
            f"Env variable '{name}' changed between initialization and now: env value is '{env_val}' and already loaded variable is '{cur_val}'. Returning the env value"
        )
        if "private" in name.lower():
            raise AttributeError(
                f"Quitting out of an abundance of caution: env vaiable '{name}' contains 'private' in its name so it's to important to allow changing it at runtime."
            )

        # check that it has the appropriate type
        assert is_bearable(env_val, self.__dataclass_fields__[name].type), env_val

        # if we were no using the freezing mechanism we could store it like
        # that but let's not
        # super().__setattr__(name, env_val)

        return env_val


# sanity check for the default values of the dataclass itself
for k, v in EnvDataclass.__dataclass_fields__.items():
    assert is_bearable(v.default, v.type), v

env = EnvDataclass()

# check that the freezing works as expected
try:
    env.WDOC_DUMMY_ENV_VAR = False
except FrozenAttributeCantBeSet as e:
    raise Exception(
        f"Something is wrong with the freezing of EnvDataclass: '{e}'"
    ) from e
if " --help" in " ".join(sys.argv):
    # just notify the user
    logger.debug("--help so using lazy loading by default")


# store the env variable instead of the default values but check their types
for k in os.environ.keys():
    if not k.lower().startswith("wdoc_"):
        continue
    v = env.__parse__(os.environ[k])

    if k not in env.__dataclass_fields__.keys():
        print(
            f"Unexpected key env variable starting by 'wdoc_': {k}. This might be a typo in your configuration!"
        )
    else:
        assert is_bearable(
            v, env.__dataclass_fields__[k].type
        ), f"Unexpected type of env variable '{k}': '{type(v)}' but expected '{env.__dataclass_fields__['k'].type}'"
        v_stored = getattr(env, k)
        setattr(env, k, v)

env.__frozen__ = True
try:
    env.WDOC_DUMMY_ENV_VAR = False
    raise Exception("Something is wrong with the freezing of EnvDataclass")
except FrozenAttributeCantBeSet:
    pass

# sanity check for the stored values
for k, v in asdict(env).items():
    # if not k.startswith("WDOC_"):
    #     continue
    assert is_bearable(v, env.__dataclass_fields__[k].type), v

# If langfuse env variables are set AND WDOC_LANGFUSE_PUBLIC_KEY etc are set: we replace langfuse's env variable to make sure any underlyng lib use wdoc's instead
for k in [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
]:
    newk = "WDOC_" + k
    if newk in os.environ and os.environ[newk]:
        os.environ[k] = os.environ[newk]
