"""
Sets the default value for environment variables, parse the actual values,
wdoc.
Also set some variables useful to access globally like is_linux for example.
"""

import platform
from textwrap import indent, dedent
from pathlib import Path
import os
import sys
from dataclasses import MISSING, dataclass, asdict, field

from loguru import logger

from beartype import BeartypeConf, beartype
from beartype.door import is_bearable
from beartype.typing import Literal, Optional, Union, List

try:
    from wdoc.utils.errors import FrozenAttributeCantBeSet
except ImportError:  # for debugging purposes
    from errors import FrozenAttributeCantBeSet

# must create it because we can't import it from typechecker.py
warn_typecheck = beartype(conf=BeartypeConf(violation_type=UserWarning))

is_linux = platform.system() == "Linux"

# useful to know if we should use tqdm or not (it can cause broken pipe errors
# otherwise) and modify the formatting output.ArithmeticError
is_input_piped = not sys.stdin.isatty()
# Also useful to modify the loglevel
is_out_piped = not sys.stdout.isatty()


@dataclass
class EnvDataclass:
    """
    This dataclass holds the env variables used by wdoc. It is frozen when
    env.py is done.
    This allows modification of env values to dynamically affect wdoc without
    having to restart the python execution or reimporting wdoc.
    """

    # stores the name of env variable that starts by WDOC_ but are not expected by EnvDataclass, to warn user only once
    __warned_unexpected__: list = field(default_factory=list)
    __frozen__: bool = False
    WDOC_DUMMY_ENV_VAR: bool = False  # used to test the __frozen__ mechanism
    WDOC_DEBUG: bool = False
    WDOC_VERBOSE: bool = False
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
    WDOC_LLM_REQUEST_TIMEOUT: int = 600
    WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE: int = 2000
    WDOC_MAX_CHUNK_SIZE: int = 16_000
    WDOC_MAX_EMBED_CONTEXT: int = 7_000
    WDOC_INTERMEDIATE_ANSWER_MAX_TOKENS: int = 4000
    WDOC_DEFAULT_MODEL: str = "openrouter/google/gemini-2.5-pro-preview"
    WDOC_DEFAULT_EMBED_MODEL: str = "openai/text-embedding-3-small"
    WDOC_DEFAULT_EMBED_DIMENSION: Optional[int] = None
    WDOC_EMBED_TESTING: bool = True
    WDOC_DISABLE_EMBEDDINGS_CACHE: bool = False
    WDOC_DEFAULT_QUERY_EVAL_MODEL: str = "openrouter/google/gemini-2.0-flash-001"
    WDOC_LANGFUSE_PUBLIC_KEY: Optional[str] = None
    WDOC_LANGFUSE_SECRET_KEY: Optional[str] = None
    WDOC_LANGFUSE_HOST: Optional[str] = None
    WDOC_LITELLM_TAGS: Optional[str] = None
    WDOC_LITELLM_USER: str = "wdoc_llm"
    WDOC_APPLY_ASYNCIO_PATCH: bool = False
    WDOC_CONTINUE_ON_INVALID_EVAL: bool = True

    @warn_typecheck
    def __parse__(self, val: str) -> Optional[Union[bool, int, str]]:
        """
        Parse a string value from environment variables into appropriate Python types.

        This method converts string values to their corresponding Python types:
        - "true" (case-insensitive) → True (boolean)
        - "false" (case-insensitive) → False (boolean)
        - String of digits → int
        - "none" (case-insensitive) or empty string → None
        - Any other string remains a string

        Args:
            val: The string value to parse

        Returns:
            The parsed value as bool, int, None, or string
        """
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

    def __check_unexpected_vars__(self) -> None:
        """
        Look for env variables that start by WDOC_ but are not defined in
        EnvDataclass. This would indicate an error in env handling. The message
        is only printed once per var.
        """
        for k in os.environ.keys():
            if not k.lower().startswith("wdoc_"):
                continue
            if (
                k not in env.__dataclass_fields__.keys()
                and k not in self.__warned_unexpected__
            ):
                self.__warned_unexpected__.append(k)
                logger.debug(
                    f"Unexpected key env variable starting by 'wdoc_': {k}. This might be a typo in your configuration!"
                )

    def __setattr__(self, name, value):
        """
        Controls attribute assignment for the EnvDataclass.

        This method enforces the frozen state of the class once it's been frozen:
        - Prevents attempts to unfreeze the instance
        - Allows normal attribute setting only when not frozen
        - Raises an exception when trying to set attributes on a frozen instance

        Args:
            name: The attribute name being set
            value: The value to assign to the attribute

        Raises:
            Exception: If attempting to unfreeze a frozen instance
            FrozenAttributeCantBeSet: If attempting to set any attribute on a frozen instance
        """
        # dont allow unfreezing
        if name == "__frozen__" and self.__frozen__ is True and value is False:
            raise Exception("Cannot unfreeze the frozen EnvDataclass instance")

        # allow setting variable values only until frozen
        if self.__frozen__ is not True:
            return super().__setattr__(name, value)

        raise FrozenAttributeCantBeSet(name, value)

    def __getattribute__(self, name):
        """
        Controls attribute access for the EnvDataclass.

        This method implements a dynamic environment variable synchronization system:
        - For special attributes, returns them directly
        - For normal attributes in a non-frozen state, returns the current value
        - For attributes in a frozen state, checks the environment variables for runtime changes
        - Enforces type safety for all values
        - Has special handling for attributes containing 'private' for security

        Args:
            name: The attribute name being accessed

        Returns:
            The attribute value, possibly updated from environment variables

        Raises:
            Exception: If there's an error getting the attribute from the class
            AttributeError: If trying to access a security-sensitive attribute that has changed
            AssertionError: If a value doesn't conform to its expected type
        """
        # non WDOC env can be gotten right away
        if name in [
            "__dataclass_fields__",
            "__frozen__",
            "__parse__",
            "__warned_unexpected__",
            "__check_unexpected_vars__",
            "__doc__",
            "__class__",
        ]:
            return super().__getattribute__(name)
        elif name.startswith("__") and name.ends_with("__"):
            logger.debug(f"Unexpected attribute of EnvDataclass was accessed: '{name}'")
            return super().__getattribute__(name)
        self.__check_unexpected_vars__()

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
    if k.startswith("WDOC_"):
        assert is_bearable(v.default, v.type), v


# add the actual documentation of each env var to the __doc__ of EnvDataclass
help_content = (Path(__file__).parent / Path("../docs/help.md")).read_text()
env_list = [
    e for e in dir(EnvDataclass) if e.startswith("WDOC_") and e != "WDOC_DUMMY_ENV_VAR"
]
# check that it's properly documented to begin with
for e in env_list:
    if not f"* `{e}`" in help_content:
        logger.error(
            f"The env variable '{e}' seems to be missing from the help.md page"
        )
help_sections = help_content.split("# Environment variables")
assert len(help_sections) == 2
doc = EnvDataclass.__doc__
indentation = len(doc.splitlines(keepends=True)[0].rstrip()) - len(
    doc.splitlines(keepends=True)[0].strip()
)
EnvDataclass.__doc__ = dedent(EnvDataclass.__doc__)
EnvDataclass.__doc__ += (
    f"\n\n## Documentation of each environment variables:\n{help_sections[1]}"
)
EnvDataclass.__doc__ = indent(EnvDataclass.__doc__, indentation * " ")

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


# if --debug -d --verbose or -v are in the command line: we set WDOC_DEBUG and WDOC_VERBOSE accordingly
def check_kwargs(arg: str, abbrv: str = None) -> bool:
    cmdline = " ".join(sys.argv[1:])
    if f" {arg}" in cmdline or f" --{arg}" in cmdline:
        return True
    if abbrv and f" -{abbrv}" in cmdline:
        return True
    return False


if check_kwargs("debug", "d"):
    logger.debug("Found 'debug' arg, setting WDOC_DEBUG and WDOC_VERBOSE to true")
    os.environ["WDOC_DEBUG"] = "true"
    os.environ["WDOC_VERBOSE"] = "true"
elif check_kwargs("verbose", "v"):
    logger.debug("Found 'verbose' arg, setting WDOC_VERBOSE to true")
    os.environ["WDOC_VERBOSE"] = "true"


# store the env variable instead of the default values but check their types
for k in os.environ.keys():
    if not k.lower().startswith("wdoc_"):
        continue
    v = env.__parse__(os.environ[k])

    if k not in env.__dataclass_fields__.keys():
        if k not in env.__warned_unexpected__:
            env.__warned_unexpected__.append(k)
            logger.debug(
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
