"""
Code related to loading the LLM instance, with an appropriate price
counting callback.
"""

import os

import litellm
from beartype.typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.misc import ModelName, get_model_max_tokens, langfuse_callback_holder
from wdoc.utils.typechecker import optional_typecheck

TESTING_LLM = "testing/testing"

# lorem ipsum is output by TESTING_LLM
LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing "
    "elit, sed do eiusmod tempor incididunt ut labore et "
    "dolore magna aliqua. Ut enim ad minim veniam, quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip "
    "ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu "
    "fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
    "non proident, sunt in culpa qui officia deserunt mollit "
    "anim id est laborum."
)


@optional_typecheck
def load_llm(
    modelname: ModelName,
    llm_verbosity: bool,
    llm_cache: Union[None, bool, BaseCache],
    api_base: Optional[str],
    private: bool,
    tags: List[str],
    **extra_model_args,
) -> Union[ChatLiteLLM, FakeListChatModel]:
    """load language model"""
    if extra_model_args is None:
        extra_model_args = {}

    # setup tags for langfuse or others
    if tags is None:
        tags = []
    if "wdoc" not in tags:
        tags.append("wdoc")
    if "tags" in extra_model_args:
        assert isinstance(extra_model_args["tags"], list)
        for t in extra_model_args["tags"]:
            if t not in tags:
                tags.append(t)

    env_tags = env.WDOC_LITELLM_TAGS
    if env_tags:
        if isinstance(env_tags, str):
            env_tags = env_tags.split(",")
        assert isinstance(env_tags, list), env_tags
        tags.extend(env_tags)

    assert "cache" not in extra_model_args
    if modelname.is_testing():
        assert modelname.original == "testing/testing"
        if llm_verbosity:
            logger.info("Loading a fake LLM using the testing/ backend")
        llm = FakeListChatModel(
            verbose=llm_verbosity,
            responses=[f"Fake answer n°{i}: {LOREM_IPSUM}" for i in range(1, 100)],
            callbacks=[PriceCountingCallback(verbose=llm_verbosity)]
            + langfuse_callback_holder,
            disable_streaming=True,  # Not needed and might break cache
            cache=False,
            **extra_model_args,
        )
        return llm
    else:
        assert "testing" not in modelname.original.lower()

    logger.debug("Loading model via litellm")

    if private:
        assert api_base, "If private is set, api_base must be set too"
        assert env.WDOC_PRIVATE_MODE
        assert (
            "WDOC_PRIVATE_MODE" in os.environ
        ), "Missing env variable WDOC_PRIVATE_MODE"
        assert (
            os.environ["WDOC_PRIVATE_MODE"] == "true"
        ), "Wrong value for env variable WDOC_PRIVATE_MODE"
    if api_base:
        logger.warning(f"Will use custom api_base {api_base}")
    if not (
        f"{modelname.backend.upper()}_API_KEY" in os.environ
        and os.environ[f"{modelname.backend.upper()}_API_KEY"]
    ):
        if modelname.backend == "ollama":
            logger.debug(
                "No OLLAMA_API_KEY env var found but ignoring because using ollama"
            )
            pass
        elif not api_base:
            raise Exception(
                f"No environment variable named {modelname.backend.upper()}_API_KEY found"
            )
        else:
            logger.debug(
                f"No environment variable named {modelname.backend.upper()}_API_KEY found. Continuing because some setups are fine with this."
            )

    # extra check for private mode
    if private:
        assert os.environ["WDOC_PRIVATE_MODE"] == "true"
        logger.warning(
            f"private is on so overwriting {modelname.backend.upper()}_API_KEY from environment variables"
        )
        assert (
            os.environ[f"{modelname.backend.upper()}_API_KEY"]
            == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
        )

        assert (
            os.environ[f"{modelname.backend.upper()}_API_KEY"]
            == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
        )
        assert (
            not langfuse_callback_holder
        ), "Private argument but langfuse_handler appears set. Something went wrong so crashing just to be safe."
    else:
        assert not env.WDOC_PRIVATE_MODE
        assert (
            "WDOC_PRIVATE_MODE" not in os.environ
            or os.environ["WDOC_PRIVATE_MODE"] == "false"
        )

    assert private == env.WDOC_PRIVATE_MODE

    max_tokens = get_model_max_tokens(modelname)
    logger.debug(f"Detected max token for model {modelname.original}: {max_tokens}")
    if "max_tokens" not in extra_model_args:
        # intentionaly limiting max tokens because it can cause bugs
        if modelname.backend != "ollama":
            extra_model_args["max_tokens"] = int(max_tokens * 0.9)
        else:
            if max_tokens <= 10_000:
                extra_model_args["max_tokens"] = int(max_tokens * 0.9)
            else:
                logger.debug(
                    f"Detected an ollama model with large max_tokens ({max_tokens}), they usually overestimate their context window capabilities so we reduce it if the user does not specify a max_tokens kwarg"
                )
                extra_model_args["max_tokens"] = int(max(max_tokens * 0.2, 4096))
    logger.debug(f"Using ChatLiteLLM backend for model {modelname.original}")
    llm = ChatLiteLLM(
        model_name=modelname.original,
        disable_streaming=True,  # Not needed and might break cache
        api_base=api_base,
        cache=llm_cache,
        verbose=llm_verbosity,
        tags=tags,
        callbacks=[PriceCountingCallback(verbose=llm_verbosity)]
        + langfuse_callback_holder,
        user=env.WDOC_LITELLM_USER,
        **extra_model_args,
    )
    litellm.drop_params = True

    if private:
        assert llm.api_base, "private is set but no api_base for llm were found"
        assert (
            llm.api_base == api_base
        ), "private is set but found unexpected llm.api_base value: '{litellm.api_base}'"

    # fix: the SQLiteCache's str appearance is cancelling its own cache lookup!
    # Simply turning '<wdoc.utils.customs.fix_llm_caching.SQLiteCacheFixed object at 0x72cd5a6d4050>' into '<wdoc.utils.customs.fix_llm_caching.SQLiteCacheFixed object at REDACTED>'
    if llm.cache:
        cur = str(llm.cache)
        if "object at " in cur:
            logger.debug(
                "The SQLiteCache of langchain is broken so we will modify ths __repr__ and __str__ of its class"
            )
            fixed = cur.split(" at ")[0] + "REDACTED"
            llm.cache.__class__.__repr__ = lambda _=None: fixed
            llm.cache.__class__.__str__ = lambda _=None: fixed

        # also try to make it so that setting extra model params does not reuse the cache
        if extra_model_args:
            cur_string = llm._get_llm_string()
            alt_string = llm._get_llm_string(extra_model_args)
            if cur_string != alt_string:
                logger.debug(
                    "Modifying the llm._get_llm_string because langchain broke it over a year ago..."
                )
                llm._get_llm_string = lambda *args, **kwargs: alt_string
                cur = str(llm._get_llm_string)
                if "object at " in cur:
                    fixed = cur.split(" at ")[0] + "REDACTED"
                    llm._get_llm_string.__class__.__repr__ = lambda _=None: fixed
                    llm._get_llm_string.__class__.__str__ = lambda _=None: fixed

    logger.debug(
        f"Extra model args there were used for {modelname.original}: '{extra_model_args}'"
    )
    return llm


@optional_typecheck
class PriceCountingCallback(BaseCallbackHandler):
    "source: https://python.langchain.com/docs/modules/callbacks/"

    def __init__(self, verbose, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.internal_reasoning_tokens = 0
        self.methods_called = []
        self.authorized_methods = [
            "on_llm_start",
            "on_chat_model_start",
            "on_llm_end",
            "on_llm_error",
            "on_chain_start",
            "on_chain_end",
            "on_chain_error",
        ]
        self.pbar = []

    def __repr__(self) -> str:
        # setting __repr__ and __str__ is important because it can
        # maybe be used for caching?
        return "PriceCountingCallback"

    def __str__(self) -> str:
        return "PriceCountingCallback"

    def _check_methods_called(self) -> bool:
        assert all(
            meth in dir(self) for meth in self.methods_called
        ), "unexpected method names!"
        wrong = [
            meth for meth in self.methods_called if meth not in self.authorized_methods
        ]
        if wrong:
            raise Exception(f"Unauthorized_method were called: {','.join(wrong)}")
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.methods_called.append("on_llm_start")
        if self.verbose:
            logger.debug(serialized)
            logger.debug(prompts)
            logger.debug(kwargs)
        self._check_methods_called()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self.methods_called.append("on_chat_model_start")
        if self.verbose:
            logger.debug(serialized)
            logger.debug(messages)
            logger.debug(kwargs)
        self._check_methods_called()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.methods_called.append("on_llm_end")
        if self.verbose:
            logger.debug(response)
            logger.debug(kwargs)

        if response.llm_output is None or response.llm_output["token_usage"] is None:
            if self.verbose:
                logger.debug("None llm_output, returning.")
            return

        new_p = response.llm_output["token_usage"]["prompt_tokens"]
        new_c = response.llm_output["token_usage"]["completion_tokens"]
        new_r = response.llm_output["token_usage"]["total_tokens"] - new_p - new_c
        self.prompt_tokens += new_p
        self.completion_tokens += new_c
        self.total_tokens += new_p + new_c + new_r
        assert (
            self.total_tokens
            == self.prompt_tokens
            + self.completion_tokens
            + self.internal_reasoning_tokens
        )
        self._check_methods_called()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.methods_called.append("on_llm_error")
        if self.verbose:
            logger.debug(error)
            logger.debug(kwargs)
        self._check_methods_called()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        if self.verbose:
            logger.debug(serialized)
            logger.debug(inputs)
            logger.debug(kwargs)
        self.methods_called.append("on_chain_start")
        self._check_methods_called()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        if self.verbose:
            logger.debug(outputs)
            logger.debug(kwargs)
        self.methods_called.append("on_chain_end")
        self._check_methods_called()
        if self.pbar:
            self.pbar[-1].update(1)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        if self.verbose:
            logger.debug(error)
            logger.debug(kwargs)
        self.methods_called.append("on_chain_error")
        self._check_methods_called()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.methods_called.append("on_llm_new_token")
        self._check_methods_called()
        raise NotImplementedError("Not expecting streaming")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.methods_called.append("on_tool_start")
        self._check_methods_called()
        raise NotImplementedError("Not expecting tool call")

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.methods_called.append("on_tool_end")
        self._check_methods_called()
        raise NotImplementedError("Not expecting tool call")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.methods_called.append("on_tool_error")
        self._check_methods_called()
        raise NotImplementedError("Not expecting tool call")

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        self.methods_called.append("on_text")
        self._check_methods_called()
        raise NotImplementedError("Not expecting to call self.on_text")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.methods_called.append("on_agent_action")
        self._check_methods_called()
        raise NotImplementedError("Not expecting agent call")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.methods_called.append("on_agent_finish")
        self._check_methods_called()
        raise NotImplementedError("Not expecting agent call")
