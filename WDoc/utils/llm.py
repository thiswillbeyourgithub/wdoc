"""
Code related to loading the LLM instance, with an appropriate price
counting callback.
"""

from typing import Union, List, Any, Optional
import os
from typing import Dict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.chat_models import ChatLiteLLM
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache
import litellm

from .logger import whi, red, yel
from .typechecker import optional_typecheck
from .flags import is_verbose
from .env import WDOC_PRIVATE_MODE

TESTING_LLM = "testing/testing"


@optional_typecheck
def load_llm(
    modelname: str,
    backend: str,
    llm_verbosity: bool,
    llm_cache: Union[None, bool, BaseCache],
    api_base: Optional[str],
    private: bool,
    **extra_model_args,
) -> Union[ChatLiteLLM, ChatOpenAI, FakeListChatModel]:
    """load language model"""
    if extra_model_args is None:
        extra_model_args = {}
    assert "cache" not in extra_model_args
    if backend == "testing":
        assert modelname == "testing/testing"
        if llm_verbosity:
            whi("Loading a fake LLM using the testing/ backend")
        lorem_ipsum = (
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
        llm = FakeListChatModel(
            verbose=llm_verbosity,
            responses=[f"Fake answer n°{i}: {lorem_ipsum}" for i in range(1, 100)],
            callbacks=[PriceCountingCallback(verbose=llm_verbosity)],
            disable_streaming=True,  # Not needed and might break cache
            cache=False,
            **extra_model_args,
        )
        return llm
    else:
        assert "testing" not in modelname.lower()

    if is_verbose:
        whi("Loading model via litellm")

    if private:
        assert api_base, "If private is set, api_base must be set too"
        assert WDOC_PRIVATE_MODE
        assert "WDOC_PRIVATE_MODE" in os.environ, "Missing env variable WDOC_PRIVATE_MODE"
        assert os.environ["WDOC_PRIVATE_MODE"] == "true", "Wrong value for env variable WDOC_PRIVATE_MODE"
    if api_base:
        red(f"Will use custom api_base {api_base}")
    if not (f"{backend.upper()}_API_KEY" in os.environ and os.environ[f"{backend.upper()}_API_KEY"]):
        if not api_base:
            raise Exception(
                f"No environment variable named {backend.upper()}_API_KEY found")
        else:
            yel(f"No environment variable named {backend.upper()}_API_KEY found. Continuing because some setups are fine with this.")

    # extra check for private mode
    if private:
        assert os.environ["WDOC_PRIVATE_MODE"] == "true"
        red(
            f"private is on so overwriting {backend.upper()}_API_KEY from environment variables")
        assert os.environ[f"{backend.upper()}_API_KEY"] == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
    else:
        assert not WDOC_PRIVATE_MODE
        assert "WDOC_PRIVATE_MODE" not in os.environ or os.environ["WDOC_PRIVATE_MODE"] == "false"

    if not private and backend == "openai" and api_base is None:
        max_tokens = litellm.get_model_info(modelname)["max_tokens"]
        if "max_tokens" not in extra_model_args:
            extra_model_args["max_tokens"] = int(max_tokens * 0.9)
        llm = ChatOpenAI(
            model_name=modelname.split("/", 1)[1],
            cache=llm_cache,
            disable_streaming=True,  # Not needed and might break cache
            verbose=llm_verbosity,
            callbacks=[PriceCountingCallback(verbose=llm_verbosity)],
            **extra_model_args,
        )
    else:
        max_tokens = litellm.get_model_info(modelname)["max_tokens"]
        if "max_tokens" not in extra_model_args:
            extra_model_args["max_tokens"] = int(max_tokens * 0.9)  # intentionaly limiting max tokens because it can cause bugs
        llm = ChatLiteLLM(
            model_name=modelname,
            disable_streaming=True,  # Not needed and might break cache
            api_base=api_base,
            cache=llm_cache,
            verbose=llm_verbosity,
            tags=["wdoc"],
            callbacks=[PriceCountingCallback(verbose=llm_verbosity)],
            **extra_model_args,
        )
        litellm.drop_params = True
    if private:
        assert llm.api_base, "private is set but no api_base for llm were found"
        assert llm.api_base == api_base, "private is set but found unexpected llm.api_base value: '{litellm.api_base}'"

    # fix: the SQLiteCache's str appearance is cancelling its own cache lookup!
    if llm.cache:
        cur = str(llm.cache)
        llm.cache.__class__.__repr__ = lambda x=None: cur.split(" at ")[0]
        llm.cache.__class__.__str__ = lambda x=None: cur.split(" at ")[0]
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
        assert all(meth in dir(self) for meth in self.methods_called), (
            "unexpected method names!")
        wrong = [
            meth for meth in self.methods_called
            if meth not in self.authorized_methods]
        if wrong:
            raise Exception(
                f"Unauthorized_method were called: {','.join(wrong)}")
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.methods_called.append("on_llm_start")
        if self.verbose:
            yel("Callback method: on_llm_start")
            yel(serialized)
            yel(prompts)
            yel(kwargs)
            yel("Callback method end: on_llm_start")
        self._check_methods_called()

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        self.methods_called.append("on_chat_model_start")
        if self.verbose:
            yel("Callback method: on_chat_model_start")
            yel(serialized)
            yel(messages)
            yel(kwargs)
            yel("Callback method end: on_chat_model_start")
        self._check_methods_called()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.methods_called.append("on_llm_end")
        if self.verbose:
            yel("Callback method: on_llm_end")
            yel(response)
            yel(kwargs)
            yel("Callback method end: on_llm_end")

        if response.llm_output is None or response.llm_output["token_usage"] is None:
            if self.verbose:
                yel("None llm_output, returning.")
            return

        new_p = response.llm_output["token_usage"]["prompt_tokens"]
        new_c = response.llm_output["token_usage"]["completion_tokens"]
        self.prompt_tokens += new_p
        self.completion_tokens += new_c
        self.total_tokens += new_p + new_c
        assert self.total_tokens == self.prompt_tokens + self.completion_tokens
        self._check_methods_called()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.methods_called.append("on_llm_error")
        if self.verbose:
            yel("Callback method: on_llm_error")
            yel(error)
            yel(kwargs)
            yel("Callback method end: on_llm_error")
        self._check_methods_called()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        if self.verbose:
            yel("Callback method: on_chain_start")
            yel(serialized)
            yel(inputs)
            yel(kwargs)
            yel("Callback method end: on_chain_start")
        self.methods_called.append("on_chain_start")
        self._check_methods_called()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        if self.verbose:
            yel("Callback method: on_chain_end")
            yel(outputs)
            yel(kwargs)
            yel("Callback method end: on_chain_end")
        self.methods_called.append("on_chain_end")
        self._check_methods_called()
        if self.pbar:
            self.pbar[-1].update(1)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        if self.verbose:
            yel("Callback method: on_chain_error")
            yel(error)
            yel(kwargs)
            yel("Callback method end: on_chain_error")
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
