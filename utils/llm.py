from typing import Union, List, Any
import time
import os
from pathlib import Path
from typing import Dict

from langchain_core.callbacks import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.llm_result import LLMResult

from .logger import whi, red
from .typechecker import optional_typecheck
from .lazy_lib_importer import lazy_import_statements, lazy_import

exec(lazy_import_statements("""
from langchain_community.llms import FakeListLLM
# from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM

import openai
"""))


Path(".cache").mkdir(exist_ok=True)

class AnswerConversationBufferMemory(ConversationBufferMemory):
    """
    quick fix from https://github.com/hwchase17/langchain/issues/5630
    """
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})


@optional_typecheck
def load_llm(
    modelname: str,
    backend: str,
    verbose: bool,
    no_cache: bool,
    **extra_model_args,
    ) -> ChatLiteLLM:
    """load language model"""
    if extra_model_args is None:
        extra_model_args = {}
    assert "cache" not in extra_model_args
    if backend == "testing":
        whi("Loading testing model")
        llm = FakeListLLM(
            verbose=verbose,
            responses=[f"Fake answer n°{i}" for i in range(1, 100)],
            callbacks=[PriceCountingCallback(verbose=verbose)],
            cache=False,
            **extra_model_args,
        )
        return llm

    whi("Loading model via litellm")
    if not (f"{backend.upper()}_API_KEY" in os.environ and os.environ[f"{backend.upper()}_API_KEY"]):
        assert Path(f"{backend.upper()}_API_KEY.txt").exists(), f"No api key found for {backend} via litellm"
        os.environ[f"{backend.upper()}_API_KEY"] = str(Path(f"{backend.upper()}_API_KEY.txt").read_text()).strip()

    # llm = ChatOpenAI(
            # model_name=modelname.split("/")[-1],
    llm = ChatLiteLLM(
            model_name=modelname,
            verbose=verbose,
            cache=not no_cache,
            callbacks=[PriceCountingCallback(verbose=verbose)],
            **extra_model_args,
            )
    return llm


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

    def _check_methods_called(self) -> None:
        assert all(meth in dir(self) for meth in self.methods_called), (
            "unexpected method names!")
        wrong = [
            meth for meth in self.methods_called
            if meth not in self.authorized_methods]
        if wrong:
            raise Exception(f"Unauthorized_method were called: {','.join(wrong)}")
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        if self.verbose:
            print("Callback method: on_llm_start")
            print(serialized)
            print(prompts)
            print(kwargs)
            print("Callback method end: on_llm_start")
        self.methods_called.append("on_llm_start")
        self._check_methods_called()

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        if self.verbose:
            print("Callback method: on_chat_model_start")
            print(serialized)
            print(messages)
            print(kwargs)
            print("Callback method end: on_chat_model_start")
        self.methods_called.append("on_chat_model_start")
        self._check_methods_called()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if self.verbose:
            print("Callback method: on_llm_end")
            print(response)
            print(kwargs)
            print("Callback method end: on_llm_end")

        new_p = response.llm_output["token_usage"]["prompt_tokens"]
        new_c = response.llm_output["token_usage"]["completion_tokens"]
        self.prompt_tokens += new_p
        self.completion_tokens += new_c
        self.total_tokens += new_p + new_c
        assert self.total_tokens == self.prompt_tokens + self.completion_tokens
        self.methods_called.append("on_llm_end")
        self._check_methods_called()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        if self.verbose:
            print("Callback method: on_llm_error")
            print(error)
            print(kwargs)
            print("Callback method end: on_llm_error")
        self.methods_called.append("on_llm_error")
        self._check_methods_called()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        if self.verbose:
            print("Callback method: on_chain_start")
            print(serialized)
            print(inputs)
            print(kwargs)
            print("Callback method end: on_chain_start")
        self.methods_called.append("on_chain_start")
        self._check_methods_called()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        if self.verbose:
            print("Callback method: on_chain_end")
            print(outputs)
            print(kwargs)
            print("Callback method end: on_chain_end")
        self.methods_called.append("on_chain_end")
        self._check_methods_called()

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        if self.verbose:
            print("Callback method: on_chain_error")
            print(error)
            print(kwargs)
            print("Callback method end: on_chain_error")
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



@optional_typecheck
def transcribe(
    audio_path: str,
    audio_hash: str,
    language: str,
    prompt: str) -> str:
    "Use whisper to transcribe an audio file"
    red(f"Calling whisper to transcribe file {audio_path}")

    assert Path("OPENAI_API_KEY.txt").exists(), "No api key found"
    os.environ["OPENAI_API_KEY"] = str(Path("OPENAI_API_KEY.txt").read_text()).strip()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    t = time.time()
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            prompt=prompt,
            language=language,
            temperature=0,
            response_format="verbose_json",
            )
    whi(f"Done transcribing {audio_path} in {int(time.time()-t)}s")
    return transcript
