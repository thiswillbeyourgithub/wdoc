import os
from pathlib import Path

import langchain
from langchain.llms import GPT4All, FakeListLLM, LlamaCpp
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.cache import SQLiteCache
from langchain.memory import ConversationBufferMemory

from typing import Dict, Any

from .logger import whi, yel, red

langchain.llm_cache = SQLiteCache(database_path=".cache/langchain.db")

class AnswerConversationBufferMemory(ConversationBufferMemory):
    """
    quick fix from https://github.com/hwchase17/langchain/issues/5630
    """
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})


def load_llm(model, local_llm_path):
    """load language model"""
    if model.lower() == "openai":
        whi("Loading openai models")
        assert Path("API_KEY.txt").exists(), "No api key found"
        os.environ["OPENAI_API_KEY"] = str(Path("API_KEY.txt").read_text()).strip()

        llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                verbose=True,
                )
        callback = get_openai_callback

    elif model.lower() == "llama":
        whi("Loading llama models")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
                model_path=local_llm_path,
                callback_manager=callback_manager,
                verbose=True,
                n_threads=4,
                )
        callback = fakecallback

    elif model.lower() == "gpt4all":
        whi(f"loading gpt4all: '{local_llm_path}'")
        local_llm_path = Path(local_llm_path)
        assert local_llm_path.exists(), "local model not found"
        callbacks = [StreamingStdOutCallbackHandler()]
        # Verbose is required to pass to the callback manager
        llm = GPT4All(
                model=str(local_llm_path.absolute()),
                n_ctx=512,
                n_threads=4,
                callbacks=callbacks,
                verbose=True,
                streaming=True,
                )
        callback = fakecallback
    elif model.lower() in ["fake", "test", "testing"]:
        llm = FakeListLLM(verbose=True, responses=[f"Fake answer n°{i}" for i in range(1, 100)])
        callback = fakecallback
    else:
        raise ValueError(model)

    whi("done loading model.\n")
    return llm, callback


class fakecallback:
    """used by gpt4all to avoid bugs"""
    total_tokens = 0
    total_cost = 0
    args = None
    kwds = None
    func = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __str__(self):
        pass
