import os
from pathlib import Path

from langchain.llms import GPT4All, FakeListLLM
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from .logger import whi, yel, red


def load_llm(model="gpt4all", gpt4all_model_path="./ggml-wizardLM-7B.q4_2.bin", **kwargs):
    """load the gpt model"""
    if model.lower() == "openai":
        whi("Loading openai models")
        assert Path("API_KEY.txt").exists(), "No api key found"
        os.environ["OPENAI_API_KEY"] = str(Path("API_KEY.txt").read_text()).strip()

        llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                verbose=True,
                streaming=True,
                )
        callback = get_openai_callback()
    elif model.lower() == "gpt4all":
        whi(f"loading gpt4all: '{gpt4all_model_path}'")
        gpt4all_model_path = Path(gpt4all_model_path)
        assert gpt4all_model_path.exists(), "local model not found"
        callbacks = [StreamingStdOutCallbackHandler()]
        # Verbose is required to pass to the callback manager
        llm = GPT4All(
                model=str(gpt4all_model_path.absolute()),
                n_ctx=512,
                n_threads=4,
                callbacks=callbacks,
                verbose=True,
                streaming=True,
                )
        callback = fakecallback()
    elif model.lower() in ["fake", "test", "testing"]:
        llm = FakeListLLM(verbose=True, responses=[f"Fake answer n°{i}" for i in range(1, 100)])
        callback = fakecallback()
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
