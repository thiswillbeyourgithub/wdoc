import time
import os
from pathlib import Path
from typing import Dict, Any

from langchain_community.llms import FakeListLLM
from langchain_community.callbacks import get_openai_callback
# from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM
from langchain.memory import ConversationBufferMemory

import openai

from .logger import whi, yel, red

Path(".cache").mkdir(exist_ok=True)

class AnswerConversationBufferMemory(ConversationBufferMemory):
    """
    quick fix from https://github.com/hwchase17/langchain/issues/5630
    """
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})


def load_llm(modelname, backend, **extra_model_args):
    """load language model"""
    if extra_model_args is None:
        extra_model_args = {}
    if backend == "testing":
        whi("Loading testing model")
        llm = FakeListLLM(
            verbose=True,
            responses=[f"Fake answer n°{i}" for i in range(1, 100)],
            **extra_model_args,
        )
        callback = fakecallback
        return llm, callback

    whi("Loading model via litellm")
    if not (f"{backend.upper()}_API_KEY" in os.environ or os.environ[f"{backend.upper()}_API_KEY"]):
        assert Path(f"{backend.upper()}_API_KEY.txt").exists(), f"No api key found for {backend} via litellm"
        os.environ[f"{backend.upper()}_API_KEY"] = str(Path(f"{backend.upper()}_API_KEY.txt").read_text()).strip()

    llm = ChatLiteLLM(
            model_name=modelname,
    # llm = ChatOpenAI(
            # model_name=modelname.split("/")[-1],
            verbose=True,
            **extra_model_args,
            )
    callback = get_openai_callback
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


def transcribe(audio_path, audio_hash, language, prompt):
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
