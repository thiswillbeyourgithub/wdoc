import fire

from .DocToolsLLM import DocToolsLLM_class

def cli_launcher() -> None:
    instance = fire.Fire(DocToolsLLM_class)
