"""
First file to be imported when import wdoc by __main__
It's purpose is to detect wether we should use import tricks like
lazyloading or threaded imports
"""

import sys

from .env import env


def trick_imports() -> None:
    import threading
    from queue import Queue

    import lazy_import

    def import_worker(q: Queue):
        while True:
            module = q.get()
            if module is None:
                return
            # print(f"Importing {module}")
            if "." in module:
                first = ".".join(module.split(".")[:-1])
                last = module.split(".")[-1]
                exec(f"from {first} import {last}")
            else:
                first = module
                exec(f"import {module}")

            assert first in sys.modules, f"Error when importing '{first}'"
            if "Lazily-loaded" in str(sys.modules[first]):
                # print(f"Module is lazy loaded so far: {first}")
                try:
                    dir(sys.modules[first])
                except Exception as e:
                    print(
                        f"Error when unlazyloading module '{first}'. Error: '{e}'"
                        "\nYou can try setting the env variable "
                        "WDOC_DISABLE_LAZYLOADING to 'true'"
                        "Don't hesitate to open an issue!"
                    )
                # print(f"Unlazyloaded module: {first}")

    if env.WDOC_IMPORT_TYPE == "thread":
        q = Queue()
        thread = threading.Thread(
            target=import_worker,
            args=(q,),
            daemon=False,
        )
        thread.start()

    def custom_loading(module: str, strict: bool = True) -> None:
        """
        Bunch of import tricks. The strict parameter allows to specify module
        that can be tricky to install or import.
        """
        if not strict:
            try:
                custom_loading(module=module)
            except Exception:
                return
        if env.WDOC_IMPORT_TYPE == "both":
            lazy_import.lazy_module(module)
            q.put(module)
        elif env.WDOC_IMPORT_TYPE == "thread":
            q.put(module)
        elif env.WDOC_IMPORT_TYPE == "lazy":
            if "module" in sys.modules:
                print(f"{module} already imported")
            lazy_import.lazy_module(module)
            assert module in sys.modules, module
        else:
            raise ValueError(
                f"Unexpected value for WDOC_IMPORT_TYPE: '{env.WDOC_IMPORT_TYPE}'"
            )

    custom_loading("langchain")
    custom_loading("langchain_community")
    custom_loading("langchain.text_splitter")
    custom_loading("langfuse")
    custom_loading("litellm")
    custom_loading("numpy")
    custom_loading("faiss")
    custom_loading("zlib")
    custom_loading("dill")
    custom_loading("sqlite3")
    custom_loading("tldextract")
    custom_loading("pyfiglet")
    custom_loading("yt_dlp", strict=False)
    custom_loading("pandas")
    custom_loading("ankipandas")
    custom_loading("ftfy")
    custom_loading("bs4")
    custom_loading("goose3")
    custom_loading("LogseqMarkdownParser")
    custom_loading("deepgram")
    custom_loading("pydub")
    custom_loading("ffmpeg")
    custom_loading("torchaudio", strict=False)
    custom_loading("playwright.sync_api")
    custom_loading("openparse")
    custom_loading("scipy")
    custom_loading("sklearn.metrics")
    custom_loading("sklearn.decomposition")
    custom_loading("sklearn.preprocessing")

    if env.WDOC_IMPORT_TYPE in ["both", "thread"]:
        q.put(None)  # kill the import worker


if env.WDOC_IMPORT_TYPE != "native":
    trick_imports()
