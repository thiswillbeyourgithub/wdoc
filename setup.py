import shutil
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install


class PostInstallCommand(install):
    def run(self):
        install.run(self)

        pip = ["uv", "pip"] if shutil.which("uv") else ["pip"]

        # do "python -m playwright install"
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        except Exception as err:
            print(f"Error when installing playwright: '{err}'")

        # do pip install --user -U --pre yt-dlp
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                ]
                + pip
                + [
                    "install",
                    "--user",
                    "-U",
                    "--pre",
                    "yt-dlp",
                ]
            )
        except Exception as err:
            print(f"Error when installing yt-dlp pre-release: '{err}'")

        # do "python -m pip install -U git+https://github.com/ahupp/python-magic/
        # see https://github.com/ahupp/python-magic/issues/261
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                ]
                + pip
                + [
                    "install",
                    "-U",
                    "git+https://github.com/ahupp/python-magic/",
                ],
            )
        except Exception as err:
            print(f"Error when pip updating python-magic from git: '{err}'")

        # do "openparse-download"
        try:
            subprocess.check_call(
                ["openparse-download"],
            )
        except Exception as err:
            print(
                "Error when trying to run 'openparse-download' to download"
                f" weights for deep learning based table detection : '{err}'"
                "\nBy default wdoc still uses pymupdf via openparse so it "
                "shouldn't matter too much.\n"
                "For more: see https://github.com/Filimoa/open-parse/"
            )

        # do "import nltk ; nltk.download('punkt_tab')"
        try:
            import nltk

            nltk.download("punkt_tab")
        except Exception as err:
            print(f"Error when downloading nltk punkt_tab: '{err}'")


with open("README.md", "r") as readme:
    long_description = readme.read()
    assert (
        '<p align="center"><img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true" width="512" style="background-color: transparent !important"></p>'
        in long_description
    ), "Unexpected HTML for the icon"
    long_description = long_description.replace(
        '<p align="center"><img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true" width="512" style="background-color: transparent !important"></p>',
        "![icon](https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true)",
    )
    assert 'align="center"' not in long_description

setup(
    name="wdoc",
    version="4.0.2",
    description="A perfect AI powered RAG for document query and summary. Supports ~all LLM and ~all filetypes (url, pdf, epub, youtube (incl playlist), audio, anki, md, docx, pptx, oe any combination!)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/wdoc/",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=[
        "RAG",
        "search",
        "summary",
        "summarize",
        "pdf",
        "documents",
        "doc",
        "docx",
        "youtube",
        "mp3",
        "embeddings",
        "AI",
        "LLM",
        "openai",
        "logseq",
        "doctools",
        "doctoolsllm",
        "winston_doc",
    ],
    entry_points={
        "console_scripts": [
            "wdoc=wdoc.__main__:cli_launcher",
        ],
    },
    python_requires=">=3.11, <=3.12",
    install_requires=[
        "sqlalchemy>=2.0.32",
        "beautifulsoup4>=4.12.3",
        "fire>=0.6.0",
        "ftfy>=6.2.0",
        "joblib>=1.4.2",
        "langchain==0.3.27",
        "langchain-community==0.3.30",
        "langchain-openai==0.3.34",
        "langchain-litellm==0.2.3",
        "langfuse>=3.6.1",  # for observability
        "litellm==v1.77.7",  # Bound because of incompatibility with langchain-litellm https://github.com/Akshay-Dongare/langchain-litellm/issues/18
        "nest_asyncio>=1.6.0",  # needed to fix ollama 'event loop closed' error thanks to https://github.com/BerriAI/litellm/pull/7625/files
        "prompt-toolkit>=3.0.47",
        "tqdm>=4.66.4",
        "faiss-cpu>=1.8.0",
        "rich>=13.8.1",
        "beartype >= 0.22.2",
        "platformdirs >= 4.2.2",
        "dill >= 0.3.8",
        "pyfiglet >= 1.0.2",  # banner
        "rtoml >= 0.11.0",
        "loguru >= 0.7.2",
        "grandalf >= 0.8",  # to print ascii graph
        "lazy-import >= 0.2.2",
        "py_ankiconnect >= 1.1.2",  # DIY wrapper to tell anki to sync just in case
        "scikit-learn >= 1.5.1",  # for semantic reordering
        "scipy >= 1.13.1",  # for semantic reordering
        # 'python-magic >= 0.4.27',  # for detecting file type  # made optional as it can help infer the filetype, and 0.4.28 is necessary for the pipe feature.
        "uuid6 >= 2025.0.1",  # for time sortable timestamp
        "PersistDict >= 0.2.14",  # by me, like a dict but an LMDB database, to fix langchain's caches
        "nltk>=3.9.2",  # needed for punkt_tab download in post-install
        "pandas >= 2.3.3",
        # some loaders are included by default:
        "playwright >= 1.45.0",  # for online_media and urls
        "openparse[ml] >= 0.5.7",  # pdf with table support
        # youtube
        "yt-dlp >= 2025.09.26",  # we actually need to install yt-dlp here otherwise readthedocs crashes. Note that in the postinstall script above it will be reinstalled using the master branch
        "youtube-transcript-api >= 0.6.2",
        # "pytube >= 15.0.0",
        # url
        "tldextract>=5.1.2",
        "goose3 >= 3.1.20",
        # online search via 'filetype=web'
        "ddgs >= 9.6.0",
        "duckduckgo-search >= 8.1.1",
        # audio/video transcription
        "deepgram-sdk >= 3.2.7",
        "httpx >= 0.27.0",  # to increase deepgram timeout
        "pydub >= 0.25.1",  # extracting audio from local video
        "audioop-lts >= 1.0.0; python_version >= '3.13'",  # audioop replacement for Python 3.13+, needed by pydub. See https://github.com/jiaaro/pydub/issues/815
        "ffmpeg-python >= 0.2.0",  # extracting audio from local video
        "torchaudio >= 2.8.0",  # silence removal from audio
        "trio >= 0.31.0",  # for some reason older versions of trio, when present are used and cause issues on python 3.11: https://github.com/python-trio/trio/issues/2317
        # many file formats
        "unstructured[all-docs]>=0.18.15",
    ],
    extras_require={
        "full": [
            # Loaders:
            # pdf
            "pdfminer.six >= 20231228",
            "pillow_heif >= 0.16.0",
            "pypdfium2 >= 4.30.0",
            "pymupdf >= 1.24.5",
            "pdfplumber >= 0.11.1",
            "pdf2image >= 1.17.0",
            # word documents
            "docx2txt >= 0.8",
            # epub
            "pandoc >= 2.4",
            # anki
            "ankipandas>=0.3.15",
            # logseq files (I'm the dev behind it)
            "LogseqMarkdownParser >= 3.3",
        ],
        "fasttext": [
            # buggy in windows so optional: https://github.com/zafercavdar/fasttext-langdetect/issues/14
            "fasttext-langdetect >= 1.0.5",
            "langdetect >= 1.0.9",
        ],
        "pdftotext": [
            # sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
            "pdftotext >= 2.2.2",
        ],
        "dev": [
            "black >= 25.1.0",
            # "isort >= 6.0.0",
            "pre-commit >= 4.1.0",
            "pytest >= 8.3.4",
            "pytest-xdist >= 3.6.1",
            "build >= 1.2.2.post1",
            "twine >= 6.1.0",
            "bumpver >= 2025.1131",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
)
