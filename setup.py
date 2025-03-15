import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install


class PostInstallCommand(install):
    def run(self):
        install.run(self)

        # do "python -m playwright install"
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        except Exception as err:
            print(f"Error when installing playwright: '{err}'")

        # do "python -m pip install git+https://github.com/ytdl-patched/ytdl-patched"
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/ytdl-patched/ytdl-patched",
                ]
            )
        except Exception as err:
            print(f"Error when installing youtube_dl_patched from git: '{err}'")

        # do "python -m pip install -U git+https://github.com/ytdl-org/youtube-dl.git"
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "git+https://github.com/ytdl-org/youtube-dl.git",
                ],
            )
        except Exception as err:
            print(f"Error when pip updating youtube_dl: '{err}'")

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
    version="2.8.0",
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
            "wdoc_parse_file=wdoc.__main__:cli_parse_file",
        ],
    },
    python_requires=">=3.11, <3.12",
    install_requires=[
        "sqlalchemy>=2.0.32",
        "beautifulsoup4>=4.12.3",
        "fire>=0.6.0",
        "ftfy>=6.2.0",
        "joblib>=1.4.2",
        "langchain>=0.3.1",
        "langchain-community>=0.3.1",
        "langchain-openai>=0.2.1",
        "langfuse>=2.59.3",  # for observability
        "litellm>=v1.62.1",
        "nest_asyncio>=1.6.0",  # needed to fix ollama 'event loop closed' error thanks to https://github.com/BerriAI/litellm/pull/7625/files
        "prompt-toolkit>=3.0.47",
        "tqdm>=4.66.4",
        "faiss-cpu>=1.8.0",
        "rich>=13.8.1",
        "beartype >= 0.19.0",
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
        # 'python-magic >= 0.4.27',  # for detecting file type  # made optionnal as it can help infer the filetype
        "uuid6",  # for time sortable timestamp
        "PersistDict >= 0.2.14",  # by me, like a dict but an LMDB database, to fix langchain's caches
        # Loaders:
        "docx2txt >= 0.8",  # word documents
        "pandoc >= 2.3",  # epub
        "unstructured[all-docs]>=0.14.6",  # many file formats
        "ankipandas>=0.3.15",  # anki
        "tldextract>=5.1.2",  # url
        "goose3 >= 3.1.19",  # url
        "youtube_dl",  # youtube_dl, we try to install yt_dl_patched using PostInstallCommand as it's not in pypi but we install yt_dl anyway just in case. Also the latest version will try to be installed from the git repo directly using the PostInstallCommand function above.
        "youtube-transcript-api >= 0.6.2",  # youtube
        # "pytube >= 15.0.0",  # youtube
        "yt-dlp >= 2024.11.2.232942.dev0",  # youtube
        "LogseqMarkdownParser >= 3.3",  # logseq files (I'm the dev behind it)
        "deepgram-sdk >= 3.2.7",  # audio transcription
        "httpx >= 0.27.0",  # to increase deepgram timeout
        "pydub >= 0.25.1",  # extracting audio from local video
        "ffmpeg-python >= 0.2.0",  # extracting audio from local video
        "torchaudio >= 2.3.1",  # silence removal from audio
        "playwright >= 1.45.0",  # for online_media and urls
        # pdf
        "pdfminer.six >= 20231228",
        "pillow_heif >= 0.16.0",
        "pypdfium2 >= 4.30.0",
        "pymupdf >= 1.24.5",
        "pdfplumber >= 0.11.1",
        "pdf2image >= 1.17.0",
        "openparse[ml] >= 0.5.7",  # pdf with table support
    ],
    extra_require={
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
            "build",
            "twine",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
)
