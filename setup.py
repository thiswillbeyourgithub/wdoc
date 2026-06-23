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

        # Only refresh yt-dlp to its pre-release if the user actually installed
        # it (i.e. picked the [youtube] extra). This keeps yt-dlp optional while
        # still letting users track YouTube extractor fixes that land in
        # pre-releases before the stable pin catches up.
        try:
            import yt_dlp  # noqa: F401

            has_yt_dlp = True
        except ImportError:
            has_yt_dlp = False
        if has_yt_dlp:
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
                        "--pre",
                        "yt-dlp",
                    ]
                )
            except Exception as err:
                print(f"Error when installing yt-dlp pre-release: '{err}'")

        # Only run "openparse-download" if openparse is actually installed.
        # openparse[ml] is in install_requires today, but guard anyway so this
        # post-install does not crash on a stripped-down install.
        try:
            import openparse  # noqa: F401

            has_openparse = True
        except ImportError:
            has_openparse = False
        if has_openparse:
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
        # Likely redundant: `unstructured` (our only nltk consumer, via
        # unstructured.nlp.tokenize) already lazily runs nltk.download("punkt_tab")
        # on first use. Kept as a safety net so the download happens at install
        # time rather than on the first parse of an office document.
        try:
            import nltk

            nltk.download("punkt_tab")
        except Exception as err:
            print(f"Error when downloading nltk punkt_tab: '{err}'")


with open("README.md", "r") as readme:
    long_description = readme.read()

    # Convert icon HTML to markdown
    assert (
        '<p align="center"><img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true" width="512" style="background-color: transparent !important"></p>'
        in long_description
    ), "Unexpected HTML for the icon"
    long_description = long_description.replace(
        '<p align="center"><img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true" width="512" style="background-color: transparent !important"></p>',
        "![icon](https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true)",
    )

    # Convert query diagram HTML to markdown
    assert (
        '<img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_query.png?raw=true" alt="Query task workflow diagram showing the flow from user inputs through Raphael the Rephraser, VectorStore, Eve the Evaluator, Anna the Answerer, and recursive combining to final output" height="400">'
        in long_description
    ), "Unexpected HTML for query diagram"
    long_description = long_description.replace(
        '<img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_query.png?raw=true" alt="Query task workflow diagram showing the flow from user inputs through Raphael the Rephraser, VectorStore, Eve the Evaluator, Anna the Answerer, and recursive combining to final output" height="400">',
        "![Query task workflow diagram showing the flow from user inputs through Raphael the Rephraser, VectorStore, Eve the Evaluator, Anna the Answerer, and recursive combining to final output](https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_query.png?raw=true)",
    )

    # Convert summary diagram HTML to markdown
    assert (
        '<img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_summary.png?raw=true" alt="Summary task workflow diagram showing the flow from user inputs through loading & chunking, Sam the Summarizer, concatenation to wdocSummary output" height="400">'
        in long_description
    ), "Unexpected HTML for summary diagram"
    long_description = long_description.replace(
        '<img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_summary.png?raw=true" alt="Summary task workflow diagram showing the flow from user inputs through loading & chunking, Sam the Summarizer, concatenation to wdocSummary output" height="400">',
        "![Summary task workflow diagram showing the flow from user inputs through loading & chunking, Sam the Summarizer, concatenation to wdocSummary output](https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_summary.png?raw=true)",
    )

    # Convert search diagram HTML to markdown
    assert (
        '<img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_search.png?raw=true" alt="Search task workflow diagram showing the flow from user inputs through Raphael the Rephraser, VectorStore, Eve the Evaluator to search output" height="400">'
        in long_description
    ), "Unexpected HTML for search diagram"
    long_description = long_description.replace(
        '<img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_search.png?raw=true" alt="Search task workflow diagram showing the flow from user inputs through Raphael the Rephraser, VectorStore, Eve the Evaluator to search output" height="400">',
        "![Search task workflow diagram showing the flow from user inputs through Raphael the Rephraser, VectorStore, Eve the Evaluator to search output](https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/diagram_search.png?raw=true)",
    )

    assert 'align="center"' not in long_description

setup(
    name="wdoc",
    version="5.1.3",
    description="A perfect AI powered RAG for document query and summary. Supports ~all LLM and ~all filetypes (url, pdf, epub, youtube (incl playlist), audio, anki, md, docx, pptx, or any combination!)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/wdoc/",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="AGPLv3",
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
    python_requires=">=3.11",
    install_requires=[
        # Core RAG engine
        "sqlalchemy>=2.0.32",
        "beautifulsoup4>=4.12.3",
        "fire>=0.6.0",
        "ftfy>=6.2.0",
        "joblib>=1.4.2",
        "langchain>=1.3.0",
        "langchain-classic>=1.0.7",
        "langchain-community>=0.4.1",
        "langchain-openai>=1.2.1",
        "langchain-litellm>=0.6.5",
        "langfuse>=3.6.1",  # for observability
        "litellm>=v1.84.0",
        "nest_asyncio>=1.6.0",  # needed to fix ollama 'event loop closed' error thanks to https://github.com/BerriAI/litellm/pull/7625/files
        "chonkie[all]>=1.4.0",  # chonkie is for the semantic embeddings
        "chonkie[semantic]>=1.4.0",
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
        "scikit-learn >= 1.5.1",  # for semantic reordering
        "scipy >= 1.13.1",  # for semantic reordering
        # 'python-magic >= 0.4.27',  # for detecting file type  # made optional as it can help infer the filetype, and 0.4.28 is necessary for the pipe feature.
        "uuid6 >= 2025.0.1",  # for time sortable timestamp
        "PersistDict >= 0.2.14",  # by me, like a dict but an LMDB database, to fix langchain's caches
        "nltk>=3.9.2",  # needed for punkt_tab download in post-install
        "blake3>=1.0.8",  # faster than sha256
        "pandas >= 2.3.3",
        "trio >= 0.31.0",  # for some reason older versions of trio, when present are used and cause issues on python 3.11: https://github.com/python-trio/trio/issues/2317
        "unstructured >= 0.18.15,<0.18.31",  # base package only, used by pdf loader for clean_extra_whitespace. The heavy [all-docs] extra is in [office].
        # PDF loading (default, since pdf is the most common filetype)
        "openparse[ml] >= 0.5.7",  # pdf with table support
        "pdfminer.six >= 20231228",
        "pillow_heif >= 0.16.0",
        "pypdfium2 >= 4.30.0",
        "pymupdf >= 1.24.5",
        "pdfplumber >= 0.11.1",
        "pdf2image >= 1.17.0",
        # URL / web loading (default, since urls are the most common filetype)
        "playwright >= 1.60.0",  # for online_media and urls
        "goose3 >= 3.1.20",
        "tldextract>=5.1.2",
        # online search via 'filetype=web'
        "ddgs >= 9.6.0",
        "duckduckgo-search >= 8.1.1",
    ],
    extras_require={
        "youtube": [
            "yt-dlp >= 2026.3.17",  # NOTE: the postinstall script reinstalls yt-dlp from the master branch
            "youtube-transcript-api >= 1.2.4",
            # "pytube >= 15.0.0",
        ],
        "audio": [
            # audio/video transcription
            "deepgram-sdk >= 3.2.7",
            "httpx >= 0.27.0",  # to increase deepgram timeout
            "pydub == 0.25.1",  # extracting audio from local video
            # audioop was removed in stdlib in Python 3.13 and pydub needs it
            # See https://github.com/jiaaro/pydub/issues/815
            "audioop-lts>=0.2.2; python_version>='3.13'",
            "ffmpeg-python == 0.2.0",  # extracting audio from local video
            "torchaudio == 2.8.0",  # silence removal from audio
        ],
        "anki": [
            "ankipandas>=0.3.15",
            "py_ankiconnect >= 1.1.2",  # DIY wrapper to tell anki to sync just in case
        ],
        "office": [
            # word, powerpoint, epub and other office formats
            "unstructured[all-docs]>=0.18.15",
            "docx2txt >= 0.8",
            "pandoc >= 2.4",  # for epub
        ],
        "logseq": [
            "LogseqMarkdownParser >= 3.3",  # I'm the dev behind it
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
        "zotero": [
            # load documents straight from a Zotero library (local API or web API)
            "pyzotero >= 1.6.0",
        ],
        "karakeep": [
            # load bookmarks straight from a Karakeep instance via its API
            "karakeep-python-api >= 1.8.0",
        ],
        "full": [
            # aggregates all loader extras (self-reference requires pip >= 21.2)
            "wdoc[youtube,audio,anki,office,logseq,zotero,karakeep]",
        ],
        "dev": [
            "ruff >= 0.14.1",
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
