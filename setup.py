from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys


class PostInstallCommand(install):
    def run(self):
        install.run(self)

        # do "playwright install"
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'playwright', 'install']
            )
        except Exception as err:
            print(f"Error when installing playwright: '{err}'")

        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install',
                    '-U", "git+https://github.com/ytdl-org/youtube-dl.git'],
            )
        except Exception as err:
            print(f"Error when pip updating youtube_dl: '{err}'")


with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="wdoc",
    version="1.1.7",
    description="A perfect AI powered RAG for document query and summary. Supports ~all LLM and ~all filetypes (url, pdf, epub, youtube (incl playlist), audio, anki, md, docx, pptx, oe any combination!)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/WDoc/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=["RAG", "search", "summary", "summarize", "pdf", "documents", "doc", "docx",
              "youtube", "mp3", "embeddings", "AI", "LLM", "openai", "logseq", "doctools", "doctoolsllm", "winston_doc"],
    entry_points={
        'console_scripts': [
            'wdoc=WDoc.__init__:cli_launcher',
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        'sqlalchemy>=2.0.29',
        'beautifulsoup4>=4.10.0',
        'fire>=0.6.0',
        'ftfy>=6.1.1',
        'joblib>=1.2.0',
        'langchain>=0.2.1,<0.2.5',
        'langchain-community>=0.2.1',
        'langchain-openai>=0.1.8',
        'langchain-mistralai>=0.1.7',
        'litellm>=1.38.10',
        'nltk>=3.8.1',
        'prompt-toolkit>=3.0.43',
        'requests>=2.25.1',
        'tiktoken>=0.6.0',
        'tqdm>=4.66.4',
        'faiss-cpu>=1.8.0',
        'llama-cpp-python>=0.2.76',
        'rich>=13.7.1',
        'beartype >= 0.19.0rc0',
        'platformdirs >= 4.2.2',
        'dill >= 0.3.7',
        'pyfiglet >= 1.0.2',   # banner
        'rtoml >= 0.11.0',
        'loguru >= 0.7.2',
        'grandalf >= 0.8',  # to print ascii graph
        'lazy-import >= 0.2.2',
        'py_ankiconnect >= 0.1.0',  # DIY wrapper to tell anki to sync just in case

        # Loaders:
        'docx2txt >= 0.8',  # word documents
        'pandoc >= 2.3',  # epub
        'unstructured[all-docs]>=0.6.2',  # many file formats
        'ankipandas>=0.3.15',  # anki
        'tldextract>=3.4.1',  # url
        'goose3 >= 3.1.16',  # url
        "youtube_dl",  # youtube_dl, the latest version will try to be installed from the git repo directly using the PostInstallCommand function above
        "youtube-transcript-api >= 0.6.2",  # youtube
        "pytube >= 15.0.0",  # youtube
        'LogseqMarkdownParser >= 2.8',  # logseq files (I'm the dev behind it)
        'deepgram-sdk >= 3.2.7',  # audio transcription
        'httpx >= 0.27.0',  # to increase deepgram timeout
        'pydub >= 0.25.1',  # extracting audio from local video
        'ffmpeg-python >= 0.2.0',  # extracting audio from local video
        'torchaudio >= 2.3.1',  # silence removal from audio
        'playwright >= 1.45.0',  # for online_media and urls
    ],
    extra_require={
        'optional_feature': [
            # buggy in windows so optional: https://github.com/zafercavdar/fasttext-langdetect/issues/14
            'fasttext-langdetect >= 1.0.5',
            'langdetect >= 1.0.9',
            # sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
            'pdftotext >= 2.2.2',

            # pdf
            'pdfminer.six >= 20231228',
            "pillow_heif >= 0.16.0",
            "pypdfium2 >= 4.30.0",
            "pymupdf >= 1.24.5",
            "pdfplumber >= 0.11.1",
            "pdf2image >= 1.17.0",
        ]
    },
    cmdclass={
        'install': PostInstallCommand,
    },

)
