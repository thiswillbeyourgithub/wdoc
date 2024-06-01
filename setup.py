from setuptools import setup, find_packages
setup(
    name="DocToolsLLM",
    version="0.14",
    description="A perfect RAG and AI summary setup for my needs. Supports all LLM, virt. any filetypes (epub, youtube_playlist, pdf, mp3, etc)",
    long_description="A perfect RAG and AI summary setup for my needs. All LLM supported. Virtually any input filetypes including epub, youtube_playlist, pdf, etc",
    url="https://github.com/thiswillbeyourgithub/DocToolsLLM/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=["RAG", "search", "summary", "summarize", "pdf", "documents", "doc", "docx", "youtube", "mp3", "embeddings", "AI", "LLM", "openai", "logseq"],
    entry_points={
        'console_scripts': ['doctoolsllm=DocToolsLLM:cli_call'],
    },
    install_requires=[
        'sqlalchemy>=2.0.29',
        'beautifulsoup4>=4.10.0',
        'fire>=0.6.0',
        'ftfy>=6.1.1',
        'joblib>=1.2.0',
        'langchain>=0.2.1',
        'langchain-community>=0.2.1',
        'langchain-openai>=0.1.8',
        'langchain-mistralai>=0.1.7',
        'litellm>=1.38.10',
        'nltk>=3.8.1',
        'openai>=1.30.3',
        'prompt-toolkit>=3.0.43',
        'requests>=2.25.1',
        'tiktoken>=0.6.0',
        'tqdm>=4.66.4',
        'faiss-cpu>=1.8.0',
        'llama-cpp-python>=0.2.76',
        'rich>=13.7.1',
        'typeguard >= 4.2.1',
        'platformdirs >= 4.2.2',
        'dill >= 0.3.7',
        'pyfiglet >= 1.0.2',   # banner
        'rtoml >= 0.10.0',
        'grandalf >= 3.1.2',  # to print ascii graph

        # Loaders:
        'docx2txt >= 0.8',  # word documents
        'pandoc >= 2.3', # epub
        'unstructured>=0.6.2',  # many file formats
        'ankipandas>=0.3.13',  # anki
        'tldextract>=3.4.1',  # url
        'goose3 >= 3.1.16',  # url
        'youtube_dl @ git+https://github.com/ytdl-org/youtube-dl.git',  # youtube
        'LogseqMarkdownParser >= 2.5',  # logseq files (I'm the dev behind it)



    ],
    extra_require={
            'optionnal_feature': [
        # buggy in windows so optional: https://github.com/zafercavdar/fasttext-langdetect/issues/14
        'fasttext-langdetect >= 1.0.5',
        'langdetect >= 1.0.9',
        'pdftotext >= 2.2.2',  # sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
        ]
    }
)
