# wdoc — Comprehensive Reference

> **This document was written for wdoc v5.0.0. If you are using a different version, some arguments, defaults, or behaviors may have changed.**

This document is a complete reference for `wdoc`, covering the CLI interface and the Python API.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
  - [Basic Syntax](#basic-syntax)
  - [Shortcuts](#shortcuts)
  - [Tasks](#tasks)
  - [Global Arguments](#global-arguments)
  - [Model Arguments](#model-arguments)
  - [Embedding Arguments](#embedding-arguments)
  - [Query Arguments](#query-arguments)
  - [Summary Arguments](#summary-arguments)
  - [Filetypes](#filetypes)
  - [Recursive Filetypes](#recursive-filetypes)
  - [Loader-Specific Arguments (DocDict)](#loader-specific-arguments-docdict)
  - [Filtering Arguments](#filtering-arguments)
  - [Other Arguments](#other-arguments)
  - [Environment Variables](#environment-variables)
  - [Shell Examples](#shell-examples)
- [Python API Reference](#python-api-reference)
  - [Importing](#importing)
  - [wdoc Class](#wdoc-class)
  - [Constructor Parameters](#constructor-parameters)
  - [Public Methods](#public-methods)
  - [Public Properties](#public-properties)
  - [parse_doc Static Method](#parse_doc-static-method)
  - [Python Examples](#python-examples)

---

## Overview

`wdoc` is a RAG (Retrieval-Augmented Generation) system for summarizing, searching, and querying documents across 15+ file types. It uses LangChain and LiteLLM as backends and supports 100+ LLM providers.

## Installation

```bash
# Full install (recommended)
pip install -U wdoc[full]

# Minimal install
pip install -U wdoc

# From git branches
pip install git+https://github.com/thiswillbeyourgithub/wdoc@main[full]
pip install git+https://github.com/thiswillbeyourgithub/wdoc@dev[full]

# Optional extras
pip install -U wdoc[pdftotext]
pip install -U wdoc[fasttext]
```

Set your API key(s): `export OPENAI_API_KEY="your_key"` (or whichever provider you use).

---

## CLI Reference

### Basic Syntax

```bash
wdoc --task=TASK --path=PATH [OPTIONS]

# Shorthand (positional args are inferred):
wdoc TASK PATH [QUERY]
```

`wdoc` also accepts shell pipes: `cat file.pdf | wdoc parse --filetype=pdf`

### Shortcuts

| Shortcut | Equivalent |
|----------|-----------|
| `wdoc web "query"` | `wdoc --task=query --filetype=ddg --path="query" --query="query"` |
| `wdoc parse FILE` | Calls `wdoc.parse_doc(path=FILE)` — no LLM, just parsing |
| `wdoc query FILE` | `wdoc --task=query --path=FILE` |
| `wdoc summarize FILE` | `wdoc --task=summarize --path=FILE` |

### Tasks

| Task | Description |
|------|-------------|
| `query` | Load documents, create embeddings, then answer questions via RAG |
| `search` | Return matching documents and metadata only (no LLM answer) |
| `summarize` | Produce a detailed markdown summary of the document |
| `summarize_then_query` | Summarize first, then open a prompt for queries |

### Global Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | *required* | One of: `query`, `search`, `summarize`, `summarize_then_query` |
| `--filetype` | str | `auto` | Document type (see [Filetypes](#filetypes)) |
| `--debug` | bool | `False` | Enable tracing, increase verbosity, disable multithreading |
| `--verbose` | bool | `False` | Increase verbosity |
| `--llm_verbosity` | bool | `False` | Print LLM intermediate reasoning steps |
| `--dollar_limit` | int | `5` | Stop if estimated cost exceeds this (summaries/embeddings only) |
| `--disable_llm_cache` | bool | `False` | Disable LLM response caching |
| `--private` | bool | `False` | Enforce that no data leaves your machine |
| `--oneoff` | bool | `False` | Exit after first result (no interactive prompt) |
| `--silent` | bool | `False` | Suppress output |
| `--version` | bool | `False` | Print version and exit |
| `--out_file` | str | `None` | Write results to this file (appends) |
| `--notification_callback` | Callable | `None` | Function receiving result string (e.g. for ntfy.sh) |

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `WDOC_DEFAULT_MODEL` | Strong LLM (litellm format: `provider/model`) |
| `--model_kwargs` | dict | `None` | Extra kwargs for the model (e.g. `{"temperature": 0}`) |
| `--query_eval_model` | str | `WDOC_DEFAULT_QUERY_EVAL_MODEL` | Cheap/fast LLM for document filtering. `None` to disable |
| `--query_eval_model_kwargs` | dict | `None` | Extra kwargs for the eval model |
| `--llms_api_bases` | dict | `None` | Override API endpoints: keys in `["model", "query_eval_model", "embeddings"]` |

### Embedding Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embed_model` | str | `WDOC_DEFAULT_EMBED_MODEL` | Embedding model (`backend/model` format) |
| `--embed_model_kwargs` | dict | `None` | Extra kwargs for the embedding model |
| `--embed_instruct` | bool | `None` | Use instruct framework for HuggingFace embeddings |
| `--save_embeds_as` | str | `"{user_dir}/latest_docs_and_embeddings"` | Save embeddings to this path |
| `--load_embeds_from` | str | `None` | Load pre-computed embeddings from this path |

### Query Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | str | `None` | Initial query string |
| `--query_retrievers` | str | `"basic_multiquery"` | Retriever(s): combine `basic`, `multiquery`, `knn`, `svm`, `parent` with `_` |
| `--query_eval_check_number` | int | `3` | Number of eval passes per document |
| `--query_relevancy` | float | `-0.5` | Minimum embedding similarity score (-1 to +1) |
| `--top_k` | int\|str | `"auto_200_500"` | Documents to retrieve. `"auto_N_M"` auto-scales from N up to M |

### Summary Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--summary_n_recursion` | int | `0` | Number of recursive summary refinement passes (0 = disabled) |
| `--summary_language` | str | `"the same language as the document"` | Output language for summaries |

### Filetypes

| Filetype | Description | Key Arguments |
|----------|-------------|---------------|
| `auto` | Guess filetype from path (default) | — |
| `anki` | Anki flashcard collection | `--anki_profile`, `--anki_deck`, `--anki_notetype`, `--anki_template`, `--anki_tag_filter` |
| `epub` | EPUB e-books | `--path` |
| `json_dict` | JSON dictionary file | `--json_dict_template`, `--json_dict_exclude_keys` |
| `local_audio` | Audio files (many formats) | `--audio_backend` (`whisper`/`deepgram`), `--audio_unsilence`, `--whisper_lang`, `--whisper_prompt`, `--deepgram_kwargs` |
| `local_html` | HTML files | `--load_functions` |
| `local_video` | Video files (audio extracted) | Same as `local_audio` |
| `logseq_markdown` | Logseq markdown pages | `--path` |
| `online_media` | Remote media via yt-dlp/playwright | Same as `local_audio` + `--online_media_url_regex`, `--online_media_resourcetype_regex` |
| `online_pdf` | PDF via URL | Same as `pdf` |
| `pdf` | PDF files (15 parsers, best auto-selected) | `--pdf_parsers`, `--doccheck_min_lang_prob`, `--doccheck_min_token`, `--doccheck_max_token` |
| `powerpoint` | .ppt/.pptx/.odp | `--path` |
| `string` | Interactive text paste | — |
| `text` | Text content passed directly as path | `--metadata` |
| `txt` | Text files (.txt, .md, etc.) | `--path` |
| `url` | Web pages | `--title` |
| `word` | .doc/.docx/.odt | `--path` |
| `youtube` | YouTube videos | `--youtube_language`, `--youtube_translation`, `--youtube_audio_backend`, `--whisper_prompt`, `--whisper_lang`, `--deepgram_kwargs` |

### Recursive Filetypes

These load multiple documents and can combine different sources:

| Filetype | Description | Key Arguments |
|----------|-------------|---------------|
| `ddg` | DuckDuckGo web search | `--ddg_max_results`, `--ddg_region`, `--ddg_safesearch` |
| `json_entries` | JSON file (one dict per line) with loader args | `--path` |
| `toml_entries` | TOML file with loader args | `--path` |
| `recursive_paths` | Glob files in a directory | `--pattern`, `--recursed_filetype`, `--include`, `--exclude` |
| `link_file` | File with one URL per line | `--out_file` |
| `youtube_playlist` | YouTube playlist | Same as `youtube` |

### Loader-Specific Arguments (DocDict)

| Argument | Used By | Description |
|----------|---------|-------------|
| `--path` | Most loaders | File path, URL, or text content |
| `--pdf_parsers` | `pdf`, `online_pdf` | Comma-separated parser names (e.g. `pymupdf,pdfplumber`) |
| `--anki_profile` | `anki` | Anki profile name |
| `--anki_deck` | `anki` | Deck name prefix (e.g. `science::physics`) |
| `--anki_notetype` | `anki` | Note type filter (case-insensitive) |
| `--anki_template` | `anki` | Template string with `{fieldName}`, `{tags}`, `{allfields}`, `{image_ocr_alt}` |
| `--anki_tag_filter` | `anki` | Regex to filter cards by tag |
| `--anki_tag_render_filter` | `anki` | Regex to filter which tags appear in output |
| `--audio_backend` | `local_audio`, `local_video` | `whisper` or `deepgram` |
| `--audio_unsilence` | `local_audio`, `local_video` | Remove silence before transcribing (default: `True`) |
| `--whisper_lang` | Audio types | Language hint for Whisper |
| `--whisper_prompt` | Audio types | Prompt for Whisper |
| `--deepgram_kwargs` | Audio types | Dict of Deepgram options |
| `--youtube_language` | `youtube` | Preferred transcript languages (e.g. `["fr","en"]`) |
| `--youtube_translation` | `youtube` | Translate transcript to this language |
| `--youtube_audio_backend` | `youtube` | `youtube`, `whisper`, or `deepgram` |
| `--json_dict_template` | `json_dict` | Template with `{key}` and `{value}` |
| `--json_dict_exclude_keys` | `json_dict` | List of keys to skip |
| `--metadata` | `text`, `json_dict` | Extra metadata as JSON dict |
| `--load_functions` | `local_html` | Python callables to preprocess text |
| `--source_tag` | All | Metadata tag for document identification |
| `--loading_failure` | All | `warn` or `crash` on load errors (default: `warn`) |
| `--pattern` | `recursive_paths` | Glob pattern for file discovery |
| `--recursed_filetype` | `recursive_paths` | Filetype for each matched file |
| `--include` | `recursive_paths` | Regex list — paths must match |
| `--exclude` | `recursive_paths` | Regex list — paths must not match |
| `--online_media_url_regex` | `online_media` | Regex matching media URLs |
| `--online_media_resourcetype_regex` | `online_media` | Regex matching resource types |

### Filtering Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--filter_metadata` | list\|str | `None` | Filter docs by metadata. Format: `[kvb][+-]regex` |
| `--filter_content` | list\|str | `None` | Filter docs by content. Format: `[+-]regex` |

**`filter_metadata` syntax:**
- `k+regex` — keep docs with a metadata **key** matching regex
- `v+regex` — keep docs with a metadata **value** matching regex
- `b+key_regex:value_regex` — keep docs where key AND value match
- Use `-` instead of `+` to exclude

**`filter_content` syntax:**
- `+regex` — keep docs whose content matches
- `-regex` — exclude docs whose content matches

### Other Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--file_loader_parallel_backend` | str | `loky` | Joblib backend: `loky`, `multiprocessing`, or `threading` |
| `--file_loader_n_jobs` | int | `-1` | Parallel jobs for loading (-1 = max, 1 = serial) |
| `--doccheck_min_lang_prob` | float | `0.5` | Min fasttext language probability for valid docs |
| `--doccheck_min_token` | int | `50` | Min tokens for a valid document |
| `--doccheck_max_token` | int | `10000000` | Max tokens for a valid document |
| `--ddg_max_results` | int | `50` | Max DuckDuckGo results |
| `--ddg_region` | str | `""` | DuckDuckGo region (e.g. `us-US`) |
| `--ddg_safesearch` | str | `off` | `on`, `moderate`, or `off` |

### Environment Variables

#### Core Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_DEFAULT_MODEL` | `openrouter/google/gemini-3.1-pro-preview` | Default strong LLM |
| `WDOC_DEFAULT_QUERY_EVAL_MODEL` | `openrouter/google/gemini-2.5-flash` | Default eval LLM |
| `WDOC_DEFAULT_EMBED_MODEL` | `openai/text-embedding-3-small` | Default embedding model |
| `WDOC_DEFAULT_EMBED_DIMENSION` | `none` | Embedding dimensions to request |

#### Behavior Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_DEBUG` | `False` | Same as `--debug=True` |
| `WDOC_VERBOSE` | `False` | Same as `--verbose=True` |
| `WDOC_TYPECHECKING` | `warn` | `disabled`, `warn`, or `crash` (via beartype) |
| `WDOC_NO_MODELNAME_MATCHING` | `True` | Disable fuzzy model name matching |
| `WDOC_ALLOW_NO_PRICE` | `False` | Don't crash if model price is unknown |
| `WDOC_STRICT_DOCDICT` | `False` | `True` = crash on unexpected DocDict args, `False` = warn, `strip` = ignore |
| `WDOC_OPEN_ANKI` | `False` | Auto-open Anki browser for found cards |
| `WDOC_DEBUGGER` | `False` | Open debugger on exceptions |
| `WDOC_EMPTY_LOADER` | `False` | Return empty string for all loaders (debug) |
| `WDOC_CONTINUE_ON_INVALID_EVAL` | `True` | Continue if eval LLM output can't be parsed |
| `WDOC_BEHAVIOR_EXCL_INCL_USELESS` | `warn` | `warn` or `crash` if include/exclude has no effect |

#### Performance & Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_LLM_MAX_CONCURRENCY` | `1` | Max concurrent LLM requests |
| `WDOC_LLM_REQUEST_TIMEOUT` | `600` | LLM request timeout in seconds |
| `WDOC_MAX_CHUNK_SIZE` | `32000` | Max tokens per chunk |
| `WDOC_MAX_EMBED_CONTEXT` | `7000` | Max tokens per chunk for embeddings |
| `WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE` | `2000` | Max tokens per semantic batch |
| `WDOC_INTERMEDIATE_ANSWER_MAX_TOKENS` | `4000` | Max tokens per intermediate answer |
| `WDOC_MAX_LOADER_TIMEOUT` | `-1` | Loader timeout in seconds (-1 = disabled) |
| `WDOC_MAX_PDF_LOADER_TIMEOUT` | `-1` | Per-PDF-parser timeout in seconds (-1 = disabled) |
| `WDOC_EXPIRE_CACHE_DAYS` | `0` | Remove cache entries older than N days (0 = keep forever) |

#### FAISS / Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_MOD_FAISS_SCORE_FN` | `True` | Normalize FAISS scores to 0–1 range |
| `WDOC_FAISS_COMPRESSION` | `True` | zlib-compress FAISS indexes |
| `WDOC_FAISS_BINARY` | `False` | Use binary embeddings (32x compression) |
| `WDOC_EMBED_TESTING` | `True` | Test embedding model on startup |
| `WDOC_DISABLE_EMBEDDINGS_CACHE` | `False` | Bypass embedding cache |

#### Whisper / Audio

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_WHISPER_ENDPOINT` | `""` | Custom Whisper API endpoint |
| `WDOC_WHISPER_API_KEY` | `""` | Custom Whisper API key |
| `WDOC_WHISPER_MODEL` | `whisper-1` | Whisper model name |
| `WDOC_WHISPER_PARALLEL_SPLITS` | `True` | Parallelize split audio transcription |

#### Import & Loading

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_IMPORT_TYPE` | `native` | `native`, `thread`, `lazy`, or `both` |
| `WDOC_LOADER_LAZY_LOADING` | `True` | Lazy-import loader functions |
| `WDOC_APPLY_ASYNCIO_PATCH` | `False` | Apply nest_asyncio patch (needed for Ollama) |
| `WDOC_IN_DOCKER` | `False` | Set automatically inside Docker |
| `WDOC_PRIVATE_MODE` | — | Set automatically by `--private`, never set manually |

#### Observability (Langfuse)

| Variable | Default | Description |
|----------|---------|-------------|
| `WDOC_LANGFUSE_PUBLIC_KEY` | `None` | Overrides `LANGFUSE_PUBLIC_KEY` |
| `WDOC_LANGFUSE_SECRET_KEY` | `None` | Overrides `LANGFUSE_SECRET_KEY` |
| `WDOC_LANGFUSE_HOST` | `None` | Overrides `LANGFUSE_HOST` |
| `WDOC_LITELLM_TAGS` | `None` | Comma-separated tags for litellm requests |
| `WDOC_LITELLM_USER` | `wdoc_llm` | User identifier for litellm requests |

### Shell Examples

```bash
# Query a PDF
wdoc --task=query --path="paper.pdf" --query="What are the main findings?"

# Query multiple PDFs in a directory
wdoc --task=query --path="papers/" --pattern="**/*.pdf" \
     --filetype=recursive_paths --recursed_filetype=pdf

# Summarize a YouTube video
wdoc --task=summarize --path="https://www.youtube.com/watch?v=VIDEO_ID" \
     --youtube_language="en"

# Web search
wdoc web "latest news on quantum computing"

# Parse a document to text (no LLM)
wdoc parse document.pdf
wdoc parse document.pdf --format=langchain_dict

# Use local models (Ollama)
wdoc --model="ollama/qwen3:8b" --query_eval_model="ollama/qwen3:8b" \
     --embed_model="ollama/snowflake-arctic-embed2" \
     --task=summarize --path=document.pdf

# Save/load embeddings for repeated queries
wdoc --task=query --path="big_corpus/" --filetype=recursive_paths \
     --pattern="**/*.pdf" --recursed_filetype=pdf \
     --save_embeds_as="my_index.pkl"
wdoc --task=query --load_embeds_from="my_index.pkl" --query="My question"

# Shell pipe
cat document.pdf | wdoc parse --filetype=pdf
echo "https://example.com" | wdoc parse

# Private mode with custom endpoints
wdoc --private --model="ollama/llama3" \
     --llms_api_bases='{"model":"http://localhost:11434","query_eval_model":"http://localhost:11434","embeddings":"http://localhost:11434"}' \
     --task=query --path=secret.pdf

# Filter documents by metadata
wdoc --task=query --load_embeds_from=index.pkl \
     --filter_metadata="v+anki" --query="My question"

# Filter documents by content
wdoc --task=query --path=docs/ --filetype=recursive_paths \
     --pattern="**/*.md" --recursed_filetype=txt \
     --filter_content="+.*machine learning.*"
```

---

## Python API Reference

### Importing

```python
from wdoc import wdoc
```

### wdoc Class

The main entry point. Instantiating `wdoc` automatically loads documents and, for summary tasks, runs the summary immediately.

### Constructor Parameters

```python
wdoc(
    task: str,                        # "query", "search", "summarize", "summarize_then_query"
    filetype: str = "auto",
    model: str = WDOC_DEFAULT_MODEL,
    model_kwargs: dict | None = None,
    query_eval_model: str | None = WDOC_DEFAULT_QUERY_EVAL_MODEL,
    query_eval_model_kwargs: dict | None = None,
    embed_model: str = WDOC_DEFAULT_EMBED_MODEL,
    embed_model_kwargs: dict | None = None,
    save_embeds_as: str | Path = "{user_cache}/latest_docs_and_embeddings",
    load_embeds_from: str | Path | None = None,
    top_k: int | str = "auto_200_500",
    query: str | None = None,
    query_retrievers: str = "basic_multiquery",
    query_eval_check_number: int = 3,
    query_relevancy: float = -0.5,
    summary_n_recursion: int = 0,
    summary_language: str = "the same language as the document",
    llm_verbosity: bool = False,
    debug: bool = False,
    verbose: bool = False,
    dollar_limit: int = 5,
    notification_callback: Callable | None = None,
    disable_llm_cache: bool = False,
    file_loader_parallel_backend: str = "loky",   # "loky", "threading", "multiprocessing"
    file_loader_n_jobs: int = -1,
    private: bool = False,
    llms_api_bases: dict | None = None,
    out_file: str | Path | None = None,
    oneoff: bool = False,
    silent: bool = False,
    version: bool = False,
    **cli_kwargs,   # DocDict / loader-specific arguments (path, include, exclude, etc.)
)
```

All CLI arguments map directly to constructor parameters.

### Public Methods

#### `query_task(query: str) -> dict`

Run a RAG query against loaded documents.

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `final_answer` | str | Combined markdown answer |
| `intermediate_answers` | list | Per-document answers |
| `relevant_filtered_docs` | list[Document] | Documents deemed relevant |
| `filtered_docs` | list[Document] | Documents passing eval filter |
| `unfiltered_docs` | list[Document] | All initially retrieved documents |
| `source_mapping` | dict | Document ID to citation number mapping |
| `all_relevant_intermediate_answers` | list | Nested merge steps |
| `total_cost` | float | Total USD cost |
| `total_model_cost` | float | Strong model cost |
| `total_eval_model_cost` | float | Eval model cost |

#### `search_task(query: str) -> dict`

Like `query_task` but returns matching documents without generating an LLM answer.

#### `summary_task() -> dict`

Run summarization. Called automatically during `__init__` for summary tasks. Results are also stored in `instance.summary_results`.

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `summary` | str | The markdown summary |
| `doc_total_tokens_str` | str | Token count |
| `doc_total_cost` | float | USD cost |
| `doc_reading_length` | float | Reading time saved (minutes) |
| `doc_total_tokens_sum` | int | Total tokens used |

### Public Properties

| Property | Type | Description |
|----------|------|-------------|
| `summary_results` | dict | Results from the latest summary |
| `loaded_docs` | list[Document] | Parsed document chunks |
| `loaded_embeddings` | object | FAISS vector store |
| `llm` | object | Main LLM instance |
| `eval_llm` | object | Eval LLM instance |
| `embedding_engine` | object | Embedding model instance |
| `model` | str | Main model name |
| `query_eval_model` | str | Eval model name |
| `embed_model` | str | Embedding model name |
| `task` | wdocTask | Current task |
| `top_k` | int\|str | Current top_k |
| `latest_cost` | float | Cost of latest operation |
| `interaction_settings` | dict | Get/set: `top_k`, `retriever`, `task`, `relevancy`, `multiline` |

### parse_doc Static Method

Parse a document without any LLM interaction.

```python
wdoc.parse_doc(
    filetype: str = "auto",
    format: str = "text",      # "text", "split_text", "xml", "langchain", "langchain_dict"
    debug: bool = False,
    verbose: bool = False,
    out_file: str | Path | None = None,
    **kwargs,                  # DocDict arguments (path, etc.)
) -> str | list[Document] | list[dict]
```

**Format options:**

| Format | Return Type | Description |
|--------|-------------|-------------|
| `text` | str | Concatenated plain text |
| `split_text` | str | Text with document split markers |
| `xml` | str | XML-formatted output |
| `langchain` | list[Document] | LangChain Document objects |
| `langchain_dict` | list[dict] | Dicts with `page_content` and `metadata` |

### Python Examples

```python
from wdoc import wdoc

# 1. Query a document
instance = wdoc(
    task="query",
    path="paper.pdf",
    model="openai/gpt-4o",
)
result = instance.query_task("What are the main contributions?")
print(result["final_answer"])
print(f"Cost: ${result['total_cost']:.4f}")

# Ask follow-up questions on the same documents
result2 = instance.query_task("What methodology was used?")

# 2. Summarize a document
instance = wdoc(
    task="summarize",
    path="paper.pdf",
    model="openai/gpt-4o",
    summary_language="en",
)
results = instance.summary_results
print(results["summary"])
print(f"Cost: ${results['doc_total_cost']:.5f}")
print(f"Time saved: {results['doc_reading_length']:.1f} min")

# 3. Parse a document (no LLM needed)
text = wdoc.parse_doc(path="document.pdf", format="text")
docs = wdoc.parse_doc(path="document.pdf", format="langchain")
dicts = wdoc.parse_doc(path="document.pdf", format="langchain_dict")

# 4. Query with local models
instance = wdoc(
    task="query",
    path="secret.pdf",
    model="ollama/qwen3:8b",
    query_eval_model="ollama/qwen3:8b",
    embed_model="ollama/snowflake-arctic-embed2",
    private=True,
)

# 5. Query multiple documents
instance = wdoc(
    task="query",
    filetype="recursive_paths",
    path="papers/",
    pattern="**/*.pdf",
    recursed_filetype="pdf",
    source_tag="research_papers",
    model="openai/gpt-4o",
)

# 6. Web search
instance = wdoc(
    task="query",
    filetype="ddg",
    path="latest quantum computing breakthroughs",
    query="What are the most recent quantum computing breakthroughs?",
)
result = instance.query_task("What are the most recent quantum computing breakthroughs?")

# 7. Save and reload embeddings
instance = wdoc(
    task="query",
    path="corpus/",
    filetype="recursive_paths",
    pattern="**/*.pdf",
    recursed_filetype="pdf",
    save_embeds_as="my_index.pkl",
)

# Later, load without re-indexing:
instance = wdoc(
    task="query",
    load_embeds_from="my_index.pkl",
)
result = instance.query_task("New question on the same corpus")

# 8. Change interaction settings at runtime
instance.interaction_settings = {
    "top_k": 100,
    "retriever": "basic_knn",
    "relevancy": 0.0,
}
```
