# WDOC Architecture

WDOC is a Retrieval-Augmented Generation (RAG) system that loads documents from 20+ source types, embeds them into a vector store, and supports querying, summarizing, searching, and parsing tasks via LLMs.

---

## Directory Structure

```
wdoc/
├── __init__.py                 # Package init, beartype setup
├── __main__.py                 # CLI entry point (Google Fire)
├── wdoc.py                     # Main orchestration class (~3000 lines)
└── utils/
    ├── env.py                  # Configuration (EnvDataclass, WDOC_* env vars)
    ├── llm.py                  # LLM loading via LiteLLM, price tracking
    ├── embeddings.py           # Embedding model setup, FAISS vector store
    ├── retrievers.py           # Retriever creation (multi-query, parent, SVM, KNN)
    ├── prompts.py              # LLM prompt templates (named personas)
    ├── batch_file_loader.py    # Parallel multi-file loading with filetype inference
    ├── load_recursive.py       # Handles compound/recursive filetypes
    ├── misc.py                 # Caching, hashing, token counting utilities
    ├── errors.py               # Custom exceptions
    ├── logger.py               # Loguru-based logging
    ├── filters.py              # Document filtering (regex, metadata)
    ├── interact.py             # Interactive CLI features
    ├── loaders/                # Per-filetype document loaders
    │   ├── __init__.py         # load_one_doc() dispatcher
    │   ├── pdf.py              # PDF (15 parser backends)
    │   ├── url.py              # URLs (jina, playwright, selenium)
    │   ├── youtube.py          # YouTube videos
    │   ├── local_audio.py      # Audio transcription (Deepgram)
    │   ├── local_video.py      # Video extraction
    │   ├── epub.py             # EPUB books
    │   ├── anki.py             # Anki flashcards
    │   ├── logseq_markdown.py  # Logseq notes
    │   ├── word.py             # Word documents
    │   ├── powerpoint.py       # PowerPoint
    │   └── json_dict.py        # JSON entries
    ├── tasks/                  # Task implementations
    │   ├── types.py            # wdocTask enum
    │   ├── query.py            # Query: retrieve → evaluate → answer → combine
    │   ├── summarize.py        # Recursive chunk summarization
    │   ├── search.py           # Similarity search (no LLM answering)
    │   ├── parse.py            # Raw document extraction
    │   └── shared_query_search.py
    └── customs/                # Custom implementations
        ├── litellm_embeddings.py          # LiteLLM-based embeddings
        ├── binary_faiss_vectorstore.py    # Binary FAISS (~32x compression)
        ├── compressed_embeddings_cacher.py
        └── fix_llm_caching.py
```

---

## Data Flow

```
User Input (CLI / Python API)
    │
    ▼
CLI Parser (__main__.py) ── Google Fire argument parsing
    │                       Piped input detection (JSON, TOML, text)
    ▼
wdoc.__init__() ── Configuration & validation
    │
    ├──► Document Loading (batch_file_loader.py)
    │      1. Infer filetype (regex rules + file magic)
    │      2. Expand recursive types (directories, playlists, JSON arrays)
    │      3. Parallel load via joblib (loaders/*.py)
    │      4. Text splitting (RecursiveCharacterTextSplitter or semantic)
    │      5. Metadata enrichment (hash, title, source, reading time)
    │
    ├──► Embedding & Vector Store (embeddings.py)
    │      1. Initialize embedding model (LiteLLMEmbeddings)
    │      2. Cache-backed embedding layer
    │      3. Build FAISS index (standard or binary-compressed)
    │
    └──► Task Execution
           │
           ├─ query/search → Retrieval pipeline (see below)
           ├─ summarize    → Chunk-by-chunk summarization
           └─ parse        → Return raw documents (no LLM)
```

---

## Three LLM Roles

| Role | Default Model | Purpose |
|------|--------------|---------|
| **Main model** | `gemini-3.1-pro` | Answering, summarizing |
| **Eval model** | `gemini-2.5-flash` | Document relevance checking (cheap/fast) |
| **Embed model** | `text-embedding-3-small` | Dense vector embeddings |

All models are loaded through **LiteLLM**, supporting 100+ providers (OpenAI, Anthropic, Google, Mistral, Ollama, OpenRouter, etc.).

---

## Query Pipeline (RAG)

The query task uses named AI personas in sequence:

1. **Raphael (Rephraser)** — Expands user query into multiple alternative phrasings
2. **Vector Store** — Embeds queries and retrieves top-k similar document chunks
3. **Eve (Evaluator)** — Checks each chunk's relevance to the query (eval model, cheap)
4. **Anna (Answerer)** — Extracts an answer from each relevant chunk (main model)
5. **Carl (Combiner)** — Hierarchically clusters and combines intermediate answers into a final response

**Smart top-k expansion**: starts at top_k (default 200); if >90% of documents are relevant, automatically increases top_k and retries until diminishing returns.

---

## Summarization Pipeline

1. Split document into chunks (with overlap)
2. **Sam (Summarizer)** summarizes each chunk, passing the previous chunk's summary as context
3. If `summary_n_recursion > 0`, recursively summarize the summaries
4. Returns a `wdocSummary` dataclass with the full result tree

---

## Configuration System

**`utils/env.py`** defines `EnvDataclass`, a frozen dataclass with 50+ fields loaded from `WDOC_*` environment variables. CLI arguments override env vars.

Key variables:
- `WDOC_DEFAULT_MODEL` / `WDOC_DEFAULT_EMBED_MODEL` / `WDOC_DEFAULT_QUERY_EVAL_MODEL`
- `WDOC_PRIVATE_MODE` — blocks outbound connections, redacts API keys
- `WDOC_MAX_CHUNK_SIZE` — max tokens per chunk
- `WDOC_FAISS_COMPRESSION` — enable binary embeddings

---

## Document Loaders

Each filetype has a dedicated loader in `utils/loaders/`. The dispatcher (`load_one_doc()`) dynamically imports the `load_{filetype}` function.

**Supported types**: `pdf`, `txt`, `word`, `powerpoint`, `epub`, `url`, `youtube`, `youtube_playlist`, `online_pdf`, `online_media`, `local_audio`, `local_video`, `anki`, `logseq_markdown`, `json_dict`, `recursive_paths`, `json_entries`, `toml_entries`, `string`, `ddg` (DuckDuckGo search).

**PDF parsing** is particularly robust: 15 different parser backends are evaluated and the best result is selected automatically.

**URL loading** has multiple fallbacks: Jina → Playwright → Selenium → others.

---

## Caching Layers

1. **LLM response cache** — `SQLiteCacheFixed` prevents duplicate LLM calls
2. **Embedding cache** — `CacheBackedEmbeddings` avoids re-embedding identical content
3. **Loader cache** — `joblib.Memory` caches parsed documents
4. **Document hashing** — content_hash / file_hash / all_hash prevent reprocessing

---

## Error Handling

Custom exceptions in `utils/errors.py`:
- `NoDocumentsRetrieved` — no documents found in vector store
- `NoDocumentsAfterLLMEvalFiltering` — all retrieved docs deemed irrelevant
- `NoRelevantIntermediateAnswers` — LLM found no useful answers
- `TimeoutPdfLoaderError` — PDF parsing timeout

Fallback strategies: multiple parser/loader backends tried in sequence before failing.

---

## Key Dependencies

| Category | Libraries |
|----------|-----------|
| RAG framework | LangChain, LangChain-Community |
| LLM abstraction | LiteLLM, LangChain-LiteLLM |
| Vector store | FAISS |
| Text splitting | Chonkie (semantic), LangChain (recursive) |
| Document parsing | Unstructured, OpenParse, BeautifulSoup, lxml |
| Media | yt-dlp, Deepgram-SDK, pydub |
| Web scraping | Playwright, Selenium, Jina |
| Parallel processing | Joblib |
| Type checking | Beartype (runtime) |
| Logging | Loguru, Rich |
| Observability | Langfuse |
| Caching/storage | SQLAlchemy, LMDB (PersistDict) |
