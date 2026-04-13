# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wdoc is a production-grade RAG system for document summarization, searching, and querying across 20+ file types. It serves as both a CLI tool (via Google Fire) and a Python library (`from wdoc import wdoc`). All LLM calls go through LiteLLM (100+ providers). See `ARCHITECTURE.md` for detailed data flow and component diagrams.

## Commands

### Install
```bash
pip install -e .[full]
```

### Test
```bash
# All tests
pytest tests --quiet

# By category (basic = no API keys needed, api = needs credentials)
pytest -m basic tests/test_wdoc.py
pytest -m api tests/test_wdoc.py

# Single test
pytest tests/test_wdoc.py::test_wdoc_version

# Parallelized
pytest -n auto -m basic tests/test_parsing.py

# CLI integration tests
cd tests && ./test_cli.sh api
```

Test environment variables: `PYTEST_IS_TESTING_WDOC=true`, `WDOC_TYPECHECKING=crash`, `WDOC_DISABLE_EMBEDDINGS_CACHE=true`.

### Lint / Format
```bash
ruff format wdoc/
pre-commit run --all-files   # runs ruff-format + pytest (on pre-merge)
```

## Architecture Essentials

**Entry points**: `__main__.py` (CLI via Google Fire) → `wdoc.py` (main orchestration class, ~3000 lines).

**Named AI personas** drive the RAG pipeline in sequence:
- **Raphael** — rephrases user query into alternatives
- **Eve** — evaluates chunk relevance (cheap eval model)
- **Anna** — extracts answers from relevant chunks (main model)
- **Carl** — combines intermediate answers hierarchically
- **Sam** — summarizes chunks (summarization task)

Each persona can be customized via `WDOC_{NAME}_INSTRUCTIONS` env vars.

**Configuration**: `utils/env.py` defines `EnvDataclass` — a frozen dataclass loading 50+ `WDOC_*` environment variables. CLI args override env vars.

**Document loading**: `utils/loaders/` has per-filetype loaders dispatched by `load_one_doc()`. PDF loading tries 15 parser backends and picks the best. URL loading has Jina → Playwright → Selenium fallback chain.

**Caching**: four layers — LLM responses (SQLite), embeddings (CacheBackedEmbeddings), parsed documents (joblib.Memory), and document content hashing.

**Type checking**: Beartype runtime checking, controlled by `WDOC_TYPECHECKING` env var (crash/warn/disabled).

## Key Conventions

- `utils/misc.py` is a large utility module (~53KB) — search it before creating new helpers.
- Piped input to CLI is auto-detected as JSON, TOML, URLs, or file paths (`__main__.py`).
- `--private` / `WDOC_PRIVATE_MODE=true` blocks all outbound connections and redacts API keys.
- Binary FAISS embeddings (`WDOC_FAISS_COMPRESSION`) give ~32x compression; implemented in `utils/customs/binary_faiss_vectorstore.py`.
- Semantic batching clusters intermediate answers before combining (scipy hierarchical clustering).
- Use `python` not `python3` in commands.
