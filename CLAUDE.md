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

## Adding New Settings

Any new setting (CLI argument or environment variable) **must** be documented in:
- `wdoc/docs/help.md` — describe the setting, its type, default value, and accepted values.
- `wdoc/docs/examples.md` — add usage examples where appropriate.
- `SKILL.md`: the comprehensive user-facing reference; mirror the new setting in the relevant table (see [Keeping `SKILL.md` in Sync](#keeping-skillmd-in-sync)).

New settings should be either:
- A **CLI argument** (defined in `wdoc.py`'s main class), or
- A **`WDOC_*` environment variable** (defined in `wdoc/utils/env.py`'s `EnvDataclass`).

**Environment variables are re-read on every access, not just at declaration.** The `EnvDataclass.__getattribute__` method checks `os.environ` at access time when the dataclass is frozen, so changing an env var between wdoc instantiations (or even at runtime) takes effect without reimporting.

## Variables in `wdoc/utils/misc.py` to Keep Updated

When adding new CLI arguments or loader-specific parameters, update these dicts in `wdoc/utils/misc.py`:
- `filetype_arg_types` — maps loader-specific argument names to their types (e.g. `"whisper_lang": str`).
- `extra_args_types` — maps extra wdoc instantiation arguments to their types. It merges `filetype_arg_types` automatically.
- `DocDict.allowed_keys` — derives from `filetype_arg_types` keys automatically, but verify new keys appear.

## Adding Support for a New Filetype

Adding a filetype is intentionally self-contained: it touches a new loader module plus a couple of registration points, never the core RAG pipeline. The recent `zotero` and `karakeep` types are good references for the two routes available:
- **Standalone loader** (most types): write a `load_*` function that returns `List[Document]` and register it (steps below).
- **Recursive fan-out** (`zotero`, `karakeep`, `recursive_paths`, ...): resolve a selector/path into many sub-documents and delegate parsing to *existing* loaders via DocDicts, so you reuse the existing parsers instead of writing new parsing code. Register the function in `recursive_types_func_mapping` in `wdoc/utils/load_recursive.py`.

Standalone loader steps:

1. **Create a loader** in `wdoc/utils/loaders/` — add a file (e.g. `myformat.py`) containing a function `load_myformat(path, file_hash, ...) -> List[Document]`. Use the `@debug_return_empty` and `@optional_strip_unexp_args` decorators (see `txt.py` for a minimal example).
2. **Register the filetype** — add `"myformat"` to the `LOADABLE_FILETYPE` list in `wdoc/utils/loaders/__init__.py`.
3. **Add loader-specific args** (if any) to `filetype_arg_types` in `wdoc/utils/misc.py`.
4. **Document it** — add the filetype and its arguments to `wdoc/docs/help.md`, add examples to `wdoc/docs/examples.md`, and add the filetype + its loader args to the tables in `SKILL.md` (see [Keeping `SKILL.md` in Sync](#keeping-skillmd-in-sync)).
5. **Auto-detection** (optional) — if the filetype corresponds to a file extension, add a mapping in the auto-detection logic so `--filetype=auto` can infer it.

## Bumping the Default Models

The two default LLM identifiers (`WDOC_DEFAULT_MODEL` and `WDOC_DEFAULT_QUERY_EVAL_MODEL`) are duplicated across `wdoc/utils/env.py`, `wdoc/docs/help.md`, `SKILL.md`, `README.md`, `ARCHITECTURE.md`, and `docker/env.example`. Use the repo-root helper instead of editing each file by hand:

```bash
./bump_default_models.sh <NEW_STRONG_MODEL> <NEW_EVAL_MODEL>           # dry-run preview
./bump_default_models.sh <NEW_STRONG_MODEL> <NEW_EVAL_MODEL> --apply   # write
```

`env.py` is the source of truth for the current values. The script replaces both the full id (`provider/path/name`) and the basename, and re-syncs the `KEY=VALUE` lines in `docker/env.example` independently (so it recovers from prior drift). It never commits; review with `git diff` and commit yourself.

## Keeping `SKILL.md` in Sync

`SKILL.md` is the comprehensive user-facing reference for the CLI and the Python API (filetypes, arguments, env vars, examples). It is **not** auto-generated, so it drifts unless updated by hand. Whenever you change user-visible behavior, keep it in sync:
- New CLI argument, env var, or loader-specific argument: add it to the relevant table (Global/Model/Embedding/Query/Summary Arguments, Environment Variables, or Loader-Specific Arguments).
- New filetype: add a row to the Filetypes or Recursive Filetypes table, plus its loader args, and ideally a shell example.
- Changed default, accepted values, or behavior of an existing setting: update the matching row.
- New install extra: update the Installation section.
- On a version bump, update the `> **This document was written for wdoc vX.Y.Z**` header line at the top.

`wdoc/docs/help.md` and `wdoc/docs/examples.md` are the source the reference mirrors; if you edited those, mirror the change here too.

## Building Documentation

To regenerate the Sphinx autodoc files, run from the repo root:
```bash
sphinx-apidoc -o docs/source/ wdoc --force
```

## Key Conventions

- `utils/misc.py` is a large utility module (~53KB) — search it before creating new helpers.
- Piped input to CLI is auto-detected as JSON, TOML, URLs, or file paths (`__main__.py`).
- `--private` / `WDOC_PRIVATE_MODE=true` blocks all outbound connections and redacts API keys.
- Binary FAISS embeddings (`WDOC_FAISS_COMPRESSION`) give ~32x compression; implemented in `utils/customs/binary_faiss_vectorstore.py`.
- Semantic batching clusters intermediate answers before combining (scipy hierarchical clustering).
- Use `python` not `python3` in commands.
