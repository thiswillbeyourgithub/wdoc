---
name: wdoc-skill
description: Comprehensive reference for wdoc, a RAG CLI and Python library that summarizes, searches, and queries documents across 20+ filetypes (PDF, YouTube, audio, Anki, web, Zotero, Karakeep, and more) through LiteLLM (100+ LLM providers). Use when the user runs or asks about the `wdoc` command, imports `from wdoc import wdoc`, or needs help with wdoc tasks (query, search, summarize, parse), CLI arguments, environment variables, filetypes, or the Python API.
---

# wdoc

> Written for **wdoc v5.1.0**. On a different version, some arguments, defaults, or behaviors may differ.

`wdoc` is a RAG (Retrieval-Augmented Generation) system for summarizing, searching, and querying documents across 20+ file types. It works as a CLI (via Google Fire) and as a Python library (`from wdoc import wdoc`), routing every LLM call through LiteLLM (100+ providers).

This SKILL.md is the quick orientation. The deep material lives in two companion files:

- **[REFERENCE.md](REFERENCE.md)**: every CLI argument, filetype, loader option, environment variable, and the full Python API.
- **[EXAMPLES.md](EXAMPLES.md)**: copy-pasteable shell and Python examples.

## Quick start

```bash
pip install -U wdoc[full]            # full install: all loaders. Plain `wdoc` ships only PDF + URL.
export ANTHROPIC_API_KEY="your_key"  # or whichever provider you use

wdoc query     paper.pdf "What are the main findings?"   # ask questions (RAG)
wdoc summarize paper.pdf                                  # detailed markdown summary
wdoc parse     paper.pdf                                  # parse to text, no LLM
wdoc web       "latest on quantum computing"             # DuckDuckGo + query
```

`uvx wdoc[full] ...` runs it without installing and sidesteps thinking about extras.

## The four tasks

| Task | What it does | Pick it when |
|------|--------------|--------------|
| `query` | Embeds docs, retrieves chunks, answers with sourced markdown | You have a question about the content |
| `search` | Returns matching docs + metadata, no LLM answer | You only need to locate relevant passages |
| `summarize` | Detailed markdown summary (author's reasoning, not vague takeaways) | You want the gist of a long document |
| `summarize_then_query` | Summarize first, then drop into a query prompt | You want both, in one run |

## Core mechanics worth knowing

- **Shortcuts:** `wdoc query FILE`, `wdoc summarize FILE`, `wdoc parse FILE`, and `wdoc web "q"` expand to longer `--task=...` forms. Positional args work too: `wdoc TASK PATH [QUERY]`.
- **Filetype is auto-detected** (`--filetype=auto`) but can be forced (`pdf`, `youtube`, `anki`, `zotero`, ...). Recursive filetypes (`recursive_paths`, `zotero`, `karakeep`, `ddg`, ...) fan one selector out into many documents.
- **Two models per run:** a strong `--model` answers, a cheap `--query_eval_model` filters chunks. Both take LiteLLM `provider/model` ids.
- **kebab or snake case:** `--query-eval-model` and `--query_eval_model` are equivalent.
- **Piped input is auto-detected:** `cat file.pdf | wdoc parse --filetype=pdf`.
- **Privacy:** `--private` (or `WDOC_PRIVATE_MODE=true`) blocks all outbound traffic and redacts API keys; pair it with local models (Ollama) and `--llms_api_bases`.
- **Reuse embeddings:** `--save_embeds_as=idx.pkl` once, then `--load_embeds_from=idx.pkl` to skip re-indexing.
- **Cost guard:** `--dollar_limit` (default 5) stops summaries/embeddings before they get expensive.

## Common patterns

```bash
# Query every PDF in a tree
wdoc --task=query --path="papers/" --filetype=recursive_paths \
     --pattern="**/*.pdf" --recursed_filetype=pdf --query="..."

# Fully local / private
wdoc --private --model="ollama/qwen3:8b" --query_eval_model="ollama/qwen3:8b" \
     --embed_model="ollama/snowflake-arctic-embed2" --task=query --path=secret.pdf

# Parse for use elsewhere (text, langchain, langchain_dict, xml, split_text)
wdoc parse document.pdf --format=langchain_dict
```

```python
from wdoc import wdoc
instance = wdoc(task="query", path="paper.pdf", model="openai/gpt-4o")
answer = instance.query_task("What are the main contributions?")
print(answer["final_answer"])
```

For anything beyond this page (exact argument types, defaults, every filetype's loader options, all `WDOC_*` env vars, the full Python API surface, and more examples), read **[REFERENCE.md](REFERENCE.md)** and **[EXAMPLES.md](EXAMPLES.md)**.
