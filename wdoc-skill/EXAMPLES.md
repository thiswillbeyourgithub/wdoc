# wdoc Examples

> Written for **wdoc v5.1.0**. On a different version, some arguments, defaults, or behaviors may differ.

Copy-pasteable shell and Python recipes. For the meaning of each argument see [REFERENCE.md](REFERENCE.md); for a quick orientation see [SKILL.md](SKILL.md).

## Shell Examples

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

# Query a Zotero collection (local Zotero app running, no API key needed)
wdoc --task=query --filetype=zotero \
     --path="Research/ML/Papers" \
     --query="What do these papers say about attention?"

# Query a Karakeep list (creds from KARAKEEP_PYTHON_API_* env vars)
wdoc --task=query --filetype=karakeep \
     --path="list:Reading" \
     --query="What do these articles say about RAG?"
```

## Python Examples

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
