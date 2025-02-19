# Examples

### Table of Contents
1. [Shell Examples](#shell-examples)
2. [Python Script Examples](#python-script-examples)

# Shell Examples

1. Query a simple PDF file
```zsh
wdoc --task=query --path="my_file.pdf" --filetype="pdf" --model='openai/gpt-4o'
```

2. Recursively query multiple PDFs in a directory
```zsh
wdoc --task=query \
     --path="my/other_dir" \
     --pattern="**/*pdf" \
     --filetype="recursive_paths" \
     --recursed_filetype="pdf" \
     --query="My question about those documents"
```

3. Summarize a YouTube video in french based on the english transcript
```zsh
wdoc --task=summary \
     --path='https://www.youtube.com/watch?v=arj7oStGLkU' \
     --youtube_language="en" \
     --summary_language="fr" \
     --disable_md_printing
```

4. Summarize a YouTube video based on the whisper transcript
```zsh
wdoc --task=summary \
     --path='https://www.youtube.com/watch?v=arj7oStGLkU' \
     --youtube_audio_backend="whisper" \
     --whisper_lang="en"
```

5. Use local models with Ollama
```zsh
wdoc --model="ollama_chat/gemma:2b" \
     --query_eval_model="ollama_chat/gemma:2b" \
     --embed_model="ollama/bge-m3" \
     my_task
```

6. Parse an Anki deck as text
```zsh
wdoc_parse_file --filetype "anki" \
                --anki_profile "Main" \
                --anki_deck "mydeck::subdeck1" \
                --anki_notetype "my_notetype" \
                --anki_template "<header>\n{header}\n</header>\n<body>\n{body}\n</body>\n<personal_notes>\n{more}\n</personal_notes>\n<tags>{tags}</tags>\n{image_ocr_alt}" \
                --anki_tag_filter "a::tag::regex::.*something.*" \
                --format=langchain_dict
```

7. Query an online PDF
```zsh
wdoc --path="https://example.com/document.pdf" \
     --task=query \
     --filetype="online_pdf" \
     --query="What does it say about X?"
```

8. Save and load embeddings for faster subsequent queries
```zsh
# First run - save embeddings
wdoc --task=query \
     --path="my_document.pdf" \
     --save_embeds_as="saved_embeddings.pkl"

# Subsequent runs - load embeddings
wdoc --task=query \
     --load_embeds_from="saved_embeddings.pkl" \
     --query="My new question"
```

# Python Script Examples

1. Basic document summarization
```python
from wdoc import wdoc

# Initialize wdoc for summarization
instance = wdoc(
    task="summary",
    path="document.pdf",
    summary_language="en",  # Optional: specify output language
    import_mode=True  # Use import mode for scripting
)

# Get summary results
results = instance.summary_results
print(f"Summary:\n{results['summary']}")
print(f"Processing cost: ${results['doc_total_cost']:.5f}")
print(f"Original reading time: {results['doc_reading_length']:.1f} minutes")
```

2. Summarize with custom model settings
```python
from wdoc import wdoc

# Use specific models for better control
instance = wdoc(
    task="summary",
    path="https://example.com/paper.pdf",
    filetype="online_pdf",
    model="openai/gpt-4o",  # Use GPT-4o for summarization
    embed_model="openai/text-embedding-3-large",  # Specify embedding model
    import_mode=True
)

results = instance.summary_results
summary_text = results['summary']
```
