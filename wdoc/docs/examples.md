# Table of Contents
1. [Shell Examples](#shell-examples)
2. [Python Script Examples](#python-script-examples)

# Shell Examples

1. Query a simple PDF file
```zsh
wdoc --task=query --path="my_file.pdf" --filetype="pdf" --modelname='openai/gpt-4o'
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

3. Summarize a YouTube video
```zsh
wdoc --task=summary \
     --path='https://www.youtube.com/watch?v=arj7oStGLkU' \
     --youtube_language="en" \
     --summary_language="fr" \
     --disable_md_printing
```

4. Use local models with Ollama
```zsh
wdoc --modelname="ollama_chat/gemma:2b" \
     --query_eval_modelname="ollama_chat/gemma:2b" \
     --embed_model="ollama/bge-m3" \
     my_task
```

5. Parse an Anki deck
```zsh
wdoc_parse_file --filetype "anki" \
                --anki_profile "Main" \
                --anki_deck "mydeck::subdeck1" \
                --anki_notetype "my_notetype" \
                --anki_template "<header>\n{header}\n</header>\n<body>\n{body}\n</body>\n<personal_notes>\n{more}\n</personal_notes>\n<tags>{tags}</tags>\n{image_ocr_alt}" \
                --anki_tag_filter "a::tag::regex::.*something.*" \
                --format=langchain_dict
```

6. Query an online PDF
```zsh
wdoc --path="https://example.com/document.pdf" \
     --task=query \
     --filetype="online_pdf" \
     --query="What does it say about X?"
```

7. Save and load embeddings for faster subsequent queries
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

1. Basic document parsing
```python
from wdoc import wdoc

# Parse a file
list_of_docs = Wdoc.parse_file(path="my_path")
```

2. Using wdoc as an imported module
```python
# Example of using wdoc in import mode
from wdoc import Wdoc

# Initialize wdoc
w = Wdoc(import_mode=True)

# Load documents
docs = w.load_documents(path="my_document.pdf", filetype="pdf")

# Query the documents
response = w.query_documents(docs, query="What is the main topic?")
```

3. Custom PDF parser implementation
```python
class CustomPDFParser:
    def __init__(self, path):
        self.path = path
    
    def load(self):
        # Your custom parsing logic here
        # Must return List[Document]
        pass

# Register the custom parser
from wdoc.utils.loaders import pdf_loaders
pdf_loaders['custom_parser'] = CustomPDFParser

# Use the custom parser
wdoc.parse_file(path="document.pdf", pdf_parsers="custom_parser")
```
