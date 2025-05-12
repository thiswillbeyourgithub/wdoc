# Walkthrough & Examples

### Table of Contents
1. [Walkthrough](#Walkthrough)
2. [Shell Examples](#shell-examples)
3. [Python Script Examples](#python-script-examples)

Note that there is [an official open-webui Tool](https://openwebui.com/t/qqqqqqqqqqqqqqqqqqqq/wdoctool) that is even simpler to use.


## Walkthrough

1. Say you want to ask a question about one pdf, that's simple: 
```bash
wdoc --task="query" --path="my_file.pdf" --filetype="pdf" --model='openai/gpt-4o'
```
Note that you could have just let `--filetype="auto"` and it would have worked the same.
* *Note: By default `wdoc` tries to parse args as kwargs so `wdoc query mydocument What's the age of the captain?` is parsed as `wdoc --task=query --path=mydocument --query "What's the age of the captain?"`. Likewise for summaries. This does not always work so use it only after getting comfortable with `wdoc`.*

2. You have several pdf? Say you want to ask a question about any pdf contained in a folder, that's not much more complicated:
```bash
wdoc --task="query" --path="my/other_dir" --pattern="**/*pdf" --filetype="recursive_paths" --recursed_filetype="pdf" --query="My question about those documents"
```
So basically you give as path the path to the dir, as pattern the globbing pattern used to find the files relative to the path, set as filetype "recursive_paths" so that `wdoc` knows what arguments to expect, and specify as recursed_filetype "pdf" so that `wdoc` knows that each found file must be treated as a pdf. You can use the same idea to glob any kind of file supported by `wdoc` like markdown etc. You can even use "auto"! Note that you can either directly ask your question with `--query="my question"`, or wait for an interactive prompt to pop up, or just pass the question as *args like so `wdoc [your kwargs] here is my question`.

3. You want more? You can write a `.json` file where each line (`#comments` and empty lines are ignored) will be parsed as a list of argument. For example one line could be:
```json
{"path": "my/other_dir", "pattern": "**/*pdf", "filetype": "recursive_paths", "recursed_filetype": "pdf"}
```
This way you can use a single json file to specify easily any number of sources. `.toml` files are also supported.

4. You can specify a "source_tag" metadata to help distinguish between documents you imported. It is EXTREMELY recommended to include a source_tag to any document you want to save: especially if using recursive filetypes. This is because after loading all documents `wdoc` use the source_tag to see if it should continue or crash. If you want to load 10_000 pdf in one go as I do, then it makes sense to continue if some failed to crash but not if a whole source_tag is missing.

5. Now say you do this with many many documents, as I do, you of course can't wait for the indexing to finish every time you have a question (even though the embeddings are cached). You should then add 
```bash
--save_embeds_as=your/saving/path
```
to save all this index in a file. Then simply do 
```bash
--load_embeds_from=your/saving/path
```
to quickly ask queries about it!

6. To know more about each argument supported by each filetype, 
```bash
wdoc --help
```

7. There is a specific recursive filetype I should mention: `--filetype="link_file"`. Basically the file designated by `--path` should contain in each line (`#comments` and empty lines are ignored) one url, that will be parsed by `wdoc`. I made this so that I can quickly use the "share" button on android from my browser to a text file (so it just appends the url to the file), this file is synced via [syncthing](https://github.com/syncthing/syncthing) to my browser and `wdoc` automatically summarize them and add them to my [Logseq](https://github.com/logseq/logseq/). Note that the url is parsed in each line, so formatting is ignored, for example it works even in markdown bullet point list.

8. If you want to only use local models, here's an example with [ollama](https://ollama.com/):
```bash
wdoc --model="ollama/qwen3:8b" --query_eval_model="ollama/qwen3:8b" --embed_model="ollama/snowflake-arctic-embed2" --task summarize --path https://situational-awareness.ai/
```
You can always add `--private` to add additional safety nets that no data will leave your local network. You can also override specific API endpoints using 
```bash
--llms_api_bases='{"model": "http://localhost:11434", "query_eval_model": "http://localhost:11434", "embeddings": "http://localhost:1434"}'
```

9. Now say you just want to summarize [Tim Urban's TED talk on procrastination](https://www.youtube.com/watch?v=arj7oStGLkU):
```bash
wdoc --task=summary --path='https://www.youtube.com/watch?v=arj7oStGLkU' --youtube_language="en" --disable_md_printing
```

<details><summary>Click to see the output</summary>


> # Summary
> ## https://www.youtube.com/watch?v=arj7oStGLkU
> - Let me take a deep breath and summarize this TED talk about procrastination:
> - [0:00-3:40] Personal experience with procrastination in college:
>     - Author's pattern with papers: planning to work steadily but actually doing everything last minute
>     - 90-page senior thesis experience:
>         - Planned to work steadily over a year
>         - Actually wrote 90 pages in 72 hours with two all-nighters
>         - *Jokingly implies* it was brilliant, then admits it was 'very, very bad'
> - [3:40-6:45] Brain comparison between procrastinators and non-procrastinators:
>     - Both have a **Rational Decision-Maker**
>     - Procrastinator's brain also has an **Instant Gratification Monkey**:
>         - Lives entirely in present moment
>         - Only cares about 'easy and fun'
>         - Works fine for animals but problematic for humans in advanced civilization
>     - **Rational Decision-Maker** capabilities:
>         - Can visualize future
>         - See big picture
>         - Make long-term plans
> - [6:45-10:55] The procrastinator's system:
>     - **Dark Playground**:
>         - Where leisure activities happen at wrong times
>         - Characterized by guilt, dread, anxiety, self-hatred
>     - **Panic Monster**:
>         - Only thing monkey fears
>         - Awakens near deadlines or threats of public embarrassment
>         - Enables last-minute productivity
>     - Personal example with TED talk preparation:
>         - Procrastinated for months
>         - Only started working when panic set in
> - [10:55-13:05] Two types of procrastination:
>     - Deadline-based procrastination:
>         - Effects contained due to Panic Monster intervention
>         - Less harmful long-term
>     - Non-deadline procrastination:
>         - More dangerous
>         - Affects important life areas without deadlines:
>             - Entrepreneurial pursuits
>             - Family relationships
>             - Health
>             - Personal relationships
>         - Can cause long-term unhappiness and regrets
> - [13:05-14:04] Concluding thoughts:
>     - *Author believes* no true non-procrastinators exist
>     - Presents **Life Calendar**:
>         - Shows 90 years in weekly boxes
>         - Emphasizes limited time available
>     - Call to action: need to address procrastination 'sometime soon'
> - Key audience response moments:
>     - Multiple instances of '(Laughter)' noted throughout
>     - Particularly strong response from PhD students relating to procrastination issues
>     - Received thousands of emails after blog post about procrastination
> Tokens used for https://www.youtube.com/watch?v=arj7oStGLkU: '4936' (in: 4307, out: 629, cost: $0.00063)
> Total cost of those summaries: 4936 tokens for $0.00063 (estimate was $0.00030)
> Total time saved by those summaries: 8.8 minutes
> Done summarizing.

</details>

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
wdoc --model="ollama/qwen3:8b" \
     --query_eval_model="ollama/qwen3:8b" \
     --embed_model="ollama/snowflake-arctic-embed2" \
     --task summarize --path https://situational-awareness.ai/
```

Note: you might find that ollama models are sometimes overly optimistic about their context length. You can pass arguments to lower it like so:
```zsh
wdoc --model="ollama/qwen3:8b" \
     --query_eval_model="ollama/qwen3:8b" \
     --model_kwargs='{"max_tokens": 4096}' \
     --query_eval_model="ollama/qwen3:8b" \
     --query_eval_model_kwargs='{"max_tokens": 4096}' \
     --embed_model="ollama/snowflake-arctic-embed2" \
     --task summarize --path https://situational-awareness.ai/
```

6. Parse an Anki deck as text
```zsh
wdoc parse \
    --filetype "anki" \
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

9. You can even use shell pipes

Data sent using shell pipes (be it for strings or binary data) will be automatically saved to a temporary file which is then passed as `--path=[temp_file]` argument. For example `cat **/*.txt | wdoc --task=query`, `echo $my_url | wdoc parse`  or even `cat my_file.pdf | wdoc parse --filetype=pdf`. For binary input it is strongly recommended to use a `--filetype` argument because `python-magic` version <=0.4.27 chokes otherwise (see [that issue](https://github.com/ahupp/python-magic/issues/261).

# Python Script Examples

1. Basic document summarization
```python
from wdoc import wdoc

# Initialize wdoc for summarization
instance = wdoc(
    task="summary",
    path="document.pdf",
    summary_language="en",  # Optional: specify output language
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
)

results = instance.summary_results
summary_text = results['summary']
```
