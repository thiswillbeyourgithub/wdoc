# wdoc

<p align="center"><img src="https://github.com/thiswillbeyourgithub/wdoc/blob/main/images/icon.png?raw=true" width="512" style="background-color: transparent !important"></p>

> *I'm wdoc. I solve RAG problems.*
> - wdoc, imitating Winston "The Wolf" Wolf

`wdoc` is a powerful RAG (Retrieval-Augmented Generation) system designed to summarize, search, and query documents across various file types. It's particularly useful for handling large volumes of diverse document types, making it ideal for researchers, students, and professionals dealing with extensive information sources. I was frustrated with all other RAG solutions for querying or summarizing, so I made my perfect solution in a single package.

*(The online documentation can be found [here](https://wdoc.readthedocs.io/en/stable))*

* **Goal and project specifications**: `wdoc` uses [LangChain](https://python.langchain.com/) to process and analyze documents. It's capable of querying **tens of thousands** of documents across [various file types](#Supported-filetypes) at the same time. The project also includes a tailored summary feature to help users efficiently keep up with large amounts of information.

* **Current status**: **Under active development**
    * Used daily by the developer for several months: but still in alpha. **I would greatly benefit from testing by users as it's the quickest way for me to find the many small bugs that are quick to fix.**
    * May have some instabilities, but issues can usually be resolved quickly
    * The main branch is more stable than the dev branch, which offers more features
    * Open to feature requests and pull requests
    * All feedbacks, including reports of typos, are highly appreciated
    * Please open an issue before making a PR, as there may be ongoing improvements in the pipeline

* **Key Features**:
    * Aims to **support *any* filetypes** and query from all of them at the same time (**15+** are already implemented!)
    * **High recall and specificity**: it was made to find A LOT of documents using carefully designed embedding search then carefully aggregate gradually each answer using semantic batch to produce a single answer that mentions the source pointing to the exact portion of the source document.
    * Supports **virtually any LLM providers**, including local ones, and even with extra layers of security for super secret stuff.
    * Use both an expensive and cheap LLM to make recall as high as possible because we can afford fetching a lot of documents per query (via embeddings)
    * **Finally a *useful* AI powered summary**: get the thought process of the author instead of nebulous takeaways.
    * **Extensible**, this is both a tool and a library.

### Table of contents
- [Ultra short guide for people in a hurry](#ultra-short-guide-for-people-in-a-hurry)
- [Features](#features)
  - [Roadmap](#roadmap)
  - [Supported filetypes](#supported-filetypes)
  - [Supported tasks](#supported-tasks)
  - [Walkthrough and examples](#walkthrough-and-examples)
- [Scripts made with wdoc](#scripts-made-with-wdoc)
- [Getting started](#getting-started)
- [FAQ](#faq)
- [Notes](#notes)
  - [Known issues](#known-issues)

## Ultra short guide for people in a hurry
<details>
<summary>
Give it to me I am in a hurry!
</summary>


``` zsh
link="https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf"

wdoc --path=$link --task=query --filetype="online_pdf" --query="What does it say about alphago?" --query_retrievers='default_multiquery' --top_k=auto_200_500
```
* This will:
    1. parse what's in --path as a link to a pdf to download (otherwise the url could simply be a webpage, but in most cases you can leave it to 'auto' by default as heuristics are in place to detect the most appropriate parser).
    2. cut the text into chunks and create embeddings for each
    3. Take the user query, create embeddings for it ('default') AND ask the default LLM to generate alternative queries and embed those
    4. Use those embeddings to search through all chunks of the text and get the 200 most appropriate documents
    5. Pass each of those documents to the smaller LLM (default: openai/gpt-4o-mini) to tell us if the document seems appropriate given the user query
    6. If More than 90% of the 200 documents are appropriate, then we do another search with a higher top_k and repeat until documents start to be irrelevant OR we it 500 documents.
    7. Then each relevant doc is sent to the strong LLM (by default, openai/gpt-4o) to extract relevant info and give one answer.
    8. Then all those "intermediate" answers are 'semantic batched' (meaning we create embeddings, do hierarchical clustering, then create small batch containing several intermediate answers) and each batch is combined into a single answer.
    9. Rinse and repeat steps 7+8 until we have only one answer, that is returned to the user.

``` zsh
link="https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf"

wdoc --path=$link --task=summarize --filetype="online_pdf"
```
* This will:
    1. Split the text into chunks
    2. pass each chunk into the strong LLM (by default openai/gpt-4o) for a very low level (=with all details) summary. The format is markdown bullet points for each idea and with logical indentation.
    3. When creating each new chunk, the LLM has access to the previous chunk for context.
    4. All summary are then concatenated and returned to the user

* For extra large documents like books for example, this summary can be recusively fed to `wdoc` using argument --summary_n_recursion=2 for example.

* Those two tasks, query and summary, can be combined with --task summarize_then_query which will summarize the document but give you a prompt at the end to ask question in case you want to clarify things.

* For more, you can jump to the section [Walkthrough and examples](#walkthrough-and-examples)

</details>

## Features
* **15+ filetypes**: also supports combination to load recursively or define complex heterogenous corpus like a list of files, list of links, using regex, youtube playlists etc. See [Supported filestypes](#Supported-filetypes). All filetype can be seamlessly combined in the same index, meaning you can query your anki collection at the same time as your work PDFs). It supports removing silence from audio files and youtube videos too!
* **100+ LLMs**: OpenAI, Mistral, Claude, Ollama, Openrouter, etc. Thanks to [litellm](https://docs.litellm.ai/). Personnaly I'm using openrouter's deepseek v3 model as strong LLM and eval LLM (previously : Claude Sonnet 3.5 and gpt-4o-mini), and openai embeddings.
* **Local and Private LLM**: take some measures to make sure no data leaves your computer and goes to an LLM provider: no API keys are used, all `api_base` are user set, cache are isolated from the rest, outgoing connections are censored by overloading sockets, etc.
* **Advanced RAG to query lots of diverse documents**:
    1. The documents are retrieved using embedding
    2. Then a weak LLM model ("Eve the Evaluator") is used to tell which of those document is not relevant
    3. Then the strong LLM is used to answer ("Anna the Answerer") the question using each individual remaining documents.
    4. Then all relevant answers are combined ("Carl the Combiner") into a single short markdown-formatted answer. Before being combined, they are batched by semantic clusters
    and semantic order using scipy's hierarchical clustering and leaf ordering, this makes it easier for the LLM to combine the answers in a manner that makes bottom up sense.
    `Eve the Evaluator`, `Anna the Answerer` and `Carl the Combiner` are the names given to each LLM in their system prompt, this way you can easily add specific additional instructions to a specific step. There's also `Sam the Summarizer` for summaries and `Raphael the Rephraser` to expand your query.
    5. Each document is identified by a unique hash and the answers are sourced, meaning you know from which document comes each information of the answer.
    * Supports a special syntax like "QE >>>> QA" were QE is a question used to filter the embeddings and QA is the actual question you want answered.
* **Advanced summary**:
    * Instead of unusable "high level takeaway" points, compress the reasoning, arguments, though process etc of the author into an easy to skim markdown file.
    * The summaries are then checked again n times for correct logical indentation etc.
    * The summary can be in the same language as the documents or directly translated.
* **Many tasks**: See [Supported tasks](#Supported-tasks).
* **Trust but verify**: The answer is sourced: `wdoc` keeps track of the hash of each document used in the answer, allowing you to verify each assertion.
* **Markdown formatted answers and summaries**: using [rich](https://github.com/Textualize/rich).
* **Sane embeddings**: By default use sophisticated embeddings like [multi query retrievers](https://python.langchain.com/docs/how_to/MultiQueryRetriever) but also include SVM, KNN, parent retriever etc. Customizable.
* **Fully documented** Lots of docstrings, lots of in code comments, detailed `--help` etc. The full usage can be found in the file [USAGE.md](./wdoc/docs/USAGE.md) or via `python -m wdoc --help`. I work hard to maintain an exhaustive documentation.
* **Scriptable / Extensible**: You can use `wdoc` in other python project using `--import_mode`. Take a look at the scripts [below](#scripts-made-with-wdoc).
* **Statically typed**: Runtime type checking. Opt out with an environment flag: `WDOC_TYPECHECKING="disabled / warn / crash" wdoc` (by default: `warn`). Thanks to [beartype](https://beartype.readthedocs.io/en/latest/) it shouldn't even slow down the code!
* **LLM (and embeddings) caching**: speed things up, as well as index storing and loading (handy for large collections).
* **Good PDF parsing** PDF parsers are notoriously unreliable, so 15 (!) different loaders are used, and the best according to a parsing scorer is kept. Including table support via [openparse](https://github.com/Filimoa/open-parse/) (no GPU needed by default) or via [UnstructuredPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/unstructured_pdfloader/).
* **Langfuse support**: If you set the appropriate langfuse environment variables they will be used. See [this guide](https://langfuse.com/docs/integrations/langchain/tracing) or [this one](https://langfuse.com/docs/integrations/litellm/tracing) to learn more (Note: this is disabled if using private_mode to avoid any leaks).
* **Document filtering**: based on regex for document content or metadata.
* **Fast**: Parallel document loading, parsing, embeddings, querying, etc.
* **Shell autocompletion** using [python-fire](https://github.com/google/python-fire/blob/master/docs/using-cli.md#completion-flag)
* **Notification callback**: Can be used for example to get summaries on your phone using [ntfy.sh](ntfy.sh).
* **Hacker mindset**: I'm a friendly dev! Just open an issue if you have a feature request or anything else.

### Roadmap

<details>
<summary>
Click to read more
</summary>

<i>This TODO list is maintained automatically by [MdXLogseqTODOSync](https://github.com/thiswillbeyourgithub/MdXLogseqTODOSync)</i>

<!-- BEGIN_TODO -->
- ## Most urgent
    - add more tests
        - add test for the private mode
        - add test for the testing models
        - add test for each loader
    - the logit bias is wrong for openai models: the token is specific to a given family of model
    - rewrite the python API to make it more useable. (also related to https://github.com/thiswillbeyourgithub/wdoc/issues/13)
        - be careful to how to use import_mode
    - make it easy to use wdoc as an openwebui pipeline (also related to https://github.com/thiswillbeyourgithub/wdoc/issues/4)
        - probably by creating a server with fastapi then a quick pipeline file
    - support other vector databases
    - make it easier to use other embeddings API than openai
    - understand why it appears that in some cases the sources id is never properly parsed
        - crash if source got lost  + arg to disable
- ### Features
    - make it easy to support other embeddings model, right now it seems only openai is easy to use. But litellm does not have an embeddings engine with langchain. Maybe there's one on github?
    - Way to add the title (or all metadata) of a document to its own text. Enabled by default. Because this would allow searching among many documents that don't refer to the original title (for example: material safety datasheets)
        - default value is "author" "page" title"
    - add a /save PATH command to save the chat and metadata to a json file
    - Accept input from stdin, to for example query directly from a manpage
        - make wdoc work if used with shell pipes
    - add image support printing via icat or via the other lib you found last time, would be useful for summaries etc
    - add an audio backend to use the subtitles from a video file directly
    - store the anki images as 'imagekeys' as the idea works for other parsers too
    - add an argument --whole_text to avoid chunking (this would just increase the chunk size to a super large number I guess)
    - add a filetype "custom_parser" and an argument "--custom_parser" containing a path to a python file. Must receive a docdict and a few other things and return a list of documents
        - then make it work with an online search engine for technical things
    - add a langchain code loader that uses aider to get the repomap
        - https://github.com/paul-gauthier/aider/issues/1043#issuecomment-2278486840
    - add a pikepdf loader because it can be used to automatically decrypt pdfs
    - add a query_branching_nb argument that asks an LLM to identify a list of keywords from the intermediate answers, then look again for documents using this keyword and filtering via the weak llm
    - write a script that shows how to use bertopic on the documents of wdoc
    - add a jina web search and async retriever https://jina.ai/news/jina-reader-for-search-grounding-to-improve-factuality-of-llms/
    - add a retriever where the LLM answer without any context
    - add support for readabilipy for parsing html
        - https://github.com/alan-turing-institute/ReadabiliPy
    - add an obsidian loader
        - https://pypi.org/project/obsidiantools/
    - add a /chat command to the prompt, it would enable starting an interactive session directly with the llm
    - make sure to expose loaders and batch_loader to make it easy to import by others
    - find a way to make it work with llm from simonw
    - make images an actual filetype
- ### Enhancements
    - use a dataclass or pydantic with validation to parse and store the backends. It would clarify the code as currently we just split the string and hope not to make a mistake
    - store the available tasks in a single var in misc.py
    - check that the task search work on things other than anki
    - create a custom custom retriever, derived from multiquery retriever that does actual parallel requests. Right now it's not the case (maybe in async but I don't plan on using async for now). This retriever seems a good part of the slow down.
    - Use an env var to drop_params of litellm
    - add more specific exceptions for file loading error. One exception for all, one for batch and one for individual loader
    - use heuristics to find the best number of clusters when doing semantic reranking
    - arg to use jina v3 embeddings for semantic batching because it allows specifying tasks that seem appropriate for that
    - add an env variable or arg to overload the backend url for whisper. Then set it always for you and mention it there: https://github.com/fedirz/faster-whisper-server/issues/5
    - find a way to set a max cost at which to crash if it exceeds a maximum cost during a query, probably via the price callback
    - anki_profile should be able to be a path
    - store wdoc's version and indexing timestamp in the metadata of the document
    - arg --oneoff that does not trigger the chat after replying. Allowing to not hog all the RAM if ran in multiple terminals for example through SSH
    - add a (high) token threshold above which two texts are not combined but just concatenated in the semantic order. It would avoid it loosing context. Use a --- separator
    - compute the cost of whisper and deepgram
    - use a pydantic basemodel for output instead of a dict
        - same for summaries, it should at least contain the method to substitute the sources and then back
    - investigate storing the vectors in a sqlite3 file
    - make a plugin to llm that looks like file-to-prompt from simonw
    - Always bind a user metadata to litellm for langfuse etc
        - Add more metadata to each request to langfuse more informative
    - add a reranker to better sort the output of the retrievers. Right now with the multiquery it returns way too many and I'm thinking it might be a bad idea to just crop at top_k as I'm doing currently
    - add a status argument that just outputs the logs location and size, the cache location and size, the number of documents etc
    - add the python magic of the file as a file metadata
    - add an env var to specify the threshold for relevant document by the query eval llm
    - find a way to return the evaluations for each document also
    - move retrievers.py in an embeddings folder
    - stop using lambda functions in the chains because it makes the code barely readable
    - when doing recursive summary: tell the model that if it's really sure that there are no modifications to do: it should just reply "EXIT" and it would save time and money instead of waiting for it to copy back the exact content
    - add image parsing as base64 metadata from pdf
    - use multiple small chains instead of one large and complicated and hard to maintain
    - add an arg to bypass query combine, useful for small models
    - break the load_embeddings into sub functions
    - tell the llm to write a special message if the parsing failed or we got a 404 or paywall etc
        - catch this text and crash
    - add check that all metadata is only made of int float and str
    - move the code that filters embeddings inside the embeddings.py file
        - this way we can dynamically refilter using the chat prompt
    - task summary then query should keep in context both the full text and the summary
    - if there's only one intermediate answer, pass it as answer without trying to recombine
    - filter_metadata should support an OR syntax
    - add a --show_models argument to display the list of available models
    - add a way to open the documents automatically, based on platform dirs etc. For ex if okular is installed, open pdfs directly at the right page
        - the best way would be to create opener.py that does a bit like loader but for all filetypes and platforms
    - add an image filetype: it will be either OCR'd using format and/or will be captioned using a multimodal llm, for example gpt4o mini
        - nanollava is a 0.5b that probably can be used for that with proper prompting
    - add a key/val arg to specify the trust we have in a doc, call this metadata context in the prompt
    - add an arg to return just the dict of all documents and embeddings. Notably useful to debug documents
    - use a class for the cli prompt, instead of a dumb function
    - arg to disable eval llm filtering
        - just answer 1 directly if no eval llm is set
    - display the number of documents and tokens in the bottom toolbar
    - add a demo gif
    - investigate asking the LLM to  add leading emojis to the bullet point for quicker reading of summaries
    - see how easy or hard it is to use an async chain
    - ability to cap the search documents capped by a number of tokens instead of a number of documents
    - for anki, allow using a query instead of loading with ankipandas
    - add a "try_all" filetype that will try each filetype and keep the first that works
    - add bespoke-minicheck from ollama to fact check when using RAG: https://ollama.com/library/bespoke-minicheck
        - or via their API directly : https://docs.bespokelabs.ai/bespoke-minicheck/api but they don't seem to properly disclose what they do with the data
    - add a way to use binary faiss index as its as efficient but faster and way more compact
<!-- END_TODO -->

</details>

### Supported filetypes
* **auto**: default, guess the filetype for you
* **url**: try many ways to load a webpage, with heuristics to find the better parsed one
* **youtube**: text is then either from the yt subtitles / translation or even better: using whisper / deepgram. Note that youtube subtitles are downloaded with the timecode (so you can ask 'when does the author talks about such and such) but at a lower sampling frequency (instead of one timecode per second, only one per 15s). Youtube chapters are also given as context to the LLM when summarizing, which probably help it a lot.
* **pdf**: 15 default loaders are implemented, heuristics are used to keep the best one and stop early. Table support via [openparse](https://github.com/Filimoa/open-parse/) or [UnstructuredPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/unstructured_pdfloader/). Easy to add more.
* **online_pdf**: via URL then treated as a **pdf** (see above)
* **anki**: any subset of an [anki](https://github.com/ankitects/anki) collection db. `alt` and `title` of images can be shown to the LLM, meaning that if you used [the ankiOCR addon](https://github.com/cfculhane/AnkiOCR) this information will help contextualize the note for the LLM.
* **string**: the cli prompts you for a text so you can easily paste something, handy for paywalled articles!
* **txt**: .txt, markdown, etc
* **text**: send a text content directly as path
* **local_html**: useful for website dumps
* **logseq_markdown**: thanks to my other project: [LogseqMarkdownParser](https://github.com/thiswillbeyourgithub/LogseqMarkdownParser) you can use your [Logseq graph](https://github.com/logseq/logseq/)
* **local_audio**: supports many file formats, can use either OpenAI's whisper or [deepgram](https://deepgram.com). Supports automatically removing silence etc. Note: audio that are too large for whisper (usually >25mb) are automatically split into smaller files, transcribed, then combined. Also, audio transcripts are converted to text containing timestamps at regular intervals, making it possible to ask the LLM when something was said.
* **local_video**: extract the audio then treat it as **local_audio**
* **online_media**: use youtube_dl to try to download videos/audio, if fails try to intercept good url candidates using playwright to load the page. Then processed as **local_audio** (but works with video too).
* **epub**: barely tested because epub is in general a poorly defined format
* **powerpoint**: .ppt, .pptx, .odp, ...
* **word**: .doc, .docx, .odt, ...
* **json_dict**: a text file containing a single json dict.

* **Recursive types**
    * **youtube playlists**: get the link for each video then process as **youtube**
    * **recursive_paths**: turns a path, a regex pattern and a filetype into all the files found recurisvely, and treated a the specified filetype (for example many PDFs or lots of HTML files etc).
    * **link_file**: turn a text file where each line contains a url into appropriate loader arguments. Supports any link, so for example webpage, link to pdfs and youtube links can be in the same file. Handy for summarizing lots of things!
    * **json_entries**: turns a path to a file where each line is a json **dict**: that contains arguments to use when loading. Example: load several other recursive types. An example can be found in `docs/json_entries_example.json`.
    * **toml_entries**: read a .toml file. An example can be found in `docs/toml_entries_example.toml`.

### Supported tasks
* **query** give documents and asks questions about it.
* **search** only returns the documents and their metadata. For anki it can be used to directly open cards in the browser.
* **summarize** give documents and read a summary. The summary prompt can be found in `utils/prompts.py`.
* **summarize_then_query** summarize the document then allow you to query directly about it.

## Walkthrough and examples

<details>
<summary>
Detailed example
</summary>

1. Say you want to ask a question about one pdf, that's simple: `wdoc --task="query" --path="my_file.pdf" --filetype="pdf" --modelname='openai/gpt-4o'`. Note that you could have just let `--filetype="auto"` and it would have worked the same.
* *Note: By default `wdoc` tries to parse args as kwargs so `wdoc query mydocument What's the age of the captain?` is parsed as `wdoc --task=query --path=mydocument --query "What's the age of the captain?"`. Likewise for summaries. This does not always work so use it only after getting comfortable with `wdoc`.*
2. You have several pdf? Say you want to ask a question about any pdf contained in a folder, that's not much more complicated : `wdoc --task="query" --path="my/other_dir" --pattern="**/*pdf" --filetype="recursive_paths" --recursed_filetype="pdf" --query="My question about those documents"`. So basically you give as path the path to the dir, as pattern the globbing pattern used to find the files relative to the path, set as filetype "recursive_paths" so that `wdoc` knows what arguments to expect, and specify as recursed_filetype "pdf" so that `wdoc` knows that each found file must be treated as a pdf. You can use the same idea to glob any kind of file supported by `wdoc` like markdown etc. You can even use "auto"! Note that you can either directly ask your question with `--query="my question"`, or wait for an interactive prompt to pop up, or just pass the question as *args like so `wdoc [your kwargs] here is my question`.
3. You want more? You can write a `.json` file where each line (`#comments` and empty lines are ignored) will be parsed as a list of argument. For example one line could be : `{"path": "my/other_dir", "pattern": "**/*pdf", "filetype": "recursive_paths", "recursed_filetype": "pdf"}`. This way you can use a single json file to specify easily any number of sources. `.toml` files are also supported.
4. You can specify a "source_tag" metadata to help distinguish between documents you imported. It is EXTREMELY recommended to include a source_tag to any document you want to save: especially if using recursive filetypes. This is because after loading all documents `wdoc` use the source_tag to see if it should continue or crash. If you want to load 10_000 pdf in one go as I do, then it makes sense to continue if some failed to crash but not if a whole source_tag is missing.
5. Now say you do this with many many documents, as I do, you of course can't wait for the indexing to finish every time you have a question (even though the embeddings are cached). You should then add `--save_embeds_as=your/saving/path` to save all this index in a file. Then simply do `--load_embeds_from=your/saving/path` to quickly ask queries about it!
6. To know more about each argument supported by each filetype, `wdoc --help`
7. There is a specific recursive filetype I should mention: `--filetype="link_file"`. Basically the file designated by `--path` should contain in each line (`#comments` and empty lines are ignored) one url, that will be parsed by `wdoc`. I made this so that I can quickly use the "share" button on android from my browser to a text file (so it just appends the url to the file), this file is synced via [syncthing](https://github.com/syncthing/syncthing) to my browser and `wdoc` automatically summarize them and add them to my [Logseq](https://github.com/logseq/logseq/). Note that the url is parsed in each line, so formatting is ignored, for example it works even in markdown bullet point list.
8. If you want to make sure your data remains private here's an example with ollama: `wdoc --private --llms_api_bases='{"model": "http://localhost:11434", "query_eval_model": "http://localhost:11434"}' --modelname="ollama_chat/gemma:2b" --query_eval_modelname="ollama_chat/gemma:2b" --embed_model="BAAI/bge-m3" my_task`
9. Now say you just want to summarize [Tim Urban's TED talk on procrastination](https://www.youtube.com/watch?v=arj7oStGLkU): `wdoc --task=summary --path='https://www.youtube.com/watch?v=arj7oStGLkU' --youtube_language="en" --disable_md_printing`:

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

</details>

## Getting started
*`wdoc` currently requires Python 3.11 to run for now. Make sure that your Python version matches this one or it will not work.*
1. To install:
    * Using pip: `pip install -U wdoc`
    * Or to get a specific git branch:
        * `dev` branch: `pip install git+https://github.com/thiswillbeyourgithub/wdoc@dev`
        * `main` branch: `pip install git+https://github.com/thiswillbeyourgithub/wdoc@main`
    * You can also use uvx or pipx. But as I'm not experiences with them I don't know if that can cause issues with for example caching etc. Do tell me if you tested it!
        * Using uvx: `uvx wdoc --help`
        * Using pipx: `pipx run wdoc --help`
    * In any case, it is recommended to try to install pdftotext with `pip install -U wdoc[pdftotext]` as well as add fasttext support with `pip install -U wdoc[fasttext]`.
    * If you plan on contributing, you will also need `wdoc[dev]` for the commit hooks.
2. Add the API key for the backend you want as an environment variable: for example `export OPENAI_API_KEY="***my_key***"`
3. Launch is as easy as using `wdoc --task=query --path=MYDOC [ARGS]`
    * If for some reason this fails, maybe try with `python -m wdoc`. And if everything fails, try with `uvx wdoc@latest`, or as last resort clone this repo and try again after `cd` inside it? Don't hesitate to open an issue.
    * To get shell autocompletion: if you're using zsh: `eval $(cat shell_completions/wdoc_completion.zsh)`. Also provided for `bash` and `fish`. You can generate your own with `wdoc -- --completion MYSHELL > my_completion_file"`.
    * Don't forget that if you're using a lot of documents (notably via recursive filetypes) it can take a lot of time (depending on parallel processing too, but you then might run into memory errors).
4. To ask questions about a local document: `wdoc query --path="PATH/TO/YOUR/FILE" --filetype="auto"`
    * If you want to reduce the startup time by directly loading the embeddings from a previous run (although the embeddings are always cached anyway): add `--saveas="some/path"` to the previous command to save the generated embeddings to a file and replace with `--loadfrom "some/path"` on every subsequent call.
5. For more: read the documentation at `wdoc --help`

## Scripts made with wdoc
* *More to come in [the scripts folder](./scripts/)*
* [Ntfy Summarizer](scripts/NtfySummarizer): automatically summarize a document from your android phone using [ntfy.sh](ntfy.sh)
* [TheFiche](scripts/TheFiche): create summaries for specific notions directly as a [logseq](https://github.com/logseq/logseq) page.
* [FilteredDeckCreator](scripts/FilteredDeckCreator): directly create an [anki](https://ankitects.github.io/) filtered deck from the cards found by `wdoc`.

## FAQ

<details>
<summary>
FAQ
</summary>

* **Who is this for?**
    * `wdoc` is for power users who want document querying on steroid, and in depth AI powered document summaries.
* **What's RAG?**
    * A RAG system (retrieval augmented generation) is basically an LLM powered search through a text corpus.
* **Why make another RAG system? Can't you use any of the others?**
    * I'm Olicorne, a medical student so I need to be able to ask medical question from **a lot** (tens of thousands) of documents, of different types (epub, pdf, [anki](https://ankitects.github.io/) database, [Logseq](https://github.com/logseq/logseq/), website dump, youtube videos and playlists, recorded conferences, audio files, etc).
* **Why is `wdoc` better than most RAG system to ask questions on documents?**
    * It uses both a strong and query_eval LLM. After finding the appropriate documents using embeddings, the query_eval LLM is used to filter through the documents that don't seem to be about the question, then the strong LLM answers the question based on each remaining documents, then combines them all in a neat markdown. Also `wdoc` is very customizable.
* **Why can `wdoc` also produce summaries?**
    * I have little free time so I needed a tailor made summary feature to keep up with the news. But most summary systems are rubbish and just try to give you the high level takeaway points, and don't handle properly text chunking. So I made my own tailor made summarizer. **The summary prompts can be found in `utils/prompts.py` and focus on extracting the arguments/reasonning/though process/arguments of the author then use markdown indented bullet points to make it easy to read.** It's really good! The prompts dataclass is not frozen so you can provide your own prompt if you want.
* **What other tasks are supported by `wdoc`?**
    * See [Supported tasks](#Supported-tasks).
* **Which LLM providers are supported by `wdoc`?**
    * `wdoc` supports virtually any LLM provider thanks to [litellm](https://docs.litellm.ai/). It even supports local LLM and local embeddings (see [Walkthrough and examples](#Walkthrough-and-examples) section).
* **What do you use `wdoc` for?**
    * I follow heterogeneous sources to keep up with the news: youtube, website, etc. So thanks to `wdoc` I can automatically create awesome markdown summaries that end up straight into my [Logseq](https://github.com/logseq/logseq/) database as a bunch of `TODO` blocks.
    * I use it to ask technical questions to my vast heterogeneous corpus of medical knowledge.
    * I use it to query my personal documents using the `--private` argument.
    * I sometimes use it to summarize a documents then go straight to asking questions about it, all in the same command.
    * I use it to ask questions about entire youtube playlists.
    * Other use case are the reason I made the [scripts made with `wdoc` section](#scripts-made-with-wdoc)
* **What's up with the name?**
    * One of my favorite character (and somewhat of a rolemodel is [Winston Wolf](https://www.youtube.com/watch?v=UeoMuK536C8) and after much hesitation I decided `WolfDoc` would be too confusing and `WinstonDoc` sounds like something micro$oft would do. Also `wd` and `wdoc` were free, whereas `doctools` was already taken. The initial name of the project was `DocToolsLLM`, a play on words between 'doctor' and 'tool'.
* **How can I improve the prompt for a specific task without coding?**
    * Each prompt of the `query` task are roleplaying as employees working for WDOC-CORP©, either as `Eve the Evaluator` (the LLM that filters out relevant documents), `Anna the Answerer` (the LLM that answers the question from a filtered document) or `Carl the Combiner` (the LLM that combines answers from Answerer as one). There's also `Sam the Summarizer` for summaries and `Raphael the Rephraser` to expand your query. They are all receiving orders from you if you talk to them in a prompt.
* **How can I use `wdoc`'s parser for my own documents?**
    * If you are in the shell cli you can easily use `wdoc parse my_file.pdf` (this actually replaces the call to call instead `wdoc_parse_file my_file.pdf`).
    add `--only_text` to only get the text and no metadata. If you're having problem with argument parsing you can try adding the `--pipe` argument.
    * If you want the document using python:
        ``` python
        from wdoc import wdoc
        list_of_docs = Wdoc.parse_file(path=my_path)
        ```
    * Another example would be to use wdoc to parse an anki deck: `wdoc_parse_file --filetype "anki" --anki_profile "Main" --anki_deck "mydeck::subdeck1" --anki_notetype "my_notetype" --anki_template "<header>\n{header}\n</header>\n<body>\n{body}\n</body>\n<personal_notes>\n{more}\n</personal_notes>\n<tags>{tags}</tags>\n{image_ocr_alt}" --anki_tag_filter "a::tag::regex::.*something.*" --only_text`
* **What should I do if my PDF are encrypted?**
    * If you're on linux you can try running `qpdf --decrypt input.pdf output.pdf`
        * I made a quick and dirty batch script for [in this repo](https://github.com/thiswillbeyourgithub/PDF_batch_decryptor)
* **How can I add my own pdf parser?**
    * Write a python class and add it there: `wdoc.utils.loaders.pdf_loaders['parser_name']=parser_object` then call `wdoc` with `--pdf_parsers=parser_name`.
        * The class has to take a `path` argument in `__init__`, have a `load` method taking
        no argument but returning a `List[Document]`. Take a look at the `OpenparseDocumentParser`
        class for an example.

* **What should I do if I keep hitting rate limits?**
    * The simplest way is to add the `debug` argument. It will disable multithreading,
        multiprocessing and LLM concurrency. A less harsh alternative is to set the
        environment variable `WDOC_LLM_MAX_CONCURRENCY` to a lower value.

* **How can I run the tests?**
    * Try `python -m pytest tests/test_wdoc.py -v -m basic` to run the basic tests, and `python -m pytest tests/test_wdoc.py -v -m api` to run the test that use external APIs.

* **How can I query a text but without chunking? / How can I query a text with the full text as context?**
    * If you set the environment variable `WDOC_MAX_CHUNK_SIZE` to a very high value and use a model with enough context according to litellm's metadata, then no chunking will happen and the LLM will have the full text as context.

</details>
