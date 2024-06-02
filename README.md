# DocToolsLLM
* **Goal** use [LangChain](https://python.langchain.com/) to summarize, search or query documents. I'm a medical student so I need to be able to query from **tens of thousands** of documents, of different types ([Supported filetypes](#Supported filetypes)). I also have little free time so I needed a tailor made summary feature to keep up with the news.
* **Current status**: **Already great but still under development**. Expect some breakage but they can be fixed usually in a few minutes if you open an issue here. The main branch is usually fine but the dev branch is usually better. I use it almost daily. I accept feature requests, issues are extremely appreciated for any reason including typos etc, I accept pull request but prefer asking me first. I have many improvements in the pipeline but do this on my spare time. Do tell me if you have specific needs.

# Table of contents
- [DocToolsLLM](#doctoolsllm)
- [DocToolsLLM in a few questions](#doctoolsllm-in-a-few-questions)
  - [What's RAG?](#what's-rag?)
  - [Why make another RAG system? Can't you use any of the others?](#why-make-another-rag-system?-can't-you-use-any-of-the-others?)
  - [Why is DocToolsLLM better than most RAG system to ask questions on documents?](#why-is-doctoolsllm-better-than-most-rag-system-to-ask-questions-on-documents?)
  - [Why can DocToolsLLM also produce summaries?](#why-can-doctoolsllm-also-produce-summaries?)
  - [What other tasks are supported by DocToolsLLM?](#what-other-tasks-are-supported-by-doctoolsllm?)
  - [Which LLM providers are supported by DocToolsLLM?](#which-llm-providers-are-supported-by-doctoolsllm?)
  - [What do you use DocToolsLLM for?](#what-do-you-use-doctoolsllm-for?)
  - [Features](#features)
    - [Supported filetypes](#supported-filetypes)
      - [Recursive types](#recursive-types)
      - [Walkthrough and examples](#walkthrough-and-examples)
    - [Supported tasks](#supported-tasks)
    - [Known issues that are not yet fixed](#known-issues-that-are-not-yet-fixed)
  - [Getting started](#getting-started)
  - [Notes](#notes)

# DocToolsLLM in a few questions
## What's RAG?
* A RAG system (retrieval augmented generation) is basically an LLM powered search through a text corpus.
## Why make another RAG system? Can't you use any of the others?
* I'm a medical student so I need to be able to ask medical question from **a lot** (tens of thousands) of documents, of different types (epub, pdf, [anki](https://ankitects.github.io/) database, [Logseq](https://github.com/logseq/logseq/), website dump, youtube videos and playlists, recorded conferences, audio files, etc).
## Why is DocToolsLLM better than most RAG system to ask questions on documents?
* It uses both a strong and query_eval LLM. After finding the appropriate documents using embeddings, the query_eval LLM is used to filter through the documents that don't seem to be about the question, then the strong LLM answers the question based on each remaining documents, then combines them all in a neat markdown. Also DocToolsLLM is very customizable.
## Why can DocToolsLLM also produce summaries?
* I have little free time so I needed a tailor made summary feature to keep up with the news. But most summary systems are rubbish and just try to give you the high level takeaway points, and don't handle properly text chunking. So I made my own tailor made summarizer. **The summary prompts can be found in `utils/prompts.py` and focus on extracting the arguments/reasonning/though process/arguments of the author then use markdown indented bullet points to make it easy to read.** It's really good!
## What other tasks are supported by DocToolsLLM?
* Summarize text from any [Supported filetypes](#Supported filetypes).
* Ask questions about a large heterogeneous corpus.
* Search the relevant documents using embeddings.
* Search the relevant documents using embeddings then filtering using a cheap LLM.
## Which LLM providers are supported by DocToolsLLM?
* DocToolsLLM supports virtually any LLM provider thanks to [litellm](https://docs.litellm.ai/). It even supports local LLM and local embeddings (see [Walkthrough and examples](#Walkthrough and examples) section).
## What do you use DocToolsLLM for?
* I follow heterogeneous sources to keep up with the news: youtube, website, etc. So thanks to DocToolsLLM I can automatically create awesome markdown summaries that end up straight into my [Logseq](https://github.com/logseq/logseq/) database as a buch of `TODO` blocks.
* I use it to ask technical questions to my vast heterogeneous corpus of medical knowledge.
* I use it to query my personal documents using the `--private` argument.
* I sometimes use it to summarize a documents then go straight to asking questions about it, all in the same command.

## Features
* **Advanced RAG**: first the documents are retrieved using embedding, then a weak LLM model is used to tell which of those document is not relevant, then the strong LLM is used to answer the question using each individual remaining documents, then all relevant answers are combined into a single short markdown-formatted answer. It even supports a special syntax like "QE // QA" were QE is a question used to filter the embeddings and QA is the actual question you want answered.
* **Advanced summary**: Instead of unusable high level points, compress the reasonning, arguments, though process etc of the author into an easy to skim markdown file.
* **Multiple LLM providers**: OpenAI, Mistral, Claude, Ollama, Openrouter, etc. Thanks to [litellm](https://docs.litellm.ai/).
* **Private LLM**: take some measures to make sure no data leaves your computer and goes to an LLM provider: no API keys are used, all `api_base` are user set, cache are isolated from the rest etc.
* **Many tasks**: See [Supported tasks](#Supported tasks).
* **Many filetypes**: also supports combination to load recursively or define complex heterogenous corpus like a list of files, list of links, using regex, youtube playlists etc. See [Supported filestypes](#Supported filetypes). All filetype can be seamlessly combined in the same index, meaning you can query your anki collection at the same time as your work PDFs).
* **Caching**: speed things up, as well as index storing and loading (handy for large collections).
* **Markdown formatted answers and summaries**: using [rich](https://github.com/Textualize/rich).
* **Document filtering**: based on regex for document content or metadata.
* **Fast**: Parallel document parsing and embedding.
* **Shell autocompletion** using [python-fire](https://github.com/google/python-fire/blob/master/docs/using-cli.md#completion-flag)
* **Statically typed**: Optional runtime type checking. Opt in with an environment flag: `DOCTOOLS_TYPECHECKING="disabled / warn / crash" python -m DocToolsLLM`.
* Very customizable, with a friendly dev! Just open an issue if you have a feature request or anything else.

### Supported filetypes
*(see [how to combine filetypes](#Recursive types))*
* **infer** (will try to guess for you)
* **youtube videos**
* **Logseq md files** (this makes uses of my other project: [LogseqMarkdownParser](https://github.com/thiswillbeyourgithub/LogseqMarkdownParser)
* **local PDF**
* **remote PDF** via URL
* **text files** (.txt, markdown, etc)
* **anki** collection
* **string** (just paste your text into the app)
* **html files** (useful for website dumps)
* **audio files** (beta-ish but mostly stable: mp3, m4a, ogg, flac)
* **epub files**
* **Microsoft Powerpoint files** (.ppt, .pptx, .odp, ...)
* **Microsoft Word documents** (.doc, .docx, .odt, ...)
* **string** (the cli prompts you for a text so you can easily paste something, including paywalled articles)

#### Recursive types
* **json_list** (you give as argument a path to a file where each line is a json_list that contains the loader arguments. This can be used for example to load several files in a row). An example can be found in `utils/json_list_example.txt`
* **recursive** (you give a path and a regex pattern and a filetype, it finds all the files)
* **link_file** (you give a text file where each line is a url, proper filetype for each url will be inferred)
* **youtube playlists** turns a youtube_playlist into a list of youtube videos

#### Walkthrough and examples
1. Say you want to ask a question about one pdf, that's simple: `python -m DocToolsLLM --task "query" --path "my_file.pdf" --filetype="pdf"`. Note that you could have just let `--filetype="infer"` and it would have worked the same.
2. You have several pdf? Say you want to ask a question about any pdf contained in a folder, that's not much more complicated : `python -m DocToolsLLM --task "query" --path "my/other_dir" --pattern "**/*pdf" --filetype "recursive" --recursed_filetype "pdf"`. So basically you give as path the path to the dir, as pattern the globbing pattern used to find the files relative to the path, set as filetype "recursive" so that DoctoolsLLM knows what arguments to expect, and specify as recursed_filetype "pdf" so that doctools knows that each found file must be treated as a pdf. You can use the same idea to glob any kind of file supported by DoctoolsLLM like markdown etc. You can even use "infer"!
3. You want more? You can write a `.json` file where each line (# comments and empty lines are ignored) will be parsed as a list of argument. For example one line could be : `{"path": "my/other_dir", "pattern": "**/*pdf", "filetype": "recursive", "recursed_filetype": "pdf"}`. This way you can use a single json file to specify easily any number of sources.
4. You can specify a "source_tag" metadata to help distinguish between documents you imported.
5. Now say you do this with many many documents, as I do, you of course can't wait for the indexing to finish every time you have a question (even though the embeddings are cached). You should then add `--save_embeds_as=your/saving/path` to save all this index in a file. Then simply do `--load_embeds_from=your/saving/path` to quickly ask queries about it!
6. To know more about each argument supported by each filetype, `python -m DoctoolsLLM --help`
7. There is a specific recursive filetype I should mention: `--filetype="link_file"`. Basically the file designated by `--path` should contain in each line (# comments and empty lines are ignored) one url, that will be parsed by DoctoolsLLM. I made this so that I can quickly use the "share" button on android from my browser to a text file (so it just appends the url to the file), this file is synced via [syncthing](https://github.com/syncthing/syncthing) to my browser and DoctoolsLLM automatically summarize them and add them to my [Logseq](https://github.com/logseq/logseq/). Note that the url is parsed in each line, so formatting is ignored, for example it works even in markdown bullet point list.
8. If you want to make sure your data remains private here's an example with ollama: `python -m DoctoolsLLM --private --llms_api_bases='{"model": "http://localhost:11434", "query_eval_model": "http://localhost:11434"}' --modelname="ollama_chat/gemma:2b" --query_eval_modelname="ollama_chat/gemma:2b" --embed_model="BAAI/bge-m3" --task=my_task`
9. Now say you just want to summarize a webpage: `python -m DocToolsLLM --task="summary" --path="https://arstechnica.com/science/2024/06/to-pee-or-not-to-pee-that-is-a-question-for-the-bladder-and-the-brain/"`.
<details><summary>Summary result:</summary>
![](./images/summary.png)
</details>


### Supported tasks
* **query** give documents and asks questions about it.
* **search** only returns the documents and their metadata. For anki it can be used to directly open cards in the browser.
* **summarize** give documents and read a summary. The summary prompt can be found in `utils/prompts.py`.
* **summarize_then_query** summarize the document then allow you to query directly about it.
* **summarize_link_file** this summarizes all the links and adds it to an output file. (logseq format is supported)

### Known issues that are not yet fixed
* whisper implementation is a bit flaky and will be improved

## Getting started
*Tested on python 3.9 and 3.11.7*
* `python -m pip install git+https://github.com/thiswillbeyourgithub/DocToolsLLM.git@dev`
* Or for the supposedly more stable branch: `python -m pip install git+https://github.com/thiswillbeyourgithub/DocToolsLLM.git@main`
* Add the API key for the backend you want to use: add a file "{BACKEND}_API_KEY.txt" to the root that contains your backend's API key. For example "OPENAI_API_KEY.txt".
* Launch using `python -m DocToolsLLM --task=query [ARGS]`
* To ask questions about a document: `python -m DoctoolsLLM --task="query" --path="PATH/TO/YOUR/FILE" --filetype="infer"`
* If you want to reduce the startup time, you can use --saveas="some/path" to save the loaded embeddings from last time and --loadfrom "some/path" on every subsequent call. (In any case, the embeddings are always cached)
* For more: read the documentation at `python -m DocToolsLLM --help`
* For shell autocompletion: `eval "$(python -m DocToolsLLM -- --completion)"`

## Notes
* Before summarizing, if the beforehand estimate of cost is above $5, the app will abort to be safe just in case you drop a few bibles in there. (Note: the tokenizer usedto count tokens to embed is the OpenAI tokenizer, which is not universal)
