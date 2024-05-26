# DocToolsLLM
* **Goal** use [LangChain](https://python.langchain.com/) to summarize, search (aka RAG), ask questions from lots of documents, and different types of documents.
* **Current status** **Under development** but the main branch is fine. Used daily by the author. Accepting pull requests, issues are extremely appreciated for any reason, including typos etc.

## Example use case
* Quickly summarize lots of diverse content (including youtube) and add it to [Logseq](https://github.com/logseq/logseq/): I use it to automatically summarize it the thought process of many articles I'd like to read.
* Ask questions about a large and diverse corpus of documents: I use it to ask medical questions in all my lessons, all my PDFs and all my anki collections, all using a single command.
* Summarize a documents then ask questions about it immediately

## Features
* Advanced RAG system: first the documents are retrieved using embedding, then a weak LLM model is used to tell which of those document is not relevant, then the strong LLM is used to answer the question using each individual remaining documents, then all relevant answers are combined into a single short markdown-formatted answer. It even supports a special syntax like "QE // QA" were QE is a question used to filter the embeddings and QA is the actual question you want answered.
* private model: take some measures to make sure no data leaves your computer and goes to an LLM provider:no API keys are used, all api_base are user set, cache are separated etc.
* Multiple type of tasks implemented. See below.
* Many supported filetype, including advanced ones like loading from list of files, list of links, using regex, youtube playlists etc. See below.
* All filetype can be seamlessly combined in the same index, meaning you can query your anki collection at the same time as your work PDFs).
* Caching is used to speed things up, as well as index storing and loading (handy for large collections).
* Markdown formatted answers
* Filtering via document metadata (e.g. to include only documents that contain "anki" in any value of any of its metadata dict: `--filter_metadata="~anki"`)
* Several LLM implemented (by default OpenAI, but Llamacpp and GPT4ALL are implemented). Adding more is very easy.
* Several embedding models implemented (by default OpenAI but sentence-transformers is implemented (including GLOVE with stop words), HuggingFace models can be used etc). Note that if using OpenAI the cost will be computed beforehand to make you confirm for embeddings larger that $1.
* Multithreaded document parsing and embedding.
* Very customizable.
* Shell completion using [python-fire](https://github.com/google/python-fire/blob/master/docs/using-cli.md#completion-flag)
* lazy loading of heavy libraries. If this is causing any issue, you can disable it with an environment flag: `DOCTOOLS_NO_LAZYLOADING="false" python DocToolsLLM.py`
* Optional runtime type checking. If this is causing any issue, you can disable it with an environment flag: `DOCTOOLS_TYPECHECKING="disabled" python DocToolsLLM.py`. Or set to 'warn' to just warn of typechecking errors.
* I'm a nice person so just open an issue if you have a feature request or anything else.
* Phone notification via [ntfy.sh](ntfy.sh) to tell you about costs, useful when using GPT-4 and cron.

### Supported filetype (see below to combine them):
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
* **string** (the cli prompts you for a text so you can easily paste something, including paywalled articles)

#### Recursive types
* **json_list** (you give as argument a path to a file where each line is a json_list that contains the loader arguments. This can be used for example to load several files in a row). An example can be found in `utils/json_list_example.txt`
* **recursive** (you give a path and a regex pattern and a filetype, it finds all the files)
* **link_file** (you give a text file where each line is a url, proper filetype for each url will be inferred)
* **youtube playlists** turns a youtube_playlist into a list of youtube videos

#### Workflow and combining filetypes
1. Say you want to ask a question about one pdf, that's simple: `python DocToolsLLM.py --task "query" --path "my_file.pdf" --filetype="pdf"`. Note that you could have just let `--filetype="infer"` and it would have worked the same.
2. You have several pdf? Say you want to ask a question about any pdf contained in a folder, that's not much more complicated : `python DocToolsLLM.py --task "query" --path "my/other_dir" --pattern "**/*pdf" --filetype "recursive" --recursed_filetype "pdf"`. So basically you give as path the path to the dir, as pattern the globbing pattern used to find the files relative to the path, set as filetype "recursive" so that DoctoolsLLM knows what arguments to expect, and specify as recursed_filetype "pdf" so that doctools knows that each found file must be treated as a pdf. You can use the same idea to glob any kind of file supported by DoctoolsLLM like markdown etc. You can even use "infer"!
3. You want more? You can write a `.json` file where each line (# comments and empty lines are ignored) will be parsed as a list of argument. For example one line could be : `{"path": "my/other_dir", "pattern": "**/*pdf", "filetype": "recursive", "recursed_filetype": "pdf"}`. This way you can use a single json file to specify easily any number of sources.
4. Now say you do this with many many documents, as I do, you of course can't wait for the indexing to finish every time you have a question (even though the embeddings are cached). You should then add `--save_embeds_as=your/saving/path` to save all this index in a file. Then simply do `--load_embeds_from=your/saving/path` to quickly ask queries about it!
5. To know more about each argument supported by each filetype, `python DoctoolsLLM --help`
6. There is a specific recursive filetype I should mention: `--filetype="link_file"`. Basically the file designated by `--path` should contain in each line (# comments and empty lines are ignored) one url, that will be parsed by DoctoolsLLM. I made this so that I can quickly use the "share" button on android from my browser to a text file (so it just appends the url to the file), this file is synced via [syncthing](https://github.com/syncthing/syncthing) to my browser and DoctoolsLLM automatically summarize them and add them to my [Logseq](https://github.com/logseq/logseq/). Note that the url is parsed in each line, so formatting is ignored, for example it works even in markdown bullet point list.
7. If you want to make sure your data remains private here's an example with ollama: `python DoctoolsLLM.py --private --llms_api_bases='{"model": "localhost:11434", "query_eval_model": "localhost:11434"}' --modelname="ollama/my_modelname" --query_eval_modelname="ollama/my_evalmodel" --embed_model="BAAI/bge-m3" --task=my_task`

### Supported tasks:
* **query** give documents and asks questions about it.
* **search** only returns the documents and their metadata. For anki it can be used to directly open cards in the browser.
* **summarize** give documents and read a summary. The summary prompt can be found in `utils/prompts.py`.
* **summarize_then_query** summarize the document then allow you to query directly about it.
* **summarize_link_file** this summarizes all the links and adds it to an output file. (logseq format is supported)

### Known issues that are not yet fixed
* filtering document by content is currently disabled
* whisper implementation is a bit flaky and will be improved

## Getting started
* `git clone`
* `python -m pip install -r requirements.txt`
* some package used to load files will not be installed by this command. Pay attention to the error message then use pip install as needed. For example :
    * for youtube: `python -m pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"` (this is the latest youtube_dl from the git repo, much more recent than their latest release).
    * for urls: `python -m pip install goose3`
* Add the API key for the backend you want to use: add a file "{BACKEND}_API_KEY.txt" to the root that contains your backend's API key. For example "REPLICATE_API_KEY" or "OPENAI_API_KEY".
* To ask questions about a document: `python ./DoctoolsLLM.py --task="query" --path="PATH/TO/YOUR/FILE" --filetype="infer"`
* If you want to reduce the startup time, you can use --saveas="some/path" to save the loaded embeddings from last time and --loadfrom "some/path" on every subsequent call. (In any case, the emebeddings are always cached)
* For more: read the documentation at `python DocToolsLLM.py --help`
* For shell autocompletion: `eval "$(python DocToolsLLM.py -- --completion)"`

## Notes
* Before summarizing, if the beforehand estimate of cost is above $5, the app will abort to be safe just in case you drop a few bibles in there. (Note: the tokenizer usedto count tokens to embed is the OpenAI tokenizer, which is not universal)
* the multilingual embeddings from [sentence transformers](https://www.sbert.net/docs/pretrained_models.html/) have a very small max token length (down to 128!) and are probably unsuitable for most documents. That's why I also implemented GLOVE embeddings which are predictably bad but still allow private use (locally on your computer). It is important to note that the current GLOVE implementation removes the stop words in the documents just before computing the "embeddings", but not at query time, making the retrieval task kinda terrible. If someone is interested I might add a query augmentation strategy. Otherwise the best bet might be to use a rolling window of sentence transformer embeddings then averaging.
