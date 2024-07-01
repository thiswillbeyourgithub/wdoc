# Global arguments

* `--task`: str
    * Accepted values:
        * `query`: means to load the input files then wait for user question.
        * `search`: means only return the document corresponding to the search
        * `summarize`: means the input will be passed through a summarization prompt.
        * `summarize_then_query`: summarize the text then open the prompt to allow querying directly the source document.

* `--filetype`: str, default `infer`
    * the type of input. Depending on the value, different other parameters
    are needed. If json_entries is used, the line of the input file can contain
    any of those parameters as long as they are as json. You can find
    an example of json_entries file in `DocToolsLLM/docs/json_entries_example.txt`

    * Supported values:
        * `infer`: will guess the appropriate filetype based on `--path`.
            Irrelevant for some filetypes, eg if `--filetype`=anki
        * `youtube`: `--path` must link to a youtube video
        * `youtube_playlist`: `--path` must link to a youtube playlist
        * `pdf`: `--path` is path to pdf
        * `txt`: `--path` is path to txt
        * `url`: `--path` must be a valid http(s) link
        * `anki`: must be set: `--anki_profile`. Optional: `--anki_deck`,
        `--anki_notetype`, `--anki_mode`. See in loader specific arguments
        below for details.
        * `string`: no other parameters needed, will provide a field where
        you must type or paste the string
        * `local_audio`: must be set: `--whisper_prompt`, `--whisper_lang`. The model used will be `whisper-1`

        * `json_entries`: `--path` is path to a txt file that contains a json
        for each line containing at least a filetype and a path key/value
        but can contain any parameters described here
        * `recursive_paths`: `--path` is the starting path `--pattern` is the globbing
        patterns to append `--exclude` and `--include` can be a list of regex
        applying to found paths (include is run first then exclude, if the
        pattern is only lowercase it will be case insensitive) `--recursed_filetype`
        is the filetype to use for each of the found path
        * `link_file`: `--path` must point to a file where each line is a link
        that will be summarized. The resulting summary will be added to `--out_file`.
        Links that have already been summarized in out_file will be skipped
        (the out_file is never overwritten). If a line is a markdown like
        like [this](link) then it will be parsed as a link.
        Empty lines and starting with # are ignored.

---

* `--modelname`: str, default `"openrouter/anthropic/claude-3.5-sonnet:beta"`
    * Keep in mind that given that the default backend used is litellm
    the part of modelname before the slash (/) is the backend name (also called provider).
    If the backend is 'testing/' then a fake LLM will be used
    for debugging purposes.
    If the value is not part of the model list of litellm, will use
    fuzzy matching to find the best match.

---

* `--embed_model`: str, default `"openai/text-embedding-3-small"`
    * Name of the model to use for embeddings. Must contain a '/'
    Everything before the slash is the backend and everything
    after the / is the model name.
    Available backends: openai, sentencetransformers,
    huggingface, llamacppembeddings

    * Note:
        * the device used by default for huggingface is 'cpu' and not 'cuda'
        * If you change this, the embedding cache will be usually
            need to be recomputed with new elements (the hash
            used to check for previous values includes the name of the model
            name)
        * If the backend if llamacppembeddings, the modelname must be the path to
            the model. For example: 'llamacppembeddings/my_model_file'

* `--embed_kwargs`: dict, default `None`
    * dictionnary of keyword arguments to pass to the embedding.

* `--save_embeds_as`: str, default `"{user_dir}/latest_docs_and_embeddings"`
    * only used if task is query
    save the latest 'inputs' to a file. Can be loaded again with
    --load_embeds_from to speed up loading time. This loads both the
    split documents and embeddings but will not update itself if the
    original files have changed.
    {user_dir} is automatically replaced by the path to the usual
    cache folder for the current user

* `--load_embeds_from`: str, default `None`
    * path to the file saved using `--save_embeds_as`

* `--top_k`: int, default `20`
    * number of chunks to look for when querying. It is high because the
    eval model is used to refilter the document after the embeddings
    first pass.

---

* `--query`: str, default `None`
    * if str, will be directly used for the first query if task in `["query", "search", "summarize_then_query"]`

* `--query_retrievers`: str, default `"default"`
    * must be a string that specifies which retriever will be used for
    queries depending on which keyword is inside this string.

    * Possible values (can be combined if separated by _):
        * `default`: cosine similarity retriever
        * `hyde`: hyde retriever
        * `knn`: knn
        * `svm`: svm
        * `parent`: parent chunk

    if contains `hyde` but modelname contains `testing` then `hyde` will
    be removed.

* `--query_eval_modelname`: str, default `"openrouter/anthropic/claude-3.5-sonnet:beta"`
    * Cheaper and quicker model than modelname. Used for intermediate
    steps in the RAG, not used in other tasks.
    If the value is not part of the model list of litellm, will use
    fuzzy matching to find the best match.
    None to disable.

* `--query_eval_check_number`: int, default `1`
    * number of pass to do with the eval llm to check if the document
    is indeed relevant to the question. The document will not
    be processed if all answers from the eval llm are 0, and will
    be processed otherwise.
    For eval llm that don't support setting `n`, multiple
    completions will be called, which costs more.

* `--query_relevancy`: float, default `0.1`
    * threshold underwhich a document cannot be considered relevant by
    embeddings alone.

* `--query_condense_question`: bool, default `True`
    * if True, will not use a special LLM call to reformulate the question
    when task is `query`. Otherwise, the query will be reformulated as
    a standalone question. Useful when you have multiple questions in
    a row.
    Disabled if using a testing model.

---

* `--summary_n_recursion`: int, default `1`
    * after summarizing, will go over the summary that many times to fix
    indentation, repetitions etc.
        * 0 means disabled.
        * 1 means that the original summary will be checked once.
        * 2 means that the original summary, will checked, then
        the check version will be checked again.
        We stop when equilibrium is reached (meaning the summary
        did not change).
    * If `--out_file` is used, each intermediate summary will be saved
    with the name `{out_file}.n.md` with n being the n-1th recursive summary.

* `--summary_language`: str, default `"the same language as the document"`
    * When writing a summary, the LLM will write using the language
    specified in this argument. If it's `[same as input]`, the LLM
    will not translate.

---

* `--llm_verbosity`: bool, default `False`
    * if True, will print the intermediate reasonning steps of LLMs
    if debug is set, llm_verbosity is also set to True

* `--debug`: bool, default `False`
    * if True will enable langchain tracing, increase verbosity,
    disable multithreading for summaries and loading files,
    crash if an error is encountered when loading a file,
    automatically trigger the debugger on exceptions.

* `--dollar_limit`: int, default `5`
    * If the estimated price is above this limit, stop instead.
    Note that the cost estimate for the embeddings is using the
    openai tokenizer, which is not universal.
    This check is skipped if the api_base url are changed.

* `--notification_callback`: Callable, default `None`
    * a function that must take as input a string and return the same
    string. Inside it you can do whatever you want with it. This
    can be used for example to send notification on your phone
    using ntfy.sh to get summaries.

* `--chat_memory`: bool, default `True`
    * if True, will remember the messages across a given chat exchange.
    Disabled if using a testing model.

* `--disable_llm_cache`: bool, default `False`
    * WARNING: The cache is temporarily ignored in non openaillms
    generations because of an error with langchain's ChatLiteLLM.
    Basically if you don't use `--private` and use llm form openai,
    DocToolsLLM will use ChatOpenAI with regular caching, otherwise
    we use ChatLiteLLM with LLM caching disabled.
    More at https://github.com/langchain-ai/langchain/issues/22389

    disable caching for LLM. All caches are stored in the usual
    cache folder for your system. This does not disable caching
    for documents.

* `--file_loader_parallel_backend`: str, default `"loky"`
    * joblib.Parallel backend to use when loading files. loky means
    multiprocessing while `threading` means multithreading.
    The number of jobs can be specified with `file_loader_n_jobs`
    but it's a loader specific kwargs.

* `--private`: bool, default `False`
    * add extra check that your data will never be sent to another
    server: for example check that the api_base was modified and used,
    check that no api keys are used, check that embedding models are
    local only. It will also use a separate cache from non private.

* `--llms_api_bases`: dict, default `None`
    * a dict with keys in `["model", "query_eval_model"]`
    The corresponding value will be used to change the url of the
    endpoint. This is needed to use local LLMs for example using
    ollama, lmstudio etc.

* `--DIY_rolling_window_embedding`: bool, default `False`
    * enables using a DIY rolling window embedding instead of using
    the default langchain SentenceTransformerEmbedding implementation

* `--import_mode`: bool, default `False`
    * if True, will return the answer from query instead of printing it

* `--cli_kwargs`: dict, optional
    * Any remaining keyword argument will be parsed as a loader
    specific argument ((see below)[#loader-specific-arguments]).
    Any unrecognized key or inappropriate value type will result in a crash. 


# Loader specific arguments
    Those arguments can be set at cli time but can also be used
    when using recursive_paths filetype combination to have arguments specific
    to a loader. They apply depending on the value of `--filetype`.
    An unexpected argument for a given filetype will result in a crash.

* `--path`: str
    * Used by most loaders. For example for `--filetype=youtube` the path
    must point to a youtube video.

* `--anki_profile`: str
    * The name of the profile
* `--anki_deck`: str
    * The beginning of the deckname
    e.g. `science::physics::freshman_year::lesson1`
* `--anki_notetype`: str
    * If it's part of the card's notetype, that notetype will be kept.
    Case insensitive.

* `--anki_fields`: List[str]
    * List of fields to keep
* `--anki_mode`: str
    * any of `window`, `concatenate`, `singlecard`: (or _ separated
    value like `concatenate_window`). By default `singlecard`
    is used.
    * Modes:
        * `singlecard`: 1 document is 1 anki card.
        * `window`: 1 documents is 5 anki note, overlapping (so
        10 anki notes will result in 5 documents)
        * `concatenate`: 1 document is all anki notes concatenated as a
        single wall of text then split like any long document.

    Whichever you choose, you can later filter out documents by metadata
    filtering over the `anki_mode` key.

* `--audio_backend`: str
    * either 'whisper' or 'deepgram' to transcribe audio.
    Not taken into account for the filetype "youtube".
    Taken into account if filetype if "local_audio" or "local_video"

* `--audio_unsilence`: bool
    * When processing audio files, remove silence before transcribing.

* `--whisper_lang`: str
    * if using whisper to transcribe an audio file, this if the language
    specified to whisper
* `--whisper_prompt`: str
    * if using whisper to transcribe an audio file, this if the prompt
    given to whisper

* `--deepgram_kwargs`: dict
    * if using deepgram for transcription, those arguments will be used.

* `--youtube_language`: List[str]
    * For youtube. e.g. `["fr","en"]` to use french transcripts if
    possible and english otherwise
* `--youtube_translation`: str
    * For youtube. e.g. `en` to use the transcripts after translation to english
* `--youtube_audio_backend`: str
    Either 'youtube', 'whisper' or 'deepgram'.
    Default is 'youtube'.
    * If 'youtube': will take the youtube transcripts as text content.
    * If 'whisper': DocToolsLLM will download
    the audio from the youtube link, and whisper will be used to turn the audio into text. whisper_prompt and whisper_lang will be used if set.
    * If 'deepgram' will download
    the audio from the youtube link, and deepgram will be used to turn the audio into text. `--deepgram_kwargs` will be used if set.

* `--include`: str
    * Only active if `--filetype` is one of 'json_entries', 'recursive_paths',
    'link_file', 'youtube_playlist'.
    `--include` can be a list of regex that must be present in the
    document PATH (not content!)
    `--exclude` can be a list of regex that if present in the PATH
    will exclude it.
    Exclude is run AFTER include
* `--exclude`: str
    * See `--include`

# Other specific arguments

* `--out_file`: str, default `None`
    * If doctools must create a summary, if out_file given the summary will
    be written to this file. Note that the file is not erased and
    Doctools will simply append to it.
    * If `--summary_n_recursion` is used, additional files will be
    created with the name `{out_file}.n.md` with n being the n-1th recursive
    summary.

* `--filter_metadata`: dict, default `None`
    * list of regex string to use as metadata filter when querying.
    Format: `[kvb][+-]your_regex`

    For example:
    * Keep only documents that contain `anki` in any value
    of any of its metadata dict:
        `--filter_metadata=v+anki`  <- at least the `filetype` key
        will have as value `anki`
    * Keep only documents that contain `anki_profile` as a key in
    its metadata dict:
        `--filter_metadata=k+anki_profile`  <- because will contain the
        key anki_profile
    * Keep only data that have a certain `source_tag` value:
        `--filter_metadata=b+source_tag:my_source_tag_regex`

    Notes:
    * Each filter must be a regex string beginning with k, v or b
    (for `key`, `value` or `both`). Followed by either `+` or `-` to:
        `+` at least one metadata should match
        `-` exclude from (no metadata should match)
    * If the string starts with k, it will filter based on the keys
    of the metadata, if it starts with a v it will filter based
    on the values, if it starts with b it will require a `:` present
    and everything left of : will be a regex to match a key key and
    right of the : will be a regex matching the matched key.
    * Filters are only relevant for task related to queries and are
    ignored for summaries.
    * Smartcasing is used: if the filter is its own lowercase version
    then insensitive casing will be used, otherwise not.
    * The function used to check the matching is `pattern.match`
    * The filtering is not done at the search time but before it. We
    first scan all the corresponding documents, then delete the useless
    embeddings from the docstore. This makes the whole search faster.
    But the embeddings are not saved afterwards so they are not lost,
    just not present in memory for this prompt.

* `--filter_content`: dict, default `None`
    * Like `--filter_metadata` but filters through the page_content of
    each document instead of the metadata.
    Syntax: `[+-]your_regex`
    Example:
    * Keep only the document that contain `doctools`
        `--filter_content=+.*doctools.*`
    * Discard the document that contain `DOCTOOLS`
        `--filter_content=-.*DOCTOOLS.*`

* `--embed_instruct`: bool, default `None`
    * when loading an embedding model using HuggingFace or
    llamacppembeddings backends, wether to wrap the input
    sentence using instruct framework or not.

* `--file_loader_n_jobs`: int, default `5`
    * number of threads to use when loading files. Set to 1 to disable
    multithreading (as it can result in out of memory error if
    using threads and overly recursive calls)

* `--load_functions`: List[str], default `None`
    * list of strings that when evaluated in python result in a list of
    callable. The first must take one input of type string and the
    last function must return one string.

    For example in the filetypes `local_html` this can be used to
    specify lambda functions that modify the text before running
    BeautifulSoup. Useful to decode html stored in .js files.
    Do tell me if you want more of this.

* `--min_lang_prob`: float, default `0.5`
    * float between 0 and 1 that sets the threshold under which to
    consider a document invalid if the estimation of
    fasttext's langdetect of any language is below that value.
    For example, setting it to 0.9 means that only documents that
    fasttext thinks have at least 90% probability of being a
    language are valid.

* `--source_tag`: str, default `None`
    * a string that will be added to the document metadata at the
    key `source_tag`. Useful when using filetype combination.

* `--loading_failure`: str, default `warn`
    * either `crash` or `warn`. Determines what to do with
    exceptions happening when loading a document. This can be set
    per document if a recursive_paths filetype is used.

# Runtime flags

* `DOCTOOLS_TYPECHECKING`
    * Setting for runtime type checking. Default value is `disabled`.
    * Possible values:
        * `disabled`: disable typechecking
        * `warn`: print a red warning if a typechecking fails
        * `crash`: crash if a typechecking fails in any function
