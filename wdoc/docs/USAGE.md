# Table of contents
- [Global arguments](#global-arguments)
- [Loader specific arguments](#loader-specific-arguments).
- [Other specific arguments](#other-specific-arguments)
- [Runtime flags / environment variables](#runtime-flags)


# Global arguments

* `--task`: str
    * Accepted values:
        * `query`: means to load the input files then wait for user question.
        * `search`: means only return the document corresponding to the search
        * `summarize`: means the input will be passed through a summarization prompt.
        * `summarize_then_query`: summarize the text then open the prompt to allow querying directly the source document.

* `--filetype`: str, default `auto`
    * the type of input. Depending on the value, different other parameters
    are needed. If json_entries is used, the line of the input file can contain
    any of those parameters as long as they are as json. You can find
    an example of json_entries file in `wdoc/docs/json_entries_example.txt`

    * Supported values and available arguments:
        *For the details of each argument, [see below](#loader-specific-arguments)*
        * `auto`: will guess the appropriate filetype based on `--path`.
            Irrelevant for some filetypes, eg if `--filetype`=anki

        * `url`
            * `--path` must be a valid http(s) link
            * Optional:
                * `--title`, otherwise we try to detect it ourselves.

        * `youtube`
            * `--path` must link to a youtube video
            * Optional:
                * `--youtube_language`
                * `--youtube_translations`
                * `--youtube_audio_backend`
                * `--whisper_prompt`
                * `--whisper_lang`
                * `--deepgram_kwargs`

        * `pdf`
            * `--path` is the filepath to pdf
            * Optional:
                * `--pdf_parsers`
                * `--doccheck_min_lang_prob`
                * `--doccheck_min_token`
                * `--doccheck_max_token`

        * `online_pdf`
            * Same arguments as for `--filetype=pdf`
                Note that the way `online_pdf` are handled is a bit different than `pdf`: we
                first try using langchain's integrated OnlinePDFLoader and if it fails,
                we download the file and parse it like if `--filetype==pdf`.

        * `anki`
            * Optional:
                * `--anki_profile`
                * `--anki_deck`
                * `--anki_notetype`
                * `--anki_template`
                * `--anki_tag_filter`
                * `--anki_tag_render_filter`

        * `string`: no parameters needed, will provide a field where
            you must type or paste the string

        * `txt`
            * `--path` is path to a .txt file

        * `text`
            * `--path` is directly the text content.
            * Optional:
                * `--metadata`

        * `local_html`
            * `--path` must points to a .html file
            * Optional:
                * `--load_functions`

        * `logseq_markdown`
            * `--path` path to the markdown file

        * `local_audio`
            * `--path`
            * `--audio_backend`
            * Optional:
                * `--audio_unsilence`
                * `--whisper_prompt`
                * `--whisper_lang`
                * `--deepgram_kwargs`

        * `local_video`
            * `--path`
            * `--audio_backend`
            * Optional:
                * `--audio_unsilence`
                * `--whisper_lang`
                * `--whisper_prompt`
                * `--deepgram_kwargs`

        * `online_media`: load the url using youtube_dl to download a media
            (video or audio) then treat it as `filetype=local_audio`.
            * If youtube_dl failed to find the media, try using playwright browser
                where any requested element that looks like a possible media will try
                be downloaded.
            * Same arguments as `local_audio` with extra arguments:
                * `--online_media_url_regex`
                * `--online_media_resourcetype_regex`
        * `epub`
            * `--path` to a .epub file

        * `word`
            * `--path` to a .doc, .docx etc

        * `json_dict`
            * `--path` to a text file containing a single json dict
            * `--json_dict_template`
            * Optional:
                * `--json_dict_exclude_keys`
                * `--metadata`

    * **Recursive types**:
        * `json_entries`
            * `--path` is path to a text file that contains a json
                for each line containing at least a filetype and a path key/value
                but can contain any parameters described here
        * `recursive_paths`
            * `--path` is the starting path
            * `--pattern` is the globbing patterns to append
            * `--exclude` and `--include` can be a list of regex
                applying to found paths (include is run first then exclude, if the
                pattern is only lowercase it will be case insensitive)
            * `--recursed_filetype` is the filetype to use for each of the found path
        * `youtube_playlist`
            * `--path` must link to a youtube playlist
        * `link_file`
            * `--path` must point to a file where each line is a link
                that will be summarized.
            * `--out_file` path to text file where the summary will be added (appended).
                Links that have already been summarized in out_file will be skipped
                (the out_file is never overwritten). If a line is a markdown like
                like [this](link) then it will be parsed as a link.
                Empty lines and starting with # are ignored.

---

* `--modelname`: str, default to value of WDOC_DEFAULT_MODELNAME
    * Keep in mind that given that the default backend used is litellm
    the part of modelname before the slash (/) is the backend name (also called provider).
    If the backend is 'testing/' then it will be parsed as 'testing/testing' and
    a fake LLM will be used for debugging purposes. It answers like a normal LLM
    but costs 0 and makes no sense.
    If the value is not part of the model list of litellm, will use
    fuzzy matching to find the best match.

---

* `--embed_model`: str, default to value of WDOC_DEFAULT_EMBED_MODEL
    * Name of the model to use for embeddings. Must contain a '/'
    Everything before the slash is the backend and everything
    after the / is the model name.
    Available backends: openai, sentencetransformers,
    huggingface

    * Note:
        * the device used by default for huggingface is 'cpu' and not 'cuda'
        * If you change this, the embedding cache will be usually
            need to be recomputed with new elements (the hash
            used to check for previous values includes the name of the model
            name)

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

* `--top_k`: Union[int, str], default `auto_200_500`
    * number of chunks to look for when querying. It is high because the
    eval model is used to refilter the document after the embeddings
    first pass.
    If top_k is a string, the format assumed is "auto_N_m" where N is the
    starting top_k and M is the max top_k value. If the number of filtered
    document is more than 90% of top_k, top_k will gradually increase up to M
    (with N and M being int, and 0<N<M).
    This way you are sure not to miss any document.

---

* `--query`: str, default `None`
    * if str, will be directly used for the first query if task in `["query", "search", "summarize_then_query"]`

* `--query_retrievers`: str, default `"default_multiquery"`
    * must be a string that specifies which retriever will be used for
    queries depending on which keyword is inside this string.

    * Possible values (can be combined if separated by _):
        * `default`: cosine similarity retriever
        * `multiquery`: retriever that uses the LLM to reformulate your
        query to get different perspectives. This uses the strong LLM
        and, as it requires complex output parsing for now it is
        recommended to not use that retriever for average models.
        * `knn`: knn
        * `svm`: svm
        * `parent`: parent chunk

    if contains `hyde` but modelname contains `testing` then `hyde` will
    be removed.

* `--query_eval_modelname`: str, default to value of WDOC_DEFAULT_QUERY_EVAL_MODELNAME
    * Cheaper and quicker model than modelname. Used for intermediate
    steps in the RAG, not used in other tasks.
    If the value is not part of the model list of litellm, will use
    fuzzy matching to find the best match.
    None to disable.

* `--query_eval_check_number`: int, default `3`
    * number of pass to do with the eval llm to check if the document
    is indeed relevant to the question. The document will not
    be processed further if the mean answer from the eval llm is too low.
    For eval llm that don't support setting `n`, multiple
    completions will be called, which costs more.

* `--query_relevancy`: float, default `0.0`
    * threshold underwhich a document cannot be considered relevant by
    embeddings alone. Keep in mind that the score is a similarity, so
    it goes from -1 (most different) to +1 (most similar), althrough
    if you set `WDOC_MOD_FAISS_SCORE_FN` to `True` it will then
    go from 0 to 1.

---

* `--summary_n_recursion`: int, default `0`
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
    display warning if an error is encountered when loading a file,
    automatically trigger the debugger on exceptions.
    Note that the parallel processing will not be disabled if you manually
    set `--file_loader_n_jobs`, allowing you to debug parallel
    processing issues.
    Because in some situation LLM calls are refused because of rate
    limiting, this can be used to slowly but always get your answer.
    It implies `--verbose=True`
    If you just want to open the debugger in case of issue, see
    below at `WDOC_DEBUGGER`.

* `--verbose`: bool, default `False`
    Increase verbosity. Implied if `--debug` is set.

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

* `--disable_llm_cache`: bool, default `False`
    * WARNING: The cache is temporarily ignored in non openaillms
    generations because of an error with langchain's ChatLiteLLM.
    Basically if you don't use `--private` and use llm form openai,
    wdoc will use ChatOpenAI with regular caching, otherwise
    we use ChatLiteLLM with LLM caching disabled.
    More at https://github.com/langchain-ai/langchain/issues/22389

    disable caching for LLM. All caches are stored in the usual
    cache folder for your system. This does not disable caching
    for documents.

* `--file_loader_parallel_backend`: str, default `"loky"`
    * joblib.Parallel backend to use when loading files. `loky` and
    `multiprocessing` refer to multiprocessing whereas `threading`
    refers to multithreading.
    The number of jobs can be specified with `--file_loader_n_jobs`
    but it's a loader specific kwargs.

* `--file_loader_n_jobs`: int, default `-1`
    * number of jobs to use when loading files in parallel (threads or process,
    depending on `--file_loader_parallel_backend`). Set to 1 to disable
    parallel processing (as it can result in out of memory error if
    using threads and overly recursive calls). Automatically set to 1 if
    `--debug` is set or if there's only one document to load.
    If -1, means use as many as possible (this is joblib's default).

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
    * if True, will return the answer from query instead of printing it.
    The idea is to use if when you import wdoc instead of running
    it from the cli. See `--silent`

* `--disable_md_printing`: bool, default `True`
    * if True, instead of using rich to display some information, default to simpler colored prints.

* `--silent`: bool, default False
    * disable almost all prints while still writing to the log.
    Can be handy if `--import_mode` is used but beware that this can
    remove important information.

* `--version`: bool, default False
    * display the version and exit

* `--cli_kwargs`: dict, optional
    * Any remaining keyword argument will be parsed as a loader
    specific argument ((see below)[#loader-specific-arguments]).
    Any unrecognized key or inappropriate value type will result in a crash. 


# Loader specific arguments
    Those arguments can be set at cli time but can also be used
    when using recursive_paths filetype combination to have arguments specific
    to a loader. They apply depending on the value of `--filetype`.
    An unexpected argument for a given filetype will result in a crash.

* `--path`: str or PosixPath
    * Used by most loaders. For example for `--filetype=youtube` the path
    must point to a youtube video.

* `--pdf_parsers`: str or List[str], default: `pymupdf`
    * list of string or comma separated list of strings where each string
    is a key of the dict `pdf_loaders` in `./utils/loaders.py`.
    The case is insensitive.
    The parsers are used in the order of this list.
    Not all parsers are tried. Instead, after each parsing we check using
    fasttext and heuristics based on doccheck_* args to rank the quality of the parsing.
    When stop if 1 parsing is high enough or take the best if 3 parsing worked.
    Note that the way `online_pdf` are handled is a bit different: we
    first try using langchain's integrated OnlinePDFLoader and if it fails,
    we download the file and parse it like if `--filetype==pdf`.

    Currently implemented:
    - Okayish metadata:
        - pymupdf
        - pdfplumber
    - Few metadata:
        - pdfminer
        - pypdfloader
        - pypdfium2
        - openparse (also has table support but quite slow)
    - pdftotext  (fastest and most basic but can be unavailable depending on your install)
    - Very slow but theoretically the best are from unstructured:
        - unstructured_fast
        - unstructured_elements_fast
        - unstructured_hires
        - unstructured_elements_hires
        - unstructured_fast_clean_table
        - unstructured_elements_fast_clean_table
        - unstructured_hires_clean_table
        - unstructured_elements_hires_clean_table
        Notes: to the best of my knowledge:
            'fast' means not AI based, as opposed to 'hires'
            'elements' means the parser returns each element of the pdf instead of collating them in the rendering
            'clean' means it tries to remove the extra whitespace
            'table' means it will try to infer table structure (AI based)

* `--anki_profile`: str
    * The name of the profile
* `--anki_deck`: str
    * The beginning of the deckname. Note that we only look at decks, filtered
    decks are not taken into acount (so a card of deck 'A' that is temporarily
    in 'B::filtered_deck' will still be considered as part of 'A'.
    e.g. `science::physics::freshman_year::lesson1`
* `--anki_notetype`: str
    * If it's part of the card's notetype, that notetype will be kept.
    Case insensitive. Note that suspended cards are always ignored.
* `--anki_template`: str
    * The template to use for the anki card. For example if you have
    a notetype with fields "fieldA","fieldB","fieldC" then you could
    set --anki_template="Question:{fieldA}\nAnswer:{fieldB}". The field
    "fieldC" would not be used and each document would look like your
    template.
    Notes:
    * '{tags}' can be used to include a '\n* ' separated
        string of the tag list. Use --anki_tag_render_filter to restrict which tag
        can be shown (to avoid privacy leakage).
        Example of what the tag formating looks like:
        "
        Anki tags:
        '''
        * my::tag1
        * my_othertag
        '''
        "
    * '{allfields}' can be used to format automatically all fields
    (not including tags). It will be replaced
    as "fieldA: 'fieldAContent'\n\nfieldB: 'fieldBContent'" etc
    The ' are added.
    * The default value is '{allfields}\n{image_ocr_alt}'.
    * '{image_ocr_alt}' if present will be replaced by any text present
    in the 'title' or 'alt' field of an html image. This is isually OCR
    so can be useful for the LLM.

* `--anki_tag_filter`: str
    Only keep the cards that have tags matchign this regex.

* `--anki_tag_render_filter`: str
    Only the tags that match this regex will be put in the template.
    Careful, this does not mean "only keep cards that have tags matching
    this filter" but rather "only mention the tags matching this filter
    in the final document".

* `--json_dict_template`: str
    String that must contain `{key} and `{value}`, that will be replaced
    by the content of the json dict so that each document correspond to
    a single key/value pair derived from the template.
* `--json_dict_exclude_keys`: list of strings
    all those keys will be ignored.

* `--metadata`: str
    either as a string that will be parsed as a json dict, or as a dict.

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
    * If 'whisper': wdoc will download
    the audio from the youtube link, and whisper will be used to turn the audio into text. whisper_prompt and whisper_lang will be used if set.
    * If 'deepgram' will download
    the audio from the youtube link, and deepgram will be used to turn the audio into text. `--deepgram_kwargs` will be used if set.

* `--include`: str
    * Only active if `--filetype` is 'recursive_paths'
    `--include` can be a list of regex that must be present in the
    document PATH (not content!)
    `--exclude` can be a list of regex that if present in the PATH
    will exclude it.
    Exclude is run AFTER include
* `--exclude`: str
    * See `--include`

# Other specific arguments

* `--out_file`: str or PosixPath, default `None`
    * If wdoc must create a summary, if out_file given the summary will
    be written to this file. Note that the file is not erased and
    wdoc will simply append to it.
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
    * Keep only the document that contain `winstondoc`
        `--filter_content=+.*winstondoc.*`
    * Discard the document that contain `winstondoc`
        `--filter_content=-.*winstondoc.*`

* `--embed_instruct`: bool, default `None`
    * when loading an embedding model using the HuggingFace backend,
    wether to wrap the input sentence using instruct framework or not.

* `--load_functions`: List[str], default `None`
    * list of strings that when evaluated in python result in a list of
    callable. The first must take one input of type string and the
    last function must return one string.

    For example in the filetypes `local_html` this can be used to
    specify lambda functions that modify the text before running
    BeautifulSoup. Useful to decode html stored in .js files.
    Do tell me if you want more of this.

* `--doccheck_min_lang_prob`: float, default `0.5`
    * float between 0 and 1 that sets the threshold under which to
    consider a document invalid if the estimation of
    fasttext's langdetect of any language is below that value.
    For example, setting it to 0.9 means that only documents that
    fasttext thinks have at least 90% probability of being a
    language are valid.
* `--doccheck_min_token`: int, default `50`
    * if we find less that that many token in a document, crash.
* `--doccheck_max_token`: int, default `10_000_000`
    * if we find more that that many token in a document, crash.

* `--online_media_url_regex`: str
    * a regex that if matching a request's url, will consider the
    request to be leading to a media. We then try to fetch those media
    using youtube_dl. The default is already a sensible value.
* `--online_media_resourcetype_regex`: str
    * Same as `--online_media_url_regex` but checking request.resource_type

* `--source_tag`: str, default `None`
    * a string that will be added to the document metadata at the
    key `source_tag`. Useful when using filetype combination.
    It is EXTREMELY recommended to include a source_tag to any document
    you want to save: especially if using recursive filetypes. This
    is because after loading all documents wdoc use the source_tag
    to see if it should continue or crash. If you want to load 10_000 pdf
    in one go as I do, then it makes sense to continue if some failed to
    crash but not if a whole source_tag is missing.

* `--loading_failure`: str, default `warn`
    * either `crash` or `warn`. Determines what to do with
    exceptions happening when loading a document. This can be set
    per document if a recursive_paths filetype is used.
    If using `wdoc_parse_file` it is by default set to `crash`.

# Runtime flags

* `WDOC_TYPECHECKING`
    * Setting for runtime type checking. Default value is `warn`.     * Possible values:
    The typing is checked using [beartype](https://beartype.readthedocs.io/en/latest/) so shouldn't slow down the runtime.
        * `disabled`: disable typechecking.
        * `warn`: print a red warning if a typechecking fails.
        * `crash`: crash if a typechecking fails in any function.

* `WDOC_NO_MODELNAME_MATCHING`
    * If "true": will bypass the model name matching. Useful for exotic
    or models that are fresh out of the oven. Default is `False`.

* `WDOC_ALLOW_NO_PRICE`
    * if "true", won't crash if no price was found for the given
    model. Useful if litellm has not yet updated its price table.
    Default is `False`.

* `WDOC_OPEN_ANKI`
    * if "true", will automatically ask wether to open the anki browser if cards are
    found in the sources. Only used if task is `query` or `search`.
    Default is `False`

* `WDOC_STRICT_DOCDICT`
    * if "True", will crash instead of printing if trying to set an unexpected argument in a DocDict.
        Otherwise, you can specify things like "anki_profile" as argument to filetype "pdf" without crashing,
        this also makes no sense but can be useful if there's a bug in wdoc that is not yet fixed
    and you want to continue in the meantime.
    * If set to "False": we print in red unexpected arguments but add them anyway.
    * If set to "strip": we print in red unexpected arguments and ignore them.
    Default is `False`.

* `WDOC_MAX_LOADER_TIMEOUT`
    * Number of seconds to wait before giving up on loading a document (this does not include recursive types, only the DocDicts).
    Default is `-1` to disable.
    Disabled if <= 0.

* `WDOC_MAX_PDF_LOADER_TIMEOUT`
    * Number of seconds to wait for each pdf loader before giving up this loader. This includes the `online_pdf` loader.
    Note that it probably makes PDF parsing substantially.
    Default is `-1` to disable.
    Disabled when using `--file_loader_parallel_backend=threading` as python does not allow it.
    Also disabled if <= 0.

* `WDOC_DEBUGGER`
    * If True, will open the debugger in case of issue. Implied by `--debug`
    Default is `False`

* `WDOC_EXPIRE_CACHE_DAYS`
    * If an int, will remove any cached value that is older than that many days.
    Otherwise keep forever. Default is `0` to disable.

* `WDOC_EMPTY_LOADER`
    * If True, loading any kind of document will return an empty string. Used for debugging. Default is `False`.

* `WDOC_BEHAVIOR_EXCL_INCL_USELESS`
    * If an "include" or "exclude" key is found in a loader but does not actually change anything, if `warn` then just print in red but
    if `crash` then raise an error. Default is `warn`.

* `WDOC_PRIVATE_MODE`
    * You should never set it yourself. It is set automatically if the `--private` argument is used, and used throughout to triple check that it's indeed fully private.

* `WDOC_IMPORT_TYPE`, default `lazy`
    * If `native` will just import the packages needed by wdoc without any tricks.
    * If `thread`, will try to use a separate thread to import packages making the startup time potentially smaller.
    * If `lazy`, will use lazy loading on some packages, making the startup time potentially smaller.
    * If `both`, will try to use both.
    All other then `native` are experimental as they rely on weird python tricks.

* `WDOC_MOD_FAISS_SCORE_FN`
    * If True, modify on the fly the FAISS vectorstores to change their scoring function. This was  inspired by [this langchain issue where users claim the default scoring function is wrong](https://github.com/langchain-ai/langchain/issues/17333)
    Default is False.

* `WDOC_LLM_MAX_CONCURRENCY`
    * Set the max_concurrency limit to give langchain. If debug is used, it is overriden and set to 1.
    Must be an int. By default is 10.

* `WDOC_SEMANTIC_BATCH_MAX_TOKEN_SIZE`
    * GPT-3.5 token size considered maximum for a batch when doing semantic batching.
    Each batch contains at least two intermediate answers so it's not an absolute limitation but increasing it should
    reduce the cost of the "combine intermediate answers" step when querying.
    Default is `500`.

* `WDOC_DEFAULT_MODELNAME`, default: `"openai/gpt-4o"`
    * Default strong LLM to use. This is the strongest model, it will be used to answer the query about each document,
    combine those answers. It can also be used by some retrievers etc.

* `WDOC_DEFAULT_QUERY_EVAL_MODELNAME`, default: `"openai/gpt-4o-mini"`
    * Default small LLM to use. It will be used to evaluate wether each document is relevant to the query or not.

* `WDOC_DEFAULT_EMBED_MODEL`, default: `"openai/text-embedding-3-small"`
    * Default model to use for embeddings.
