import json
import pyfiglet
import copy
from textwrap import indent
from functools import wraps
from typing import List, Union, Any, Optional, Callable
from typeguard import check_type, TypeCheckError
import tldextract
from joblib import Parallel, delayed
from threading import Lock
from pathlib import Path
import time
from datetime import datetime
import re
import textwrap
import os
import asyncio
from tqdm import tqdm

from langchain.globals import set_verbose, set_debug, set_llm_cache
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.docstore.document import Document
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import (
        DocumentCompressorPipeline)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import KNNRetriever, SVMRetriever
from langchain_community.cache import SQLiteCache
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import chain
from langchain_core.runnables.base import RunnableEach
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation, ChatGeneration
import litellm

from .utils.llm import load_llm, AnswerConversationBufferMemory
from .utils.batch_file_loader import batch_load_doc
from .utils.loaders import (
    get_tkn_length,
    average_word_length,
    wpm,
    get_splitter,
    check_docs_tkn_length,
    )

from .utils.embeddings import load_embeddings
from .utils.retrievers import create_hyde_retriever, create_parent_retriever
from .utils.logger import whi, yel, red, md_printer, log
from .utils.cli import ask_user
from .utils.misc import ankiconnect, debug_chain, model_name_matcher, cache_dir
from .utils.tasks.summary import do_summarize
from .utils.tasks.query import format_chat_history, refilter_docs, check_intermediate_answer, parse_eval_output, doc_eval_cache
from .utils.typechecker import optional_typecheck
from .utils.prompts import PR_CONDENSE_QUESTION, PR_EVALUATE_DOC, PR_ANSWER_ONE_DOC, PR_COMBINE_INTERMEDIATE_ANSWERS
from .utils.errors import NoDocumentsRetrieved, NoDocumentsAfterLLMEvalFiltering


os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

extra_args = {
    "anki_profile": str,
    "anki_notetype": str,
    "anki_fields": str,
    "anki_deck": str,
    "anki_mode": str,
    "whisper_lang": str,
    "whisper_prompt": str,
    "path": str,
    "include": str,
    "exclude": str,
    "out_file": str,
    "out_file_logseq_mode": str,
    "youtube_language": str,
    "youtube_translation": str,
    "out_check_file": str,
    "embed_instruct": str,
    "file_loader_n_jobs": int,
    "load_functions": List[str],
    "filter_metadata": Union[List[str], str],
    # "filter_content": Union[List[str, str]],
    "source_tag": str,
}

class DocToolsLLM_class:
    VERSION: str = "0.16"

    @optional_typecheck
    def __init__(
        self,
        modelname: str = "openai/gpt-4o",
        # modelname: str = "openai/gpt-3.5-turbo-0125",
        # modelname: str = "mistral/mistral-large-latest",
        task: str = "query",
        query: Optional[Union[str, bool]] = None,
        filetype: str = "infer",
        embed_model: str = "openai/text-embedding-3-small",
        # embed_model: str =  "sentencetransformers/BAAI/bge-m3",
        # embed_model: str =  "sentencetransformers/paraphrase-multilingual-mpnet-base-v2",
        # embed_model: str =  "sentencetransformers/distiluse-base-multilingual-cased-v1",
        # embed_model: str =  "sentencetransformers/msmarco-distilbert-cos-v5",
        # embed_model: str =  "sentencetransformers/all-mpnet-base-v2",
        # embed_model: str =  "huggingface/google/gemma-2b",
        embed_kwargs: Optional[dict] = None,
        save_embeds_as: str = "{user_cache}/latest_docs_and_embeddings",
        load_embeds_from: Optional[str] = None,

        top_k: int = 20,
        query_retrievers: str = "default",
        query_eval_modelname: Optional[str] = "openai/gpt-3.5-turbo-0125",
        # query_eval_modelname: str = "mistral/open-mixtral-8x7b",
        # query_eval_modelname: str = "mistral/open-small",
        query_eval_check_number: int = 3,
        query_relevancy: float = 0.1,
        n_recursive_summary: int = 0,

        n_summaries_target: int = -1,
        summary_language: str = "[same as input]",

        dollar_limit: int = 5,
        debug: bool = False,
        llm_verbosity: bool = False,
        notification_callback: Optional[Callable] =  None,
        condense_question: bool = True,
        chat_memory: bool = True,
        no_llm_cache: bool = False,
        file_loader_parallel_backend: str = "loky",
        private: bool = False,
        llms_api_bases: Optional[Union[dict, str]] = None,
        DIY_rolling_window_embedding: bool = False,

        help: bool = False,
        h: bool = False,
        import_mode: bool = False,
        **kwargs: Any,
        ) -> None:
        """
        Parameters
        ----------
        --task str, default query
            possibilities:
                * query means to load the input files then wait for user question.
                * search means only return the document corresponding to the search
                * summarize means the input will be passed through a summarization prompt.
                * summarize_then_query
                * summarize_link_file takes in --filetype must be link_file

        --query str, default None
            if str, will be directly used for the first query if task in ["query", "search"]

        --filetype str, default infer
            the type of input. Depending on the value, different other parameters
            are needed. If json_list is used, the line of the input file can contain
            any of those parameters as long as they are as json. You can find
            an example of json_list file in utils/json_list_example.txt

            Supported values => relevant parameters
                * infer => will guess the appropriate filetype based on --path (so does not work with all filetypes, for example not with --filetype=anki)
                * youtube => --path must link to a youtube video
                * youtube_playlist => --path must link to a youtube playlist
                * pdf => --path is path to pdf
                * txt => --path is path to txt
                * url => --path must be a valid http(s) link
                * anki => must be set: --anki_profile. Optional: --anki_deck, --anki_notetype, --anki_mode. See in loader specific arguments below for details.
                * string => no other parameters needed, will provide a field where you must type or paste the string
                * local_audio => must be set: --whisper_prompt, --whisper_lang

                * json_list => --path is path to a txt file that contains a json for each line containing at least a filetype and a path key/value but can contain any parameters described here
                * recursive => --path is the starting path --pattern is the globbing patterns to append --exclude and --include can be a list of regex applying to found paths (include is run first then exclude, if the pattern is only lowercase it will be case insensitive) --recursed_filetype is the filetype to use for each of the found path
                * link_file => --path must point to a file where each line is a link that will be summarized. The resulting summary will be added to --out_file. Links that have already been summarized in out_file will be skipped (the out_file is never overwritten). If a line is a markdown linke like [this](link) then it will be parsed as a link. Empty lines and starting with # are ignored. If argument --out_file_logseq_mode is present, the formatting will be compatible with logseq.


        --modelname str, default openai/gpt-4o
            Keep in mind that given that the default backend used is litellm
            the part of modelname before the slash (/) is the server name.
            If the backend is 'testing/' then a fake LLM will be used
            for debugging purposes.
            If the value is not part of the model list of litellm, will use
            fuzzy matching to find the best match.

        --embed_model str, default "openai/text-embedding-3-small"
            Name of the model to use for embeddings. Must contain a '/'
            Everything before the slash is the backend and everything
            after the / is the model name.
            Available backends: openai, sentencetransformers,
            huggingface, llamacppembeddings

            Note:
            * the device used by default for huggingface is 'cpu' and not 'cuda'
            * If you change this, the embedding cache will be usually
              need to be recomputed with new elements (the hash
              used to check for previous values includes the name of the model
              name)
            * If the backend if llamacppembeddings, the modelname must be the path to
              the model. For example: 'llamacppembeddings/my_model_file'

        --embed_kwargs: dict, default None
            dictionnary of keyword arguments to pass to the embedding.

        --save_embeds_as str, default {user_dir}/latest_docs_and_embeddings
            only used if task is query
            save the latest 'inputs' to a file. Can be loaded again with
            --load_embeds_from to speed up loading time. This loads both the
            split documents and embeddings but will not update itself if the
            original files have changed.
            {user_dir} is automatically replaced by the path to the usual
            cache folder for the current user

        --load_embeds_from str, default None
            path to the file saved using --save_embeds_as

        --top_k int, default 20
            number of chunks to look for when querying. It is high because the
            eval model is used to refilter the document after the embeddings
            first pass.

        --query_retrievers: str, default 'default'
            must be a string that specifies which retriever will be used for
            queries depending on which keyword is inside this string:
                "default": cosine similarity retriever
                "hyde": hyde retriever
                "knn": knn
                "svm": svm
                "parent": parent chunk

            if contains 'hyde' but modelname contains "testing" then hyde will
            be removed.

        --query_eval_modelname str, default openai/gpt-3.5-turbo-0125
            Cheaper and quicker model than modelname. Used for intermediate
            steps in the RAG, not used in other tasks.
            If the value is not part of the model list of litellm, will use
            fuzzy matching to find the best match.
            None to disable.

        --query_eval_check_number: int, default 3
            number of pass to do with the eval llm to check if the document
            is indeed relevant to the question. The document will not
            be processed if all answers from the eval llm are 0, and will
            be processed otherwise.
            For eval llm that don't support setting 'n', multiple
            completions will be called, which costs more.

        --query_relevancy: float, default 0.1
            threshold underwhich a document cannot be considered relevant by
            embeddings alone.

        --n_recursive_summary: int, default 0
            will recursively summarize the summary this many times.
            1 means that the original summary will be summarize. 0 means disabled.

        --n_summaries_target: int, default -1
            Only active if query is 'summarize_link_file' and
            --out_file_logseq_mode is True. Set a limit to
            the number of links that will be summarized. If the number of
            TODO in the output is higher, exit. If it's lower, only do the
            difference. -1 to disable.

        --summary_language: str, default "[same as input]"
            When writing a summary, the LLM will write using the language
            specified in this argument. If it's '[same as input]', the LLM
            will not translate.

        --dollar_limit: int, default 5
            If the estimated price is above this limit, stop instead.
            Note that the cost estimate for the embeddings is using the
            openai tokenizer, which is not universal.
            This check is skipped if the api_base url are changed.

        --debug: bool, default False
            if True will enable langchain tracing, increase verbosity etc.
            Will also disable multithreading for summaries and for loading
            files.

        --llm_verbosity: bool, default False
            if True, will print the intermediate reasonning steps of LLMs
            if debug is set, llm_verbosity is also set to True

        --notification_callback: Callable, default None
            a function that must take as input a string and return the same
            string. Inside it you can do whatever you want with it. This
            can be used for example to send notification on your phone
            using ntfy.sh to get summaries.

        --condense_question: bool, default True
            if True, will not use a special LLM call to reformulate the question
            when task is "query". Otherwise, the query will be reformulated as
            a standalone question. Useful when you have multiple questions in
            a row.
            Disabled if using a testing model.

        --chat_memory: bool, default True
            if True, will remember the messages across a given chat exchange.
            Disabled if using a testing model.

        --private: bool, default False
            add extra check that your data will never be sent to another
            server: for example check that the api_base was modified and used,
            check that no api keys are used, check that embedding models are
            local only. It will also use a separate cache from non private.

        --no_llm_cache: bool, default False
            WARNING: The cache is temporarily ignored in non openaillms
            generations because of an error with langchain's ChatLiteLLM.
            Basically if you don't use --private and use llm form openai,
            DocToolsLLM will use ChatOpenAI with regular caching, otherwise
            we use ChatLiteLLM with LLM caching disabled.
            More at https://github.com/langchain-ai/langchain/issues/22389

            disable caching for LLM. All caches are stored in the usual
            cache folder for your system. This does not disable caching
            for documents.

        --file_loader_parallel_backend: str, default "loky"
            joblib.Parallel backend to use when loading files. loky means
            multiprocessing while "threading" means multithreading.
            The number of jobs can be specified with 'file_loader_n_jobs'
            but it's a loader specific kwargs.

        --llms_api_bases: dict, default None
            a dict with keys in ["model", "query_eval_model"]
            The corresponding value will be used to change the url of the
            endpoint. This is needed to use local LLMs for example using
            ollama, lmstudio etc.

        --DIY_rolling_window_embedding: bool, default False
            enables using a DIY rolling window embedding instead of using
            the default langchain SentenceTransformerEmbedding implementation

        --import_mode: bool, default False
            if True, will return the answer from query instead of printing it

        --help or -h: bool, default False
            if True, will return this documentation.


        Loader specific arguments
        --------------------------
        (meaning they apply depending on the value of --filetype):

        --path
            Used by most loaders. For example for --filetype=youtube the path
            must point to a youtube video.

        --anki_profile
            The name of the profile
        --anki_deck
            The beginning of the deckname
            e.g. "science::physics::freshman_year::lesson1"
        --anki_notetype: str
            If it's part of the card's notetype, that notetype will be kept.
            Case insensitive.

        --anki_fields
            List of fields to keep
        --anki_mode:
            any of 'window', 'concatenate', 'single_note': (or _ separated value like 'concatenate_window'). By default 'window_single_note' is used.
                * 'single_note': 1 document is 1 anki note.
                * 'window': 1 documents is 5 anki note, overlapping
                * 'concatenate': 1 document is all anki notes concatenated as a single wall of text then split like any long document.
            Whatever you choose, you can later filter out documents by metadata
            filtering over the 'anki_mode' key.

        --whisper_lang
            if using whisper to transcribe an audio file, this if the language
            specified to whisper
        --whisper_prompt
            if using whisper to transcribe an audio file, this if the prompt
            given to whisper

        --youtube_language
            For youtube. e.g. ["fr","en"] to use french transcripts if
            possible and english otherwise
        --youtube_translation
            For youtube. e.g. "en" to use the transcripts after translation to english

        --include
            Only active if --filetype is one of json_list, recursive,
            link_file, youtube_playlist.
            --include can be a list of regex that must be present in the
            document PATH (not content!)
            --exclude can be a list of regex that if present in the PATH
            will exclude it.
            Exclude is run AFTER include
        --exclude
            See --include

        Other specific arguments
        ------------------------
        --out_file
            If doctools must create a summary, if out_file given the summary will
            be written to this file. Note that the file is not erased and
            Doctools will simply append to it.
            Related: see --out_file_logseq_mode

        --out_file_logseq_mode
            If --out_file is specified, this argument tells Doctools to export
            in a logseq friendly format. This means adding metadata of the run
            as block properties as well as setting TODO states.

        --out_check_file
            If --out_file_logseq_mode is True and --out_check_file is set:
            it must point to a file where each present TODO string will be
            counted and taken into account when calculating --n_summaries_target

        --filter_metadata
            list of regex string to use as metadata filter when querying.
            Format: "[kvb][+-]your_regex"

            For example:
            * Keep only documents that contain "anki" in any value
            of any of its metadata dict:
                --filter_metadata="v+anki"  <- at least the 'filetype' key
                will have as value 'anki'
            * Keep only documents that contain "anki_profile" as a key in
            its metadata dict:
                --filter_metadata="k+anki_profile"  <- because will contain the
                key anki_profile
            * Keep only data that have a certain 'source_tag' value:
                --filter_metadata="b+source_tag:my_source_tag_regex"

            Notes:
            * Each filter must be a regex string beginning with k, v or b
            (for 'key', 'value' or 'both'). Followed by either '+' or '-' to:
                '+' at least one metadata should match
                '-' exclude from (no metadata should match)
            * If the string starts with k, it will filter based on the keys
            of the metadata, if it starts with a v it will filter based
            on the values, if it starts with b it will require a ':' present
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

        --filter_content
            Like --filter_metadata but filters through the page_content of
            each document instead of the metadata.
            Syntax: "[+-]your_regex""
            Example:
            * Keep only the document that contain "doctools"
                --filter_content="+.*doctools.*"
            * Discard the document that contain "DOCTOOLS"
                --filter_content="-.*DOCTOOLS.*"

        --embed_instruct: bool
            when loading an embedding model using HuggingFace or
            llamacppembeddings backends, wether to wrap the input
            sentence using instruct framework or not.

        --file_loader_n_jobs: int, default 5
            number of threads to use when loading files. Set to 1 to disable
            multithreading (as it can result in out of memory error if
            using threads and overly recursive calls)

        --load_functions: List[str], default None
            list of strings that when evaluated in python result in a list of
            callable. The first must take one input of type string and the
            last function must return one string.

            For example in the filetypes 'local_html' this can be used to
            specify lambda functions that modify the text before running
            BeautifulSoup. Useful to decode html stored in .js files.
            Do tell me if you want more of this.

        --source_tag: str, default None
            a string that will be added to the document metadata at the
            key "source_tag". Useful when using filetype combination.

        Runtime flags
        -------------

        DOCTOOLS_TYPECHECKING="crash"
            default: "warn"
            Possible values:
                crash: crash if a typechecking fails
                warn: print a red warning if a typechecking fails
                disabled: disable typechecking

        """
        if help or h:
            print(self.__init__.__doc__)
            raise SystemExit()

        # make sure the extra args are valid
        for k in kwargs:
            if k not in extra_args:
                raise Exception(red(f"Found unexpected keyword argument: '{k}'"))

            # type checking of extra args
            if os.environ["DOCTOOLS_TYPECHECKING"] in ["crash", "warn"]:
                val = kwargs[k]
                curr_type = type(val)
                expected_type = extra_args[k]
                if not check_type(val, expected_type):
                    if os.environ["DOCTOOLS_TYPECHECKING"] == "warn":
                        red(f"Invalid type in kwargs: '{k}' is {val} of type {curr_type} instead of {expected_type}")
                    elif os.environ["DOCTOOLS_TYPECHECKING"] == "crash":
                        raise TypeCheckError(f"Invalid type in kwargs: '{k}' is {val} of type {curr_type} instead of {expected_type}")

        # checking argument validity
        assert "loaded_docs" not in kwargs, "'loaded_docs' cannot be an argument as it is used internally"
        assert "loaded_embeddings" not in kwargs, "'loaded_embeddings' cannot be an argument as it is used internally"
        task = task.replace("summary", "summarize")
        assert task in ["query", "search", "summarize", "summarize_then_query", "summarize_link_file"], "invalid task value"
        if task in ["summarize", "summarize_then_query"]:
            assert not load_embeds_from, "can't use load_embeds_from if task is summary"
        if task in ["query", "search", "summarize_then_query"]:
            assert query_eval_modelname is not None, "query_eval_modelname can't be None if doing RAG"
        else:
            query_eval_modelname = None
        assert (task == "summarize_link_file" and filetype == "link_file"
                ) or (task != "summarize_link_file" and filetype != "link_file"
                        ), "summarize_link_file must be used with filetype link_file"
        if task == "summarize_link_file":
            assert "path" in kwargs, 'missing path arg for summarize_link_file'
            assert "out_file" in kwargs, 'missing "out_file" arg for summarize_link_file'
            assert kwargs["out_file"] != kwargs["path"], "can't use same 'path' and 'out_file' arg"
        if filetype == "infer":
            assert "path" in kwargs and kwargs["path"], "If filetype is 'infer', a --path must be given"
        assert "/" in embed_model, "embed model must contain slash"
        assert embed_model.split("/", 1)[0] in ["openai", "sentencetransformers", "huggingface", "llamacppembeddings"], "Backend of embeddings must be either openai, sentencetransformers, huggingface of llamacppembeddings"
        if embed_kwargs is None:
            embed_kwargs = {}
        if isinstance(embed_kwargs, str):
            try:
                embed_kwargs = json.loads(embed_kwargs)
            except Exception as err:
                raise Exception(f"Failed to parse embed_kwargs: '{embed_kwargs}'")
        assert isinstance(embed_kwargs, dict), f"Not a dict but {type(embed_kwargs)}"
        assert query_eval_check_number > 0, "query_eval_check_number value"

        if filetype == "string":
            top_k = 1
            red("Input is 'string' so setting 'top_k' to 1")

        if llms_api_bases is None:
            llms_api_bases = {}
        elif isinstance(llms_api_bases, str):
            try:
                llms_api_bases = json.loads(llms_api_bases)
            except Exception as err:
                raise Exception(f"Error when parsing llms_api_bases as a dict: {err}")
        assert isinstance(llms_api_bases, dict), "llms_api_bases must be a dict"
        for k in llms_api_bases:
            assert k in ["model", "query_eval_model"], (
                f"Invalid k of llms_api_bases: {k}")
        for k in ["model", "query_eval_model"]:
            if k not in llms_api_bases:
                llms_api_bases[k] = None
        if llms_api_bases["model"] == llms_api_bases["query_eval_model"] and llms_api_bases["model"]:
            red(f"Setting litellm wide api_base because it's the same for model and query_eval_model")
            litellm.api_base = llms_api_bases["model"]
        assert isinstance(private, bool), "private arg should be a boolean, not {private}"
        if private:
            assert llms_api_bases["model"], "private is set but llms_api_bases['model'] is not set"
            assert llms_api_bases["query_eval_model"], "private is set but llms_api_bases['query_eval_model'] is not set"
            os.environ["DOCTOOLS_PRIVATEMODE"] = "true"
        else:
            os.environ["DOCTOOLS_PRIVATEMODE"] = "false"

        if (not modelname.startswith("testing/")) and (not llms_api_bases["model"]):
            modelname = model_name_matcher(modelname)
        if (query_eval_modelname is not None) and (not llms_api_bases["query_eval_model"]):
            if modelname.startswith("testing/"):
                if not query_eval_modelname.startswith("testing/"):
                    query_eval_modelname = "testing/testing"
                    red(f"modelname uses 'testing' backend so setting query_eval_modelname to '{query_eval_modelname}'")
            else:
                assert not query_eval_modelname.startswith("testing/"), "query_eval_modelname can't use 'testing' backend if modelname isn't set to testing too"
                query_eval_modelname = model_name_matcher(query_eval_modelname)

        if query is True:
            # otherwise specifying --query and forgetting to add text fails
            query = None
        if isinstance(query, str):
            query = query.strip() or None
        assert file_loader_parallel_backend in ["loky", "threading"], "Invalid value for file_loader_parallel_backend"
        if "{user_cache}" in save_embeds_as:
            save_embeds_as = save_embeds_as.replace("{user_cache}", str(cache_dir))

        if debug:
            llm_verbosity = True

        # storing as attributes
        self.modelbackend = modelname.split("/")[0].lower() if "/" in modelname else "openai"
        self.modelname = modelname
        if query_eval_modelname is not None:
            self.query_eval_modelbackend = query_eval_modelname.split("/")[0].lower() if "/" in modelname else "openai"
            self.query_eval_modelname = query_eval_modelname
        self.task = task
        self.filetype = filetype
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs
        self.save_embeds_as = save_embeds_as
        self.load_embeds_from = load_embeds_from
        self.top_k = top_k
        self.query_retrievers = query_retrievers if "testing" not in modelname else query_retrievers.replace("hyde", "")
        self.query_eval_check_number = int(query_eval_check_number)
        self.query_relevancy = query_relevancy
        self.debug = debug
        self.kwargs = kwargs
        self.llm_verbosity = llm_verbosity
        self.n_recursive_summary = n_recursive_summary
        self.n_summaries_target = n_summaries_target
        self.summary_language = summary_language
        self.dollar_limit = dollar_limit
        self.condense_question = bool(condense_question) if "testing" not in modelname else False
        self.chat_memory = chat_memory if "testing" not in modelname else False
        self.private = bool(private)
        self.no_llm_cache = bool(no_llm_cache)
        self.file_loader_parallel_backend = file_loader_parallel_backend
        self.llms_api_bases = llms_api_bases
        self.DIY_rolling_window_embedding = bool(DIY_rolling_window_embedding)
        self.import_mode = import_mode

        if not no_llm_cache:
            if not private:
                set_llm_cache(SQLiteCache(database_path=cache_dir / "langchain.db"))
            else:
                set_llm_cache(SQLiteCache(database_path=cache_dir / "private_langchain.db"))

        if llms_api_bases["model"]:
            red(f"Disabling price computation for model because api_base was modified")
            self.llm_price = [0, 0]
        elif modelname in litellm.model_cost:
            self.llm_price = [
                litellm.model_cost[modelname]["input_cost_per_token"],
                litellm.model_cost[modelname]["output_cost_per_token"]
            ]
        elif modelname.split("/")[1] in litellm.model_cost:
            self.llm_price = [
                litellm.model_cost[modelname.split("/")[1]]["input_cost_per_token"],
                litellm.model_cost[modelname.split("/")[1]]["output_cost_per_token"]
            ]
        else:
            raise Exception(red(f"Can't find the price of {modelname}"))
        if query_eval_modelname is not None:
            if llms_api_bases["query_eval_model"]:
                red(f"Disabling price computation for query_eval_model because api_base was modified")
                self.query_evalllm_price = [0, 0]
            elif query_eval_modelname in litellm.model_cost:
                self.query_evalllm_price = [
                    litellm.model_cost[query_eval_modelname]["input_cost_per_token"],
                    litellm.model_cost[query_eval_modelname]["output_cost_per_token"]
                ]
            elif query_eval_modelname.split("/")[1] in litellm.model_cost:
                self.query_evalllm_price = [
                    litellm.model_cost[query_eval_modelname.split("/")[1]]["input_cost_per_token"],
                    litellm.model_cost[query_eval_modelname.split("/")[1]]["output_cost_per_token"]
                ]
            else:
                raise Exception(red(f"Can't find the price of {query_eval_modelname}"))

        if notification_callback is not None:
            @optional_typecheck
            def ntfy(text: str) -> str:
                out = notification_callback(text)
                assert out == text, "The notification callback must return the same string"
                return out
            ntfy("Starting DocToolsLLM")
        else:
            @optional_typecheck
            def ntfy(text: str) -> str:
                return text

        if self.debug:
            # os.environ["LANGCHAIN_TRACING_V2"] = "true"
            set_verbose(True)
            set_debug(True)
            kwargs["file_loader_n_jobs"] = 1
            litellm.set_verbose=True
        else:
            litellm.set_verbose=False
            # fix from https://github.com/BerriAI/litellm/issues/2256
            import logging
            for logger_name in ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM"]:
                logger = logging.getLogger(logger_name)
                # logger.setLevel(logging.CRITICAL + 1)
                logger.setLevel(logging.WARNING)

        # don't crash if extra arguments are used for a model
        # litellm.drop_params = True  # drops parameters that are not used by some models

        # compile include / exclude regex
        if "include" in self.kwargs:
            for i, inc in enumerate(self.kwargs["include"]):
                if inc == inc.lower():
                    self.kwargs["include"][i] = re.compile(inc, flags=re.IGNORECASE)
                else:
                    self.kwargs["include"][i] = re.compile(inc)
        if "exclude" in self.kwargs:
            for i, exc in enumerate(self.kwargs["exclude"]):
                if exc == exc.lower():
                    self.kwargs["exclude"][i] = re.compile(exc, flags=re.IGNORECASE)
                else:
                    self.kwargs["exclude"][i] = re.compile(exc)

        # loading llm
        self.llm = load_llm(
            modelname=modelname,
            backend=self.modelbackend,
            no_llm_cache=self.no_llm_cache,
            temperature=0,
            verbose=self.llm_verbosity,
            api_base=self.llms_api_bases["model"],
            private=self.private,
        )

        # if task is to summarize lots of links, check first if there are
        # links already summarized as it would greatly reduce the number of
        # documents to load
        if self.task == "summarize_link_file" and "out_file_logseq_mode" in kwargs:
            if not Path(self.kwargs["out_file"]).exists():
                Path(self.kwargs["out_file"]).touch()
            with open(self.kwargs["out_file"], "r") as f:
                output_content = f.read()

            if self.n_summaries_target > 0:
                self.n_todos_present = output_content.count("- TODO ")

            if "out_check_file" in self.kwargs:
                # this is an undocumented function for the author. It
                # allows to specify a second path for which to check if
                # a document has already been summaried. I use this because
                # I made a script to automatically move my DONE tasks
                # from logseq to another near by file.
                assert Path(self.kwargs["out_check_file"]).exists()
                with open(self.kwargs["out_check_file"], "r") as f:
                    output_content += f.read()

            # parse just the links already present in the output
            doclist = output_content.splitlines()
            doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
            doclist = [p.strip() for p in doclist if p.strip() and not p.strip().startswith("#") and "http" in p]
            links_regex = re.compile(r'(https?://\S+)')
            doclist = [
                    matched.group(0)
                    for d in doclist
                    if (matched := links_regex.search(d).strip())
            ]

            self.done_links = " ".join(doclist)
            self.kwargs["done_links"] = doclist
            self.kwargs["n_summaries_target"] = self.n_summaries_target

        # loading documents
        if not load_embeds_from:
            self.loaded_docs = batch_load_doc(
                filetype=self.filetype,
                debug=self.debug,
                task=self.task,
                backend=self.file_loader_parallel_backend,
                **self.kwargs)

            # check that the hash are unique
            if len(self.loaded_docs) > 1:
                ids = [id(d.metadata) for d in self.loaded_docs]
                assert len(ids) == len(set(ids)), (
                        "Same metadata object is used to store information on "
                        "multiple documents!")

                hashes = [d.metadata["hash"] for d in self.loaded_docs]
                uniq_hashes = list(set(hashes))
                removed_paths = []
                removed_docs = []
                counter = {h: hashes.count(h) for h in uniq_hashes}
                if len(hashes) != len(uniq_hashes):
                    red("Found duplicate hashes after loading documents:")

                    for i, doc in enumerate(tqdm(self.loaded_docs, desc="Looking for duplicates")):
                        h = doc.metadata['hash']
                        n = counter[h]
                        if n > 1:
                            removed_docs.append(self.loaded_docs[i])
                            self.loaded_docs[i] = None
                            counter[h] -= 1
                        assert counter[h] > 0
                    red(f"Removed {len(removed_docs)}/{len(hashes)} documents because they had the same hash")

                    # check if deduplication likely amputated documents
                    self.loaded_docs = [d for d in self.loaded_docs if d is not None]
                    present_path = [d.metadata["path"] for d in self.loaded_docs]

                    intersect = set(removed_paths).intersection(set(present_path))
                    if intersect:
                        red(f"Found {len(intersect)} documents that were only partially removed, this results in incomplete documents.")
                        for i, inte in enumerate(intersect):
                            red(f"  * #{i + 1}: {inte}")
                        raise Exception()
                    else:
                        red(f"Removed {len(removed_paths)}/{len(hashes)} documents because they had the same hash")

        else:
            self.loaded_docs = None  # will be loaded when embeddings are loaded

        if self.task in ["summarize_link_file", "summarize", "summarize_then_query"]:
            self.summary_task()

            if self.task == "summary_then_query":
                whi("Done summarizing. Switching to query mode.")
                if self.modelbackend == "openai":
                    del self.llm.model_kwargs["logit_bias"]
            else:
                whi("Done summarizing. Exiting.")
                raise SystemExit()

        assert self.task in ["query", "search", "summary_then_query"], f"Invalid task: {self.task}"
        self.prepare_query_task()

        if not self.import_mode:
            while True:
                self.query(query=query)
                query = None
        else:
            whi("Ready to query, call self.query(your_question)")

    def summary_task(self):
        # storing links in dict instead of set to keep the original ordering
        links_todo = {}
        # failed = []

        # get the list of documents from the same source. Also checks if
        # it's not part of the output file if task is "summarize_link_file"
        if self.task == "summarize_link_file":

            for d in self.loaded_docs:
                assert "subitem_link" in d.metadata, "missing 'subitem_link' in a doc metadata"

                link = d.metadata["subitem_link"]
                if link in self.done_links or link in links_todo:
                    continue

                if self.n_summaries_target == -1:
                    links_todo[link] = None
                else:
                    if len(links_todo) < self.n_summaries_target:
                        links_todo[link] = None
                    else:
                        whi(ntfy("'n_summaries_target' limit reached, will not add more links to summarize for this run."))
                        break

            # comment out the links that are marked as already done
            # if self.done_links:
            #     with open(self.kwargs["path"], "r") as f:
            #         temp = f.read().split("\n")
            #     with open(self.kwargs["path"], "w") as f:
            #         for t in temp:
            #             for done_link in self.done_links:
            #                 if done_link in t:
            #                     t = f"#already done as of {today}# {t}"
            #                     break
            #             f.write(t.strip() + "\n")

            if self.n_summaries_target > 0:
                # allows to run DocTools to summarise from a link file
                # only if there are less than 'n_summaries_target' TODOS
                # blocks in the target file. This way we can have a
                # list of TODOS that will never be larger than this.
                # Avoiding both having too many summaries and not enough
                # as it allows to run this frequently
                n_todos_desired = self.n_summaries_target
                if self.n_todos_present >= n_todos_desired:
                    return red(ntfy(f"Found {self.n_todos_present} in the output file(s) which is >= {n_todos_desired}. Exiting without summarising."))
                else:
                    self.n_summaries_target = n_todos_desired - self.n_todos_present
                    red(ntfy(f"Found {self.n_todos_present} in output file(s) which is under {n_todos_desired}. Will summarize only {self.n_summaries_target}"))
                    assert self.n_summaries_target > 0

                while len(links_todo) > self.n_summaries_target:
                    del links_todo[list(links_todo.keys())[-1]]

            # estimate price before summarizing, in case you put the bible in there
            docs_tkn_cost = {}
            for doc in self.loaded_docs:
                meta = doc.metadata["subitem_link"]
                if meta in links_todo:
                    if meta not in docs_tkn_cost:
                        docs_tkn_cost[meta] = get_tkn_length(doc.page_content)
                    else:
                        docs_tkn_cost[meta] += get_tkn_length(doc.page_content)

        else:
            for d in self.loaded_docs:
                links_todo[d.metadata["path"]] = None
            assert len(links_todo) == 1, f"Invalid length of links_todo for this task: '{len(links_todo)}'"

            docs_tkn_cost = {}
            for doc in self.loaded_docs:
                meta = doc.metadata["path"]
                if meta not in docs_tkn_cost:
                    docs_tkn_cost[meta] = get_tkn_length(doc.page_content)
                else:
                    docs_tkn_cost[meta] += get_tkn_length(doc.page_content)

        full_tkn = sum(list(docs_tkn_cost.values()))
        red("Token price of each document:")
        for k, v in docs_tkn_cost.items():
            pr = v * (self.llm_price[0] * 4 + self.llm_price[1]) / 5 / 1000
            red(f"- {v:>6}: {k:>10} - ${pr:04f}")

        red(f"Total number of tokens in documents to summarize: '{full_tkn}'")
        # a conservative estimate is that it takes 4 times the number
        # of tokens of a document to summarize it
        price = (self.llm_price[0] * 3 + self.llm_price[1] * 2) / 5
        estimate_dol = full_tkn / 1000 * price * 1.1
        if self.n_recursive_summary:
            for i in range(1, self.n_recursive_summary + 1):
                estimate_dol += full_tkn / 1000 * ((2/5) ** i) * price * 1.1
        whi(ntfy(f"Conservative estimate of the LLM cost to summarize: ${estimate_dol:.4f} for {full_tkn} tokens."))
        if estimate_dol > self.dollar_limit:
            if self.llms_api_bases["model"]:
                raise Exception(red(ntfy(f"Cost estimate ${estimate_dol:.5f} > ${self.dollar_limit} which is absurdly high. Has something gone wrong? Quitting.")))
            else:
                red(f"Cost estimate > limit but the api_base was modified so not crashing.")

        if self.modelbackend == "openai":
            # increase likelyhood that chatgpt will use indentation by
            # biasing towards adding space.
            logit_val = 3
            self.llm.model_kwargs["logit_bias"] = {
                    12: logit_val,  # '-'
                    # 220: logit_val,  # ' '
                    # 532: logit_val,  # ' -'
                    # 9: logit_val,  # '*'
                    # 1635: logit_val,  # ' *'
                    197: logit_val,  # '\t'
                    334: logit_val,  # '**'
                    # 25: logit_val,  # ':'
                    # 551: logit_val,  # ' :'
                    # 13: -1,  # '.'
                    }
            self.llm.model_kwargs["frequency_penalty"] = 0.5
            self.llm.model_kwargs["temperature"] = 0.0

        @optional_typecheck
        def threaded_summary(link: str, lock: Lock) -> dict:
            if self.task == "summarize_link_file":
                # get only the docs that match the link
                relevant_docs = [d for d in self.loaded_docs if d.metadata["subitem_link"] == link]
            else:
                relevant_docs = self.loaded_docs
            assert relevant_docs, 'Empty relevant_docs!'

            # parse metadata from the doc
            metadata = []
            if "http" in link:
                item_name = tldextract.extract(link).registered_domain
            elif "/" in link and Path(link).exists():
                item_name = Path(link).name
            else:
                item_name = link

            if "title" in relevant_docs[0].metadata:
                item_name = f"{relevant_docs[0].metadata['title'].strip()} - {item_name}"
            else:
                metadata.append(f"Title: '{item_name.strip()}'")


            # replace # in title as it would be parsed as a tag
            item_name = item_name.replace("#", r"\#")

            if "docs_reading_time" in relevant_docs[0].metadata:
                doc_reading_length = relevant_docs[0].metadata["docs_reading_time"]
                metadata.append(f"Reading length: {doc_reading_length:.1f} minutes")
            else:
                doc_reading_length = None
            if "author" in relevant_docs[0].metadata:
                author = relevant_docs[0].metadata["author"].strip()
                metadata.append(f"Author: '{author}'")
            else:
                author = None

            if metadata:
                metadata = "- Text metadata:\n\t- " + "\n\t- ".join(metadata) + "\n"
                metadata += "\t- Section number: [PROGRESS]\n"
            else:
                metadata = ""

            # summarize each chunk of the link and return one text
            summary, n_chunk, doc_total_tokens, doc_total_cost = do_summarize(
                    docs=relevant_docs,
                    metadata=metadata,
                    language=self.summary_language,
                    modelbackend=self.modelbackend,
                    llm=self.llm,
                    llm_price=self.llm_price,
                    verbose=self.llm_verbosity,
                    )

            # get reading length of the summary
            real_text = "".join([letter for letter in list(summary) if letter.isalpha()])
            sum_reading_length = len(real_text) / average_word_length / wpm
            whi(f"{item_name} reading length is {sum_reading_length:.1f}")

            n_recursion_done = 0
            if self.n_recursive_summary > 0:
                splitter = get_splitter("recursive_summary")
                summary_text = summary

                for n_recur in range(1, self.n_recursive_summary + 1):
                    red(f"Doing recursive summary #{n_recur} of {item_name}")

                    # remove any chunk count that is not needed to summarize
                    sp = summary_text.split("\n")
                    for i, l in enumerate(sp):
                        if l.strip() == "- ---":
                            sp[i] = None
                        elif re.search(r"- Chunk \d+/\d+", l):
                            sp[i] = None
                        elif l.strip().startswith("- BEFORE RECURSION #"):
                            for new_i in range(i, len(sp)):
                                sp[new_i] = None
                            break
                    summary_text = "\n".join([s.rstrip() for s in sp if s])
                    assert "- ---" not in summary_text, "Found chunk separator"
                    assert "- Chunk " not in summary_text, "Found chunk marker"
                    assert "- BEFORE RECURSION # " not in summary_text, "Found recursion block"

                    summary_docs = [Document(page_content=summary_text)]
                    summary_docs = splitter.transform_documents(summary_docs)
                    try:
                        check_docs_tkn_length(summary_docs, item_name)
                    except Exception as err:
                        red(f"Exception when checking if {item_name} could be recursively summarized for the #{n_recur} time: {err}")
                        break
                    summary_text, n_chunk, new_doc_total_tokens, new_doc_total_cost = do_summarize(
                            docs=summary_docs,
                            metadata=metadata,
                            language=self.summary_language,
                            modelbackend=self.modelbackend,
                            llm=self.llm,
                            llm_price=self.llm_price,
                            verbose=self.llm_verbosity,
                            n_recursion=n_recur,
                            logseq_mode="out_file_logseq_mode" in self.kwargs,
                            )
                    doc_total_tokens += new_doc_total_tokens
                    doc_total_cost += new_doc_total_cost
                    n_recursion_done += 1

                    # clean text again to compute the reading length
                    sp = summary_text.split("\n")
                    for i, l in enumerate(sp):
                        if l.strip() == "- ---":
                            sp[i] = None
                        elif re.search(r"- Chunk \d+/\d+", l):
                            sp[i] = None
                        elif l.strip().startswith("- BEFORE RECURSION #"):
                            for new_i in range(i, len(sp)):
                                sp[new_i] = None
                            break
                    real_text = "\n".join([s.rstrip() for s in sp if s])
                    assert "- ---" not in real_text, "Found chunk separator"
                    assert "- Chunk " not in real_text, "Found chunk marker"
                    assert "- BEFORE RECURSION # " not in real_text, "Found recursion block"
                    real_text = "".join([letter for letter in list(real_text) if letter.isalpha()])
                    sum_reading_length = len(real_text) / average_word_length / wpm
                    whi(f"{item_name} reading length after recursion #{n_recur} is {sum_reading_length:.1f}")
                summary = summary_text

            with lock:
                print("\n\n")
                md_printer("# Summary")
                md_printer(f'## {link}')
                md_printer(summary)

                red(f"Tokens used for {link}: '{doc_total_tokens}' (${doc_total_cost:.5f})")

            summary_tkn_length = get_tkn_length(summary)

            if "out_file_logseq_mode" in self.kwargs:
                header = f"\n- TODO {item_name}"
                header += "\n  collapsed:: true"
                header += "\n  block_type:: DocToolsLLM_summary"
                header += f"\n  DocToolsLLM_version:: {self.VERSION}"
                header += f"\n  DocToolsLLM_model:: {self.modelname} of {self.modelbackend}"
                header += f"\n  DocToolsLLM_parameters:: n_recursion_summary={self.n_recursive_summary};n_recursion_done={n_recursion_done}"
                header += f"\n  summary_date:: {today}"
                header += f"\n  summary_timestamp:: {int(time.time())}"
                header += f"\n  token_cost:: {doc_total_tokens}"
                header += f"\n  dollar_cost:: {doc_total_cost:.5f}"
                header += f"\n  summary_token_length:: {summary_tkn_length}"
                header += f"\n  summary_reading_time:: {sum_reading_length:.1f}"
                header += f"\n  link:: {link}"
                if doc_reading_length:
                    header += f"\n  doc_reading_time:: {doc_reading_length:.1f}"
                    header += f"\n  reading_time_prct_speedup:: {int(sum_reading_length/doc_reading_length * 100)}%"
                if n_chunk > 1:
                    header += f"\n  chunks:: {n_chunk}"
                if author:
                    header += f"\n  author:: {author}"
                header += f"\n  language:: {self.summary_language}"

            else:
                header = f"\n- {item_name}    cost: {doc_total_tokens} (${doc_total_cost:.5f})"
                if doc_reading_length:
                    header += f"    {doc_reading_length:.1f} minutes"
                if author:
                    header += f"    by '{author}'"
                header += f"    original link: '{link}'"
                header += f"    DocToolsLLM version {self.VERSION} with model {self.modelname} of {self.modelbackend}"
                header += f"    parameters: n_recursion_summary={self.n_recursive_summary};n_recursion_done={n_recursion_done}"

            # save to output file
            if "out_file" in self.kwargs:
                Path(self.kwargs["out_file"]).touch()  # create file if missing
                with lock:
                    with open(self.kwargs["out_file"], "a") as f:
                        f.write(header)
                        for bulletpoint in summary.split("\n"):
                            f.write("\n")
                            bulletpoint = bulletpoint.rstrip()
                            # # make sure the line begins with a bullet point
                            # if not bulletpoint.lstrip().startswith("- "):
                            #     begin_space = re.search(r"^(\s+)", bulletpoint)
                            #     if not begin_space:
                            #         begin_space = [""]
                            #     bulletpoint = begin_space[0] + "- " + bulletpoint
                            f.write(f"\t{bulletpoint}")
                        f.write("\n\n\n")
            return {
                    "link": link,
                    "sum_reading_length": sum_reading_length,
                    "doc_reading_length": doc_reading_length,
                    "doc_total_tokens": doc_total_tokens,
                    "doc_total_cost": doc_total_cost,
                    "summary": summary,
                    }

        lock = Lock()
        results = Parallel(
                n_jobs=3 if not self.debug else 1,
                backend="threading",
                )(delayed(threaded_summary)(
                    link=link,
                    lock=lock,
                    ) for link in tqdm(
                        links_todo,
                        desc="Summarizing documents",
                        # disable=(not len(links_todo) - 1) or self.debug,
                        colour="magenta",
                        ))
        total_tkn_cost = sum([x["doc_total_tokens"] for x in results])
        total_dol_cost = sum([x["doc_total_cost"] for x in results])
        total_docs_length = sum([x["doc_reading_length"] for x in results])
        # total_summary_length = sum([x["sum_reading_length"] for x in results])

        red(ntfy(f"Total cost of those summaries: '{total_tkn_cost}' (${total_dol_cost:.5f}, estimate was ${estimate_dol:.5f})"))
        red(ntfy(f"Total time saved by those summaries: {total_docs_length:.1f} minutes"))

        # if "out_file" in self.kwargs:
        #     # after summarizing all links, append to output file the total cost
        #     if total_tkn_cost != 0 and total_dol_cost != 0:
        #         with open(self.kwargs["out_file"], "a") as f:
        #             f.write(f"- Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})\n")
        #             f.write(f"- Total time saved by this run: {total_docs_length - total_summary_length:.1f} minutes\n\n\n")

        # and write to input file a summary too
        # if "out_file" in self.kwargs:
        #     try:
        #         with open(self.kwargs["path"], "a") as f:
        #             f.write(f"\n\n")
        #             f.write(f"- Done with summaries of {today}\n")
        #             f.write(f"    - Number of links summarized: {len(links_todo) - len(failed)}/{len(links_todo) + len(self.done_links)}\n")
        #             if failed:
        #                 f.write(f"    - Number of links failed: {len(failed)}:\n")
        #                 for f in failed:
        #                     f.write(f"        - {f}\n")
        #             # f.write(f"    - Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})\n")
        #             # f.write(f"    - Total time saved by this run: plausibly {total_docs_length:.1f} minutes\n")
        #     except Exception as err:
        #         red(f"Exception when writing end of run details to input file: '{err}'")

    def prepare_query_task(self):
        # load embeddings for querying
        self.loaded_embeddings, self.embeddings = load_embeddings(
            embed_model=self.embed_model,
            embed_kwargs=self.embed_kwargs,
            load_embeds_from=self.load_embeds_from,
            save_embeds_as=self.save_embeds_as,
            debug=self.debug,
            loaded_docs=self.loaded_docs,
            dollar_limit=self.dollar_limit,
            private=self.private,
            use_rolling=self.DIY_rolling_window_embedding,
            kwargs=self.kwargs,
        )

        # conversational memory
        self.memory = AnswerConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True)

        # set default ask_user argument
        self.cli_settings = {
                "top_k": self.top_k,
                "multiline": False,
                "retriever": self.query_retrievers,
                "task": self.task,
                "relevancy": self.query_relevancy,
                }
        self.all_texts = [v.page_content for k, v in self.loaded_embeddings.docstore._dict.items()]

        # parse filters as callable for faiss filtering
        if "filter_metadata" in self.kwargs or "filter_content" in self.kwargs:
            if "filter_metadata" in self.kwargs:
                if isinstance(self.kwargs["filter_metadata"], str):
                    filter_metadata = self.kwargs["filter_metadata"].split(",")
                else:
                    filter_metadata = self.kwargs["filter_metadata"]
                assert isinstance(filter_metadata, list), f"filter_metadata must be a list, not {self.kwargs['filter_metadata']}"

                # storing fast as list then in tupples for faster iteration
                filters_k_plus = []
                filters_k_minus = []
                filters_v_plus = []
                filters_v_minus = []
                filters_b_plus_keys = []
                filters_b_plus_values = []
                filters_b_minus_keys = []
                filters_b_minus_values = []
                for f in filter_metadata:
                    assert isinstance(f, str), f"Filter must be a string: '{f}'"
                    kvb = f[0]
                    assert kvb in ["k", "v", "b"], f"filter 1st character must be k, v or b: '{f}'"
                    incexc = f[1]
                    assert incexc in ["+", "-"], f"filter 2nd character must be + or -: '{f}'"
                    incexc_str = "plus" if incexc == "+" else "minus"
                    assert f[2:].strip(), f"Filter can't be an empty regex: '{f}'"
                    pattern = f[2:].strip()
                    if kvb == "b":
                        assert ":" in f, (
                            "Filter starting with b must contain "
                            "a ':' to distinguish the key regex and the value "
                            f"regex: '{f}'")
                        key_pat, value_pat = pattern.split(":", 1)
                        if key_pat == key_pat.lower():
                            key_pat = re.compile(key_pat, flags=re.IGNORECASE)
                        else:
                            key_pat = re.compile(key_pat)
                        if value_pat == value_pat.lower():
                            value_pat = re.compile(value_pat, flags=re.IGNORECASE)
                        else:
                            value_pat = re.compile(value_pat)
                        assert key_pat not in locals()[f"filters_b_{incexc_str}_keys"], (
                            f"Can't use several filters for the same key "
                            "regex. Use a single but more complex regex"
                            f": '{f}'"
                        )
                        locals()[f"filters_b_{incexc_str}_keys"].append(key_pat)
                        locals()[f"filters_b_{incexc_str}_values"].append(value_pat)
                    else:
                        if pattern == pattern.lower():
                            pattern = re.compile(pattern, flags=re.IGNORECASE)
                        else:
                            pattern = re.compile(pattern)
                        locals()[f"filters_{kvb}_{incexc_str}"].append(pattern)
                assert len(filters_b_plus_keys) == len(filters_b_plus_values)
                assert len(filters_b_minus_keys) == len(filters_b_minus_values)

                # store as tuple for faster iteration
                filters_k_plus = tuple(filters_k_plus)
                filters_k_minus = tuple(filters_k_minus)
                filters_v_plus = tuple(filters_v_plus)
                filters_v_minus = tuple(filters_v_minus)
                filters_b_plus_keys = tuple(filters_b_plus_keys)
                filters_b_plus_values = tuple(filters_b_plus_values)
                filters_b_minus_keys = tuple(filters_b_minus_keys)
                filters_b_minus_values = tuple(filters_b_minus_values)

                def filter_meta(meta: dict) -> bool:
                    # match keys
                    for inc in filters_k_plus:
                        if not any(inc.match(k) for k in meta.keys()):
                            return False
                    for exc in filters_k_minus:
                        if any(exc.match(k) for k in meta.keys()):
                            return False

                    # match values
                    for inc in filters_v_plus:
                        if not any(inc.match(v) for v in meta.values()):
                            return False
                    for exc in filters_v_minus:
                        if any(exc.match(v) for v in meta.values()):
                            return False

                    # match both
                    for kp, vp in zip(filters_b_plus_keys, filters_b_plus_values):
                        good_keys = (k for k in meta.keys() if kp.match(k))
                        gk_checked = 0
                        for gk in good_keys:
                            if vp.match(meta[gk]):
                                gk_checked += 1
                                break
                        if not gk_checked:
                            return False
                    for kp, vp in zip(filters_b_minus_keys, filters_b_minus_values):
                        good_keys = (k for k in meta.keys() if kp.match(k))
                        gk_checked = 0
                        for gk in good_keys:
                            if vp.match(meta[gk]):
                                return False
                            gk_checked += 1
                        if not gk_checked:
                            return False

                    return True
            else:
                def filter_meta(meta: dict) -> bool:
                    return True

            if "filter_content" in self.kwargs:
                if isinstance(self.kwargs["filter_content"], str):
                    filter_content = self.kwargs["filter_content"].split(",")
                else:
                    filter_content = self.kwargs["filter_content"]
                assert isinstance(filter_content, list), f"filter_content must be a list, not {self.kwargs['filter_content']}"

                # storing fast as list then in tupples for faster iteration
                filters_cont_plus = []
                filters_cont_minus = []

                for f in filter_content:
                    assert isinstance(f, str), f"Filter must be a string: '{f}'"
                    incexc = f[0]
                    assert incexc in ["+", "-"], f"filter 1st character must be + or -: '{f}'"
                    incexc_str = "plus" if incexc == "+" else "minus"
                    assert f[1:].strip(), f"Filter can't be an empty regex: '{f}'"
                    pattern = f[1:].strip()
                    if pattern == pattern.lower():
                        pattern = re.compile(pattern, flags=re.IGNORECASE)
                    else:
                        pattern = re.compile(pattern)
                    locals()[f"filters_cont_{incexc_str}"].append(pattern)
                filters_cont_plus = tuple(filters_cont_plus)
                filters_cont_minus = tuple(filters_cont_minus)

                def filter_cont(cont: str) -> bool:
                    if not all(inc.match(cont) for inc in filters_cont_plus):
                        return False
                    if any(exc.match(cont) for exc in filters_cont_minus):
                        return False
                    return True

            else:
                def filter_cont(cont: str) -> bool:
                    return True

            # check filtering is valid
            checked = 0
            good = 0
            ids_to_del = []
            for doc_id, doc in tqdm(
                self.loaded_embeddings.docstore._dict.items(),
                desc="filtering",
                unit="docs",
                ):
                checked += 1
                if filter_meta(doc.metadata) and filter_cont(doc.page_content):
                    good += 1
                else:
                    ids_to_del.append(doc_id)
            red(f"Keeping {good}/{checked} documents from vectorstore after filtering")
            if good == checked:
                red("Your filter matched all stored documents!")
            assert good, "No documents in the vectorstore match the given filter"

            # directly remove the filtered documents from the docstore
            # but first store the docstore before altering it to allow
            # unfiltering in the prompt
            self.unfiltered_docstore = self.loaded_embeddings.serialize_to_bytes()
            status = self.loaded_embeddings.delete(ids_to_del)

            # checking deletiong want well
            if status is False:
                raise Exception("Vectorstore filtering failed")
            elif status is None:
                raise Exception("Vectorstore filtering not implemented")
            assert len(self.loaded_embeddings.docstore._dict) == checked - len(ids_to_del), "Something went wrong when deleting filtered out documents"
            assert len(self.loaded_embeddings.docstore._dict), "Something went wrong when deleting filtered out documents: no document left"
            assert len(self.loaded_embeddings.docstore._dict) == len(self.loaded_embeddings.index_to_docstore_id), "Something went wrong when deleting filtered out documents"


    #@optional_typecheck
    def query(self, query: Optional[str]) -> Optional[str]:
        if not query:
            query, self.cli_settings = ask_user(self.cli_settings)
            if "do_reset_memory" in self.cli_settings:
                assert self.cli_settings["do_reset_memory"]
                del self.cli_settings["do_reset_memory"]
                self.memory = AnswerConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True)
        assert all(
            retriev in ["default", "hyde", "knn", "svm", "parent"]
            for retriev in self.cli_settings["retriever"].split("_")
        ), f"Invalid retriever value: {self.cli_settings['retriever']}"
        retrievers = []
        if "hyde" in self.cli_settings["retriever"].lower():
            retrievers.append(
                    create_hyde_retriever(
                        query=query,

                        llm=self.llm,
                        top_k=self.cli_settings["top_k"],
                        relevancy=self.cli_settings["relevancy"],

                        embeddings=self.embeddings,
                        loaded_embeddings=self.loaded_embeddings,
                        )
                    )

        if "knn" in self.cli_settings["retriever"].lower():
            retrievers.append(
                    KNNRetriever.from_texts(
                        self.all_texts,
                        self.embeddings,
                        relevancy_threshold=self.cli_settings["relevancy"],
                        k=self.cli_settings["top_k"],
                        )
                    )
        if "svm" in self.cli_settings["retriever"].lower():
            retrievers.append(
                    SVMRetriever.from_texts(
                        self.all_texts,
                        self.embeddings,
                        relevancy_threshold=self.cli_settings["relevancy"],
                        k=self.cli_settings["top_k"],
                        )
                    )
        if "parent" in self.cli_settings["retriever"].lower():
            retrievers.append(
                    create_parent_retriever(
                        task=self.task,
                        loaded_embeddings=self.loaded_embeddings,
                        loaded_docs=self.loaded_docs,
                        top_k=self.cli_settings["top_k"],
                        relevancy=self.cli_settings["relevancy"],
                        )
                    )

        if "default" in self.cli_settings["retriever"].lower():
            retrievers.append(
                    self.loaded_embeddings.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={
                            "k": self.cli_settings["top_k"],
                            "distance_metric": "cos",
                            "score_threshold": self.cli_settings["relevancy"],
                            })
                        )

        assert retrievers, "No retriever selected. Probably cause by a wrong cli_command or query_retrievers arg."
        if len(retrievers) == 1:
            retriever = retrievers[0]
        else:
            retriever = MergerRetriever(retrievers=retrievers)

            # remove redundant results from the merged retrievers:
            filtered = EmbeddingsRedundantFilter(
                    embeddings=self.embeddings,
                    similarity_threshold=0.999,
                    )
            pipeline = DocumentCompressorPipeline(transformers=[filtered])
            retriever = ContextualCompressionRetriever(
                base_compressor=pipeline, base_retriever=retriever
            )

        if " // " in query:
            sp = query.split(" // ")
            assert len(sp) == 2, "The query must contain a maximum of 1 // symbol"
            query_fe = sp[0].strip()
            query_an = sp[1].strip()
        else:
            query_fe, query_an = copy.copy(query), copy.copy(query)
        whi(f"Query for the embeddings: {query_fe}")
        whi(f"Question to answer: {query_an}")

        # the eval doc chain needs its own caching
        if not self.no_llm_cache:
            eval_cache_wrapper = doc_eval_cache.cache
        else:
            def eval_cache_wrapper(func): return func

        @chain
        @optional_typecheck
        @eval_cache_wrapper
        def evaluate_doc_chain(
                inputs: dict,
                query_nb: int = self.query_eval_check_number,
                eval_model_name: str = self.query_eval_modelname,
            ) -> List[str]:
            if "n" in self.eval_llm_params or self.query_eval_check_number == 1:
                out = self.eval_llm._generate(PR_EVALUATE_DOC.format_messages(**inputs))
                outputs = [gen.text for gen in out.generations]
                assert outputs, "No generations found by query eval llm"
                outputs = [parse_eval_output(o) for o in outputs]
                new_p = out.llm_output["token_usage"]["prompt_tokens"]
                new_c = out.llm_output["token_usage"]["completion_tokens"]
            else:
                outputs = []
                new_p = 0
                new_c = 0
                async def eval(inputs):
                    return await self.eval_llm._agenerate(PR_EVALUATE_DOC.format_messages(**inputs))
                outs = [
                    eval(inputs)
                    for i in range(self.query_eval_check_number)
                ]
                try:
                    loop = asyncio.get_event_loop()
                except:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                outs = loop.run_until_complete(asyncio.gather(*outs))
                for out in outs:
                    assert len(out.generations) == 1, f"Query eval llm produced more than 1 evaluations: '{out.generations}'"
                    outputs.append(out.generations[0].text)
                    new_p += out.llm_output["token_usage"]["prompt_tokens"]
                    new_c += out.llm_output["token_usage"]["completion_tokens"]
                assert outputs, "No generations found by query eval llm"
                outputs = [parse_eval_output(o) for o in outputs]

            assert len(outputs) == self.query_eval_check_number, f"query eval model failed to produce {self.query_eval_check_number} outputs"

            self.eval_llm.callbacks[0].prompt_tokens += new_p
            self.eval_llm.callbacks[0].completion_tokens += new_c
            self.eval_llm.callbacks[0].total_tokens += new_p + new_c
            return outputs

        if self.task == "search":
            if self.query_eval_modelname:
                # uses in most places to increase concurrency limit
                multi = {"max_concurrency": 50 if not self.debug else 1}

                # answer 0 or 1 if the document is related
                if not hasattr(self, "eval_llm"):
                    self.eval_llm_params = litellm.get_supported_openai_params(
                        model=self.query_eval_modelname,
                        custom_llm_provider=self.query_eval_modelbackend,
                    )
                    eval_args = {}
                    if "n" in self.eval_llm_params:
                        eval_args["n"] = self.query_eval_check_number
                    else:
                        red(f"Model {self.query_eval_modelname} does not support parameter 'n' so will be called multiple times instead. This might cost more.")
                    if "max_tokens" in self.eval_llm_params:
                        eval_args["max_tokens"] = 2
                    else:
                        red(f"Model {self.query_eval_modelname} does not support parameter 'max_token' so the result might be of less quality.")
                    self.eval_llm = load_llm(
                        modelname=self.query_eval_modelname,
                        backend=self.query_eval_modelbackend,
                        no_llm_cache=self.no_llm_cache,
                        verbose=self.llm_verbosity,
                        temperature=1,
                        api_base=self.llms_api_bases["query_eval_model"],
                        private=self.private,
                        **eval_args,
                    )

                # for some reason I needed to have at least one chain object otherwise rag_chain is a dict
                @chain
                def retrieve_documents(inputs):
                    return {
                            "unfiltered_docs": retriever.get_relevant_documents(inputs["question_for_embedding"]),
                            "question_to_answer": inputs["question_to_answer"],
                    }
                    return inputs

                refilter_documents =  {
                    "filtered_docs": (
                            RunnablePassthrough.assign(
                                evaluations=RunnablePassthrough.assign(
                                    doc=lambda inputs: inputs["unfiltered_docs"],
                                    q=lambda inputs: [inputs["question_to_answer"] for i in range(len(inputs["unfiltered_docs"]))],
                                    )
                                | RunnablePassthrough.assign(
                                    inputs=lambda inputs: [
                                        {"doc":d.page_content, "q":q}
                                        for d, q in zip(inputs["doc"], inputs["q"])])
                                    | itemgetter("inputs")
                                    | RunnableEach(bound=evaluate_doc_chain.with_config(multi)).with_config(multi)
                        )
                        | refilter_docs
                    ),
                    "unfiltered_docs": itemgetter("unfiltered_docs"),
                    "question_to_answer": itemgetter("question_to_answer")
                }
                rag_chain = (
                    retrieve_documents
                    | refilter_documents
                )
                output = rag_chain.invoke(
                    {
                        "question_for_embedding": query_fe,
                        "question_to_answer": query_an,
                    }
                )
                docs = output["filtered_docs"]
            else:

                docs = retriever.get_relevant_documents(query)
                if len(docs) < self.cli_settings["top_k"]:
                    red(f"Only found {len(docs)} relevant documents")


            md_printer("\n\n# Documents")
            anki_cid = []
            to_print = ""
            for id, doc in enumerate(docs):
                to_print += f"## Document #{id + 1}\n"
                content = doc.page_content.strip()
                wrapped = "\n".join(textwrap.wrap(content, width=240))
                to_print += "```\n" + wrapped + "\n ```\n"
                for k, v in doc.metadata.items():
                    to_print += f"* **{k}**: `{v}`\n"
                to_print += "\n"
                if "anki_cid" in doc.metadata:
                    cid_str = str(doc.metadata["anki_cid"]).split(" ")
                    for cid in cid_str:
                        if cid not in anki_cid:
                            anki_cid.append(cid)
            md_printer(to_print)
            if self.query_eval_modelname:
                red(f"Number of documents using embeddings: {len(output['unfiltered_docs'])}")
                red(f"Number of documents after query eval filter: {len(output['filtered_docs'])}")

            if anki_cid:
                open_answ = input(f"\nAnki cards found, open in anki? (yes/no/debug)\n(cids: {anki_cid})\n> ")
                if open_answ == "debug":
                    breakpoint()
                elif open_answ in ["y", "yes"]:
                    whi("Opening anki.")
                    query = f"cid:{','.join(anki_cid)}"
                    ankiconnect(
                            action="guiBrowse",
                            query=query,
                            )
            all_filepaths = []
            for doc in docs:
                if "path" in doc.metadata:
                    path = doc.metadata["path"]
                    try:
                        path = str(Path(path).resolve().absolute())
                    except Exception as err:
                        pass
                    all_filepaths.append(path)
            if all_filepaths:
                md_printer("### All file paths")
                md_printer("* " + "\n* ".join(all_filepaths))

        else:
            if self.condense_question:
                loaded_memory = RunnablePassthrough.assign(
                    chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history"),
                )
                standalone_question = {
                    "question_to_answer": RunnablePassthrough(),
                    "question_for_embedding": {
                        "question_for_embedding": lambda x: x["question_for_embedding"],
                        "chat_history": lambda x: format_chat_history(x["chat_history"]),
                    }
                        | PR_CONDENSE_QUESTION
                        | self.llm
                        | StrOutputParser()
                }

            # uses in most places to increase concurrency limit
            multi = {"max_concurrency": 50 if not self.debug else 1}

            # answer 0 or 1 if the document is related
            if not hasattr(self, "eval_llm"):
                self.eval_llm_params = litellm.get_supported_openai_params(
                    model=self.query_eval_modelname,
                    custom_llm_provider=self.query_eval_modelbackend,
                )
                eval_args = {}
                if "n" in self.eval_llm_params:
                    eval_args["n"] = self.query_eval_check_number
                else:
                    red(f"Model {self.query_eval_modelname} does not support parameter 'n' so will be called multiple times instead. This might cost more.")
                if "max_tokens" in self.eval_llm_params:
                    eval_args["max_tokens"] = 2
                else:
                    red(f"Model {self.query_eval_modelname} does not support parameter 'max_token' so the result might be of less quality.")
                self.eval_llm = load_llm(
                    modelname=self.query_eval_modelname,
                    backend=self.query_eval_modelbackend,
                    no_llm_cache=self.no_llm_cache,
                    verbose=self.llm_verbosity,
                    temperature=1,
                    api_base=self.llms_api_bases["query_eval_model"],
                    private=self.private,
                    **eval_args,
                )

            # the eval doc chain needs its own caching
            if self.no_llm_cache:
                def eval_cache_wrapper(func): return func
            else:
                eval_cache_wrapper = doc_eval_cache.cache

            @chain
            @optional_typecheck
            @eval_cache_wrapper
            def evaluate_doc_chain(
                    inputs: dict,
                    query_nb: int = self.query_eval_check_number,
                    eval_model_name: str = self.query_eval_modelname,
                ) -> List[str]:
                if "n" in self.eval_llm_params or self.query_eval_check_number == 1:
                    out = self.eval_llm._generate(PR_EVALUATE_DOC.format_messages(**inputs))
                    outputs = [gen.text for gen in out.generations]
                    assert outputs, "No generations found by query eval llm"
                    outputs = [parse_eval_output(o) for o in outputs]
                    new_p = out.llm_output["token_usage"]["prompt_tokens"]
                    new_c = out.llm_output["token_usage"]["completion_tokens"]
                else:
                    outputs = []
                    new_p = 0
                    new_c = 0
                    async def eval(inputs):
                        return await self.eval_llm._agenerate(PR_EVALUATE_DOC.format_messages(**inputs))
                    outs = [
                        eval(inputs)
                        for i in range(self.query_eval_check_number)
                    ]
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    outs = loop.run_until_complete(asyncio.gather(*outs))
                    for out in outs:
                        assert len(out.generations) == 1, f"Query eval llm produced more than 1 evaluations: '{out.generations}'"
                        outputs.append(out.generations[0].text)
                        new_p += out.llm_output["token_usage"]["prompt_tokens"]
                        new_c += out.llm_output["token_usage"]["completion_tokens"]
                    assert outputs, "No generations found by query eval llm"
                    outputs = [parse_eval_output(o) for o in outputs]

                assert len(outputs) == self.query_eval_check_number, f"query eval model failed to produce {self.query_eval_check_number} outputs"

                self.eval_llm.callbacks[0].prompt_tokens += new_p
                self.eval_llm.callbacks[0].completion_tokens += new_c
                self.eval_llm.callbacks[0].total_tokens += new_p + new_c
                return outputs

            # for some reason I needed to have at least one chain object otherwise rag_chain is a dict
            @chain
            def retrieve_documents(inputs):
                return {
                        "unfiltered_docs": retriever.get_relevant_documents(inputs["question_for_embedding"]),
                        "question_to_answer": inputs["question_to_answer"],
                }
                return inputs
            refilter_documents =  {
                "filtered_docs": (
                        RunnablePassthrough.assign(
                            evaluations=RunnablePassthrough.assign(
                                doc=lambda inputs: inputs["unfiltered_docs"],
                                q=lambda inputs: [inputs["question_to_answer"] for i in range(len(inputs["unfiltered_docs"]))],
                                )
                            | RunnablePassthrough.assign(
                                inputs=lambda inputs: [
                                    {"doc":d.page_content, "q":q}
                                    for d, q in zip(inputs["doc"], inputs["q"])])
                                | itemgetter("inputs")
                                | RunnableEach(bound=evaluate_doc_chain.with_config(multi)).with_config(multi)
                    )
                    | refilter_docs
                ),
                "unfiltered_docs": itemgetter("unfiltered_docs"),
                "question_to_answer": itemgetter("question_to_answer")
            }
            answer_each_doc_chain = (
                PR_ANSWER_ONE_DOC
                | self.llm.bind(max_tokens=1000)
                | StrOutputParser()
            )
            combine_answers = (
                PR_COMBINE_INTERMEDIATE_ANSWERS
                | self.llm.bind(max_tokens=2000)
                | StrOutputParser()
            )

            answer_all_docs = RunnablePassthrough.assign(
                inputs=lambda inputs: [
                    {"context":d.page_content, "question_to_answer":q}
                    for d, q in zip(
                        inputs["filtered_docs"],
                        [inputs["question_to_answer"]] * len(inputs["filtered_docs"])
                    )
                ]
            ) | {
                    "intermediate_answers": itemgetter("inputs") | RunnableEach(bound=answer_each_doc_chain),
                    "question_to_answer": itemgetter("question_to_answer"),
                    "filtered_docs": itemgetter("filtered_docs"),
                    "unfiltered_docs": itemgetter("unfiltered_docs"),
                }

            final_answer_chain = RunnablePassthrough.assign(
                        final_answer=RunnablePassthrough.assign(
                            question=lambda inputs: inputs["question_to_answer"],
                            # remove answers deemed irrelevant
                            intermediate_answers=lambda inputs: "\n".join(
                                [
                                    inp
                                    for inp in inputs["intermediate_answers"]
                                    if check_intermediate_answer(inp)
                                ]
                            )
                        )
                        | combine_answers,
                )
            if self.condense_question:
                rag_chain = (
                    loaded_memory
                    | standalone_question
                    | retrieve_documents
                    | refilter_documents
                    | answer_all_docs
                )
            else:
                rag_chain = (
                    retrieve_documents
                    | refilter_documents
                    | answer_all_docs
                )

            if self.debug:
                rag_chain.get_graph().print_ascii()

            chain_time = 0
            try:
                start_time = time.time()
                output = rag_chain.invoke(
                    {
                        "question_for_embedding": query_fe,
                        "question_to_answer": query_an,
                    }
                )
                chain_time = time.time() - start_time
            except NoDocumentsRetrieved as err:
                return md_printer(f"## No documents were retrieved with query '{query_fe}'", color="red")
            except NoDocumentsAfterLLMEvalFiltering as err:
                return md_printer(f"## No documents remained after query eval LLM filtering using question '{query_an}'", color="red")

            # group the intermediate answers by batch, then do a batch reduce mapping
            batch_size = 5
            intermediate_answers = output["intermediate_answers"]
            all_intermediate_answers = [intermediate_answers]
            while len(intermediate_answers) > batch_size:
                batches = [[]]
                for ia in intermediate_answers:
                    if not check_intermediate_answer(ia):
                        continue
                    if len(batches[-1]) >= batch_size:
                        batches.append([])
                    if len(batches[-1]) < batch_size:
                        batches[-1].append(ia)
                batch_args = [
                    {"question_to_answer": query_an, "intermediate_answers": b}
                    for b in batches]
                intermediate_answers = [a["final_answer"] for a in final_answer_chain.batch(batch_args)]
            all_intermediate_answers.append(intermediate_answers)
            final_answer = final_answer_chain.invoke({"question_to_answer": query_an, "intermediate_answers": intermediate_answers})["final_answer"]
            output["final_answer"] = final_answer
            output["all_intermediate_answeers"] = all_intermediate_answers
            # output["intermediate_answers"] = intermediate_answers  # better not to overwrite that

            output["relevant_filtered_docs"] = []
            output["relevant_intermediate_answers"] = []
            for ia, a in enumerate(output["intermediate_answers"]):
                if check_intermediate_answer(a):
                    output["relevant_filtered_docs"].append(output["filtered_docs"][ia])
                    output["relevant_intermediate_answers"].append(a)

            md_printer("\n\n# Intermediate answers for each document:")
            counter = 0
            to_print = ""
            for ia, doc in zip(output["relevant_intermediate_answers"], output["relevant_filtered_docs"]):
                counter += 1
                to_print += f"## Document #{counter}\n"
                content = doc.page_content.strip()
                wrapped = "\n".join(textwrap.wrap(content, width=240))
                to_print += "```\n" + wrapped + "\n ```\n"
                for k, v in doc.metadata.items():
                    to_print += f"* **{k}**: `{v}`\n"
                to_print += indent("### Intermediate answer:\n" + ia, "> ")
                to_print += "\n"
            md_printer(to_print)

            md_printer(indent(f"# Answer:\n{output['final_answer']}\n", "> "))

            red(f"Number of documents using embeddings: {len(output['unfiltered_docs'])}")
            red(f"Number of documents after query eval filter: {len(output['filtered_docs'])}")
            red(f"Number of documents found relevant by eval llm: {len(output['relevant_filtered_docs'])}")
            if chain_time:
                red(f"Time took by the chain: {chain_time:.2f}s")

            if self.import_mode:
                return output

            assert len(self.llm.callbacks) == 1, "Unexpected number of callbacks for llm"
            llmcallback = self.llm.callbacks[0]
            total_cost = self.llm_price[0] * llmcallback.prompt_tokens + self.llm_price[1] * llmcallback.completion_tokens
            yel(f"Tokens used by strong model: '{llmcallback.total_tokens}' (${total_cost:.5f})")

            assert len(self.eval_llm.callbacks) == 1, "Unexpected number of callbacks for eval_llm"
            evalllmcallback = self.eval_llm.callbacks[0]
            wtotal_cost = self.query_evalllm_price[0] * evalllmcallback.prompt_tokens + self.query_evalllm_price[1] * evalllmcallback.completion_tokens
            yel(f"Tokens used by query_eval model: '{evalllmcallback.total_tokens}' (${wtotal_cost:.5f})")

            red(f"Total cost: ${total_cost + wtotal_cost:.5f}")

def cli_launcher():
    import fire
    red(pyfiglet.figlet_format("DocToolsLLM"))
    log.info("Starting DocToolsLLM")
    instance = fire.Fire(DocToolsLLM_class)

if __name__ == "__main__":
    cli_launcher()
