"""
Main class.
"""

from __future__ import annotations
import copy
import json
import os
import re
import sys
import time
import traceback
from operator import itemgetter
from pathlib import Path

from beartype.door import is_bearable
from beartype.typing import Any, Callable, Dict, List, Literal, Optional, Union
from langchain.docstore.document import Document
from langchain.globals import set_debug, set_llm_cache, set_verbose
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.runnables.base import RunnableEach
from tqdm import tqdm
from loguru import logger as logger

# import this first because it sets the logging level
from wdoc.utils.logger import (
    log_dir,
    md_printer,
    set_help_md_as_docstring,
    set_parse_doc_help_md_as_docstring,
    debug_exceptions,
)

from wdoc.utils.batch_file_loader import batch_load_doc
from wdoc.utils.env import env, is_out_piped
from wdoc.utils.errors import (
    NoDocumentsAfterLLMEvalFiltering,
    NoDocumentsRetrieved,
    ShouldIncreaseTopKAfterLLMEvalFiltering,
)

from wdoc.utils.llm import TESTING_LLM, load_llm

from wdoc.utils.misc import (  # debug_chain,
    cache_dir,
    ModelName,
    create_langfuse_callback,
    disable_internet,
    extra_args_types,
    get_model_price,
    get_supported_model_params,
    get_tkn_length,
    model_name_matcher,
    query_eval_cache,
    set_func_signature,
    thinking_answer_parser,
    tasks_list,
)

logger.info("Starting wdoc")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@set_help_md_as_docstring
class wdoc:
    """
    This docstring is dynamically updated with the content of wdoc/docs/help.md
    """

    VERSION: str = "4.0.2"
    allowed_extra_args = extra_args_types
    __import_mode__: bool = True

    @set_func_signature
    def __init__(
        self,
        task: Literal["query", "search", "summarize", "summarize_then_query"],
        filetype: str = "auto",
        model: str = env.WDOC_DEFAULT_MODEL,
        model_kwargs: Optional[dict] = None,
        query_eval_model: Optional[str] = env.WDOC_DEFAULT_QUERY_EVAL_MODEL,
        query_eval_model_kwargs: Optional[dict] = None,
        embed_model: str = env.WDOC_DEFAULT_EMBED_MODEL,
        embed_model_kwargs: Optional[dict] = None,
        save_embeds_as: Union[str, Path] = "{user_cache}/latest_docs_and_embeddings",
        load_embeds_from: Optional[Union[str, Path]] = None,
        top_k: Union[str, int] = "auto_200_500",
        query: Optional[str] = None,
        query_retrievers: str = "basic_multiquery",
        query_eval_check_number: int = 3,
        query_relevancy: Union[float, int] = -0.5,
        summary_n_recursion: int = 0,
        summary_language: str = "the same language as the document",  # <- the LLM will understand
        llm_verbosity: Union[bool, int] = False,
        debug: Union[bool, int] = False,
        verbose: Union[bool, int] = False,
        dollar_limit: int = 5,
        notification_callback: Optional[Callable] = None,
        disable_llm_cache: Union[bool, int] = False,
        file_loader_parallel_backend: Literal[
            "loky", "threading", "multiprocessing"
        ] = "loky",
        file_loader_n_jobs: int = -1,
        private: Union[bool, int] = False,
        llms_api_bases: Optional[Union[dict, str]] = None,
        out_file: Optional[Union[str, Path]] = None,
        oneoff: bool = False,
        silent: bool = False,
        version: bool = False,
        **cli_kwargs,
    ) -> None:
        """
        This docstring is dynamically updated with the content of wdoc/docs/help.md
        """
        import litellm

        if version:
            print(self.VERSION)
            return
        if notification_callback is not None:

            def ntfy(text: str) -> str:
                out = notification_callback(text)
                assert (
                    out == text
                ), "The notification callback must return the same string"
                return out

            ntfy("Starting wdoc")
        else:

            def ntfy(text: str) -> str:
                return text

        self.ntfy = ntfy

        if debug or env.WDOC_DEBUGGER or env.WDOC_DEBUG:
            debug_exceptions(instance=self)

        elif notification_callback:

            def print_exception(exc_type, exc_value, exc_traceback):
                if not issubclass(exc_type, KeyboardInterrupt):
                    message = "An error has occured:\n"
                    message += "\n".join(
                        [line for line in traceback.format_tb(exc_traceback)]
                    )
                    message += "\n" + str(exc_type) + " : " + str(exc_value)
                    self.ntfy(message)
                    sys.exit(1)

            sys.excepthook = print_exception
            import faulthandler

            faulthandler.enable()

        from loguru import logger  # for some reason I have to reimport

        # loguru here otherwise the next line fails!
        import pyfiglet

        logger.warning(pyfiglet.figlet_format("wdoc"))

        # make sure the extra args are valid
        for k in cli_kwargs:
            if k not in self.allowed_extra_args:
                logger.exception(
                    f"Found unexpected keyword argument: '{k}'\nThe allowed arguments are {','.join(self.allowed_extra_args)}"
                )
                raise Exception(
                    f"Found unexpected keyword argument: '{k}'\nThe allowed arguments are {','.join(self.allowed_extra_args)}"
                )

            # type checking of extra args
            if env.WDOC_TYPECHECKING in ["crash", "warn"]:
                val = cli_kwargs[k]
                # curr_type = type(val)
                expected_type = self.allowed_extra_args[k]
                if expected_type is str:
                    assert val.strip(), f"Empty string found for cli_kwargs: '{k}'"
                if isinstance(val, list):
                    assert val, f"Empty list found for cli_kwargs: '{k}'"
                if not is_bearable(val, expected_type):
                    raise Exception(
                        f"Cli_kwargs '{k}' is of type '{type(val)}' instead of '{expected_type}'"
                    )

        if (
            model == TESTING_LLM
            or model == "testing"
            or model.lower().startswith("testing")
        ):
            logger.warning(f"Detected 'testing' model in {model}")
            model = TESTING_LLM
            if query_eval_model != TESTING_LLM:
                logger.warning(f"Setting the query_eval_model to {TESTING_LLM} too")
                query_eval_model = TESTING_LLM
            if "multiquery" in query_retrievers:
                logger.warning(
                    f"Removing 'multiquery' from the query_retrievers because using {TESTING_LLM}"
                )
                query_retrievers = query_retrievers.replace("multiquery", "")
            logger.warning(
                f"Setting the query_relevancy to -1.0 because using {TESTING_LLM}"
            )
            query_relevancy = -1.0
        query_retrievers = query_retrievers.replace("_", " ").strip().replace(" ", "_")

        # checking argument validity
        assert (
            "loaded_docs" not in cli_kwargs
        ), "'loaded_docs' cannot be an argument as it is used internally"
        assert (
            "loaded_embeddings" not in cli_kwargs
        ), "'loaded_embeddings' cannot be an argument as it is used internally"
        task = task.replace("summary", "summarize")
        assert task in tasks_list, f"invalid task value: {task}"
        if task in ["summarize", "summarize_then_query"]:
            assert not load_embeds_from, "can't use load_embeds_from if task is summary"
        if filetype == "ddg":
            assert task == "query", "Only 'query' task is supported for 'ddg' filetype"
        if task in ["query", "search", "summarize_then_query"]:
            assert (
                query_eval_model is not None
            ), "query_eval_model can't be None if doing RAG"
        else:
            query_eval_model = None
        if filetype == "auto":
            assert (
                "path" in cli_kwargs and cli_kwargs["path"]
            ), "If filetype is 'auto', a --path must be given"
        elif filetype == "string":
            cli_kwargs["path"] = "empty placeholder"
        assert (
            "/" in model
        ), "model must be in litellm format: provider/model. For example 'openai/gpt-4o'"
        if model != TESTING_LLM and model.split("/", 1)[0] not in list(
            litellm.models_by_provider.keys()
        ):
            raise Exception(
                f"For model '{model}': backend not found in "
                "litellm nor 'testing'.\nList of litellm providers/backend:\n"
                f"{litellm.models_by_provider.keys()}"
            )
        assert (
            query_eval_check_number > 0
        ), "query_eval_check_number value must be greater than 0"

        # parse the model kwargs
        if model_kwargs is None:
            model_kwargs = {}
        if isinstance(model_kwargs, str):
            try:
                model_kwargs = json.loads(model_kwargs)
            except Exception as err:
                raise Exception(
                    f"Failed to parse model_kwargs: '{model_kwargs}'"
                ) from err
        assert isinstance(model_kwargs, dict), f"Not a dict but {type(model_kwargs)}"
        if "tags" not in model_kwargs:
            model_kwargs["tags"] = ["strong_model"]
        else:
            assert isinstance(
                model_kwargs["tags"], list
            ), f"Model kwargs 'tags' value must be a list. Got '{model_kwargs['tags']}'"
            model_kwargs["tags"].append("strong_model")
        self.model_kwargs = model_kwargs
        if query_eval_model_kwargs is None:
            query_eval_model_kwargs = {}
        if isinstance(query_eval_model_kwargs, str):
            try:
                query_eval_model_kwargs = json.loads(query_eval_model_kwargs)
            except Exception as err:
                raise Exception(
                    f"Failed to parse query_eval_model_kwargs: '{query_eval_model_kwargs}'"
                ) from err
        assert isinstance(
            query_eval_model_kwargs, dict
        ), f"Not a dict but {type(query_eval_model_kwargs)}"
        assert (
            "n" not in query_eval_model_kwargs
        ), "Trying to set the 'n' argument using query_eval_model_kwargs, you should instead use the query_eval_check_number argument"
        self.query_eval_model_kwargs = query_eval_model_kwargs
        if "tags" not in query_eval_model_kwargs:
            query_eval_model_kwargs["tags"] = ["eval_model"]
        else:
            assert isinstance(
                query_eval_model_kwargs["tags"], list
            ), f"Model kwargs 'tags' value must be a list. Got '{query_eval_model_kwargs['tags']}'"
            query_eval_model_kwargs["tags"].append("eval_model")
        if embed_model_kwargs is None:
            embed_model_kwargs = {}
        if isinstance(embed_model_kwargs, str):
            try:
                embed_model_kwargs = json.loads(embed_model_kwargs)
            except Exception as err:
                raise Exception(
                    f"Failed to parse embed_model_kwargs: '{embed_model_kwargs}'"
                ) from err
        assert isinstance(
            embed_model_kwargs, dict
        ), f"Not a dict but {type(embed_model_kwargs)}"
        self.embed_model_kwargs = embed_model_kwargs

        if llms_api_bases is None:
            llms_api_bases = {}
        elif isinstance(llms_api_bases, str):
            try:
                llms_api_bases = json.loads(llms_api_bases)
            except Exception as err:
                raise Exception(f"Error when parsing llms_api_bases as a dict: {err}")
        assert isinstance(
            llms_api_bases, dict
        ), "llms_api_bases must be a dict or be a string that can be parsed as a dict"
        for k in llms_api_bases:
            assert k in [
                "model",
                "query_eval_model",
                "embeddings",
            ], f"Invalid k of llms_api_bases not in 'model', 'query_eval_model', 'embeddings': {k}"
        for k in ["model", "query_eval_model", "embeddings"]:
            if k not in llms_api_bases:
                llms_api_bases[k] = None
        if (
            llms_api_bases["model"] == llms_api_bases["query_eval_model"]
            and llms_api_bases["model"] == llms_api_bases["embeddings"]
            and llms_api_bases["model"]
        ):
            logger.warning(
                "Setting litellm wide api_base because it's the same for model, query_eval_model and embeddings"
            )
            litellm.api_base = llms_api_bases["model"]
        assert isinstance(
            private, bool
        ), "private arg should be a boolean, not {private}"
        assert private == env.WDOC_PRIVATE_MODE
        if private:
            assert llms_api_bases[
                "model"
            ], "private is set but llms_api_bases['model'] is not set"
            assert llms_api_bases[
                "query_eval_model"
            ], "private is set but llms_api_bases['query_eval_model'] is not set"
            os.environ["WDOC_PRIVATE_MODE"] = "true"
            for k in dict(os.environ):
                if (
                    k.endswith("_API_KEY")
                    or k.endswith("_API_KEYS")
                    and (not k.startswith("WDOC_"))
                ):
                    logger.warning(
                        f"private mode enabled: overwriting '{k}' from environment variables just in case"
                    )
                    os.environ[k] = "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"

            if any("LANGFUSE_" in k for k in os.environ.keys()):
                logger.warning("Disabled langfuse because using private mode.")
            os.environ["LANGFUSE_PUBLIC_KEY"] = "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
            os.environ["LANGFUSE_SECRET_KEY"] = "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
            os.environ["LANGFUSE_HOST"] = "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
            os.environ["WDOC_LANGFUSE_PUBLIC_KEY"] = (
                "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
            )
            os.environ["WDOC_LANGFUSE_SECRET_KEY"] = (
                "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
            )
            os.environ["WDOC_LANGFUSE_HOST"] = "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
            if litellm.success_callback or litellm.failure_callback:
                logger.warning("Disabled litellm callbacks because using private mode.")
            litellm.success_callback = []
            litellm.failure_callback = []

            # to be extra safe, let's try to block any remote connection
            disable_internet(
                allowed=llms_api_bases,
            )

        else:
            os.environ["WDOC_PRIVATE_MODE"] = "false"
            create_langfuse_callback(f"wdoc_{wdoc.VERSION}")

        if (model != TESTING_LLM) and (not llms_api_bases["model"]):
            model = model_name_matcher(model)
        if (query_eval_model is not None) and (not llms_api_bases["query_eval_model"]):
            if model == TESTING_LLM:
                assert query_eval_model == TESTING_LLM
            else:
                query_eval_model = model_name_matcher(query_eval_model)

        if isinstance(query, bool) and query is True:
            # otherwise specifying --query and forgetting to add text fails
            query = None

        elif isinstance(query, str):
            query = query.strip() or None
        assert file_loader_parallel_backend in [
            "loky",
            "threading",
            "multiprocessing",
        ], "Invalid value for file_loader_parallel_backend"
        assert isinstance(file_loader_n_jobs, int), "file_loader_n_jobs mus be an int"
        if "{user_cache}" in save_embeds_as:
            save_embeds_as = save_embeds_as.replace("{user_cache}", str(cache_dir))
        if query_relevancy is None:
            query_relevancy = 0.0
        query_relevancy = float(query_relevancy)

        # parsing top_k value
        if isinstance(top_k, str):
            try:
                starting_top_k, max_top_k = top_k.split("_")[1:]
                starting_top_k = int(starting_top_k)
                max_top_k = int(max_top_k)
                assert max_top_k > starting_top_k, "M<=N"
                assert starting_top_k > 0, "N<=0"

            except Exception as err:
                raise Exception(
                    "Failed to parse string top_k value. If top_k "
                    "is a string, the expected format is 'auto_N_M' with N "
                    f"and M ints and M>N. Received: {top_k}"
                ) from err
            top_k = starting_top_k
            self.max_top_k = max_top_k
        else:
            self.max_top_k = None

        # storing as attributes
        self.model = ModelName(model)
        self.model_supported_params = get_supported_model_params(self.model)
        if query_eval_model is not None:
            self.query_eval_model = ModelName(query_eval_model)
        else:
            self.query_eval_model = None
        self.task = task
        self.filetype = filetype
        self.embed_model = ModelName(embed_model)
        self.embed_model_kwargs = embed_model_kwargs
        self.save_embeds_as = save_embeds_as
        self.load_embeds_from = load_embeds_from
        self.top_k = top_k
        self.query_retrievers = query_retrievers
        self.query_eval_check_number = int(query_eval_check_number)
        self.query_relevancy = query_relevancy
        self.debug = debug
        self.verbose = verbose
        self.cli_kwargs = cli_kwargs
        self.llm_verbosity = llm_verbosity
        self.out_file = out_file
        self.summary_n_recursion = summary_n_recursion
        self.summary_language = summary_language
        self.dollar_limit = dollar_limit
        self.private = bool(private)
        self.disable_llm_cache = bool(disable_llm_cache)
        self.file_loader_parallel_backend = file_loader_parallel_backend
        self.file_loader_n_jobs = file_loader_n_jobs
        self.llms_api_bases = llms_api_bases
        self.oneoff = oneoff
        self.latest_cost = 0  # used to keep track of the costs overall
        if debug:
            os.environ["WDOC_DEBUG"] = "true"
            os.environ["WDOC_VERBOSE"] = "true"
        elif verbose:
            os.environ["WDOC_VERBOSE"] = "true"

        if disable_llm_cache:
            self.llm_cache = False
        else:
            from wdoc.utils.customs.fix_llm_caching import SQLiteCacheFixed

            if not private:
                self.llm_cache = SQLiteCacheFixed(
                    database_path=(cache_dir / "langchain_db").resolve().absolute(),
                    verbose=env.WDOC_VERBOSE,
                )
            else:
                self.llm_cache = SQLiteCacheFixed(
                    database_path=(cache_dir / "private_langchain_db")
                    .resolve()
                    .absolute(),
                    verbose=env.WDOC_VERBOSE,
                )
            set_llm_cache(self.llm_cache)

        if llms_api_bases["model"]:
            logger.warning(
                f"Disabling price computation for model because api_base for 'model' was modified to {llms_api_bases['model']}"
            )
            self.llm_price = {"prompt": 0, "completion": 0, "internal_reasoning": 0}
        else:
            self.llm_price = get_model_price(self.model)
        logger.debug(f"Detected price of '{self.model.original}': {self.llm_price}")
        assert "prompt" in self.llm_price
        assert "completion" in self.llm_price
        assert "internal_reasoning" in self.llm_price

        if self.query_eval_model is not None:
            if llms_api_bases["query_eval_model"]:
                logger.warning(
                    "Disabling price computation for query_eval_model because api_base was modified"
                )
                self.query_evalllm_price = {
                    "prompt": 0,
                    "completion": 0,
                    "internal_reasoning": 0,
                }
            else:
                self.query_evalllm_price = get_model_price(self.query_eval_model)
            logger.debug(
                f"Detected price of '{self.query_eval_model.original}': {self.query_evalllm_price}"
            )

            assert "prompt" in self.query_evalllm_price
            assert "completion" in self.query_evalllm_price
            assert "internal_reasoning" in self.query_evalllm_price

        if env.WDOC_VERBOSE:
            set_verbose(True)
            # litellm is way too verbose
            # os.environ["LITELLM_LOG"] = "DEBUG"
            # litellm._turn_on_debug()

            llm_verbosity = True
            logger.info(f"Cache location: {cache_dir.absolute()}")
            logger.info(f"Log location: {log_dir.absolute()}")
        else:
            set_verbose(False)
            set_debug(False)
            litellm.set_verbose = False
            # fix from https://github.com/BerriAI/litellm/issues/2256
            import logging

            for logger_name in [
                "LiteLLM Proxy",
                "LiteLLM Router",
                "LiteLLM",
                "litellm",
                "httpx",
            ]:
                logger = logging.getLogger(logger_name)
                # logger.setLevel(logging.CRITICAL + 1)
                logger.setLevel(logging.WARNING)
            for logger_name in ["bs4"]:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
        if debug:
            assert env.WDOC_VERBOSE
            assert env.WDOC_DEBUG
            # os.environ["LANGCHAIN_TRACING_V2"] = "true"
            set_debug(True)

        # don't crash if extra arguments are used for a model
        # litellm.drop_params = True  # drops parameters that are not used by some models

        # compile include / exclude regex
        if "include" in self.cli_kwargs:
            for i, inc in enumerate(self.cli_kwargs["include"]):
                if inc == inc.lower():
                    self.cli_kwargs["include"][i] = re.compile(inc, flags=re.IGNORECASE)
                else:
                    self.cli_kwargs["include"][i] = re.compile(inc)
        if "exclude" in self.cli_kwargs:
            for i, exc in enumerate(self.cli_kwargs["exclude"]):
                if exc == exc.lower():
                    self.cli_kwargs["exclude"][i] = re.compile(exc, flags=re.IGNORECASE)
                else:
                    self.cli_kwargs["exclude"][i] = re.compile(exc)

        # loading llm
        self.llm = load_llm(
            modelname=self.model,
            llm_cache=self.llm_cache,
            temperature=0,
            llm_verbosity=self.llm_verbosity,
            api_base=self.llms_api_bases["model"],
            private=self.private,
            **self.model_kwargs,
        )
        # if "anthropic" in self.model.lower() or "anthropic" in self.backend.lower():
        #     prompts.enable_prompt_caching("answer")
        #     prompts.enable_prompt_caching("combine")
        #     prompts.enable_prompt_caching("multiquery")

        # loading documents
        if not load_embeds_from:
            # remove args that are indented only for the instanciation and
            # not as loading argument
            filtered_cli_kwargs = self.cli_kwargs.copy()
            for k in [
                "embed_instruct",
                "filter_content",
                "filter_metadata",
            ]:
                if k in filtered_cli_kwargs:
                    del filtered_cli_kwargs[k]

            self.loaded_docs = batch_load_doc(
                llm_name=self.model,
                filetype=self.filetype,
                task=self.task,
                backend=self.file_loader_parallel_backend,
                n_jobs=self.file_loader_n_jobs if not env.WDOC_DEBUG else 1,
                **filtered_cli_kwargs,
            )
        else:
            self.loaded_docs = None  # will be loaded when embeddings are loaded

        # flag to know if we already filtered or not
        self._is_vectorstore_filtered = False

        if self.__import_mode__:
            logger.debug(
                "Ready to query or summarize, call your_instance.query_task(your_question)"
            )
            return

        if self.task in ["summarize", "summarize_then_query"]:
            self.summary_task()

            if self.task == "summary_then_query":
                logger.info("Done summarizing. Switching to query mode.")
            else:
                logger.info("Done summarizing.")
                return

        else:
            assert self.task in [
                "query",
                "search",
                "summary_then_query",
            ], f"Invalid task: {self.task}"
            if self.oneoff:
                self._query_or_search_task(query=query)
            elif is_out_piped:
                self._query_or_search_task(query=query)
                logger.debug(
                    "Exited query_task because we don't loop the queries when the output is a shell pipe"
                )
            else:
                # import at last minute to reduce load time
                from wdoc.utils.interact import ask_user

                if not query:
                    query, self.interaction_settings = ask_user(
                        self.interaction_settings
                    )
                while True:
                    self._query_or_search_task(query=query)
                    query, self.interaction_settings = ask_user(
                        self.interaction_settings
                    )

    def summary_task(self) -> "wdoc.utils.tasks.summarize.wdocSummary":
        from wdoc.utils.tasks.summarize import summarize_documents

        docs_tkn_cost = {}
        for doc in self.loaded_docs:
            meta = doc.metadata["path"]
            if meta not in docs_tkn_cost:
                docs_tkn_cost[meta] = get_tkn_length(doc.page_content)
            else:
                docs_tkn_cost[meta] += get_tkn_length(doc.page_content)

        full_tkn = sum(list(docs_tkn_cost.values()))
        logger.warning("Token price of each document:")
        for k, v in docs_tkn_cost.items():
            pr = v * (self.llm_price["prompt"] * 4 + self.llm_price["completion"]) / 5
            logger.warning(f"- {v:>6}: {k:>10} - ${pr:04f}")

        logger.warning(
            f"Total number of tokens in documents to summarize: '{full_tkn}'"
        )
        # use an heuristic to estimate the price to summarize
        compr_ratio = 0.28
        prompt_tkn = 1000
        estimate_dol = (prompt_tkn + full_tkn) * self.llm_price[
            "prompt"
        ] + full_tkn * compr_ratio * self.llm_price["completion"]
        if self.summary_n_recursion:
            for i in range(self.summary_n_recursion):
                estimate_dol += (prompt_tkn + full_tkn) * (
                    compr_ratio**i
                ) * self.llm_price["prompt"] + full_tkn * (
                    compr_ratio ** (i + 1)
                ) * self.llm_price[
                    "completion"
                ]
        logger.info(
            self.ntfy(
                f"Summary cost estimation: ${estimate_dol:.4f} for {full_tkn} tokens."
            )
        )
        if estimate_dol > self.dollar_limit:
            if self.llms_api_bases["model"]:
                raise Exception(
                    logger.warning(
                        self.ntfy(
                            f"Cost estimate ${estimate_dol:.5f} > ${self.dollar_limit} which is absurdly high. Has something gone wrong? Quitting."
                        )
                    )
                )
            else:
                logger.warning(
                    "Cost estimate > limit but the api_base was modified so not crashing."
                )

        results = summarize_documents(
            path=self.cli_kwargs["path"],
            relevant_docs=self.loaded_docs,
            summary_language=self.summary_language,
            model=self.model,
            llm=self.llm,
            llm_verbosity=self.llm_verbosity,
            summary_n_recursion=self.summary_n_recursion,
            llm_price=self.llm_price,
            in_import_mode=self.__import_mode__,
            out_file=self.out_file,
            wdoc_version=self.VERSION,
        )

        logger.info(
            self.ntfy(
                f"Total cost of those summaries: {results['doc_total_tokens_str']} tokens for ${results['doc_total_cost']:.5f} (estimate was ${estimate_dol:.5f})"
            )
        )
        logger.info(
            self.ntfy(
                f"Total time saved by those summaries: {results['doc_reading_length']:.1f} minutes"
            )
        )

        llmcallback = self.llm.callbacks[0]
        total_cost = (
            self.llm_price["prompt"] * llmcallback.prompt_tokens
            + self.llm_price["completion"] * llmcallback.completion_tokens
        )
        if llmcallback.total_tokens != results["doc_total_tokens_sum"]:
            logger.warning(
                f"Cost discrepancy? Tokens used according to the callback: '{llmcallback.total_tokens}' vs in the result: '{results['doc_total_tokens_sum']}' (${total_cost:.5f})"
            )
        self.summary_results = results
        self.latest_cost = total_cost
        return results

    def query_task(self, query) -> Dict[str, Any]:
        self.task = "query"
        return self._query_or_search_task(query=query)

    def search_task(self, query) -> Dict[str, Any]:
        self.task = "search"
        return self._query_or_search_task(query=query)

    def _query_or_search_task(self, query: str) -> dict:
        from wdoc.utils.retrievers import create_retrievers
        from wdoc.utils.tasks.shared_query_search import (
            split_query_parts,
            create_evaluate_doc_chain,
        )
        from wdoc.utils.prompts import prompts

        # load embeddings for querying
        if not hasattr(self, "embedding_engine"):
            from wdoc.utils.embeddings import load_embeddings_engine

            self.embedding_engine = load_embeddings_engine(
                modelname=self.embed_model,
                cli_kwargs=self.cli_kwargs,
                api_base=self.llms_api_bases["embeddings"],
                embed_kwargs=self.embed_model_kwargs,
                private=self.private,
                do_test=env.WDOC_EMBED_TESTING,
            )
        if not hasattr(self, "loaded_embeddings"):
            from wdoc.utils.embeddings import create_embeddings

            self.loaded_embeddings = create_embeddings(
                modelname=self.embed_model,
                cached_embeddings=self.embedding_engine,
                load_embeds_from=self.load_embeds_from,
                save_embeds_as=self.save_embeds_as,
                loaded_docs=self.loaded_docs,
                dollar_limit=self.dollar_limit,
                private=self.private,
            )

        # set default ask_user argument
        self.interaction_settings = {
            "top_k": self.top_k,
            "multiline": False,
            "retriever": self.query_retrievers,
            "task": self.task,
            "relevancy": self.query_relevancy,
        }

        # parse filters as callable for faiss filtering
        if not self._is_vectorstore_filtered:
            if (
                "filter_metadata" in self.cli_kwargs
                or "filter_content" in self.cli_kwargs
            ):
                from wdoc.utils.filters import filter_vectorstore

                self.loaded_embeddings = filter_vectorstore(
                    loaded_embeddings=self.loaded_embeddings,
                    cli_kwargs=self.cli_kwargs,
                )
                self._is_vectorstore_filtered = True

        assert query.strip(), "Cannot accept empty query"
        assert all(
            retriev in ["basic", "multiquery", "knn", "svm", "parent"]
            for retriev in self.interaction_settings["retriever"].split("_")
        ), f"Invalid retriever value: {self.interaction_settings['retriever']}"

        retriever = create_retrievers(
            query_retrievers=self.interaction_settings["retriever"],
            loaded_embeddings=self.loaded_embeddings,
            embedding_engine=self.embedding_engine,
            llm=self.llm,
            top_k=self.interaction_settings["top_k"],
            relevancy=self.interaction_settings["relevancy"],
            task=self.interaction_settings["task"],
            loaded_docs=self.loaded_docs,
        )

        query_fe, query_an = split_query_parts(query)

        assert query.strip(), "Received empty 'query'"
        logger.debug(f"Query for the embeddings: {query_fe}")
        logger.debug(f"Question to answer: {query_an}")

        # answer 0 or 1 if the document is related
        if not hasattr(self, "eval_llm"):
            self.eval_llm_params = get_supported_model_params(self.query_eval_model)
            eval_args = copy.deepcopy(self.query_eval_model_kwargs)
            if "n" in self.eval_llm_params:
                eval_args["n"] = self.query_eval_check_number
            if self.query_eval_check_number > 1:
                logger.warning(
                    f"Model {self.query_eval_model.original} does not support parameter 'n' so will be called multiple times instead. This might cost more."
                )
            self.eval_llm = load_llm(
                modelname=self.query_eval_model,
                llm_cache=False,  # disables caching because another caching is used on top
                llm_verbosity=self.llm_verbosity,
                temperature=0 if self.query_eval_check_number == 1 else 1,
                api_base=self.llms_api_bases["query_eval_model"],
                private=self.private,
                **eval_args,
            )
            # if "anthropic" in self.query_eval_model.original.lower() or "anthropic" in self.query_eval_model.backend.lower():
            #     prompts.enable_prompt_caching("evaluate")

        # the eval doc chain needs its own caching
        if self.llm_cache:
            eval_cache_wrapper = query_eval_cache.cache
        else:

            def eval_cache_wrapper(func: Callable) -> Callable:
                return func

        if " object at " in self.llm._get_llm_string():
            logger.warning(
                "Found llm._get_llm_string() value that potentially "
                f"invalidates the cache: '{self.llm._get_llm_string()}'\n"
                f"Related github issue: 'https://github.com/langchain-ai/langchain/issues/23257'"
            )
        if " object at " in self.eval_llm._get_llm_string():
            logger.warning(
                "Found eval_llm._get_llm_string() value that potentially "
                f"invalidates the cache: '{self.eval_llm._get_llm_string()}'\n"
                f"Related github issue: 'https://github.com/langchain-ai/langchain/issues/23257'"
            )

        evaluate_doc_chain = create_evaluate_doc_chain(
            eval_llm=self.eval_llm,
            eval_llm_params=self.eval_llm_params,
            query_eval_check_number=self.query_eval_check_number,
            eval_cache_wrapper=eval_cache_wrapper,
            prompts=prompts,
        )

        # uses in most places to increase concurrency limit
        multi = {
            "max_concurrency": env.WDOC_LLM_MAX_CONCURRENCY if not self.debug else 1
        }

        if self.task == "search":
            return self._actual_search_task(
                retriever=retriever,
                query=query,
                query_fe=query_fe,
                query_an=query_an,
                evaluate_doc_chain=evaluate_doc_chain,
                multi=multi,
            )
        else:
            return self._actual_query_task(
                retriever=retriever,
                query_fe=query_fe,
                query_an=query_an,
                evaluate_doc_chain=evaluate_doc_chain,
                multi=multi,
            )

    def _actual_query_task(
        self,
        retriever: Any,
        query_fe: str,
        query_an: str,
        evaluate_doc_chain: Any,
        multi: Dict[str, Any],
    ) -> Dict[str, Any]:
        from wdoc.utils.tasks.query import (
            autoincrease_top_k,
            check_intermediate_answer,
            collate_relevant_intermediate_answers,
            pbar_chain,
            pbar_closer,
            refilter_docs,
            retrieve_documents_for_query,
            semantic_batching,
            sieve_documents,
            source_replace,
        )
        from langchain_core.output_parsers.string import StrOutputParser
        from textwrap import indent
        from wdoc.utils.prompts import prompts

        # for some reason I needed to have at least one chain object otherwise rag_chain is a dict
        retrieve_documents = retrieve_documents_for_query(retriever)

        def create_autoincrease_top_k_chain(inputs):
            filtered_docs = inputs
            return autoincrease_top_k(filtered_docs, self.top_k, self.max_top_k)

        create_autoincrease_top_k_chain = chain(create_autoincrease_top_k_chain)

        meta_refilter_docs = {
            "filtered_docs": (
                RunnablePassthrough.assign(
                    evaluations=RunnablePassthrough.assign(
                        doc=lambda inputs: inputs["unfiltered_docs"],
                        q=lambda inputs: [
                            inputs["question_to_answer"]
                            for i in range(len(inputs["unfiltered_docs"]))
                        ],
                    )
                    | RunnablePassthrough.assign(
                        inputs=lambda inputs: [
                            {"doc": d.page_content, "q": q}
                            for d, q in zip(inputs["doc"], inputs["q"])
                        ]
                    )
                    | itemgetter("inputs")
                    | RunnableEach(
                        bound=evaluate_doc_chain.with_config(multi)
                    ).with_config(multi)
                )
                | refilter_docs
                | create_autoincrease_top_k_chain
            ),
            "unfiltered_docs": itemgetter("unfiltered_docs"),
            "question_to_answer": itemgetter("question_to_answer"),
        }
        answer_each_doc_chain = (
            prompts.answer
            | self.llm.bind(max_tokens=env.WDOC_INTERMEDIATE_ANSWER_MAX_TOKENS)
            | StrOutputParser()
        )

        answer_all_docs = RunnablePassthrough.assign(
            inputs=lambda inputs: [
                {"context": d.page_content, "question_to_answer": q}
                for d, q in zip(
                    inputs["filtered_docs"],
                    [inputs["question_to_answer"]] * len(inputs["filtered_docs"]),
                )
            ]
        ) | {
            "intermediate_answers": itemgetter("inputs")
            | RunnableEach(bound=answer_each_doc_chain),
            "question_to_answer": itemgetter("question_to_answer"),
            "filtered_docs": itemgetter("filtered_docs"),
            "unfiltered_docs": itemgetter("unfiltered_docs"),
        }

        logger.debug("Defining the rag_chain")
        rag_chain = (
            retrieve_documents
            | sieve_documents(instance=self)
            | pbar_chain(
                llm=self.eval_llm,
                len_func="len(inputs['unfiltered_docs'])",
                desc="LLM evaluation",
                unit="doc",
            )
            | meta_refilter_docs
            | pbar_closer(llm=self.eval_llm)
            | pbar_chain(
                llm=self.llm,
                len_func="len(inputs['filtered_docs'])",
                desc="Answering each",
                unit="doc",
            )
            | answer_all_docs
            | pbar_closer(llm=self.llm)
        )

        if env.WDOC_VERBOSE:
            rag_chain.get_graph().print_ascii()

        chain_time = 0
        start_time = time.time()
        tried_top_k = []
        while True:
            if len(tried_top_k) > 10:
                raise Exception(f"Tried more than 10 top_k: {tried_top_k}")
            try:
                assert self.top_k not in tried_top_k
                tried_top_k.append(self.top_k)
                logger.debug("Calling the rag_chain")
                output = rag_chain.invoke(
                    {
                        "question_for_embedding": query_fe,
                        "question_to_answer": query_an,
                    }
                )
                break
            except ShouldIncreaseTopKAfterLLMEvalFiltering:
                if self.max_top_k is None:
                    raise
                elif self.max_top_k == self.top_k:
                    break
                assert self.max_top_k > self.top_k
                new_top_k = min(int(self.top_k * 1.5), self.max_top_k)
                assert new_top_k > self.top_k
                assert new_top_k not in tried_top_k
                assert new_top_k <= self.max_top_k
                self.top_k = new_top_k
            except NoDocumentsRetrieved:
                return {
                    "error": logger.error(
                        md_printer(
                            f"## No documents were retrieved with query '{query_fe}'",
                            color="red",
                        )
                    )
                }
            except NoDocumentsAfterLLMEvalFiltering:
                return {
                    "error": logger.error(
                        md_printer(
                            f"## No documents remained after query eval LLM filtering using question '{query_an}'",
                            color="red",
                        )
                    )
                }
        chain_time = time.time() - start_time

        assert len(output["intermediate_answers"]) == len(output["filtered_docs"])

        output["relevant_filtered_docs"] = []
        output["relevant_intermediate_answers"] = []
        for ia, a in enumerate(output["intermediate_answers"]):
            if check_intermediate_answer(a):
                output["relevant_filtered_docs"].append(output["filtered_docs"][ia])
                output["relevant_intermediate_answers"].append(a)

        # Create consistent document identifiers using WDOC_N format
        output["source_mapping"] = {}
        for ifd, fd in enumerate(output["relevant_filtered_docs"]):
            doc_id = f"WDOC_{ifd + 1}"
            output["source_mapping"][doc_id] = ifd + 1
            ia = output["relevant_intermediate_answers"][ifd]
            output["relevant_intermediate_answers"][
                ifd
            ] = f"<doc id=[[{doc_id}]]>\n{ia}\n</doc>"

        all_rlvt_interim_ans = [output["relevant_intermediate_answers"]]

        if len(output["relevant_intermediate_answers"]) > 1:
            # next step is to combine the intermediate answers into a single answer
            final_answer_chain = RunnablePassthrough.assign(
                final_answer=RunnablePassthrough.assign(
                    question=lambda inputs: inputs["question_to_answer"],
                    intermediate_answers=lambda inputs: collate_relevant_intermediate_answers(
                        list_ia=inputs["relevant_intermediate_answers"],
                    ),
                )
                | prompts.combine
                | self.llm
                | StrOutputParser()
            )

            llmcallback = self.llm.callbacks[0]
            cost_before_combine = (
                self.llm_price["prompt"] * llmcallback.prompt_tokens
                + self.llm_price["completion"] * llmcallback.completion_tokens
                + self.llm_price["internal_reasoning"]
                * llmcallback.internal_reasoning_tokens
            )

            # group the intermediate answers by batch, then do a batch reduce mapping
            # each batch is at least 2 intermediate answers and maxes at
            # batch_tkn_size tokens to avoid losing anything because of
            # the context
            pbar = tqdm(
                desc="Combining answers",
                unit="answer",
                total=len(output["relevant_intermediate_answers"]),
                # disable=not env.WDOC_VERBOSE,
                # disable=is_out_piped,
            )
            temp_interm_answ = output["relevant_intermediate_answers"]
            temp_interm_answ = [
                thinking_answer_parser(a)["answer"] for a in temp_interm_answ
            ]
            while True:
                batches = [[]]
                batches = semantic_batching(temp_interm_answ, self.embedding_engine)
                batch_args = [
                    {
                        "question_to_answer": query_an,
                        "relevant_intermediate_answers": b,
                    }
                    for b in batches
                ]
                temp_interm_answ = []
                batch_result = final_answer_chain.batch(batch_args)
                n_trial = 2
                for ia, a in enumerate(batch_result):
                    for trial in range(1, n_trial + 1):
                        try:
                            answer_text = a["final_answer"]
                            o = thinking_answer_parser(
                                answer_text,
                                strict=True,
                            )["answer"]
                            temp_interm_answ.append(o)
                            break
                        except Exception as e:
                            logger.warning(
                                f"Error at trial {trial} when separating "
                                "thinking from answer from LLM output.\n\n"
                                f"The full answer is: '{a}'\n\n"
                                f"The error was: '{e}'\n\n"
                                "Retrying "
                                "this specific batch to make sure we don't"
                                " loose intermediate answers"
                            )
                            # modify the batch slightly to bypass the cache
                            altered_batch = batch_args[ia]
                            altered_batch["question_to_answer"] += "."
                            a = final_answer_chain.batch([altered_batch])[0]

                if len(temp_interm_answ) == 0 and trial > 0:
                    logger.warning(
                        f"Couldn't continue merging documents. This is likely because the intermediate answers got too large. As a cheap workaround I'll concatenate them in semantic order. The latest batch contains {len(batch_result)} intermediate answers. The number of trial was {trial}/{n_trial}."
                    )
                    assert batch_result, trial
                    concat = "\n---\n".join([b["final_answer"] for b in batch_result])
                    temp_interm_answ.append(concat)

                all_rlvt_interim_ans.append(temp_interm_answ)
                pbar.n = pbar.total - len(temp_interm_answ) + 1
                pbar.update(0)
                if len(temp_interm_answ) == 1:
                    break

            assert pbar.n == pbar.total
            pbar.close()
            assert len(all_rlvt_interim_ans[-1]) == 1

            final_answer = all_rlvt_interim_ans[-1][0]
            output["all_relevant_intermediate_answers"] = all_rlvt_interim_ans

        elif not len(output["relevant_intermediate_answers"]):
            raise Exception(
                f"No 'relevant_intermediate_answers' found. Output was: '{str(output)[:1000]}...'"
            )

        else:
            final_answer = output["relevant_intermediate_answers"][0]

            # Apply source replacement to final answer
            final_answer = f"Source identifier: [[{doc_id}]]\n{final_answer}"
            output["all_relevant_intermediate_answers"] = [
                output["relevant_intermediate_answers"][0]
            ]

        # check that all sources used as intermediates are mentionned in the final answer
        collates = "\n".join(
            ["\n".join(d) for d in output["all_relevant_intermediate_answers"]]
        )
        missing_mapper = []
        for k, v in output["source_mapping"].items():
            if k in collates and k not in final_answer:
                missing_mapper.append(v)

        # prepare the content of the output
        final_answer = source_replace(final_answer, output["source_mapping"])
        output["final_answer"] = final_answer

        # if out_file is specified then we write the summary there too.
        if self.out_file:

            def output_handler(text: str) -> str:
                # the raw markdown is written to the file
                # and the rendered markdown is printed on screen
                with open(self.out_file, "a") as f:
                    f.write(text + "\n")
                return md_printer(text)

        else:

            def output_handler(text: str) -> str:
                return md_printer(text)

        # display sources (i.e. documents used to answer)
        output_handler("---")
        if not output["relevant_intermediate_answers"]:
            logger.error(
                output_handler(
                    "\n\n# No document filtered so no intermediate answers to combine.\nThe answer will be based purely on the LLM's internal knowledge.",
                    color="red",
                )
            )
            logger.error(
                output_handler(
                    "\n\n# No document filtered so no intermediate answers to combine"
                )
            )
        else:
            output_handler("\n\n# Intermediate answers for each document:")
        n = len(output["relevant_intermediate_answers"])
        for counter, (ia, doc) in enumerate(
            zip(
                output["relevant_intermediate_answers"][::-1],
                output["relevant_filtered_docs"][::-1],
            )
        ):
            to_print = f"## Document #{n - counter}\n"
            content = doc.page_content.strip()
            to_print += "```\n" + content + "\n ```\n"
            for k, v in doc.metadata.items():
                to_print += f"* **{k}**: `{v}`\n"
            ia = thinking_answer_parser(ia)
            # print either both thinking and answer or just answer
            # ia = "### Thinking:\n" + ia["thinking"] + "\n\n" + "### Answer:\n" + ia["answer"]
            ia = ia["answer"]
            to_print += indent("### Intermediate answer:\n" + ia, "> ")
            output_handler(source_replace(to_print, output["source_mapping"]))

        # print the final answer
        fa = thinking_answer_parser(output["final_answer"])
        # fa = "### Thinking:\n" + fa["thinking"] + "\n\n" + "### Answer:\n" + fa["answer"]
        fa = fa["answer"]
        output_handler("---")
        output_handler(
            indent(f"# Answer:\n{source_replace(fa, output['source_mapping'])}\n", "> ")
        )

        if missing_mapper:
            logger.warning(
                f"Found some source mappers in intermediate answers that are missing from the final answer. Here are the documents #: {','.join(map(str, missing_mapper))}"
            )

        # print the breakdown of documents used and chain time
        logger.info(
            f"Number of documents using embeddings: {len(output['unfiltered_docs'])}"
        )
        logger.info(
            f"Number of documents found relevant by eval LLM: {len(output['filtered_docs'])}"
        )
        logger.info(
            f"Number of documents found relevant by answer LLM: {len(output['relevant_filtered_docs'])}"
        )
        if len(all_rlvt_interim_ans) > 1:
            extra = "->".join([str(len(ia)) for ia in all_rlvt_interim_ans])
            extra = f"({extra})"
        else:
            extra = ""
        logger.debug(
            f"Number of steps to combine intermediate answers: {len(all_rlvt_interim_ans) - 1} {extra}"
        )
        logger.debug(f"Time took by the chain: {chain_time:.2f}s")

        evalllmcallback = self.eval_llm.callbacks[0]
        etotal_cost = (
            self.query_evalllm_price["prompt"] * evalllmcallback.prompt_tokens
            + self.query_evalllm_price["completion"] * evalllmcallback.completion_tokens
            + self.query_evalllm_price["internal_reasoning"]
            * evalllmcallback.internal_reasoning_tokens
        )
        llmcallback = self.llm.callbacks[0]
        total_cost = (
            self.llm_price["prompt"] * llmcallback.prompt_tokens
            + self.llm_price["completion"] * llmcallback.completion_tokens
            + self.llm_price["internal_reasoning"]
            * llmcallback.internal_reasoning_tokens
        )
        logger.debug(
            f"Tokens used by query_eval model: '{evalllmcallback.total_tokens}' (${etotal_cost:.5f})"
        )

        if "cost_before_combine" in locals():
            combine_cost = total_cost - cost_before_combine
            logger.debug(
                f"Tokens used by strong model to combine the intermediate answers: ${combine_cost:.5f}"
            )

        logger.debug(
            f"Total tokens used by strong model: '{llmcallback.total_tokens}' (${total_cost:.5f})"
        )

        logger.info(f"Total cost: ${total_cost + etotal_cost:.5f}")

        assert total_cost + etotal_cost >= self.latest_cost
        self.latest_cost = total_cost + etotal_cost

        output["total_cost"] = self.latest_cost
        output["total_model_cost"] = total_cost
        output["total_eval_model_cost"] = etotal_cost

        return output

    def _actual_search_task(
        self,
        retriever: Any,
        query: str,
        query_fe: str,
        query_an: str,
        evaluate_doc_chain: Any,
        multi: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.query_eval_model is not None:
            from wdoc.utils.tasks.search import retrieve_documents_for_search
            from wdoc.utils.tasks.query import (
                autoincrease_top_k,
                pbar_chain,
                pbar_closer,
                refilter_docs,
                sieve_documents,
            )

            # for some reason I needed to have at least one chain object otherwise rag_chain is a dict
            retrieve_documents = retrieve_documents_for_search(retriever)

            def create_autoincrease_top_k_chain(inputs):
                filtered_docs = inputs
                return autoincrease_top_k(filtered_docs, self.top_k, self.max_top_k)

            create_autoincrease_top_k_chain = chain(create_autoincrease_top_k_chain)

            meta_refilter_docs = {
                "filtered_docs": (
                    RunnablePassthrough.assign(
                        evaluations=RunnablePassthrough.assign(
                            doc=lambda inputs: inputs["unfiltered_docs"],
                            q=lambda inputs: [
                                inputs["question_to_answer"]
                                for i in range(len(inputs["unfiltered_docs"]))
                            ],
                        )
                        | RunnablePassthrough.assign(
                            inputs=lambda inputs: [
                                {"doc": d.page_content, "q": q}
                                for d, q in zip(inputs["doc"], inputs["q"])
                            ],
                        )
                        | itemgetter("inputs")
                        | RunnableEach(
                            bound=evaluate_doc_chain.with_config(multi)
                        ).with_config(multi)
                    )
                    | refilter_docs
                    | create_autoincrease_top_k_chain
                ),
                "unfiltered_docs": itemgetter("unfiltered_docs"),
                "question_to_answer": itemgetter("question_to_answer"),
            }

            logger.debug("Defining the rag_chain")
            rag_chain = (
                retrieve_documents
                | sieve_documents(instance=self)
                | pbar_chain(
                    llm=self.eval_llm,
                    len_func="len(inputs['unfiltered_docs'])",
                    desc="LLM evaluation",
                    unit="doc",
                )
                | meta_refilter_docs
                | pbar_closer(llm=self.eval_llm)
            )
            tried_top_k = []
            while True:
                if len(tried_top_k) > 10:
                    raise Exception(f"Tried more than 10 top_k: {tried_top_k}")
                try:
                    assert self.top_k not in tried_top_k
                    tried_top_k.append(self.top_k)
                    logger.debug("Calling the rag_chain")
                    output = rag_chain.invoke(
                        {
                            "question_for_embedding": query_fe,
                            "question_to_answer": query_an,
                        }
                    )
                    break
                except ShouldIncreaseTopKAfterLLMEvalFiltering:
                    if self.max_top_k is None:
                        raise
                    assert (
                        self.max_top_k != self.top_k
                    ), f"Something went wrong: top_k: {self.top_k} ; max_top_k: {self.max_top_k}"
                    assert (
                        self.max_top_k > self.top_k
                    )  # if it's equal, it should have returned normally
                    new_top_k = min(int(self.top_k * 1.5), self.max_top_k)
                    assert new_top_k > self.top_k
                    assert new_top_k not in tried_top_k
                    assert new_top_k <= self.max_top_k
                    self.top_k = new_top_k

            docs = output["filtered_docs"]
        else:
            docs = retriever.invoke(query)
            if len(docs) < self.interaction_settings["top_k"]:
                logger.warning(f"Only found {len(docs)} relevant documents")

        if "unfiltered_docs" in output:
            logger.info(
                f"Number of documents using embeddings: {len(output['unfiltered_docs'])}"
            )
        if "filtered_docs" in output:
            logger.info(
                f"Number of documents found relevant by eval LLM: {len(output['filtered_docs'])}"
            )
        if "relevant_filtered_docs" in output:
            logger.info(
                f"Number of documents found relevant by answer LLM: {len(output['relevant_filtered_docs'])}"
            )

        evalllmcallback = self.eval_llm.callbacks[0]
        etotal_cost = (
            self.query_evalllm_price["prompt"] * evalllmcallback.prompt_tokens
            + self.query_evalllm_price["completion"] * evalllmcallback.completion_tokens
            + self.query_evalllm_price["internal_reasoning"]
            * evalllmcallback.internal_reasoning_tokens
        )
        logger.debug(
            f"Tokens used by query_eval model: '{evalllmcallback.total_tokens}' (${etotal_cost:.5f})"
        )

        llmcallback = self.llm.callbacks[0]
        total_cost = (
            self.llm_price["prompt"] * llmcallback.prompt_tokens
            + self.llm_price["completion"] * llmcallback.completion_tokens
            + self.llm_price["internal_reasoning"]
            * llmcallback.internal_reasoning_tokens
        )
        logger.debug(
            f"Total tokens used by strong model: '{llmcallback.total_tokens}' (${total_cost:.5f})"
        )
        logger.warning(f"Total cost: ${total_cost + etotal_cost:.5f}")

        if self.__import_mode__:
            return output

        md_printer("\n\n# Documents")
        if env.WDOC_OPEN_ANKI:
            anki_nids = []
            to_print = ""
        for id, doc in enumerate(docs):
            to_print += f"## Document #{id + 1}\n"
            content = doc.page_content.strip()
            to_print += "```\n" + content + "\n ```\n"
            for k, v in doc.metadata.items():
                to_print += f"* **{k}**: `{v}`\n"
            to_print += "\n"
            if env.WDOC_OPEN_ANKI and "anki_nid" in doc.metadata:
                nid_str = str(doc.metadata["anki_nid"]).split(" ")
                for nid in nid_str:
                    if nid not in anki_nids:
                        anki_nids.append(nid)
        md_printer(to_print)
        if self.query_eval_model is not None:
            logger.warning(
                f"Number of documents using embeddings: {len(output['unfiltered_docs'])}"
            )
            logger.warning(
                f"Number of documents after query eval filter: {len(output['filtered_docs'])}"
            )

        if env.WDOC_OPEN_ANKI and anki_nids:
            open_answ = input(
                f"\nAnki notes found, open in anki? (yes/no/debug)\n(nids: {anki_nids})\n> "
            )
            if open_answ == "debug":
                breakpoint()
            elif open_answ in ["y", "yes"]:
                logger.info("Opening anki.")
                query = f"nid:{','.join(anki_nids)}"
                try:
                    from py_ankiconnect import PyAnkiconnect

                    ankiconnect = PyAnkiconnect()
                    ankiconnect(
                        action="guiBrowse",
                        query=query,
                    )
                except Exception as e:
                    logger.warning(f"Error when trying to open Anki: '{e}'")
        all_filepaths = []
        for doc in docs:
            if "path" in doc.metadata:
                path = doc.metadata["path"]
                try:
                    path = str(Path(path).resolve().absolute())
                except Exception:
                    pass
                all_filepaths.append(path)
        if all_filepaths:
            md_printer("### All file paths")
            md_printer("* " + "\n* ".join(all_filepaths))

        evalllmcallback = self.eval_llm.callbacks[0]
        etotal_cost = (
            self.query_evalllm_price["prompt"] * evalllmcallback.prompt_tokens
            + self.query_evalllm_price["completion"] * evalllmcallback.completion_tokens
            + self.query_evalllm_price["internal_reasoning"]
            * evalllmcallback.internal_reasoning_tokens
        )
        logger.debug(
            f"Tokens used by query_eval model: '{evalllmcallback.total_tokens}' (${etotal_cost:.5f})"
        )

        logger.warning(f"Total cost: ${etotal_cost:.5f}")
        self.latest_cost = etotal_cost

    @staticmethod
    @set_parse_doc_help_md_as_docstring
    def parse_doc(*args, **kwargs) -> Union[List[Document], str, List[dict]]:
        from wdoc.utils.tasks.parse import parse_doc

        return parse_doc(*args, **kwargs)
