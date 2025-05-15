"""
Main class.
"""

import asyncio
import copy
import faulthandler
import json
import os
import pdb
import re
import sys
import time
import traceback
from dataclasses import MISSING
from datetime import date
from operator import itemgetter
from pathlib import Path
from textwrap import indent

import litellm
import pyfiglet
import tldextract
from beartype.door import is_bearable
from beartype.typing import Any, Callable, List, Literal, Optional, Union
from langchain.docstore.document import Document
from langchain.globals import set_debug, set_llm_cache, set_verbose
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.retrievers import KNNRetriever, SVMRetriever
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.runnables.base import RunnableEach
from tqdm import tqdm
from loguru import logger as logger

from wdoc.utils.batch_file_loader import batch_load_doc
from wdoc.utils.customs.fix_llm_caching import SQLiteCacheFixed
from wdoc.utils.embeddings import create_embeddings, load_embeddings_engine
from wdoc.utils.env import env, is_out_piped
from wdoc.utils.errors import (
    NoDocumentsAfterLLMEvalFiltering,
    NoDocumentsRetrieved,
    ShouldIncreaseTopKAfterLLMEvalFiltering,
)

from wdoc.utils.interact import ask_user
from wdoc.utils.llm import TESTING_LLM, load_llm

# import this first because it sets the logging level
from wdoc.utils.logger import (
    log_dir,
    md_printer,
    set_help_md_as_docstring,
    set_parse_file_help_md_as_docstring,
)
from wdoc.utils.misc import (  # debug_chain,
    cache_dir,
    DocDict,
    ModelName,
    average_word_length,
    check_docs_tkn_length,
    create_langfuse_callback,
    disable_internet,
    extra_args_types,
    get_model_price,
    get_splitter,
    get_supported_model_params,
    get_tkn_length,
    model_name_matcher,
    query_eval_cache,
    set_func_signature,
    thinking_answer_parser,
    wpm,
    tasks_list,
)
from wdoc.utils.prompts import prompts
from wdoc.utils.retrievers import create_multiquery_retriever, create_parent_retriever
from wdoc.utils.tasks.query import (
    check_intermediate_answer,
    collate_relevant_intermediate_answers,
    parse_eval_output,
    pbar_chain,
    pbar_closer,
    refilter_docs,
    semantic_batching,
    sieve_documents,
)
from wdoc.utils.tasks.summarize import do_summarize
from wdoc.utils.typechecker import optional_typecheck

logger.info("Starting wdoc")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@optional_typecheck
@set_help_md_as_docstring
class wdoc:
    """
    This docstring is dynamically updated with the content of wdoc/docs/help.md
    """

    VERSION: str = "3.2.4"
    allowed_extra_args = extra_args_types
    __import_mode__: bool = True

    @optional_typecheck
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
        if version:
            print(self.VERSION)
            return
        if notification_callback is not None:

            @optional_typecheck
            def ntfy(text: str) -> str:
                out = notification_callback(text)
                assert (
                    out == text
                ), "The notification callback must return the same string"
                return out

            ntfy("Starting wdoc")
        else:

            @optional_typecheck
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
            faulthandler.enable()

        from loguru import logger  # for some reason I have to reimport

        # loguru here otherwise the next line fails!
        logger.warning(pyfiglet.figlet_format("wdoc"))

        # make sure the extra args are valid
        for k in cli_kwargs:
            if k not in self.allowed_extra_args:
                raise Exception(
                    logger.warning(
                        f"Found unexpected keyword argument: '{k}'\nThe allowed arguments are {','.join(self.allowed_extra_args)}"
                    )
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
                if k.endswith("_API_KEY") or k.endswith("_API_KEYS"):
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

        if query is True:
            # otherwise specifying --query and forgetting to add text fails
            query = None

        if isinstance(query, str):
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
            os.environ["LITELLM_LOG"] = "DEBUG"
            litellm._turn_on_debug()

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

        if self.task in ["query", "search", "summary_then_query"]:
            self.prepare_query_task()

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
            while True:
                self.query_task(query=query)
                query = None

    @optional_typecheck
    def summary_task(self) -> dict:
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
                f"Estimate of the LLM cost to summarize: ${estimate_dol:.4f} for {full_tkn} tokens."
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

        @optional_typecheck
        def summarize_documents(
            path: Any,
            relevant_docs: List,
        ) -> dict:
            assert relevant_docs, "Empty relevant_docs!"

            # parse metadata from the doc
            metadata = []
            if "http" in path:
                item_name = tldextract.extract(path).registered_domain
            elif "/" in path and Path(path).exists():
                item_name = Path(path).name
            else:
                item_name = path

            if "title" in relevant_docs[0].metadata:
                item_name = (
                    f"{relevant_docs[0].metadata['title'].strip()} - {item_name}"
                )
            else:
                metadata.append(f"<title>\n{item_name.strip()}\n</title>")

            # replace # in title as it would be parsed as a tag
            item_name = item_name.replace("#", r"\#")

            if "doc_reading_time" in relevant_docs[0].metadata:
                doc_reading_length = relevant_docs[0].metadata["doc_reading_time"]
                metadata.append(
                    f"<reading_length>\n{doc_reading_length:.1f} minutes\n</reading_length>"
                )
            else:
                doc_reading_length = 0
            if "author" in relevant_docs[0].metadata:
                author = relevant_docs[0].metadata["author"].strip()
                metadata.append(f"<author>\n{author}\n</author>")
            else:
                author = None
            if "yt_chapters" in relevant_docs[0].metadata:
                chapters = json.dumps(relevant_docs[0].metadata["yt_chapters"])
                metadata.append(f"<youtube_chapters>\n{chapters}\n</youtube_chapters>")
            metadata.append(f"<today>\n{date.today().isoformat()}\n</today>")

            if metadata:
                metadata = "<text_metadata>\n" + "\n".join(metadata) + "\n"
                metadata += "<section_number>\n[PROGRESS]\n</section_number>\n"
                metadata += "</text_metadata>"
            else:
                metadata = "<text_metadata><section_number>[PROGRESS]</section_number></text_metadata>"

            # summarize each chunk of the link and return one text
            (
                summary,
                n_chunk,
                doc_total_tokens,
            ) = do_summarize(
                docs=relevant_docs,
                metadata=metadata,
                language=self.summary_language,
                modelbackend=self.model.backend,
                llm=self.llm,
                verbose=self.llm_verbosity,
            )

            # get reading length of the summary
            real_text = "".join(
                [letter for letter in list(summary) if letter.isalpha()]
            )
            sum_reading_length = len(real_text) / average_word_length / wpm
            logger.info(f"{item_name} reading length is {sum_reading_length:.1f}")

            recursive_summaries = {0: summary}
            prev_real_text = MISSING
            if self.summary_n_recursion > 0:
                for n_recur in range(1, self.summary_n_recursion + 1):
                    summary_text = copy.deepcopy(recursive_summaries[n_recur - 1])
                    logger.warning(f"Doing summary check #{n_recur} of {item_name}")

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
                    assert (
                        "- BEFORE RECURSION # " not in summary_text
                    ), "Found recursion block"

                    splitter = get_splitter(
                        "recursive_summary",
                        modelname=self.model,
                    )
                    summary_docs = [Document(page_content=summary_text)]
                    summary_docs = splitter.transform_documents(summary_docs)
                    assert summary_docs != relevant_docs
                    try:
                        check_docs_tkn_length(summary_docs, item_name)
                    except Exception as err:
                        logger.warning(
                            f"Exception when checking if {item_name} could be recursively summarized for the #{n_recur} time: {err}"
                        )
                        break
                    (
                        summary_text,
                        n_chunk,
                        new_doc_total_tokens,
                    ) = do_summarize(
                        docs=summary_docs,
                        metadata=metadata,
                        language=self.summary_language,
                        modelbackend=self.model.backend,
                        llm=self.llm,
                        verbose=self.llm_verbosity,
                        n_recursion=n_recur,
                    )

                    # aggregate the token count
                    for k, v in new_doc_total_tokens.items():
                        doc_total_tokens[k] += v

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
                    assert (
                        "- BEFORE RECURSION # " not in real_text
                    ), "Found recursion block"
                    real_text = "".join(
                        [letter for letter in list(real_text) if letter.isalpha()]
                    )
                    sum_reading_length = len(real_text) / average_word_length / wpm
                    logger.info(
                        f"{item_name} reading length after recursion #{n_recur} is {sum_reading_length:.1f}"
                    )
                    if prev_real_text is not MISSING:
                        if real_text == prev_real_text:
                            logger.warning(
                                f"Identical summary after {n_recur} "
                                "recursion, adding more recursion will not "
                                "help so stopping here"
                            )
                            recursive_summaries[n_recur] = summary_text
                            break
                    prev_real_text = real_text

                    assert n_recur not in recursive_summaries
                    if summary_text not in recursive_summaries:
                        logger.warning(
                            f"Identical summary after {n_recur} "
                            "recursion, adding more recursion will not "
                            "help so stopping here"
                        )
                        recursive_summaries[n_recur] = summary_text
                        break
                    else:
                        recursive_summaries[n_recur] = summary_text

            best_sum_i = max(list(recursive_summaries.keys()))
            if not self.__import_mode__:
                print("\n\n")
                md_printer("# Summary")
                md_printer(f"## {path}")
                md_printer(recursive_summaries[best_sum_i])

            # the price computation needs to happen as late as possible to avoid
            # underflow errors
            doc_total_cost = 0
            doc_total_tokens_str = ""
            for k, v in doc_total_tokens.items():
                if self.llm_price[k]:  # to avoid underflow errors:
                    doc_total_cost += v * self.llm_price[k]
                doc_total_tokens_str += f"{k.title()}: {v} "
            doc_total_tokens_str = doc_total_tokens_str.strip()
            logger.info(
                f"Tokens used for {path}: ({doc_total_tokens_str}, cost: ${doc_total_cost:.5f})"
            )

            summary_tkn_length = get_tkn_length(recursive_summaries[best_sum_i])

            header = f"\n- {item_name}    cost: ${doc_total_cost:.5f} ({doc_total_tokens_str})"
            if doc_reading_length:
                header += f"    {doc_reading_length:.1f} minutes"
            if author:
                header += f"    by '{author}'"
            header += f"    original path: '{path}'"
            header += f"    wdoc version {self.VERSION} with model {self.model} on {date.today().isoformat()}"

            # save to output file
            if self.out_file:
                if self.__import_mode__:
                    logger.warning(
                        f"Detected use of out_file arg while in __import_mode__. This is unexpected and might lead to issues."
                    )
                for nrecur, sum in recursive_summaries.items():
                    out_file = Path(self.out_file)
                    if len(recursive_summaries) > 1 and nrecur < max(
                        list(recursive_summaries.keys())
                    ):
                        # also store intermediate summaries if present
                        out_file = out_file.parent / (
                            out_file.stem + f".{nrecur + 1}.md"
                        )

                    with open(str(out_file), "a") as f:
                        if out_file.exists() and out_file.read_text().strip():
                            f.write("\n\n\n")
                        f.write(header)
                        if len(recursive_summaries) > 1:
                            f.write(
                                f"\n    Recursive summary pass: {nrecur + 1}/{len(recursive_summaries)}"
                            )

                        for bulletpoint in sum.split("\n"):
                            f.write("\n")
                            bulletpoint = bulletpoint.rstrip()
                            f.write(f"    {bulletpoint}")

            return {
                "path": path,
                "sum_reading_length": sum_reading_length,
                "sum_tkn_length": summary_tkn_length,
                "doc_reading_length": doc_reading_length,
                "doc_total_tokens": doc_total_tokens,
                "doc_total_tokens_str": doc_total_tokens_str,
                "doc_total_cost": doc_total_cost,
                "summary": recursive_summaries[best_sum_i],
                "recursive_summaries": recursive_summaries,
                "author": author,
                "n_chunk": n_chunk,
            }

        results = summarize_documents(
            path=self.cli_kwargs["path"],
            relevant_docs=self.loaded_docs,
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
        if llmcallback.total_tokens != results["doc_total_tokens"]:
            logger.warning(
                f"Cost discrepancy? Tokens used according to the callback: '{llmcallback.total_tokens}' vs in the result: '{results['doc_total_tokens']}' (${total_cost:.5f})"
            )
        self.summary_results = results
        self.latest_cost = total_cost
        return results

    @optional_typecheck
    def prepare_query_task(self) -> None:
        # load embeddings for querying
        self.embedding_engine = load_embeddings_engine(
            modelname=self.embed_model,
            cli_kwargs=self.cli_kwargs,
            api_base=self.llms_api_bases["embeddings"],
            embed_kwargs=self.embed_model_kwargs,
            private=self.private,
            do_test=env.WDOC_EMBED_TESTING,
        )
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
        self.all_texts = [
            v.page_content for k, v in self.loaded_embeddings.docstore._dict.items()
        ]

        # parse filters as callable for faiss filtering
        if "filter_metadata" in self.cli_kwargs or "filter_content" in self.cli_kwargs:
            if "filter_metadata" in self.cli_kwargs:
                # get the list of all metadata to see if a filter was not misspelled
                all_metadata_keys = set()
                for doc in tqdm(
                    self.loaded_embeddings.docstore._dict.values(),
                    desc="gathering metadata keys",
                    unit="doc",
                    disable=(not env.WDOC_VERBOSE) or is_out_piped,
                ):
                    for k in doc.metadata.keys():
                        all_metadata_keys.add(k)
                assert (
                    all_metadata_keys
                ), "No metadata keys found in any metadata, something went wrong!"

                if isinstance(self.cli_kwargs["filter_metadata"], str):
                    filter_metadata = self.cli_kwargs["filter_metadata"].split(",")
                else:
                    filter_metadata = self.cli_kwargs["filter_metadata"]
                assert isinstance(
                    filter_metadata, list
                ), f"filter_metadata must be a list, not {self.cli_kwargs['filter_metadata']}"

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
                    assert kvb in [
                        "k",
                        "v",
                        "b",
                    ], f"filter 1st character must be k, v or b: '{f}'"
                    incexc = f[1]
                    assert incexc in [
                        "+",
                        "-",
                    ], f"filter 2nd character must be + or -: '{f}'"
                    incexc_str = "plus" if incexc == "+" else "minus"
                    assert f[2:].strip(), f"Filter can't be an empty regex: '{f}'"
                    pattern = f[2:].strip()
                    if kvb == "b":
                        assert ":" in f, (
                            "Filter starting with b must contain "
                            "a ':' to distinguish the key regex and the value "
                            f"regex: '{f}'"
                        )
                        key_pat, value_pat = pattern.split(":", 1)
                        if key_pat == key_pat.lower():
                            key_pat = re.compile(key_pat, flags=re.IGNORECASE)
                        else:
                            key_pat = re.compile(key_pat)
                        if value_pat == value_pat.lower():
                            value_pat = re.compile(value_pat, flags=re.IGNORECASE)
                        else:
                            value_pat = re.compile(value_pat)
                        assert (
                            key_pat not in locals()[f"filters_b_{incexc_str}_keys"]
                        ), (
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

                # check that all key filter indeed match metadata keys
                for k in (
                    filters_k_plus
                    + filters_k_minus
                    + filters_b_plus_keys
                    + filters_b_minus_keys
                ):
                    assert any(
                        k.match(str(key)) for key in all_metadata_keys
                    ), f"Key {k} didn't match any key in the metadata"

                @optional_typecheck
                def filter_meta(meta: dict) -> bool:
                    # match keys
                    for inc in filters_k_plus:
                        if not any(inc.match(str(k)) for k in meta.keys()):
                            return False
                    for exc in filters_k_minus:
                        if any(exc.match(str(k)) for k in meta.keys()):
                            return False

                    # match values
                    for inc in filters_v_plus:
                        if not any(inc.match(str(v)) for v in meta.values()):
                            return False
                    for exc in filters_v_minus:
                        if any(exc.match(str(v)) for v in meta.values()):
                            return False

                    # match both
                    for kp, vp in zip(filters_b_plus_keys, filters_b_plus_values):
                        good_keys = (k for k in meta.keys() if kp.match(str(k)))
                        gk_checked = 0
                        for gk in good_keys:
                            if vp.match(str(meta[gk])):
                                gk_checked += 1
                                break
                        if not gk_checked:
                            return False
                    for kp, vp in zip(filters_b_minus_keys, filters_b_minus_values):
                        good_keys = (k for k in meta.keys() if kp.match(str(k)))
                        gk_checked = 0
                        for gk in good_keys:
                            if vp.match(str(meta[gk])):
                                return False
                            gk_checked += 1
                        if not gk_checked:
                            return False

                    return True

            else:

                @optional_typecheck
                def filter_meta(meta: dict) -> bool:
                    return True

            if "filter_content" in self.cli_kwargs:
                if isinstance(self.cli_kwargs["filter_content"], str):
                    filter_content = self.cli_kwargs["filter_content"].split(",")
                else:
                    filter_content = self.cli_kwargs["filter_content"]
                assert isinstance(
                    filter_content, list
                ), f"filter_content must be a list, not {self.cli_kwargs['filter_content']}"

                # storing fast as list then in tupples for faster iteration
                filters_cont_plus = []
                filters_cont_minus = []

                for f in filter_content:
                    assert isinstance(f, str), f"Filter must be a string: '{f}'"
                    incexc = f[0]
                    assert incexc in [
                        "+",
                        "-",
                    ], f"filter 1st character must be + or -: '{f}'"
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

                @optional_typecheck
                def filter_cont(cont: str) -> bool:
                    if not all(inc.match(cont) for inc in filters_cont_plus):
                        return False
                    if any(exc.match(cont) for exc in filters_cont_minus):
                        return False
                    return True

            else:

                @optional_typecheck
                def filter_cont(cont: str) -> bool:
                    return True

            # check filtering is valid
            checked = 0
            good = 0
            ids_to_del = []
            for doc_id, doc in tqdm(
                self.loaded_embeddings.docstore._dict.items(),
                desc="Filtering",
                unit="docs",
                disable=(not env.WDOC_VERBOSE) or is_out_piped,
            ):
                checked += 1
                if filter_meta(doc.metadata) and filter_cont(doc.page_content):
                    good += 1
                else:
                    ids_to_del.append(doc_id)
            logger.warning(
                f"Keeping {good}/{checked} documents from vectorstore after filtering"
            )
            if good == checked:
                logger.warning("Your filter matched all stored documents!")
            assert good, "No documents in the vectorstore match the given filter"

            # directly remove the filtered documents from the docstore
            # but first store the docstore before altering it to allow
            # unfiltering in the prompt
            # self.unfiltered_docstore = self.loaded_embeddings.serialize_to_bytes()
            status = self.loaded_embeddings.delete(ids_to_del)

            # checking deletions went well
            if status is False:
                raise Exception("Vectorstore filtering failed")
            elif status is None:
                raise Exception("Vectorstore filtering not implemented")
            assert len(self.loaded_embeddings.docstore._dict) == checked - len(
                ids_to_del
            ), "Something went wrong when deleting filtered out documents"
            assert len(
                self.loaded_embeddings.docstore._dict
            ), "Something went wrong when deleting filtered out documents: no document left"
            assert len(self.loaded_embeddings.docstore._dict) == len(
                self.loaded_embeddings.index_to_docstore_id
            ), "Something went wrong when deleting filtered out documents"

    @optional_typecheck
    def query_task(self, query: Optional[str] = None) -> dict:
        if not query:
            if self.oneoff:
                sys.exit(0)
            if is_out_piped:
                logger.debug(
                    "Exited query_task because we don't loop the queries when the output is a shell pipe"
                )
                sys.exit(0)
            query, self.interaction_settings = ask_user(self.interaction_settings)
        assert all(
            retriev in ["basic", "multiquery", "knn", "svm", "parent"]
            for retriev in self.interaction_settings["retriever"].split("_")
        ), f"Invalid retriever value: {self.interaction_settings['retriever']}"
        retrievers = []
        if "multiquery" in self.interaction_settings["retriever"].lower():
            retrievers.append(
                create_multiquery_retriever(
                    llm=self.llm,
                    retriever=self.loaded_embeddings.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={
                            "k": self.interaction_settings["top_k"],
                            "score_threshold": self.interaction_settings["relevancy"],
                        },
                    ),
                )
            )

        if "knn" in self.interaction_settings["retriever"].lower():
            retrievers.append(
                KNNRetriever.from_texts(
                    self.all_texts,
                    self.embedding_engine,
                    relevancy_threshold=self.interaction_settings["relevancy"],
                    k=self.interaction_settings["top_k"],
                )
            )
        if "svm" in self.interaction_settings["retriever"].lower():
            retrievers.append(
                SVMRetriever.from_texts(
                    self.all_texts,
                    self.embedding_engine,
                    relevancy_threshold=self.interaction_settings["relevancy"],
                    k=self.interaction_settings["top_k"],
                )
            )
        if "parent" in self.interaction_settings["retriever"].lower():
            retrievers.append(
                create_parent_retriever(
                    task=self.task,
                    loaded_embeddings=self.loaded_embeddings,
                    loaded_docs=self.loaded_docs,
                    top_k=self.interaction_settings["top_k"],
                    relevancy=self.interaction_settings["relevancy"],
                )
            )

        if "basic" in self.interaction_settings["retriever"].lower():
            retrievers.append(
                self.loaded_embeddings.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": self.interaction_settings["top_k"],
                        "score_threshold": self.interaction_settings["relevancy"],
                    },
                )
            )

        assert (
            retrievers
        ), "No retriever selected. Probably cause by a wrong cli_command or query_retrievers arg."

        if len(retrievers) == 1:
            retriever = retrievers[0]
        else:
            merge_retriever = MergerRetriever(retrievers=retrievers)

            # remove redundant results from the merged retrievers:
            filtered = EmbeddingsRedundantFilter(
                embeddings=self.embedding_engine,
                similarity_threshold=0.999,
            )
            filter_pipeline = DocumentCompressorPipeline(transformers=[filtered])
            retriever = ContextualCompressionRetriever(
                base_compressor=filter_pipeline, base_retriever=merge_retriever
            )

        if ">>>>" in query:
            sp = query.split(">>>>")
            assert (
                len(sp) == 2
            ), "The query must contain a maximum of 1 occurence of '>>>>'"
            query_fe = sp[0].strip()
            query_an = sp[1].strip()
        else:
            query_fe, query_an = copy.copy(query), copy.copy(query)
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

            @optional_typecheck
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

        @chain
        @optional_typecheck
        def autoincrease_top_k(filtered_docs: List[Document]) -> List[Document]:
            if not self.max_top_k:
                return filtered_docs
            ratio = len(filtered_docs) / self.top_k
            if ratio >= 0.9:
                if self.top_k < self.max_top_k:
                    raise ShouldIncreaseTopKAfterLLMEvalFiltering(
                        logger.warning(
                            f"Number of documents found: {len(filtered_docs)}, "
                            f"top_k is {self.top_k} so ratio={ratio:.1f}, hence "
                            f"top_k should be increased. Max_top_k is {self.max_top_k}"
                        )
                    )
                else:
                    logger.warning(
                        f"Number of documents found: {len(filtered_docs)}, "
                        f"top_k is {self.top_k} so ratio={ratio:.1f}, hence "
                        f"top_k should be increased but we eached "
                        f"max_top_k ({self.max_top_k}) so continuing."
                    )
            return filtered_docs

        @chain
        @optional_typecheck
        @eval_cache_wrapper
        def evaluate_doc_chain(
            inputs: dict,
            query_nb: int = self.query_eval_check_number,
            eval_model_string: str = self.eval_llm._get_llm_string(),  # just for caching
            eval_prompt: str = str(prompts.evaluate.to_json()),
        ) -> List[str]:
            if isinstance(self.eval_llm, FakeListChatModel):
                outputs = ["10" for i in range(self.query_eval_check_number)]
                new_p = 0
                new_c = 0
                new_r = 0

            elif "n" in self.eval_llm_params or self.query_eval_check_number == 1:

                def _parse_outputs(out) -> List[str]:
                    reasons = [
                        gen.generation_info["finish_reason"] for gen in out.generations
                    ]
                    outputs = [gen.text for gen in out.generations]
                    # don't always crash if finish_reason is not stop, because it can sometimes still be parsed.
                    if not all(r == "stop" for r in reasons):
                        logger.warning(
                            f"Unexpected generation finish_reason: '{reasons}' for generations: '{outputs}'. Expected 'stop'"
                        )
                    assert outputs, "No generations found by query eval llm"
                    # parse_eval_output will crash if the output is bad anyway
                    outputs = [parse_eval_output(o) for o in outputs]
                    return outputs

                try:
                    out = self.eval_llm._generate_with_cache(
                        prompts.evaluate.format_messages(**inputs),
                        request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                    )
                    outputs = _parse_outputs(out)
                except Exception:  # retry without cache
                    logger.debug(
                        f"Failed to run eval_llm on an input. Retrying without cache."
                    )
                    out = self.eval_llm._generate(
                        prompts.evaluate.format_messages(**inputs),
                        request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                    )
                    outputs = _parse_outputs(out)

                if out.llm_output:
                    new_p = out.llm_output["token_usage"]["prompt_tokens"]
                    new_c = out.llm_output["token_usage"]["completion_tokens"]
                    new_r = (
                        out.llm_output["token_usage"]["total_tokens"] - new_p - new_c
                    )
                else:
                    new_p = 0
                    new_c = 0
                    new_r = 0

            else:
                outputs = []
                new_p = 0
                new_c = 0
                new_r = 0

                async def do_eval(subinputs):
                    try:
                        val = await self.eval_llm._agenerate_with_cache(
                            prompts.evaluate.format_messages(**subinputs),
                            request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                        )
                    except Exception:  # retry without cache
                        val = await self.eval_llm._agenerate(
                            prompts.evaluate.format_messages(**subinputs),
                            request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                        )
                    return val

                outs = [do_eval(inputs) for i in range(self.query_eval_check_number)]
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                outs = loop.run_until_complete(asyncio.gather(*outs))
                for out in outs:
                    assert (
                        len(out.generations) == 1
                    ), f"Query eval llm produced more than 1 evaluations: '{out.generations}'"
                    outputs.append(out.generations[0].text)
                    finish_reason = out.generations[0].generation_info["finish_reason"]
                    if finish_reason not in ["stop", "length"]:
                        logger.warning(
                            f"Unexpected finish_reason: '{finish_reason}' for generation '{outputs[-1]}'"
                        )
                    if out.llm_output:
                        new_p += out.llm_output["token_usage"]["prompt_tokens"]
                        new_c += out.llm_output["token_usage"]["completion_tokens"]
                        new_r += (
                            out.llm_output["token_usage"]["total_tokens"]
                            - new_p
                            - new_c
                        )
                assert outputs, "No generations found by query eval llm"
                outputs = [parse_eval_output(o) for o in outputs]

            if len(outputs) < self.query_eval_check_number and len(outputs) == 1:
                logger.warning(
                    f"query eval model produced 1 output instead of {self.query_eval_check_number}). Output: '{outputs}'\nThis is usually because the model is wrongly specified by litellm as having a modifiable `n` parameter. To avoid this use another model or set the query_eval_check_number to 1."
                )
                if "n" in self.eval_llm_params:
                    self.eval_llm_params.remove("n")
                outputs = outputs * self.query_eval_check_number
            assert (
                len(outputs) == self.query_eval_check_number
            ), f"Query eval model produced an unexpected number of outputs ({outputs} but expected {self.query_eval_check_number} outputs).\nInputs: {inputs}'"

            self.eval_llm.callbacks[0].prompt_tokens += new_p
            self.eval_llm.callbacks[0].completion_tokens += new_c
            self.eval_llm.callbacks[0].internal_reasoning_tokens += new_r
            self.eval_llm.callbacks[0].total_tokens += new_p + new_c + new_r
            if self.eval_llm.callbacks[0].pbar:
                self.eval_llm.callbacks[0].pbar[-1].update(1)
            return outputs

        # uses in most places to increase concurrency limit
        multi = {
            "max_concurrency": env.WDOC_LLM_MAX_CONCURRENCY if not self.debug else 1
        }

        if self.task == "search":
            if self.query_eval_model is not None:
                # for some reason I needed to have at least one chain object otherwise rag_chain is a dict
                @chain
                @optional_typecheck
                def retrieve_documents(inputs):
                    return {
                        "unfiltered_docs": retriever.invoke(
                            inputs["question_for_embedding"]
                        ),
                        "question_to_answer": inputs["question_to_answer"],
                    }
                    return inputs

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
                        | autoincrease_top_k
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
                + self.query_evalllm_price["completion"]
                * evalllmcallback.completion_tokens
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
                + self.query_evalllm_price["completion"]
                * evalllmcallback.completion_tokens
                + self.query_evalllm_price["internal_reasoning"]
                * evalllmcallback.internal_reasoning_tokens
            )
            logger.debug(
                f"Tokens used by query_eval model: '{evalllmcallback.total_tokens}' (${etotal_cost:.5f})"
            )

            logger.warning(f"Total cost: ${etotal_cost:.5f}")
            self.latest_cost = etotal_cost

        else:
            # for some reason I needed to have at least one chain object otherwise rag_chain is a dict
            @chain
            @optional_typecheck
            def retrieve_documents(inputs):
                return {
                    "unfiltered_docs": retriever.invoke(
                        inputs["question_for_embedding"]
                    ),
                    "question_to_answer": inputs["question_to_answer"],
                }
                return inputs

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
                    | autoincrease_top_k
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

            @optional_typecheck
            def source_replace(
                input: str, mapping: dict = output["source_mapping"]
            ) -> str:
                # Make a copy of the input to avoid modifying the original string during iteration
                result = input
                # substitude in reverse order to avoid WDOC_2 replacing WDOC_21
                doc_ids = list(mapping.keys())
                for doc_id in doc_ids[::-1]:
                    doc_num = str(mapping[doc_id])
                    result = result.replace(doc_id, doc_num)
                return result

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
                    disable=is_out_piped,
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
                        concat = "\n---\n".join(
                            [b["final_answer"] for b in batch_result]
                        )
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
            final_answer = source_replace(final_answer)
            output["final_answer"] = final_answer

            # if out_file is specified then we write the summary there too.
            if self.out_file:

                def output_handler(text: str) -> None:
                    # the raw markdown is written to the file
                    # and the rendered markdown is printed on screen
                    with open(self.out_file, "a") as f:
                        f.write(text + "\n")
                    return md_printer(text)

            else:

                def output_handler(text: str) -> None:
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
                output_handler(source_replace(to_print))

            # print the final answer
            fa = thinking_answer_parser(output["final_answer"])
            # fa = "### Thinking:\n" + fa["thinking"] + "\n\n" + "### Answer:\n" + fa["answer"]
            fa = fa["answer"]
            output_handler("---")
            output_handler(indent(f"# Answer:\n{source_replace(fa)}\n", "> "))

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
                + self.query_evalllm_price["completion"]
                * evalllmcallback.completion_tokens
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

    @staticmethod
    @optional_typecheck
    @set_parse_file_help_md_as_docstring
    def parse_file(
        path: Optional[Union[str, Path]] = None,
        filetype: str = "auto",
        format: str = "text",
        cli_kwargs: Optional[dict] = None,
        debug: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Union[List[Document], str, List[dict]]:
        """
        This docstring is dynamically updated with the content of wdoc/docs/parse_file_help.md
        """
        assert format in [
            "text",
            "xml",
            "langchain",
            "langchain_dict",
        ], f"Unexpected --format value: '{format}'"
        default_cli_kwargs = {
            "llm_name": ModelName("testing/testing"),
            "task": "query",
            "backend": "loky",  # doesn't matter because n_jobs is 1 anyway
            "n_jobs": 1,
            "loading_failure": "crash",
        }

        if debug:
            debug_exceptions()

        if cli_kwargs is not None:
            default_cli_kwargs.update(cli_kwargs)

        if "task" in kwargs:
            assert (
                kwargs["task"] == "parse"
            ), f"Unexpected task when parsing. Expected 'parse' but got '{kwargs['task']}'"
            del kwargs["task"]
        assert (
            "task" not in kwargs
        ), "Cannot give --task argument if we are only parsing"

        if kwargs:
            kwargs = DocDict(kwargs)

        # all loaders need a path arg except anki
        if filetype == "anki" and path:
            logger.warning(
                "You supplied a --path argument even though the filetype is `anki`, we must ignore `path` in that case."
            )
        else:
            if not path:
                logger.warning(
                    "You did not specify a --path argument, this will probably cause issues."
                )
            else:
                kwargs["path"] = path

        out = batch_load_doc(
            filetype=filetype,
            **default_cli_kwargs,
            **kwargs,
        )
        if format == "text":
            n = len(out)
            if n > 1:
                return (
                    "Parsed documents:\n"
                    + "\n".join(
                        [
                            f"Doc #{i + 1}/{n}\n{d.page_content}\n\n"
                            for i, d in enumerate(out)
                        ]
                    ).rstrip()
                )
            else:
                return f"Parsed document:\n{out[0].page_content.strip()}"
        elif format == "xml":
            return (
                "<documents>\n"
                + "\n".join(
                    [
                        f"<doc id={i}>\n{d.page_content}\n</doc>"
                        for i, d in enumerate(out)
                    ]
                )
                + "\n</documents>"
            )
        elif format == "langchain":
            return out
        elif format == "langchain_dict":
            return [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in out
            ]
        else:
            raise ValueError(format)


def debug_exceptions(instance: Optional[wdoc] = None) -> None:
    "open a debugger is --debug is set"

    def handle_exception(exc_type, exc_value, exc_traceback):
        if not issubclass(exc_type, KeyboardInterrupt):

            @optional_typecheck
            def p(message: str) -> None:
                "print error, in red if possible"
                if instance:
                    logger.warning(instance.ntfy(message))
                else:
                    try:
                        logger.warning(message)
                    except Exception:
                        print(message)

            p(
                "\n--verbose was used so opening debug console at the "
                "appropriate frame. Press 'c' to continue to the frame "
                "of this print."
            )
            p(
                "Please open an issue on github and include the trace. It's "
                "tremendously useful for me as there are many small bugs that "
                "can be quickly squashed if users just told me about it :)"
            )
            [p(line) for line in traceback.format_tb(exc_traceback)]
            p(str(exc_type) + " : " + str(exc_value))
            if hasattr(exc_value, "__cause__") and hasattr(
                exc_value.__cause__, "__traceback__"
            ):
                p("Detected a cause to the exception, opening the cause first")
                pdb.post_mortem(exc_value.__cause__.__traceback__)
                p("Out of the __cause__, now debugging the higher traceback:")
            pdb.post_mortem(exc_traceback)
            p("You are now in the exception handling frame.")
            breakpoint()
            sys.exit(1)

    sys.excepthook = handle_exception
    faulthandler.enable()
