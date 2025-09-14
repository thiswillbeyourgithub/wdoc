"""
Shared utilities for query and search tasks.
"""

import asyncio
import copy
from beartype.typing import Callable, List, Tuple

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.runnables import chain
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.misc import log_and_time_fn
from wdoc.utils.tasks.query import parse_eval_output


@log_and_time_fn
def split_query_parts(query: str) -> Tuple[str, str]:
    """
    Split query into parts for embedding search and answering.

    If the query contains ">>>>", splits it into:
    - query_for_embedding: part before >>>>
    - query_to_answer: part after >>>>

    Otherwise returns the same query for both purposes.

    Parameters
    ----------
    query : str
        The input query string

    Returns
    -------
    Tuple[str, str]
        A tuple of (query_for_embedding, query_to_answer)

    Raises
    ------
    AssertionError
        If query contains more than one occurrence of ">>>>"
    """
    if ">>>>" in query:
        sp = query.split(">>>>")
        assert len(sp) == 2, "The query must contain a maximum of 1 occurence of '>>>>'"
        query_fe = sp[0].strip()
        query_an = sp[1].strip()
    else:
        query_fe, query_an = copy.copy(query), copy.copy(query)

    return query_fe, query_an


@log_and_time_fn
def create_evaluate_doc_chain(
    eval_llm,
    eval_llm_params: List[str],
    query_eval_check_number: int,
    eval_cache_wrapper: Callable,
    prompts,
):
    """
    Create a document evaluation chain for assessing document relevance.

    This function creates a chain that evaluates documents for relevance to a query
    using an LLM. It handles different model configurations and caching strategies.

    Parameters
    ----------
    eval_llm : object
        The evaluation LLM instance
    eval_llm_params : List[str]
        List of supported parameters for the evaluation LLM
    query_eval_check_number : int
        Number of evaluation checks to perform
    eval_cache_wrapper : Callable
        Function to wrap the evaluation for caching
    prompts : object
        Prompts object containing the evaluation prompt

    Returns
    -------
    chain
        A langchain chain object for document evaluation
    """

    @eval_cache_wrapper
    def evaluate_doc_chain(
        inputs: dict,
        query_nb: int = query_eval_check_number,
        eval_model_string: str = eval_llm._get_llm_string(),  # just for caching
        eval_prompt: str = str(prompts.evaluate.to_json()),
    ) -> List[str]:
        if isinstance(eval_llm, FakeListChatModel):
            outputs = ["10" for i in range(query_eval_check_number)]
            new_p = 0
            new_c = 0
            new_r = 0

        elif "n" in eval_llm_params or query_eval_check_number == 1:

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
                out = eval_llm._generate_with_cache(
                    prompts.evaluate.format_messages(**inputs),
                    request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                )
                outputs = _parse_outputs(out)
            except Exception:  # retry without cache
                logger.debug(
                    "Failed to run eval_llm on an input. Retrying without cache."
                )
                out = eval_llm._generate(
                    prompts.evaluate.format_messages(**inputs),
                    request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                )
                outputs = _parse_outputs(out)

            if out.llm_output:
                new_p = out.llm_output["token_usage"]["prompt_tokens"]
                new_c = out.llm_output["token_usage"]["completion_tokens"]
                new_r = out.llm_output["token_usage"]["total_tokens"] - new_p - new_c
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
                    val = await eval_llm._agenerate_with_cache(
                        prompts.evaluate.format_messages(**subinputs),
                        request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                    )
                except Exception:  # retry without cache
                    val = await eval_llm._agenerate(
                        prompts.evaluate.format_messages(**subinputs),
                        request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT,
                    )
                return val

            outs = [do_eval(inputs) for i in range(query_eval_check_number)]
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
                        out.llm_output["token_usage"]["total_tokens"] - new_p - new_c
                    )
            assert outputs, "No generations found by query eval llm"
            outputs = [parse_eval_output(o) for o in outputs]

        if len(outputs) < query_eval_check_number and len(outputs) == 1:
            logger.warning(
                f"query eval model produced 1 output instead of {query_eval_check_number}). Output: '{outputs}'\nThis is usually because the model is wrongly specified by litellm as having a modifiable `n` parameter. To avoid this use another model or set the query_eval_check_number to 1."
            )
            if "n" in eval_llm_params:
                eval_llm_params.remove("n")
            outputs = outputs * query_eval_check_number
        assert (
            len(outputs) == query_eval_check_number
        ), f"Query eval model produced an unexpected number of outputs ({outputs} but expected {query_eval_check_number} outputs).\nInputs: {inputs}'"

        eval_llm.callbacks[0].prompt_tokens += new_p
        eval_llm.callbacks[0].completion_tokens += new_c
        eval_llm.callbacks[0].internal_reasoning_tokens += new_r
        eval_llm.callbacks[0].total_tokens += new_p + new_c + new_r
        if eval_llm.callbacks[0].pbar:
            eval_llm.callbacks[0].pbar[-1].update(1)
        return outputs

    evaluate_doc_chain = chain(evaluate_doc_chain)
    return evaluate_doc_chain
