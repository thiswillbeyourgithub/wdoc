"""
Filter functions for VectorStore documents.

This module provides functions to filter VectorStore documents (e.g., FAISS) on the fly,
since the langchain implementation does not support native filtering. These functions
allow filtering by regex patterns on document content and metadata.
"""

from tqdm import tqdm
import re
import time
from beartype.typing import Tuple, Callable
from langchain_core.vectorstores.base import VectorStore
from loguru import logger

from wdoc.utils.env import env, is_out_piped


def filter_vectorstore(
    loaded_embeddings: VectorStore,
    cli_kwargs: dict,
) -> VectorStore:
    if "filter_metadata" in cli_kwargs:
        filter_meta = create_metadata_filter(
            loaded_embeddings=loaded_embeddings,
            cli_kwargs=cli_kwargs,
        )
    else:

        def filter_meta(meta: dict) -> bool:
            return True

    if "filter_content" in cli_kwargs:
        filter_cont = create_content_filter(cli_kwargs)
    else:

        def filter_cont(cont: str) -> bool:
            return True

    # check filtering is valid
    checked = 0
    good = 0
    ids_to_del = []
    for doc_id, doc in tqdm(
        loaded_embeddings.docstore._dict.items(),
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

    # commented because it's taking quite long
    # # first store the docstore before altering it to allow
    # # unfiltering in the prompt
    # start_time = time.time()
    # unfiltered_docstore_bytes = loaded_embeddings.serialize_to_bytes()
    # serialize_time = time.time() - start_time
    # logger.debug(f"Serializing unfiltered docstore took {serialize_time:.3f} seconds")

    # directly remove the filtered documents from the docstore
    start_time = time.time()
    status = loaded_embeddings.delete(ids_to_del)
    delete_time = time.time() - start_time
    logger.debug(f"Deleting {len(ids_to_del)} documents took {delete_time:.3f} seconds")

    # checking deletions went well
    if status is False:
        raise Exception("Vectorstore filtering failed")
    elif status is None:
        raise Exception("Vectorstore filtering not implemented")
    assert len(loaded_embeddings.docstore._dict) == checked - len(
        ids_to_del
    ), "Something went wrong when deleting filtered out documents"
    assert len(
        loaded_embeddings.docstore._dict
    ), "Something went wrong when deleting filtered out documents: no document left"
    assert len(loaded_embeddings.docstore._dict) == len(
        loaded_embeddings.index_to_docstore_id
    ), "Something went wrong when deleting filtered out documents"

    return loaded_embeddings


def create_metadata_filter(
    loaded_embeddings: VectorStore,
    cli_kwargs: dict,
) -> Callable:
    # get the list of all metadata to see if a filter was not misspelled
    all_metadata_keys = set()
    for doc in tqdm(
        loaded_embeddings.docstore._dict.values(),
        desc="gathering metadata keys",
        unit="doc",
        disable=(not env.WDOC_VERBOSE) or is_out_piped,
    ):
        for k in doc.metadata.keys():
            all_metadata_keys.add(k)
    assert (
        all_metadata_keys
    ), "No metadata keys found in any metadata, something went wrong!"

    if isinstance(cli_kwargs["filter_metadata"], str):
        filter_metadata = cli_kwargs["filter_metadata"].split(",")
    else:
        filter_metadata = cli_kwargs["filter_metadata"]
    assert isinstance(
        filter_metadata, list
    ), f"filter_metadata must be a list, not {cli_kwargs['filter_metadata']}"

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

    # check that all key filter indeed match metadata keys
    for k in (
        filters_k_plus + filters_k_minus + filters_b_plus_keys + filters_b_minus_keys
    ):
        assert any(
            k.match(str(key)) for key in all_metadata_keys
        ), f"Key {k} didn't match any key in the metadata"

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

    return filter_meta


def create_content_filter(
    cli_kwargs: dict,
) -> Callable:
    if isinstance(cli_kwargs["filter_content"], str):
        filter_content = cli_kwargs["filter_content"].split(",")
    else:
        filter_content = cli_kwargs["filter_content"]
    assert isinstance(
        filter_content, list
    ), f"filter_content must be a list, not {cli_kwargs['filter_content']}"

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

    def filter_cont(cont: str) -> bool:
        if not all(inc.match(cont) for inc in filters_cont_plus):
            return False
        if any(exc.match(cont) for exc in filters_cont_minus):
            return False
        return True

    return filter_cont
