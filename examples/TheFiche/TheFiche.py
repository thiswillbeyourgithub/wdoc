"""
TheFiche: A module for generating and exporting Logseq pages based on WDoc queries.

This module provides functionality to create structured Logseq pages
with content generated from WDoc queries, including metadata and properties.
"""

import json
from LogseqMarkdownParser import LogseqBlock, LogseqPage, parse_file, parse_text
import time
from textwrap import indent
from WDoc import WDoc
import fire
from pathlib import Path, PosixPath
from datetime import datetime
from beartype import beartype
from typing import Union, Tuple
from loguru import logger
from joblib import Memory
import re

# logger
logger.add(
    "logs.txt",
    rotation="100MB",
    retention=5,
    format='{time} {level} {thread} TheFiche {process} {function} {line} {message}',
    level="DEBUG",
    enqueue=True,
    colorize=False,
)
def p(*args):
    "simple way to log to logs.txt and to print"
    print(*args)
    logger.info(*args)

VERSION = "0.5"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

mem = Memory(".cache", verbose=False)

def run_wdoc(query: str, kwargs2: dict) -> Tuple[WDoc, dict]:
    "call to wdoc, optionaly cached"
    instance = WDoc(
        task="query",
        import_mode=True,
        query=query,
        **kwargs2,
    )
    fiche = instance.query_task(query=query)
    assert "error" not in fiche

    if len(fiche["all_intermediate_answers"]) > 1:
        extra = '->'.join(
            [str(len(ia)) for ia in fiche["all_intermediate_answers"]]
        )
        extra = f"({extra})"
    else:
        extra = ""

    props = {
        "block_type": "WDoc_the_fiche",
        "WDoc_version": instance.VERSION,
        "WDoc_model": f"{instance.modelname} of {instance.modelbackend}",
        "WDoc_evalmodel": f"{instance.query_eval_modelname} of {instance.query_eval_modelbackend}",
        "WDoc_evalmodel_check_nb": instance.query_eval_check_number,
        "WDoc_cost": f"{float(instance.latest_cost):.5f}",
        "WDoc_n_docs_found": len(fiche["unfiltered_docs"]),
        "WDoc_n_docs_filtered": len(fiche["filtered_docs"]),
        "WDoc_n_docs_used": len(fiche["relevant_filtered_docs"]),
        "WDoc_n_combine_steps": str(len(fiche["all_intermediate_answers"])) + " " + extra,
        "WDoc_kwargs": json.dumps(kwargs2),
        "the_fiche_version": VERSION,
        "the_fiche_date": today,
        "the_fiche_timestamp": int(time.time()),
        "the_fiche_query": query,
    }
    return fiche, props

@beartype
class TheFiche:
    """
    A class for generating and exporting Logseq pages based on WDoc queries.

    This class encapsulates the process of creating a Logseq page with content
    generated from a WDoc query, including metadata and properties.
    """
    def __init__(
        self,
        query: str,
        logseq_page: Union[str, PosixPath],
        overwrite: bool = False,
        top_k: int = 300,
        sources_location: str = "as_pages",
        sources_ref_as_prop: bool = False,
        use_cache: bool = True,
        **kwargs,
        ):
        """
        Initialize a TheFiche instance and generate a Logseq page.

        Args:
            query (str): The query to be processed by WDoc.
            logseq_page (Union[str, PosixPath]): The path to the Logseq page file.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False. If False, will append to the file instead of overwriting. Else, will also overwrite sources if present.
            top_k (int, optional): The number of top documents to consider. Defaults to 300.
            sources_location (str): If 'as_pages', will store each source as its own page in a 'TheFiche___' namespace. If 'below', sources will be written at the end of the page.
            sources_ref_as_prop (bool): if True, make sure the sources appear as block properties instead of leaving them as is.
            use_cache (bool): set to False to bypass the cache
            **kwargs: Additional keyword arguments to pass to WDoc.

        Raises:
            AssertionError: If the file exists and overwrite is False, or if the ratio of used/found documents is too high.
        """
        p(f"Starting TheFiche with args: {query}, {logseq_page}, {overwrite}, {top_k}, {sources_location} and kwargs: {kwargs}")
        assert "top_k" not in kwargs
        assert sources_location in ["as_pages", "below"]
        logseq_page = Path(logseq_page)

        all_kwargs = kwargs.copy()
        all_kwargs.update({"top_k": top_k})

        # posixpath can't be serialized to json
        for k, v in all_kwargs.items():
            if isinstance(v, PosixPath):
                all_kwargs[k] = str(v)

        if use_cache:
            cached = mem.cache(run_wdoc)
        else:
            cached = run_wdoc

        fiche, props = cached(query=query, kwargs2=all_kwargs)

        p(f"Fiche properties: {props}")

        n_used = props["WDoc_n_docs_used"]
        n_found = props["WDoc_n_docs_found"]
        ratio = n_used / n_found
        assert ratio <= 0.9, (
            f"Ratio of number of docs used/found is {ratio:.1f} > 0.9 ({n_used}, {n_found})"
        )
        assert n_used / top_k <= 0.9, f"Used {n_used} documents but asked {top_k}, you should ask more"

        text = fiche["final_answer"]
        assert text.strip()

        # prepare sources hash dict
        doc_hash = {
            d.metadata["content_hash"][:5]: d.metadata
            for d in fiche["filtered_docs"]
        }
        # discard sources that are not used
        used_hash = []
        for d, dm in doc_hash.items():
            if d in text:
                used_hash.append(d)
            # note: we replace content_hash by all_hash to avoid collisions, as sources are originally refered only by the first 5 letters of content_hash, we now use 5 of all_hash
            new_h = dm["all_hash"][:5]
            text = text.replace(d, f" [[{new_h}]] ")
        text = text.replace(" , ", ", ")
        if not used_hash:
            p("No documents seem to be sourced")

        # create logseq page but don't save it yet
        if not text.startswith("- "):
            text = "- " + text
        content = parse_text(text)
        content.page_properties.update(props)
        assert content.page_properties

        # add documents source
        if sources_location == "below":
            content.blocks.append(LogseqBlock("- # Sources\n  collapsed:: true"))
            for dh, dm in doc_hash.items():
                if dh not in used_hash:
                    continue
                cont = [
                    fd.page_content for fd in fiche["filtered_docs"]
                    if fd.metadata["all_hash"] == dm["all_hash"]
                ]
                assert len(cont) == 1, f"Found multiple sources with the same hash! {cont}"
                cont = cont[0].strip()
                new_block = LogseqBlock(f"- [[{dh}]]: {indent(cont, '  ').strip()}")
                for k, v in dm.items():
                    new_block.set_property(k, v)
                diff = (4 - new_block.indentation_level % 4)
                new_block.indentation_level += 4 + diff
                content.blocks.append(new_block)

        elif sources_location == "as_pages":
            logseq_dir = logseq_page.parent
            for dh, dm in doc_hash.items():
                if dh not in used_hash:
                    continue
                source_path = logseq_dir / ("TheFicheSources___" + dm["all_hash"] + ".md")
                cont = [
                    fd.page_content for fd in fiche["filtered_docs"]
                    if fd.metadata["all_hash"] == dm["all_hash"]
                ]
                assert len(cont) == 1, f"Found multiple sources with the same hash! {cont}"
                cont = cont[0].strip()
                new_h = dm["all_hash"][:5]

                if overwrite:
                    source_path.unlink(missing_ok=True)

                if source_path.exists():
                    p(f"Warning: a source for {dh} ({new_h}) already exists at {source_path}")
                    prev_source = parse_file(source_path)
                    if prev_source.page_properties["content_hash"][:5].startswith(dh):
                        if not prev_source.page_properties["all_hash"].startswith(new_h):
                            raise Exception(f"Found previous source with the same name and overlapping content but different alias: page: {source_path}, dh: {dh}, new_h: {new_h}")
                    else:
                        raise Exception(f"Found previous source with the same name but does not contain the new content: {source_path}, dh: {dh}, new_h: {new_h}")
                else:
                    logger.info(f"Creating source page for {dh} ({new_h}) at {source_path}")
                    cont = indent(cont, "  ").strip()
                    if not cont.startswith("- "):
                        cont = "- " + cont
                    source_page = parse_text(cont)
                    source_page.page_properties.update(dm.copy())
                    source_page.page_properties["alias"] = new_h
                    source_page.export_to(
                        file_path=source_path.absolute(),
                        overwrite=False,
                        allow_empty=False,
                    )
        else:
            raise ValueError(sources_location)

        if sources_ref_as_prop:
            # make it so that the sources appear as block properties instead of in the content
            for ib, b in enumerate(content.blocks):
                for dh, dm in doc_hash.items():
                    new_h = dm["all_hash"][:5]
                    assert dh not in b.content
                    if new_h in b.content:
                        assert f"[[{new_h}]]" in b.content
                        content.blocks[ib].content = re.sub(
                            f"[, ]*\[\[{new_h}\]\][, ]*",
                            "",
                            b.content,
                        )
                        assert f"[[{new_h}]]" not in b.content
                        assert new_h not in content.blocks[ib].content
                        content.blocks[ib].content = content.blocks[ib].content.replace(" []", "")
                        content.blocks[ib].content = content.blocks[ib].content.replace("[]", "")
                        content.blocks[ib].set_property("source", f"[[{new_h}]]")


        # save to file
        if (not logseq_page.absolute().exists()) or overwrite:
            content.export_to(
                file_path=logseq_page.absolute(),
                overwrite=False if not overwrite else True,
                allow_empty=False,
            )
        else:
            prev_content = parse_file(logseq_page)
            prev_content.blocks.append(LogseqBlock("- ---"))

            new_block = LogseqBlock(f"- # {today}")
            for k, v in content.page_properties.items():
                new_block.set_property(k, v)

            prev_content.blocks.append(new_block)
            for block in content.blocks:
                diff = (4 - block.indentation_level % 4)
                block.indentation_level += 4 + diff
                prev_content.blocks.append(block)
            prev_content.export_to(
                file_path=logseq_page.absolute(),
                overwrite=True,
                allow_empty=False,
            )
        p("Done")


if __name__ == "__main__":
    fire.Fire(TheFiche)
