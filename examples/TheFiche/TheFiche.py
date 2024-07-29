"""
TheFiche: A module for generating and exporting Logseq pages based on WDoc queries.

This module provides functionality to create structured Logseq pages
with content generated from WDoc queries, including metadata and properties.
"""

import json
from LogseqMarkdownParser import LogseqBlock, LogseqPage
import time
from WDoc import WDoc
import fire
from pathlib import Path, PosixPath
from datetime import datetime
from beartype import beartype
from typing import Union

VERSION = "0.3"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"


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
        **kwargs,
        ):
        """
        Initialize a TheFiche instance and generate a Logseq page.

        Args:
            query (str): The query to be processed by WDoc.
            logseq_page (Union[str, PosixPath]): The path to the Logseq page file.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False. If False, will append to the file instead of overwriting.
            top_k (int, optional): The number of top documents to consider. Defaults to 300.
            **kwargs: Additional keyword arguments to pass to WDoc.

        Raises:
            AssertionError: If the file exists and overwrite is False, or if the ratio of used/found documents is too high.
        """
        assert "top_k" not in kwargs
        logseq_page = Path(logseq_page)

        instance = WDoc(
            task="query",
            import_mode=True,
            query=query,
            top_k=top_k,
            **kwargs,
        )
        fiche = instance.query_task(query=query)

        all_kwargs = kwargs.copy()
        all_kwargs.update({"top_k": top_k})

        if len(fiche["all_intermediate_answers"]) > 1:
            extra = '->'.join(
                [str(len(ia)) for ia in fiche["all_intermediate_answers"]]
            )
            extra = f"({extra})"
        else:
            extra = ""

        props = {
            "collapsed": "false",
            "block_type": "WDoc_the_fiche",
            "WDoc_version": instance.VERSION,
            "WDoc_model": f"{instance.modelname} of {instance.modelbackend}",
            "WDoc_evalmodel": f"{instance.query_eval_modelname} of {instance.query_eval_modelbackend}",
            "WDoc_evalmodel_check_nb": instance.query_eval_check_number,
            "WDoc_cost": instance.latest_cost,
            "WDoc_n_docs_found": len(fiche["unfiltered_docs"]),
            "WDoc_n_docs_filtered": len(fiche["filtered_docs"]),
            "WDoc_n_docs_used": len(fiche["relevant_filtered_docs"]),
            "WDoc_n_combine_steps": str(len(fiche["all_intermediate_answers"])) + " " + extra,
            "WDoc_kwargs": json.dumps(all_kwargs),
            "the_fiche_version": VERSION,
            "the_fiche_date": today,
            "the_fiche_timestamp": int(time.time()),
            "the_fiche_query": query,
        }

        n_used = props["WDoc_n_docs_used"]
        n_found = props["WDoc_n_docs_found"]
        ratio = n_used / n_found
        assert ratio <= 0.9, (
            f"Ratio of number of docs used/found is {ratio:.1f} > 0.9 ({n_used}, {n_found})"
        )

        text = fiche["final_answer"]
        assert text.strip()

        content = LogseqPage(
            text,
            check_parsing=False,
            verbose=False,
        )
        content.page_properties.update(props)
        assert content.page_properties

        # add documents source
        doc_hash = {
            d.metadata["content_hash"][:5]: d.metadata
            for d in fiche["filtered_docs"]
        }
        content.blocks.append(LogseqBlock("- # Sources"))
        for dh, dm in doc_hash.items():
            new_block = LogseqBlock(f"{dh}:")
            for k, v in dm.items():
                new_block.properties[k] = v
            content.blocks.append(new_block)

        # save to file
        if not logseq_page.absolute().exists():
            content.export_to(
                file_path=logseq_page.absolute(),
                overwrite=False,
                allow_empty=False,
            )
        else:
            prev_content = LogseqPage(
                logseq_page.read_text(),
                check_parsing=False,
                verbose=False,
            )
            prev_content.blocks.append(LogseqBlock("- ---"))
            new_block = LogseqBlock(content=f"# {today}")
            for k, v in props.items():
                new_block.properties[k] = v
            prev_content.blocks.append(new_block)
            for block in content.blocks:
                block.indentation_level += 2
                prev_content.blocks.append(block)
            prev_content.export_to(
                file_path=logseq_page.absolute(),
                overwrite=True,
                allow_empty=False,
            )
        logseq_page.write_text(
            logseq_page.read_text().replace("  ", "\t")
        )


if __name__ == "__main__":
    fire.Fire(TheFiche)
