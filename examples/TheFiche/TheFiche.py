import json
import LogseqMarkdownParser
import time
from WDoc import WDoc
import fire
from pathlib import Path, PosixPath
from datetime import datetime
from beartype import beartype
from typing import Union

VERSION = "0.1"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"


@beartype
class TheFiche:
    def __init__(
        self,
        query: str,
        logseq_page: Union[str, PosixPath],
        overwrite: bool = False,
        top_k: int = 300,
        **kwargs,
        ):
        assert "top_k" not in kwargs
        logseq_page = Path(logseq_page)
        if not overwrite:
            assert not (logseq_page).exists()

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
        props = {
            "collapsed": "false",
            "block_type": "WDoc_the_fiche",
            "WDoc_version": instance.VERSION,
            "WDoc_model": f"{instance.modelname} of {instance.modelbackend}",
            "WDoc_evalmodel": f"{instance.query_eval_modelname} of {instance.query_eval_modelbackend}",
            "WDoc_n_docs_found": len(fiche["unfiltered_docs"]),
            "WDoc_n_docs_filtered": len(fiche["filtered_docs"]),
            "WDoc_n_docs_used": len(fiche["relevant_filtered_docs"]),
            "WDoc_n_combine_steps": len(fiche["all_intermediate_answers"]),
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

        content = LogseqMarkdownParser.LogseqPage(
            text,
            check_parsing=False,
            verbose=True,
        )
        content.page_properties.update(props)
        assert content.page_properties

        content.export_to(
            file_path=logseq_page.absolute(),
            overwrite=overwrite,
            allow_empty=False,
        )
        logseq_page.write_text(
            logseq_page.read_text().replace("  ", "\t")
        )


if __name__ == "__main__":
    fire.Fire(TheFiche)
