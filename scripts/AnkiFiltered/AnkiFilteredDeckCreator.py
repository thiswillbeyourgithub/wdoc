"""
Simple script to create a filtered deck from a wdoc search
"""

import os
from typing import List, Tuple

import fire
from langchain.docstore.document import Document
from loguru import logger
from py_ankiconnect import PyAnkiconnect

from wdoc import wdoc
from wdoc.utils.env import WDOC_DEFAULT_MODELNAME
from wdoc.utils.typechecker import optional_typecheck

VERSION = "1.1"

# logger
logger.add(
    "logs.txt",
    rotation="100MB",
    retention=5,
    format="{time} {level} {thread} FilteredDeckCreator {process} {function} {line} {message}",
    level="DEBUG",
    enqueue=True,
    colorize=False,
)


def p(*args):
    "simple way to log to logs.txt and to print"
    print(*args)
    logger.info(*args)


# default anki host
if "PYTHON_PY_ANKICONNECT_HOST" in os.environ:
    host = os.environ["PYTHON_PY_ANKICONNECT_HOST"]
else:
    host: str = "http://localhost"
if "PYTHON_PY_ANKICONNECT_PORT" in os.environ:
    port = int(os.environ["PYTHON_PY_ANKICONNECT_PORT"])
else:
    port = 8775

akc = PyAnkiconnect(
    default_host=host,
    default_port=port,
)


@optional_typecheck
class FilteredDeckCreator:
    def __init__(
        self,
        deckname: str,  # will have 'wdoc_filtered_deck::' prepended to it
        query: str,
        task: str = "search",  # could also be 'query', would be more expensive but would use the large LLM
        filtered_deck_query: str = "",
        new_tag: str = "",  # will have 'wdoc_filtered_deck::' prepended to each space separated tag
        reschedule: bool = False,
        sort_order: int = 8,
        create_empty: bool = False,
        query_eval_modelname: str = WDOC_DEFAULT_MODELNAME,  # by default, use the same model as we use normally for querying
        **kwargs,
    ) -> None:
        akc("sync")
        p(f"Started with query '{query}'")
        deckname = "wdoc_filtered_deck::" + deckname
        decknames = akc("deckNames")
        assert (
            deckname not in decknames
        ), f"Deckname {deckname} is already used. You have to delete it manually"

        instance = wdoc(
            query_eval_modelname=query_eval_modelname,
            task=task,
            import_mode=True,
            query=query,
            **kwargs,
        )
        found = instance.query_task(query=query)

        if "relevant_filtered_docs" in found:
            docs = found["relevant_filtered_docs"]
        else:
            docs = found["filtered_docs"]
        nids, cids = self.find_anki_docs(
            docs=docs,
        )

        query = filtered_deck_query
        query += " (nid:" + ",".join([str(n) for n in nids]) + ")"

        if new_tag:
            if " " not in new_tag:
                new_tag = "wdoc_filtered_deck::" + new_tag
            else:
                new_tag = new_tag.replace(" ", " wdoc_filtered_deck::")
            p(f"Adding tags {new_tag} to nids:{nids}")
            akc("addTags", notes=nids, tags=new_tag)
            p("Done adding tags")

        p("Creating filtered deck")
        self.create_filtered_deck(
            deckname=deckname,
            search_query=query,
            gather_count=len(cids) + 1,
            reschedule=reschedule,
            sort_order=sort_order,
            create_empty=create_empty,
        )
        p(f"Done, created filtered deck {deckname}")
        akc("sync")

    @classmethod
    def find_anki_docs(self, docs: List[Document]) -> Tuple[List[int], List[int]]:
        """
        goes through the metadata of each langchain Document to find which
        correspond to anki cards.
        Note that the order will NOT be respected (i.e. nids, cids and documents
        will all have their own order and length)
        Returns [ nids, cids ]
        """
        anki_docs = []
        for doc in docs:
            if any("anki_" in k for k in doc.metadata.keys()):
                anki_docs.append(doc)
                continue
        assert anki_docs, "No anki notes found in documents"

        present_anki_docs = []
        cids = []
        nids = []
        for idoc, doc in enumerate(anki_docs):
            if "anki_cid" in doc.metadata:
                cid = int(doc.metadata["anki_cid"])
                if akc("findCards", query=f"cid:{cid}"):
                    present_anki_docs.append(doc)
                    cids.append(int(cid))
            if "anki_nid" in doc.metadata:
                nid = int(doc.metadata["anki_nid"])
                nids.append(nid)
                temp_cids = [int(c) for c in akc("findCards", query=f"nid:{nid}")]
                if temp_cids:
                    present_anki_docs.append(doc)
                    cids.extend(temp_cids)

        assert present_anki_docs, "No anki notes after filtering"
        assert cids, "No cids found"
        cids = list(set(cids))
        nids = list(set(nids))
        p(f"Found cids:{cids}")

        p(f"Ratio of cid per document: {len(cids)/len(docs):.4f}")
        p(f"Ratio of nid per document: {len(nids)/len(docs):.4f}")

        return nids, cids

    @classmethod
    def create_filtered_deck(
        self,
        deckname: str,
        search_query: str,
        gather_count: int = 100,
        reschedule: bool = False,
        sort_order: int = 8,
        create_empty: bool = False,
    ) -> None:
        logger.info(
            f"Creating filtered deck {deckname}, {search_query}, {gather_count}, {reschedule}, {sort_order}, {create_empty}"
        )
        akc(
            action="createFilteredDeck",
            newDeckName=deckname,
            searchQuery=search_query,
            gatherCount=gather_count,
            reschedule=reschedule,
            sortOrder=sort_order,
            createEmpty=create_empty,
        )


if __name__ == "__main__":
    fire.Fire(FilteredDeckCreator)
