"""
Simple script to create a filtered deck from a wdoc search
"""
import os
from wdoc import wdoc
from wdoc.utils.typechecker import optional_typecheck
import fire
from loguru import logger
from typing import List
from langchain.docstore.document import Document

from py_ankiconnect import PyAnkiconnect

VERSION = "0.1"

# logger
logger.add(
    "logs.txt",
    rotation="100MB",
    retention=5,
    format='{time} {level} {thread} FilteredDeckCreator {process} {function} {line} {message}',
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
        deckname: str,
        query: str,
        top_k: int,
        filtered_deck_query: str = "",
        new_tag: str = "",
        reschedule: bool = False,
        sort_order: int = 8,
        create_empty: bool = False,
        **kwargs,
        ) -> None:
        akc("sync")
        p(f"Started with query '{query}'")
        decknames = akc("deckNames")
        assert deckname not in decknames, f"Deckname {deckname} is already used. You have to delete it manually"

        instance = wdoc(
            task="search",
            import_mode=True,
            query=query,
            top_k=top_k,
            **kwargs,
        )
        found = instance.query_task(query=query)

        cids = self.find_anki_docs(
            docs=found["relevant_filtered_docs"]
        )

        query = filtered_deck_query
        query += " (cid:" + ",".join([str(c) for c in cids]) + ")"

        if new_tag:
            nids = akc("cardsToNotes", cards=[int(c) for c in cids])
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
    def find_anki_docs(self, docs: List[Document]) -> List[int]:
        """
        goes through the metadata of each langchain Document to find which
        correspond to anki cards
        """
        anki_docs = []
        for doc in docs:
            if any("anki_" in k for k in doc.metadata.keys()):
                anki_docs.append(doc)
                continue
        assert anki_docs, "No anki notes found in documents"

        present_anki_docs = []
        cids = []
        for idoc, doc in enumerate(anki_docs):
            if "anki_cid" in doc.metadata:
                cid = doc.metadata["anki_cid"]
                if akc("findCards", query=f"cid:{cid}"):
                    present_anki_docs.append(doc)
                    cids.append(cid)
            elif "anki_nid" in doc.metadata:
                nid = doc.metadata["anki_nid"]
                temp_cids =  akc("findCards", query=f"nid:{nid}")
                if temp_cids:
                    present_anki_docs.append(doc)
                    cids.extend(temp_cids)

        assert present_anki_docs, "No anki notes after filtering"
        assert cids, "No cids found"
        p(f"Found cids:{cids}")

        return cids

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
        logger.info(f"Creating filtered deck {deckname}, {search_query}, {gather_count}, {reschedule}, {sort_order}, {create_empty}")
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
