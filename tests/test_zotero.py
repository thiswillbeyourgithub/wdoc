"""Tests for the `zotero` recursive filetype.

The `basic` tests use a fake pyzotero client so they need no network and no
Zotero install. The `api` test needs real credentials (or a running local
Zotero) and a collection name provided via the ZOTERO_COLLECTION_NAME env var.
"""

import os
import sys

import pytest

os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"
os.environ["WDOC_TYPECHECKING"] = "crash"

from wdoc.utils.load_recursive import parse_zotero, recursive_types_func_mapping
from wdoc.utils.loaders import zotero as zmod
from wdoc.utils.misc import DocDict


# --- a minimal fake pyzotero client -----------------------------------------

_ITEM = {
    "key": "ITEM1",
    "data": {
        "itemType": "journalArticle",
        "title": "Great Paper",
        "creators": [{"firstName": "Ada", "lastName": "Lovelace"}],
        "date": "1843",
        "DOI": "10.1/great",
        "tags": [{"tag": "cs"}, {"tag": "history"}],
        "abstractNote": "An abstract about analytical engines.",
        "url": "http://example.org/paper",
    },
    "links": {"alternate": {"href": "https://zotero.org/users/1/items/ITEM1"}},
}

_ATTACHMENT = {
    "key": "ATT1",
    "data": {
        "itemType": "attachment",
        "linkMode": "imported_file",
        "contentType": "application/pdf",
        "filename": "paper.pdf",
        "title": "paper.pdf",
    },
}

_NOTE = {
    "key": "NOTE1",
    "data": {"itemType": "note", "note": "<p>my handwritten note</p>"},
}


class FakeZot:
    """Just enough of the pyzotero surface used by the zotero helpers."""

    def everything(self, x):
        return x

    def collections(self, limit=None):
        return [{"key": "COL1", "data": {"name": "Papers", "parentCollection": False}}]

    def collection_items(self, key):
        return [_ITEM] if key == "COL1" else []

    def top(self):
        return [_ITEM]

    def items(self, **kwargs):
        return [_ITEM]

    def item(self, key):
        return _ITEM

    def children(self, key):
        return [_ATTACHMENT, _NOTE] if key == "ITEM1" else []

    def file(self, key):
        return b"%PDF-1.4 fake pdf bytes"

    def fulltext_item(self, key):
        return {"content": "the indexed full text body"}

    def searches(self):
        return [
            {
                "key": "S1",
                "data": {
                    "name": "MySearch",
                    "conditions": [{"condition": "tag", "value": "cs"}],
                },
            }
        ]


@pytest.fixture
def fake_client(monkeypatch):
    fake = FakeZot()
    monkeypatch.setattr(zmod, "get_zotero_client", lambda **kw: fake)
    return fake


# --- selector parsing (pure, no client) -------------------------------------


@pytest.mark.basic
def test_zotero_selector_parsing():
    assert zmod.parse_selector("library") == ("library", None)
    assert zmod.parse_selector("*") == ("library", None)
    assert zmod.parse_selector("tag:foo, bar") == ("tag", ["foo", "bar"])
    assert zmod.parse_selector("items:AAA,BBB") == ("items", ["AAA", "BBB"])
    assert zmod.parse_selector("key:AAA") == ("items", ["AAA"])
    assert zmod.parse_selector("search:My Search") == ("search", "My Search")
    assert zmod.parse_selector("Research/ML/Papers") == (
        "collection",
        "Research/ML/Papers",
    )
    assert zmod.parse_selector("collection:Research%%ML") == (
        "collection",
        "Research%%ML",
    )
    # nested path splitting accepts both separators
    assert zmod._split_collection_path("A%%B/C") == ["A", "B", "C"]


@pytest.mark.basic
def test_zotero_registered():
    assert recursive_types_func_mapping["zotero"] is parse_zotero


@pytest.mark.basic
def test_zotero_docdict_accepts_args():
    """The zotero_* args and title must be valid DocDict keys."""
    d = DocDict(
        {
            "path": "p",
            "filetype": "zotero",
            "title": "t",
            "zotero_library_id": "1",
            "zotero_attachment_text": "fulltext",
            "zotero_include_notes": True,
        },
        strict=True,
    )
    assert d["zotero_attachment_text"] == "fulltext"


# --- fan-out behaviour (fake client) ----------------------------------------


def _filetypes(docs):
    return sorted(d["filetype"] for d in docs)


@pytest.mark.basic
def test_zotero_fanout_wdoc_backend(fake_client):
    """Default backend: a metadata txt doc + a pdf doc handed to wdoc's loader."""
    docs = parse_zotero(cli_kwargs={"path": "Papers"}, path="Papers")
    assert all(isinstance(d, DocDict) for d in docs)
    assert _filetypes(docs) == ["pdf", "txt"]

    # one shared recur_parent_id across the fan-out
    assert len({d["recur_parent_id"] for d in docs}) == 1

    pdf = next(d for d in docs if d["filetype"] == "pdf")
    assert pdf["title"] == "paper.pdf"
    assert os.path.exists(pdf["path"])
    with open(pdf["path"], "rb") as fh:
        assert fh.read().startswith(b"%PDF")

    meta = next(d for d in docs if d["filetype"] == "txt")
    assert meta["title"] == "Great Paper (metadata)"
    text = open(meta["path"]).read()
    assert "Great Paper" in text
    assert "Ada Lovelace" in text
    assert "analytical engines" in text  # abstract included
    assert "10.1/great" in text  # DOI included


@pytest.mark.basic
def test_zotero_fanout_fulltext_backend(fake_client):
    """fulltext backend: attachment becomes a txt doc with the indexed text."""
    docs = parse_zotero(
        cli_kwargs={"path": "Papers"},
        path="Papers",
        zotero_attachment_text="fulltext",
    )
    # metadata txt + fulltext txt, no pdf
    assert _filetypes(docs) == ["txt", "txt"]
    bodies = [open(d["path"]).read() for d in docs]
    assert any("the indexed full text body" in b for b in bodies)
    # the fulltext doc also carries the bib header
    assert any("Great Paper" in b and "the indexed full text body" in b for b in bodies)


@pytest.mark.basic
def test_zotero_fanout_with_notes(fake_client):
    """Notes are off by default and added when requested."""
    without = parse_zotero(cli_kwargs={"path": "Papers"}, path="Papers")
    assert not any(d["title"].endswith("(note)") for d in without)

    withn = parse_zotero(
        cli_kwargs={"path": "Papers"},
        path="Papers",
        zotero_include_notes=True,
    )
    note = next((d for d in withn if d["title"].endswith("(note)")), None)
    assert note is not None
    assert "my handwritten note" in open(note["path"]).read()


@pytest.mark.basic
def test_zotero_fanout_no_metadata(fake_client):
    """Metadata doc can be disabled."""
    docs = parse_zotero(
        cli_kwargs={"path": "Papers"},
        path="Papers",
        zotero_include_metadata=False,
    )
    assert _filetypes(docs) == ["pdf"]


@pytest.mark.basic
def test_zotero_tag_selector(fake_client):
    """A tag selector resolves through the items() endpoint."""
    docs = parse_zotero(cli_kwargs={"path": "tag:cs"}, path="tag:cs")
    assert _filetypes(docs) == ["pdf", "txt"]


@pytest.mark.basic
def test_zotero_fanout_is_resilient(fake_client):
    """Every emitted sub-document is marked loading_failure='warn'.

    Regression: a fan-out over a whole library hits items that fail to load
    (e.g. a sparse bibliographic entry too short for wdoc's minimum length
    check). Those must warn and be skipped rather than crash the entire
    selection, so the fan-out overrides the inherited 'crash' setting.
    """
    docs = parse_zotero(cli_kwargs={"path": "Papers"}, path="Papers")
    assert docs
    assert all(d["loading_failure"] == "warn" for d in docs)


# --- real library (needs creds / local Zotero) ------------------------------


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv) or not os.getenv("ZOTERO_COLLECTION_NAME"),
    reason=(
        "Needs '-m api' and a Zotero collection name in ZOTERO_COLLECTION_NAME, "
        "plus a reachable local Zotero or the standard ZOTERO_API_KEY/"
        "ZOTERO_LIBRARY_ID credentials."
    ),
)
@pytest.mark.parametrize("route", ["fulltext", "wdoc"])
def test_zotero_real_library(route):
    """Both attachment-text routes must fan a real collection out into documents.

    `fulltext` pulls Zotero's indexed text (a txt doc); `wdoc` downloads the
    attachment and runs it through wdoc's own pdf loader. Either way the
    bibliographic metadata doc must be present and carry the item title.
    """
    from wdoc.wdoc import wdoc

    docs = wdoc.parse_doc(
        path=os.environ["ZOTERO_COLLECTION_NAME"],
        filetype="zotero",
        zotero_attachment_text=route,
        format="langchain",
    )
    assert len(docs) > 0
    assert any(d.page_content.strip() for d in docs)
    # every document inherits a non-empty title from its Zotero item
    assert all(d.metadata.get("title") for d in docs)
