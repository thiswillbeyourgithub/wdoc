"""Tests for the `karakeep` recursive filetype.

The `basic` tests use a fake Karakeep client so they need no network and no
Karakeep instance. The `api` test needs real credentials and a small selector
provided via the KARAKEEP_TEST_SELECTOR env var (e.g. a list or tag name).
"""

import os
import sys
import uuid

import pytest

os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"
os.environ["WDOC_TYPECHECKING"] = "crash"

from wdoc.utils.load_recursive import parse_karakeep, recursive_types_func_mapping
from wdoc.utils.loaders import karakeep as kmod
from wdoc.utils.misc import DocDict


# --- a minimal fake Karakeep client -----------------------------------------

_LINK = {
    "id": "BL",
    "modifiedAt": "2024-01-01T00:00:00Z",
    "title": "Great Article",
    "note": "my handwritten note",
    "summary": "an ai summary",
    "tags": [{"id": "T1", "name": "cs"}, {"id": "T2", "name": "history"}],
    "content": {
        "type": "link",
        "url": "http://example.org/article",
        "htmlContent": "<html><body><h1>Hello</h1><p>Body paragraph here.</p></body></html>",
        "author": "Ada Lovelace",
        "publisher": "Analytical Press",
        "datePublished": "1843",
    },
    "assets": [],
}

_TEXT = {
    "id": "BT",
    "modifiedAt": "2024-01-02T00:00:00Z",
    "title": "My Thought",
    "tags": [],
    "content": {"type": "text", "text": "some free text content I wrote down"},
    "assets": [],
}

_ASSET = {
    "id": "BA",
    "modifiedAt": "2024-01-03T00:00:00Z",
    "title": "Saved Paper",
    "tags": [],
    "content": {
        "type": "asset",
        "assetType": "pdf",
        "assetId": "ASSET1",
        "fileName": "paper.pdf",
    },
    "assets": [],
}

_BY_ID = {b["id"]: b for b in (_LINK, _TEXT, _ASSET)}


class FakeKarakeep:
    """Just enough of the KarakeepAPI surface used by the karakeep helpers."""

    def __init__(self):
        self.asset_calls = 0

    def _page(self, bookmarks):
        return {"bookmarks": bookmarks, "nextCursor": None}

    def get_all_lists(self):
        return [{"id": "L1", "name": "Reading", "parentId": None}]

    def get_bookmarks_in_the_list(self, list_id, cursor=None):
        return self._page([_LINK, _TEXT, _ASSET] if list_id == "L1" else [])

    def get_all_tags(self):
        return [{"id": "T1", "name": "cs"}]

    def get_bookmarks_with_the_tag(self, tag_id, cursor=None):
        return self._page([_LINK] if tag_id == "T1" else [])

    def search_bookmarks(self, q, cursor=None):
        return self._page([_LINK])

    def get_a_single_bookmark(self, bookmark_id, **kw):
        return _BY_ID[bookmark_id]

    def get_all_bookmarks(self, cursor=None, favourited=None, archived=None, **kw):
        return self._page([_LINK, _TEXT, _ASSET])

    def get_a_single_asset(self, asset_id):
        self.asset_calls += 1
        return b"%PDF-1.4 fake pdf bytes"


@pytest.fixture
def fake_client(monkeypatch):
    fake = FakeKarakeep()
    monkeypatch.setattr(kmod, "get_karakeep_client", lambda **kw: fake)
    return fake


# --- selector parsing (pure, no client) -------------------------------------


@pytest.mark.basic
def test_karakeep_selector_parsing():
    assert kmod.parse_selector("library") == ("library", None)
    assert kmod.parse_selector("*") == ("library", None)
    assert kmod.parse_selector("favourites") == ("favourites", None)
    assert kmod.parse_selector("archived") == ("archived", None)
    assert kmod.parse_selector("tag:cs") == ("tag", "cs")
    assert kmod.parse_selector("search:hello world") == ("search", "hello world")
    assert kmod.parse_selector("ids:AAA, BBB") == ("ids", ["AAA", "BBB"])
    assert kmod.parse_selector("id:AAA") == ("ids", ["AAA"])
    assert kmod.parse_selector("list:Reading") == ("list", "Reading")
    # a bare value is treated as a list name
    assert kmod.parse_selector("Reading") == ("list", "Reading")


@pytest.mark.basic
def test_karakeep_registered():
    assert recursive_types_func_mapping["karakeep"] is parse_karakeep


@pytest.mark.basic
def test_karakeep_docdict_accepts_args():
    """The karakeep_* args and title must be valid DocDict keys."""
    d = DocDict(
        {
            "path": "p",
            "filetype": "karakeep",
            "title": "t",
            "karakeep_api_endpoint": "http://localhost/api/v1/",
            "karakeep_content_source": "native",
            "karakeep_verify_ssl": False,
        },
        strict=True,
    )
    assert d["karakeep_content_source"] == "native"


# --- fan-out behaviour (fake client) ----------------------------------------


def _filetypes(docs):
    return sorted(d["filetype"] for d in docs)


@pytest.mark.basic
def test_karakeep_fanout_link(fake_client):
    """A link bookmark fans out to a local_html doc with header + html."""
    docs = parse_karakeep(cli_kwargs={"path": "ids:BL"}, path="ids:BL")
    assert _filetypes(docs) == ["local_html"]
    doc = docs[0]
    assert doc["title"] == "Great Article"
    assert os.path.exists(doc["path"])
    text = open(doc["path"]).read()
    assert "Great Article" in text  # header
    assert "Ada Lovelace" in text  # author in header
    assert "cs, history" in text  # tags in header
    assert "my handwritten note" in text  # note in header
    assert "<h1>Hello</h1>" in text  # raw html preserved for wdoc to parse


@pytest.mark.basic
def test_karakeep_fanout_text(fake_client):
    """A text bookmark fans out to a txt doc with header + text."""
    docs = parse_karakeep(cli_kwargs={"path": "ids:BT"}, path="ids:BT")
    assert _filetypes(docs) == ["txt"]
    text = open(docs[0]["path"]).read()
    assert "My Thought" in text
    assert "some free text content I wrote down" in text


@pytest.mark.basic
def test_karakeep_fanout_asset_pdf(fake_client):
    """An asset/pdf bookmark fans out to a pdf doc holding the downloaded bytes."""
    docs = parse_karakeep(cli_kwargs={"path": "ids:BA"}, path="ids:BA")
    assert _filetypes(docs) == ["pdf"]
    with open(docs[0]["path"], "rb") as fh:
        assert fh.read().startswith(b"%PDF")


@pytest.mark.basic
def test_karakeep_fanout_list_mixed(fake_client):
    """A list selector fans the three bookmark shapes into three docs."""
    docs = parse_karakeep(cli_kwargs={"path": "list:Reading"}, path="list:Reading")
    assert _filetypes(docs) == ["local_html", "pdf", "txt"]
    # one shared recur_parent_id across the fan-out
    assert len({d["recur_parent_id"] for d in docs}) == 1


@pytest.mark.basic
def test_karakeep_tag_selector(fake_client):
    """A tag selector resolves through the tag endpoint."""
    docs = parse_karakeep(cli_kwargs={"path": "tag:cs"}, path="tag:cs")
    assert _filetypes(docs) == ["local_html"]


@pytest.mark.basic
def test_karakeep_native_skips_link_without_html(fake_client, monkeypatch):
    """content_source=native drops a link bookmark that has no stored html."""
    stripped = dict(_LINK)
    stripped["content"] = {"type": "link", "url": "http://x", "htmlContent": ""}
    monkeypatch.setitem(_BY_ID, "BL", stripped)
    with pytest.raises(AssertionError, match="produced no loadable documents"):
        parse_karakeep(
            cli_kwargs={"path": "ids:BL"},
            path="ids:BL",
            karakeep_content_source="native",
        )
    monkeypatch.setitem(_BY_ID, "BL", _LINK)


@pytest.mark.basic
def test_karakeep_fanout_is_resilient(fake_client):
    """Every emitted sub-document is marked loading_failure='warn'.

    Regression: a fan-out over a whole library hits bookmarks that fail to load
    (e.g. an empty crawl or a corrupt asset). Those must warn and be skipped
    rather than crash the entire selection, so the fan-out overrides the
    inherited 'crash' setting.
    """
    docs = parse_karakeep(cli_kwargs={"path": "library"}, path="library")
    assert docs
    assert all(d["loading_failure"] == "warn" for d in docs)


@pytest.mark.basic
def test_karakeep_content_cache():
    """Resolving the same (id, modifiedAt) twice hits the joblib cache."""
    fake = FakeKarakeep()
    # a fresh id so the on-disk cache is guaranteed cold for this run
    bid = f"CACHE-{uuid.uuid4()}"
    bookmark = {
        "id": bid,
        "modifiedAt": "2024-01-03T00:00:00Z",
        "title": "Saved Paper",
        "tags": [],
        "content": {"type": "asset", "assetType": "pdf", "assetId": "ASSET1"},
        "assets": [],
    }

    def _resolve():
        return kmod.cached_karakeep_content(
            bid,
            bookmark["modifiedAt"],
            "auto",
            None,
            client=fake,
            bookmark=bookmark,
        )

    first = _resolve()
    second = _resolve()
    assert first["filetype"] == "pdf"
    assert second["data"] == first["data"]
    # the asset was downloaded once; the second call came from the cache
    assert fake.asset_calls == 1


# --- real instance (needs creds) --------------------------------------------
#
# Rather than depend on the live library already containing a known bookmark,
# the api test creates its own temporary text bookmark (mirroring
# karakeep_python_api's own create/delete lifecycle tests), runs wdoc's loader
# against it, then deletes it. A text bookmark is deterministic: its content is
# set at creation, unlike a link bookmark whose html is populated asynchronously
# by Karakeep's crawler. This exercises the real schema end to end, so it also
# guards against the basic-test fakes going stale: if Karakeep renames/removes a
# field the loader reads, the round-trip below breaks loudly.

# the bookmark-level keys the loader (and the FakeKarakeep fixtures) rely on
_CONTRACT_KEYS = {"id", "modifiedAt", "title", "tags", "content", "assets"}


@pytest.fixture
def karakeep_api_client():
    """A real KarakeepAPI client, or skip when credentials are absent."""
    endpoint = os.environ.get("KARAKEEP_PYTHON_API_ENDPOINT")
    api_key = os.environ.get("KARAKEEP_PYTHON_API_KEY")
    if " -m api" not in " ".join(sys.argv) or not endpoint or not api_key:
        pytest.skip(
            "Needs '-m api' and the standard KARAKEEP_PYTHON_API_ENDPOINT / "
            "KARAKEEP_PYTHON_API_KEY credentials."
        )
    from karakeep_python_api import KarakeepAPI

    return KarakeepAPI(
        api_endpoint=endpoint,
        api_key=api_key,
        disable_response_validation=True,  # plain dicts, like the loader uses
    )


@pytest.mark.api
def test_karakeep_real_text_bookmark_roundtrip(karakeep_api_client):
    """Create a text bookmark, load it through wdoc's loader, then delete it."""
    import time

    from wdoc.wdoc import wdoc

    client = karakeep_api_client
    marker = f"wdoc karakeep roundtrip {uuid.uuid4()}"
    title = f"wdoc test bookmark {int(time.time())}"
    created = kmod._as_dict(
        client.create_a_new_bookmark(type="text", text=marker, title=title)
    )
    bid = created["id"]
    try:
        # freshness guard: the live bookmark must carry the structural keys our
        # fakes encode, so the basic tests cannot silently drift from reality
        live = kmod._as_dict(client.get_a_single_bookmark(bid))
        missing = _CONTRACT_KEYS - set(live)
        assert not missing, (
            f"Karakeep bookmark schema changed: missing {missing}. The "
            "FakeKarakeep fixtures in this file are out of date."
        )
        assert (live.get("content") or {}).get("type") == "text"

        docs = wdoc.parse_doc(
            path=f"ids:{bid}",
            filetype="karakeep",
            format="langchain",
        )
        assert len(docs) == 1
        assert marker in docs[0].page_content  # the text content came through
        assert title in docs[0].metadata.get("title", "")
    finally:
        client.delete_a_bookmark(bookmark_id=bid)
