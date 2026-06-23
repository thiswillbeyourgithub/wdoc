"""
Helpers for the `karakeep` recursive filetype.

A Karakeep instance is not a single document: one list / tag / search / the whole
library fans out into many bookmarks, each carrying a link (with crawled
`htmlContent`), free text, or an uploaded asset (pdf / image), plus a title,
note, AI summary and tags. The actual fan-out into `DocDict`s lives in
`wdoc.utils.load_recursive.parse_karakeep`; this module only holds the
Karakeep-API-facing helpers so that `load_recursive.py` stays lean.

We deliberately do NOT re-implement any extraction here: a bookmark is resolved
to one of Karakeep's *stored* artifacts (the saved HTML, the pre-extracted text,
or a downloaded stored PDF asset) and handed back to wdoc's own `local_html` /
`txt` / `pdf` loaders, which are far more capable and already cached. The online
resource is never re-fetched, so the loader works offline and under
`--private` mode against a local Karakeep instance.

Per-bookmark content resolution is wrapped in a joblib cache keyed on the
bookmark id + its `modifiedAt` (youtube-loader style), so re-running wdoc on the
same selection does not re-hit the Karakeep server or re-download assets.
"""

import os
from urllib.parse import urlparse

from beartype.typing import Dict, List, Optional, Tuple
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.misc import doc_loaders_cache

KARAKEEP_INSTALL_HINT = (
    "karakeep-python-api is required for the 'karakeep' filetype. "
    "Install it with 'pip install wdoc[karakeep]'."
)


def _import_karakeep():
    try:
        from karakeep_python_api import KarakeepAPI
    except ImportError as err:  # pragma: no cover - exercised only without extra
        raise ImportError(KARAKEEP_INSTALL_HINT) from err
    return KarakeepAPI


def _is_local_host(endpoint: str) -> bool:
    """True if the endpoint points at a loopback / private-network host."""
    host = (urlparse(endpoint).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    if host.startswith("172."):
        # 172.16.0.0 - 172.31.255.255
        try:
            second = int(host.split(".")[1])
            return 16 <= second <= 31
        except (IndexError, ValueError):
            return False
    return False


def get_karakeep_client(
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    verify_ssl: bool = True,
):
    """Return a connected ``KarakeepAPI`` client.

    Credentials fall back to the karakeep-python-api standard environment
    variables ``KARAKEEP_PYTHON_API_ENDPOINT`` / ``KARAKEEP_PYTHON_API_KEY`` /
    ``KARAKEEP_PYTHON_API_VERIFY_SSL`` when the corresponding argument is absent.

    Response validation is disabled so the client returns plain dicts, which is
    both robust to upstream schema drift and convenient for the fan-out.

    Private-mode guard: this loader never reaches the live bookmarked URL, only
    the Karakeep instance itself. A local instance is therefore allowed under
    ``WDOC_PRIVATE_MODE``; a remote endpoint is blocked.
    """
    KarakeepAPI = _import_karakeep()

    resolved_endpoint = api_endpoint or os.environ.get("KARAKEEP_PYTHON_API_ENDPOINT")
    assert resolved_endpoint, (
        "The Karakeep API needs an endpoint (karakeep_api_endpoint or the "
        "KARAKEEP_PYTHON_API_ENDPOINT env var, e.g. "
        "'https://karakeep.example.com/api/v1/')."
    )

    if env.WDOC_PRIVATE_MODE:
        assert _is_local_host(resolved_endpoint), (
            "Private mode is enabled but the Karakeep endpoint "
            f"'{resolved_endpoint}' is not a local/loopback host. Point "
            "karakeep_api_endpoint at a local instance to use it under "
            "--private."
        )

    logger.info(f"Connecting to the Karakeep API at {resolved_endpoint}")
    return KarakeepAPI(
        api_endpoint=resolved_endpoint,
        api_key=api_key,  # None lets the client read KARAKEEP_PYTHON_API_KEY
        verify_ssl=verify_ssl,
        disable_response_validation=True,
    )


def parse_selector(path: str) -> Tuple[str, object]:
    """Turn the ``--path`` selector into a ``(kind, value)`` pair.

    Accepted forms:
    - ``library`` / ``*`` / ``all`` -> the whole library
    - ``favourites`` / ``favorites`` -> favourited bookmarks
    - ``archived`` -> archived bookmarks
    - ``tag:foo`` -> bookmarks with that tag (by name)
    - ``search:query`` -> a Karakeep search query
    - ``ids:ID1,ID2`` (or ``id:``) -> explicit bookmark ids
    - ``list:Name`` or a bare value -> a list by name
    """
    s = str(path).strip()
    low = s.lower()
    if low in ("library", "*", "all"):
        return ("library", None)
    if low in ("favourites", "favorites", "favourited", "favorited"):
        return ("favourites", None)
    if low == "archived":
        return ("archived", None)

    prefix, sep, rest = s.partition(":")
    if sep:
        p = prefix.lower().strip()
        rest = rest.strip()
        if p == "tag":
            return ("tag", rest)
        if p in ("id", "ids", "bookmark", "bookmarks"):
            return ("ids", [t.strip() for t in rest.split(",") if t.strip()])
        if p == "search":
            return ("search", rest)
        if p in ("list", "lst"):
            return ("list", rest)

    return ("list", s)


def _as_dict(obj) -> Dict:
    """Normalize a pydantic model or dict to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _bookmarks_page(page) -> Tuple[List[Dict], Optional[str]]:
    """Extract (bookmarks, nextCursor) from a paginated response."""
    page = _as_dict(page)
    bookmarks = [_as_dict(b) for b in page.get("bookmarks", [])]
    return bookmarks, page.get("nextCursor") or None


def _fetch_all(fetch_page) -> List[Dict]:
    """Follow Karakeep's cursor pagination, returning every bookmark.

    ``fetch_page(cursor)`` must call the relevant client method and return its
    raw paginated response.
    """
    out: List[Dict] = []
    cursor = None
    seen_cursors = set()
    while True:
        bookmarks, next_cursor = _bookmarks_page(fetch_page(cursor))
        out.extend(bookmarks)
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return out


def _resolve_list_id(client, name: str) -> str:
    # get_all_lists may return a bare list or a {"lists": [...]} dict
    raw = client.get_all_lists()
    if isinstance(raw, list):
        lists = [_as_dict(x) for x in raw]
    else:
        lists = [_as_dict(x) for x in _as_dict(raw).get("lists", [])]
    match = next((x for x in lists if x.get("name") == name), None)
    assert match is not None, (
        f"Karakeep list '{name}' not found. Available: "
        + ", ".join(sorted(x.get("name", "?") for x in lists))
    )
    return match["id"]


def _resolve_tag_id(client, name: str) -> str:
    raw = client.get_all_tags()
    if isinstance(raw, list):
        tags = [_as_dict(x) for x in raw]
    else:
        tags = [_as_dict(x) for x in _as_dict(raw).get("tags", [])]
    match = next((x for x in tags if x.get("name") == name), None)
    assert match is not None, (
        f"Karakeep tag '{name}' not found. Available: "
        + ", ".join(sorted(x.get("name", "?") for x in tags))
    )
    return match["id"]


def resolve_bookmarks(client, selector: Tuple[str, object]) -> List[Dict]:
    """Resolve a parsed selector into a list of bookmark dicts."""
    kind, value = selector

    if kind == "library":
        logger.info("Loading all Karakeep bookmarks")
        return _fetch_all(lambda c: client.get_all_bookmarks(cursor=c))

    if kind == "favourites":
        logger.info("Loading favourited Karakeep bookmarks")
        return _fetch_all(lambda c: client.get_all_bookmarks(favourited=True, cursor=c))

    if kind == "archived":
        logger.info("Loading archived Karakeep bookmarks")
        return _fetch_all(lambda c: client.get_all_bookmarks(archived=True, cursor=c))

    if kind == "tag":
        assert value, "No tag given (expected 'tag:Name')"
        tag_id = _resolve_tag_id(client, value)
        logger.info(f"Loading Karakeep bookmarks with tag '{value}'")
        return _fetch_all(lambda c: client.get_bookmarks_with_the_tag(tag_id, cursor=c))

    if kind == "search":
        assert value, "No query given (expected 'search:terms')"
        logger.info(f"Searching Karakeep bookmarks: '{value}'")
        return _fetch_all(lambda c: client.search_bookmarks(q=value, cursor=c))

    if kind == "ids":
        assert value, "No bookmark id given (expected 'ids:ID1,ID2')"
        logger.info(f"Loading explicit Karakeep bookmark ids: {value}")
        return [_as_dict(client.get_a_single_bookmark(i)) for i in value]

    if kind == "list":
        list_id = _resolve_list_id(client, value)
        logger.info(f"Loading Karakeep list '{value}'")
        return _fetch_all(lambda c: client.get_bookmarks_in_the_list(list_id, cursor=c))

    raise ValueError(f"Unknown selector kind: '{kind}'")


def _tags_str(bookmark: Dict) -> str:
    return ", ".join(
        t.get("name", "") for t in bookmark.get("tags", []) if t.get("name")
    )


def format_header(bookmark: Dict) -> str:
    """A short, human + RAG friendly metadata header for a bookmark."""
    content = bookmark.get("content") or {}
    lines = []

    def add(label, value):
        if value:
            lines.append(f"{label}: {value}")

    add("Title", bookmark.get("title") or content.get("title"))
    add("URL", content.get("url") or content.get("sourceUrl"))
    add("Author", content.get("author"))
    add("Publisher", content.get("publisher"))
    add("Date", content.get("datePublished"))
    add("Tags", _tags_str(bookmark))
    add("Note", bookmark.get("note"))
    add("Summary", bookmark.get("summary"))
    add("Karakeep id", bookmark.get("id"))
    return "\n".join(lines)


def bookmark_web_url(bookmark: Dict, api_endpoint: Optional[str]) -> str:
    """Best-effort permalink for a bookmark (used as subitem_link)."""
    bid = bookmark.get("id")
    if api_endpoint and bid:
        base = api_endpoint.split("/api/v1")[0].rstrip("/")
        return f"{base}/dashboard/preview/{bid}"
    content = bookmark.get("content") or {}
    return content.get("url") or content.get("sourceUrl") or ""


def _link_asset_id(bookmark: Dict, content: Dict) -> Optional[str]:
    """The best stored PDF/archive asset for a link bookmark, or None."""
    direct = content.get("pdfAssetId") or content.get("fullPageArchiveAssetId")
    if direct:
        return direct
    for a in bookmark.get("assets", []):
        a = _as_dict(a)
        if a.get("assetType") in ("pdf", "fullPageArchive", "precrawledArchive"):
            return a.get("id")
    return None


@doc_loaders_cache.cache(ignore=["client", "bookmark"])
def cached_karakeep_content(
    bookmark_id: str,
    modified_at: Optional[str],
    content_source: str,
    api_endpoint: Optional[str],
    *,
    client,
    bookmark: Dict,
) -> Optional[Dict]:
    """Resolve a bookmark to a loadable artifact, cached on id + modifiedAt.

    Returns a descriptor ``{"filetype", "text", "data", "suffix"}`` (``text`` for
    a doc written from a string, ``data`` for downloaded bytes) or ``None`` when
    the bookmark has no usable stored content.

    The live ``client`` and ``bookmark`` are excluded from the cache key (the key
    is ``bookmark_id`` + ``modified_at`` + ``content_source`` + ``api_endpoint``)
    so the secret api key never lands in the cache and a re-run on an unchanged
    bookmark skips the Karakeep round-trip and any asset download.
    """
    content = bookmark.get("content") or {}
    ctype = content.get("type")
    header = format_header(bookmark)

    def with_header(body: str) -> str:
        return f"{header}\n\n{body}" if header else body

    def _download(asset_id: str, suffix: str) -> Optional[Dict]:
        try:
            data = client.get_a_single_asset(asset_id)
        except Exception as err:
            logger.warning(
                f"Could not download Karakeep asset {asset_id} of bookmark "
                f"{bookmark_id}: {err}"
            )
            return None
        if not data:
            return None
        return {"filetype": "pdf", "text": None, "data": data, "suffix": suffix}

    if ctype == "text":
        text = (content.get("text") or "").strip()
        if text:
            return {
                "filetype": "txt",
                "text": with_header(text),
                "data": None,
                "suffix": ".txt",
            }
        return None

    if ctype == "asset":
        extracted = (content.get("content") or "").strip()
        if content_source in ("auto", "native") and extracted:
            return {
                "filetype": "txt",
                "text": with_header(extracted),
                "data": None,
                "suffix": ".txt",
            }
        if content_source in ("auto", "wdoc"):
            asset_id = content.get("assetId")
            if asset_id and content.get("assetType") == "pdf":
                got = _download(asset_id, ".pdf")
                if got:
                    return got
        # asset has only extracted text but we are in wdoc mode, fall back to it
        if extracted:
            return {
                "filetype": "txt",
                "text": with_header(extracted),
                "data": None,
                "suffix": ".txt",
            }
        return None

    if ctype == "link":
        html = content.get("htmlContent") or ""
        has_html = bool(html.strip())

        if content_source == "native":
            if has_html:
                return {
                    "filetype": "local_html",
                    "text": with_header(html),
                    "data": None,
                    "suffix": ".html",
                }
            return None

        if content_source == "wdoc":
            # prefer the stored underlying pdf/archive, else the stored html
            asset_id = _link_asset_id(bookmark, content)
            if asset_id:
                got = _download(asset_id, ".pdf")
                if got:
                    return got
            if has_html:
                return {
                    "filetype": "local_html",
                    "text": with_header(html),
                    "data": None,
                    "suffix": ".html",
                }
            return None

        # auto: stored html first (cheap), else the stored pdf/archive asset
        if has_html:
            return {
                "filetype": "local_html",
                "text": with_header(html),
                "data": None,
                "suffix": ".html",
            }
        asset_id = _link_asset_id(bookmark, content)
        if asset_id:
            got = _download(asset_id, ".pdf")
            if got:
                return got
        return None

    # unknown content type
    return None
