"""
Helpers for the `zotero` recursive filetype.

A Zotero library is not a single document: one collection / tag / saved search
fans out into many items, each of which can carry several attachments (PDFs,
linked files, web links), notes and bibliographic metadata. The actual fan-out
into `DocDict`s lives in `wdoc.utils.load_recursive.parse_zotero`; this module
only holds the Zotero-API-facing helpers so that `load_recursive.py` stays lean.

We deliberately do NOT re-implement PDF extraction here (unlike the reference
project `openwebui-knowledgesync-zotero-python`): attachments are written to a
temp file and handed back to wdoc's own `pdf`/`auto` loaders, which are far more
capable (15 parser backends) and already cached.

Connection follows pyzotero: the local Zotero HTTP API (http://localhost:23119,
works offline and in --private mode) is tried first, with a fall back to the
Web API (api key + numeric library id) when configured.
"""

import os
from pathlib import Path

from beartype.typing import Dict, List, Literal, Optional, Tuple
from loguru import logger

from wdoc.utils.env import env

ZOTERO_INSTALL_HINT = (
    "pyzotero is required for the 'zotero' filetype. "
    "Install it with 'pip install wdoc[zotero]'."
)

# item types that are not standalone documents but children of an item
_NON_PARENT_ITEMTYPES = ("attachment", "note")


def _import_pyzotero():
    try:
        from pyzotero import zotero
    except ImportError as err:  # pragma: no cover - exercised only without extra
        raise ImportError(ZOTERO_INSTALL_HINT) from err
    return zotero


def get_zotero_client(
    connection: Literal["auto", "local", "web"] = "auto",
    library_id: Optional[str] = None,
    library_type: Literal["user", "group"] = "user",
    api_key: Optional[str] = None,
):
    """Return a connected ``pyzotero.zotero.Zotero`` instance.

    Credentials fall back to the pyzotero-standard environment variables
    ``ZOTERO_LIBRARY_ID`` / ``ZOTERO_API_KEY`` / ``ZOTERO_LIBRARY_TYPE`` when the
    corresponding argument is not provided.

    - ``connection="local"``: only the local Zotero HTTP API (requires the Zotero
      desktop app running). Probed eagerly so failures are reported up-front.
    - ``connection="web"``: only the Web API (needs library id + api key). Blocked
      under ``WDOC_PRIVATE_MODE`` since it reaches out to zotero.org.
    - ``connection="auto"`` (default): try local, fall back to web.
    """
    zotero = _import_pyzotero()

    library_type = library_type or "user"
    library_id = library_id or os.environ.get("ZOTERO_LIBRARY_ID")
    api_key = api_key or os.environ.get("ZOTERO_API_KEY")
    library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", library_type)

    def _make_local():
        # The local API serves the running desktop library; auth is ignored, so a
        # placeholder library_id is fine when none is configured.
        return zotero.Zotero(library_id or "0", library_type, api_key or "", local=True)

    def _make_web():
        assert not env.WDOC_PRIVATE_MODE, (
            "Private mode is enabled but the Zotero Web API would reach out to "
            "zotero.org. Use the local Zotero API instead (set "
            "zotero_connection='local' and keep the Zotero desktop app running)."
        )
        assert library_id, (
            "The Zotero Web API needs a library id (zotero_library_id or the "
            "ZOTERO_LIBRARY_ID env var)."
        )
        assert api_key, (
            "The Zotero Web API needs an api key (zotero_api_key or the "
            "ZOTERO_API_KEY env var)."
        )
        return zotero.Zotero(library_id, library_type, api_key)

    if connection == "web":
        logger.info("Connecting to the Zotero Web API")
        return _make_web()

    if connection == "local":
        logger.info("Connecting to the local Zotero API (localhost:23119)")
        zot = _make_local()
        # probe so an unreachable local server fails clearly instead of later
        zot.collections(limit=1)
        return zot

    # auto: local first, then web
    try:
        zot = _make_local()
        zot.collections(limit=1)
        logger.info("Connected to the local Zotero API (localhost:23119)")
        return zot
    except Exception as err:
        logger.warning(
            f"Local Zotero API unavailable ({err}); falling back to the Web API"
        )
        return _make_web()


def parse_selector(path: str) -> Tuple[str, object]:
    """Turn the ``--path`` selector into a ``(kind, value)`` pair.

    Accepted forms:
    - ``library`` / ``*`` / ``all`` -> the whole library
    - ``tag:foo,bar`` -> items matching those tags
    - ``items:KEY1,KEY2`` (or ``key:``) -> explicit item keys
    - ``search:Name`` -> a Zotero saved search by name
    - ``collection:Path`` or a bare value -> a collection by name or nested path
    """
    s = str(path).strip()
    low = s.lower()
    if low in ("library", "*", "all"):
        return ("library", None)

    prefix, sep, rest = s.partition(":")
    if sep:
        p = prefix.lower().strip()
        rest = rest.strip()
        if p == "tag":
            return ("tag", [t.strip() for t in rest.split(",") if t.strip()])
        if p in ("item", "items", "key", "keys"):
            return ("items", [t.strip() for t in rest.split(",") if t.strip()])
        if p in ("search", "saved_search", "savedsearch"):
            return ("search", rest)
        if p in ("collection", "col"):
            return ("collection", rest)

    return ("collection", s)


def _split_collection_path(path: str) -> List[str]:
    # accept both "A/B/C" and the reference tool's "A%%B%%C" separators
    norm = path.replace("%%", "/")
    return [p.strip() for p in norm.split("/") if p.strip()]


def _resolve_collection_key(all_collections: List[Dict], path: str) -> str:
    """Walk the collection tree to find the key of a (possibly nested) path."""
    parts = _split_collection_path(path)
    assert parts, f"Empty collection path: '{path}'"

    parent = None  # top-level collections have a falsy parentCollection
    key = None
    for part in parts:
        match = None
        for col in all_collections:
            data = col.get("data", {})
            col_parent = data.get("parentCollection") or None
            if data.get("name") == part and col_parent == parent:
                match = col
                break
        assert match is not None, (
            f"Collection '{part}' (from path '{path}') not found under parent "
            f"'{parent}'."
        )
        key = match["key"]
        parent = key
    return key


def _descendant_collection_keys(all_collections: List[Dict], key: str) -> List[str]:
    """Return ``key`` plus the keys of all its (recursive) subcollections."""
    keys = [key]
    children = [
        c["key"]
        for c in all_collections
        if (c.get("data", {}).get("parentCollection") or None) == key
    ]
    for child in children:
        keys.extend(_descendant_collection_keys(all_collections, child))
    return keys


def _is_parent_item(item: Dict) -> bool:
    return item.get("data", {}).get("itemType") not in _NON_PARENT_ITEMTYPES


def _saved_search_items(zot, name: str) -> List[Dict]:
    """Best-effort execution of a Zotero saved search by name.

    The Zotero API cannot run a saved search server-side, so we translate the
    common conditions (tag, collection, itemType, quicksearch) into an item
    query. Unsupported conditions are warned about and ignored.
    """
    searches = zot.searches()
    match = next((s for s in searches if s.get("data", {}).get("name") == name), None)
    assert match is not None, (
        f"Saved search '{name}' not found. Available: "
        + ", ".join(sorted(s.get("data", {}).get("name", "?") for s in searches))
    )
    conditions = match.get("data", {}).get("conditions", [])
    params: Dict[str, object] = {}
    tags: List[str] = []
    for cond in conditions:
        field = cond.get("condition")
        value = cond.get("value")
        if field == "tag":
            tags.append(value)
        elif field in ("title", "quicksearch", "quicksearch-everything"):
            params["q"] = value
        elif field == "itemType":
            params["itemType"] = value
        else:
            logger.warning(
                f"Saved search '{name}': ignoring unsupported condition '{field}'"
            )
    if tags:
        params["tag"] = tags if len(tags) > 1 else tags[0]
    items = zot.everything(zot.items(**params)) if params else zot.everything(zot.top())
    return [it for it in items if _is_parent_item(it)]


def resolve_items(zot, selector: Tuple[str, object]) -> List[Dict]:
    """Resolve a parsed selector into a list of parent Zotero items."""
    kind, value = selector

    if kind == "library":
        logger.info("Loading the whole Zotero library")
        return [it for it in zot.everything(zot.top()) if _is_parent_item(it)]

    if kind == "tag":
        assert value, "No tag given (expected 'tag:foo,bar')"
        logger.info(f"Loading Zotero items with tags: {value}")
        tag_param = value if len(value) > 1 else value[0]
        items = zot.everything(zot.items(tag=tag_param))
        return [it for it in items if _is_parent_item(it)]

    if kind == "items":
        assert value, "No item key given (expected 'items:KEY1,KEY2')"
        logger.info(f"Loading explicit Zotero item keys: {value}")
        return [zot.item(k) for k in value]

    if kind == "search":
        logger.info(f"Loading Zotero saved search: '{value}'")
        return _saved_search_items(zot, value)

    if kind == "collection":
        all_collections = zot.everything(zot.collections())
        key = _resolve_collection_key(all_collections, value)
        col_keys = _descendant_collection_keys(all_collections, key)
        logger.info(
            f"Loading Zotero collection '{value}' "
            f"({len(col_keys)} collection(s) including subcollections)"
        )
        items: List[Dict] = []
        seen = set()
        for ck in col_keys:
            for it in zot.everything(zot.collection_items(ck)):
                if it["key"] in seen or not _is_parent_item(it):
                    continue
                seen.add(it["key"])
                items.append(it)
        return items

    raise ValueError(f"Unknown selector kind: '{kind}'")


def item_children(zot, item: Dict) -> List[Dict]:
    """Return the child items (attachments + notes) of a Zotero item."""
    try:
        return zot.children(item["key"])
    except Exception as err:
        logger.warning(f"Could not fetch children of item {item['key']}: {err}")
        return []


def attachment_fulltext(zot, attachment_key: str) -> Optional[str]:
    """Return Zotero's pre-indexed fulltext for an attachment, or None."""
    try:
        return zot.fulltext_item(attachment_key)["content"]
    except Exception as err:
        logger.debug(f"No indexed fulltext for {attachment_key}: {err}")
        return None


def attachment_to_file(
    zot, attachment: Dict, temp_dir: Path
) -> Optional[Tuple[str, str]]:
    """Materialise an attachment into something a wdoc loader can read.

    Returns a ``(filetype, path_or_url)`` tuple:
    - ``("pdf", path)`` / ``("auto", path)`` for a file on disk (downloaded or linked)
    - ``("url", url)`` for a linked web URL
    or ``None`` if the attachment cannot be turned into a document.
    """
    data = attachment.get("data", {})
    key = attachment["key"]
    link_mode = data.get("linkMode")
    content_type = (data.get("contentType") or "").lower()
    filename = data.get("filename") or data.get("title") or key

    def _filetype_for(name: str) -> str:
        if content_type == "application/pdf" or str(name).lower().endswith(".pdf"):
            return "pdf"
        return "auto"

    if link_mode == "linked_url":
        url = data.get("url")
        if url:
            return ("url", url)
        logger.warning(f"Linked-URL attachment {key} has no url, skipping")
        return None

    if link_mode == "linked_file":
        raw = data.get("path") or ""
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return (_filetype_for(p.name), str(p))
        logger.warning(
            f"Linked-file attachment {key} points at '{raw}' which is not an "
            "accessible absolute path, skipping"
        )
        return None

    # imported_file / imported_url: download the bytes via the API
    try:
        content = zot.file(key)
    except Exception as err:
        logger.warning(f"Could not download attachment {key}: {err}")
        return None
    safe = str(filename).replace("/", "_").replace("\\", "_")
    out = Path(temp_dir) / f"{key}_{safe}"
    out.write_bytes(content)
    return (_filetype_for(out.name), str(out))


def _creators_str(item_data: Dict) -> str:
    names = []
    for c in item_data.get("creators", []):
        if c.get("name"):
            names.append(c["name"])
        else:
            full = " ".join(
                p for p in (c.get("firstName"), c.get("lastName")) if p
            ).strip()
            if full:
                names.append(full)
    return ", ".join(names)


def _tags_str(item_data: Dict) -> str:
    return ", ".join(
        t.get("tag", "") for t in item_data.get("tags", []) if t.get("tag")
    )


def item_web_url(item: Dict) -> Optional[str]:
    """Best-effort permalink for a Zotero item (used as subitem_link)."""
    href = item.get("links", {}).get("alternate", {}).get("href")
    if href:
        return href
    key = item.get("key")
    return f"zotero://select/library/items/{key}" if key else None


def format_bib_header(item: Dict) -> str:
    """A short, human + RAG friendly bibliographic header for an item."""
    data = item.get("data", {})
    lines = []

    def add(label, value):
        if value:
            lines.append(f"{label}: {value}")

    add("Title", data.get("title"))
    add("Authors", _creators_str(data))
    add("Date", data.get("date"))
    add("Item type", data.get("itemType"))
    add("Publication", data.get("publicationTitle"))
    add("DOI", data.get("DOI"))
    add("URL", data.get("url"))
    add("Tags", _tags_str(data))
    add("Zotero key", item.get("key"))
    return "\n".join(lines)


def metadata_document_text(item: Dict) -> str:
    """Full text for the always-on per-item metadata document."""
    header = format_bib_header(item)
    abstract = item.get("data", {}).get("abstractNote", "")
    if abstract:
        return f"{header}\n\nAbstract:\n{abstract}"
    return header
