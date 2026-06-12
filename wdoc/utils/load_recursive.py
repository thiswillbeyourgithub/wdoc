import json
import re
import tempfile
from loguru import logger
from pathlib import Path
import uuid
from functools import cache as memoizer
from beartype.typing import List, Optional, Tuple, Union, Literal
from wdoc.utils.loaders import (
    markdownlink_regex,
)
from wdoc.utils.misc import (
    DocDict,
)
from wdoc.utils.env import env


def parse_recursive_paths(
    cli_kwargs: dict,
    path: Union[str, Path],
    pattern: str,
    recursed_filetype: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **extra_args,
) -> List[Union[DocDict, dict]]:
    """
    Turn a DocDict that has `filetype==recursive_paths` into the DocDict of
    individual files in that path.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The directory path to search recursively
        pattern: Glob pattern to match files (e.g., "*.pdf", "**/*.txt")
        recursed_filetype: The filetype to assign to found files
        include: Optional list of regex patterns to include files, default=None
        exclude: Optional list of regex patterns to exclude files, default=None
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict or dict objects, each representing a found file
    """
    logger.info(f"Parsing recursive load_filetype: '{path}'")
    assert recursed_filetype not in [
        "recursive_paths",
        "json_entries",
        "youtube",
        "anki",
    ], (
        "'recursed_filetype' cannot be 'recursive_paths', 'json_entries', 'anki' or 'youtube'"
    )

    if not Path(path).exists() and Path(path.replace(r"\ ", " ")).exists():
        logger.info(r"File was not found so replaced '\ ' by ' '")
        path = path.replace(r"\ ", " ")
    assert Path(path).exists(), f"not found: {path}"
    doclist = [p for p in Path(path).rglob(pattern)]
    assert doclist, f"No document found by pattern {pattern}"
    doclist = [str(p).strip() for p in doclist if p.is_file()]
    assert doclist, "No document after filtering by file"
    doclist = [p for p in doclist if p]
    assert doclist, "No document after removing nonemtpy"
    doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
    recur_parent_id = str(uuid.uuid4())

    if include:
        for iinc, inc in enumerate(include):
            if isinstance(inc, str):
                if inc == inc.lower():
                    inc = re.compile(inc, flags=re.IGNORECASE)
                else:
                    inc = re.compile(inc)
                include[iinc] = inc
        ndoclist = len(doclist)
        doclist = [d for d in doclist if any(inc.search(d) for inc in include)]
        if not len(doclist) < ndoclist:
            mess = f"Include rules were useless and didn't filter out anything.\nInclude rules: '{include}'"
            if env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "warn":
                logger.warning(mess)
            elif env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "crash":
                logger.warning(mess)
                raise Exception(mess)

    if exclude:
        for iexc, exc in enumerate(exclude):
            if isinstance(exc, str):
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                exclude[iexc] = exc
            ndoclist = len(doclist)
            doclist = [d for d in doclist if not exc.search(d)]
            if not len(doclist) < ndoclist:
                mess = f"Exclude rule '{exc}' was useless and didn't filter out anything.\nExclude rules: '{exclude}'"
                if env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "warn":
                    logger.warning(mess)
                elif env.WDOC_BEHAVIOR_EXCL_INCL_USELESS == "crash":
                    logger.warning(mess)
                    raise Exception(mess)

    for i, d in enumerate(doclist):
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = recursed_filetype
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        if doc_kwargs["filetype"] not in recursive_types_func_mapping:
            doclist[i] = DocDict(doc_kwargs)
        else:
            doclist[i] = doc_kwargs
    return doclist


def parse_json_entries(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[Union[DocDict, dict]]:
    """
    Turn a DocDict that has `filetype==json_entries` into the individual
    DocDict mentionned inside the json file.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The path to the JSON file containing document entries
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict or dict objects, each representing an entry from the JSON file
    """
    logger.info(f"Loading json_entries: '{path}'")
    doclist = str(Path(path).read_text()).splitlines()
    doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
    doclist = [
        p.strip() for p in doclist if p.strip() and not p.strip().startswith("#")
    ]
    recur_parent_id = str(uuid.uuid4())

    for i, d in enumerate(doclist):
        meta = cli_kwargs.copy()
        meta["filetype"] = "auto"
        meta.update(json.loads(d.strip()))
        for k, v in cli_kwargs.copy().items():
            if k not in meta:
                meta[k] = v
        if meta["path"] == path:
            del meta["path"]
        meta.update(extra_args)
        meta["recur_parent_id"] = recur_parent_id
        if meta["filetype"] not in recursive_types_func_mapping:
            doclist[i] = DocDict(meta)
        else:
            doclist[i] = meta
    return doclist


def parse_toml_entries(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[Union[DocDict, dict]]:
    """
    Turn a DocDict that has `filetype==toml_entries` into the individual
    DocDict mentionned inside the toml file.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The path to the TOML file containing document entries
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict or dict objects, each representing an entry from the TOML file
    """
    import rtoml

    logger.info(f"Loading toml_entries: '{path}'")
    content = rtoml.load(toml=Path(path))
    assert isinstance(content, dict)
    doclist = list(content.values())
    assert all(len(d) == 1 for d in doclist)
    doclist = [d[0] for d in doclist]
    assert all(isinstance(d, dict) for d in doclist)
    recur_parent_id = str(uuid.uuid4())

    for i, d in enumerate(doclist):
        meta = cli_kwargs.copy()
        meta["filetype"] = "auto"
        meta.update(d)
        for k, v in cli_kwargs.items():
            if k not in meta:
                meta[k] = v
        if meta["path"] == path:
            del meta["path"]
        meta.update(extra_args)
        meta["recur_parent_id"] = recur_parent_id
        if meta["filetype"] not in recursive_types_func_mapping:
            doclist[i] = DocDict(meta)
        else:
            doclist[i] = meta
    return doclist


def parse_link_file(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==link_file` into the individual
    DocDict of each url, where there is one url per line inside the
    `link_file` file. Note that bullet points are stripped (i.e. "- [the url]" is
    treated the same as "the url"), and commented lines (i.e. starting with "#")
    are ignored.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The path to the link file containing URLs
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each representing a URL from the link file
    """
    logger.info(f"Loading link_file: '{path}'")
    doclist = str(Path(path).read_text()).splitlines()
    doclist = [p[1:].strip() if p.startswith("-") else p.strip() for p in doclist]
    doclist = [
        p.strip()
        for p in doclist
        if p.strip() and not p.strip().startswith("#") and "http" in p
    ]
    doclist = [
        matched.group(0)
        for d in doclist
        if (matched := markdownlink_regex.search(d).strip())
    ]

    recur_parent_id = str(uuid.uuid4())
    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["subitem_link"] = d
        doc_kwargs["filetype"] = "auto"
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        doclist[i] = DocDict(doc_kwargs)
    return doclist


def parse_youtube_playlist(
    cli_kwargs: dict,
    path: Union[str, Path],
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==youtube_playlist` into the individual
    DocDict of each youtube video part of that playlist.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The YouTube playlist URL
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each representing a YouTube video from the playlist
    """
    from wdoc.utils.loaders.youtube import load_youtube_playlist, yt_link_regex

    if "\\" in path:
        logger.warning(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")
    logger.info(f"Loading youtube playlist: '{path}'")
    video = load_youtube_playlist(path)

    playlist_title = video["title"].strip().replace("\n", "")
    assert "duration" not in video, (
        f'"duration" found when loading youtube playlist. This might not be a playlist: {path}'
    )
    doclist = [ent["webpage_url"] for ent in video["entries"]]
    doclist = [li for li in doclist if yt_link_regex.search(li)]

    recur_parent_id = str(uuid.uuid4())
    for i, d in enumerate(doclist):
        assert "http" in d, f"Link does not appear to be a link: '{d}'"
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = d
        doc_kwargs["filetype"] = "youtube"
        doc_kwargs["subitem_link"] = d
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        doclist[i] = DocDict(doc_kwargs)

    assert doclist, f"No video found in youtube playlist: {path}"
    for idoc, doc in enumerate(doclist):
        if "playlist_title" in doc.metadata and doc.metadata["playlist_title"]:
            if playlist_title not in doc.metadata["playlist_title"]:
                doc.metadata["playlist_title"] += " - " + playlist_title
        else:
            doc.metadata["playlist_title"] = playlist_title
        doclist[idoc]
    return doclist


def parse_ddg_search(
    cli_kwargs: dict,
    path: Union[str, Path],
    ddg_max_results: int = 50,
    ddg_region: str = "",
    ddg_safesearch: Literal["on", "off", "moderate"] = "off",
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==ddg` into the individual
    DocDict of the webpage of each DuckDuckGo search result, treating the
    `path` as a search query.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The search query string
        ddg_max_results: Maximum number of search results to return, default=50
        ddg_region: DuckDuckGo search region, default=''
        ddg_safesearch: SafeSearch setting ("on", "moderate", "off"), default='off'
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each representing a URL from search results
    """
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    query = str(path).strip()
    logger.info(f"Performing DuckDuckGo search: '{query}'")

    # Set up DuckDuckGo search
    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=ddg_max_results,
        safesearch=ddg_safesearch,
        source="text",
        region=ddg_region,
    )

    tool = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        response_format="content_and_artifact",
        output_format="list",
        max_results=ddg_max_results,
        region=ddg_region,
    )
    tool.max_results = ddg_max_results

    # Perform the search
    try:
        search_results = tool.invoke(query)
    except Exception as err:
        raise Exception(f"DuckDuckGo search failed for query '{query}': {err}") from err

    if not search_results:
        logger.warning(f"No search results found for query: '{query}'")
        return []

    logger.info(f"Found {len(search_results)} search results for: '{query}'")

    # Create a unique parent ID for tracking this search batch
    recur_parent_id = str(uuid.uuid4())

    doclist = []
    for i, result in enumerate(search_results):
        if "link" not in result:
            logger.warning(f"Search result #{i + 1} missing 'link' field: {result}")
            continue

        url = result["link"].strip()
        if not url or not url.startswith("http"):
            logger.warning(f"Invalid URL in search result #{i + 1}: '{url}'")
            continue

        # Create document kwargs
        doc_kwargs = cli_kwargs.copy()
        doc_kwargs["path"] = url
        doc_kwargs["subitem_link"] = url
        doc_kwargs["filetype"] = "auto"  # Let wdoc auto-detect the URL content type
        doc_kwargs["recur_parent_id"] = recur_parent_id

        # Add DuckDuckGo search metadata
        # doc_kwargs["ddg_search_query"] = query
        # doc_kwargs["ddg_search_rank"] = i + 1
        # if "title" in result:
        #     doc_kwargs["ddg_title"] = result["title"]
        # if "snippet" in result:
        #     doc_kwargs["ddg_snippet"] = result["snippet"]

        # Apply any extra arguments
        doc_kwargs.update(extra_args)

        doclist.append(DocDict(doc_kwargs))

    if not doclist:
        raise Exception(
            f"No valid URLs found in DuckDuckGo search results for query: '{query}'"
        )

    # it seems sometimes ddg returns duplicate result
    assert len(doclist) == len(set(doclist)), f"Duplicate elements: {doclist}"

    return doclist


def parse_zotero(
    cli_kwargs: dict,
    path: Union[str, Path],
    zotero_connection: Literal["auto", "local", "web"] = "auto",
    zotero_library_id: Optional[str] = None,
    zotero_library_type: Literal["user", "group"] = "user",
    zotero_api_key: Optional[str] = None,
    zotero_attachment_text: Literal["wdoc", "fulltext", "hybrid"] = "wdoc",
    zotero_include_notes: bool = False,
    zotero_include_metadata: bool = True,
    **extra_args,
) -> List[DocDict]:
    """
    Turn a DocDict that has `filetype==zotero` into one DocDict per loadable
    sub-document of the selected Zotero items.

    The `path` carries the selector (see `parse_selector`): a collection name or
    nested path by default, or one of `tag:...`, `items:...`, `search:...`,
    `library`/`*`. Each selected item fans out into:
      - one DocDict per attachment (a `pdf`/`auto`/`url` doc pointing at the
        downloaded/linked file, so wdoc's own loaders handle the extraction; or a
        `txt` doc holding Zotero's indexed fulltext when `zotero_attachment_text`
        requests it);
      - an always-on (`zotero_include_metadata`) `txt` doc with the item's
        bibliographic header + abstract;
      - optionally (`zotero_include_notes`) one `txt` doc per attached note.

    Args:
        cli_kwargs: Base CLI arguments to inherit
        path: The Zotero selector (see above)
        zotero_connection: 'auto' (local then web), 'local', or 'web'
        zotero_library_id: Zotero numeric library id (web API), else ZOTERO_LIBRARY_ID
        zotero_library_type: 'user' or 'group'
        zotero_api_key: Zotero api key (web API), else ZOTERO_API_KEY
        zotero_attachment_text: how to get attachment text ('wdoc'=reuse wdoc
            loaders on the file, 'fulltext'=Zotero indexed fulltext, 'hybrid'=
            fulltext then fall back to the file)
        zotero_include_notes: also emit a doc per Zotero note
        zotero_include_metadata: also emit a per-item metadata/abstract doc
        **extra_args: Additional arguments to pass to each document

    Returns:
        List of DocDict objects, each a loadable sub-document
    """
    from wdoc.utils.loaders.zotero import (
        attachment_fulltext,
        attachment_to_file,
        format_bib_header,
        get_zotero_client,
        item_children,
        item_web_url,
        metadata_document_text,
        parse_selector,
        resolve_items,
    )

    logger.info(f"Loading zotero selector: '{path}'")
    selector = parse_selector(path)
    zot = get_zotero_client(
        connection=zotero_connection,
        library_id=zotero_library_id,
        library_type=zotero_library_type,
        api_key=zotero_api_key,
    )
    items = resolve_items(zot, selector)
    assert items, f"No Zotero items found for selector '{path}'"

    temp_dir = Path(tempfile.mkdtemp(prefix="wdoc_zotero_"))
    recur_parent_id = str(uuid.uuid4())
    doclist: List[DocDict] = []

    def _emit(filetype: str, location: str, title: str, item: dict, subitem=None):
        doc_kwargs = cli_kwargs.copy()
        for k in list(doc_kwargs.keys()):
            if k.startswith("zotero_"):
                del doc_kwargs[k]
        doc_kwargs.pop("filetype", None)
        doc_kwargs["path"] = location
        doc_kwargs["filetype"] = filetype
        doc_kwargs["title"] = title
        doc_kwargs["subitem_link"] = subitem or item_web_url(item) or ""
        doc_kwargs.update(extra_args)
        doc_kwargs["recur_parent_id"] = recur_parent_id
        # A fan-out over a whole library will inevitably hit individual items
        # that fail to load (a sparse bibliographic entry too short for wdoc's
        # minimum length check, a corrupt or DRM'd attachment, an unsupported
        # attachment type, ...). Degrade gracefully: warn and skip that
        # sub-document instead of crashing the entire selection. The batch
        # still raises if every single sub-document fails.
        doc_kwargs["loading_failure"] = "warn"
        doclist.append(DocDict(doc_kwargs))

    def _write_text(name: str, text: str) -> str:
        out = temp_dir / name
        out.write_text(text)
        return str(out)

    for item in items:
        data = item.get("data", {})
        title = data.get("title") or "Untitled"
        key = item["key"]
        children = item_children(zot, item)
        attachments = [
            c for c in children if c.get("data", {}).get("itemType") == "attachment"
        ]

        if zotero_include_metadata:
            meta_text = metadata_document_text(item)
            if meta_text.strip():
                _emit(
                    "txt",
                    _write_text(f"{key}_metadata.txt", meta_text),
                    f"{title} (metadata)",
                    item,
                )

        for att in attachments:
            att_title = att.get("data", {}).get("title") or title
            att_key = att["key"]

            if zotero_attachment_text in ("fulltext", "hybrid"):
                fulltext = attachment_fulltext(zot, att_key)
                if fulltext:
                    header = format_bib_header(item)
                    content = f"{header}\n\n{fulltext}" if header else fulltext
                    _emit(
                        "txt",
                        _write_text(f"{att_key}_fulltext.txt", content),
                        att_title,
                        item,
                    )
                    continue
                if zotero_attachment_text == "fulltext":
                    logger.warning(
                        f"No indexed fulltext for attachment {att_key} of "
                        f"'{title}', skipping (zotero_attachment_text=fulltext)"
                    )
                    continue
                # hybrid: fall through to materialising the file

            materialised = attachment_to_file(zot, att, temp_dir)
            if materialised is None:
                continue
            ftype, location = materialised
            _emit(ftype, location, att_title, item, subitem=location)

        if zotero_include_notes:
            notes = [c for c in children if c.get("data", {}).get("itemType") == "note"]
            for note in notes:
                note_html = note.get("data", {}).get("note", "") or ""
                if not note_html.strip():
                    continue
                header = format_bib_header(item)
                content = f"{header}\n\nNote:\n{note_html}" if header else note_html
                _emit(
                    "txt",
                    _write_text(f"{note['key']}_note.txt", content),
                    f"{title} (note)",
                    item,
                )

    assert doclist, (
        f"Zotero selector '{path}' produced no loadable documents "
        f"(items found: {len(items)})"
    )
    return doclist


@memoizer
def parse_load_functions(load_functions: Tuple[str, ...]) -> bytes:
    load_functions = list(load_functions)
    assert isinstance(load_functions, list), "load_functions must be a list"
    assert all(isinstance(lf, str) for lf in load_functions), (
        "load_functions elements must be strings"
    )
    import dill

    try:
        for ilf, lf in enumerate(load_functions):
            load_functions[ilf] = eval(lf)
    except Exception as err:
        raise Exception(f"Error when evaluating load_functions #{ilf}: {lf} '{err}'")
    load_functions = tuple(load_functions)
    pickled = dill.dumps(load_functions)
    return pickled


recursive_types_func_mapping = {
    "recursive_paths": parse_recursive_paths,
    "json_entries": parse_json_entries,
    "toml_entries": parse_toml_entries,
    "link_file": parse_link_file,
    "youtube_playlist": parse_youtube_playlist,
    "ddg": parse_ddg_search,
    "zotero": parse_zotero,
    "auto": None,
}
