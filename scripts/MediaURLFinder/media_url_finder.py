#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wdoc[full]",
#     "fire",
#     "yt-dlp",
# ]
# ///
"""
This script utilizes the `find_online_media` function from `wdoc.utils.loaders`
to automatically discover URLs for video and audio content embedded within a
given webpage URL. It serves as a fallback mechanism when `yt-dlp` is unable
to directly identify the media links on the page.

It can be run without installing anything thanks to the inline script metadata
above, which always resolves the latest wdoc:

    uv run --refresh scripts/MediaURLFinder/media_url_finder.py --url="..."

Subtitle tracks (.srt, .vtt) are dropped from every format: the browser reports
them alongside the media they belong to, and they are never what is wanted.

Three output formats are available via --output_format:
  - metadata (default): the full yt-dlp metadata, grouped by discovery bucket,
    plus the browser tab title under "page_title"
  - urls: one media URL per line, ready to pipe into `yt-dlp -a -`
  - json: a JSON list of {"url", "title", "page_title", "filesize", "duration",
    "ext"}, one per link, in discovery order. `filesize` is what lets a caller
    tell the real video apart from the trailer or the ad that a page often
    loads first, for instance with `jq 'sort_by(.filesize // -1) | reverse'`,
    and `page_title` is the browser tab title, which is usually a far better
    name for the downloaded file than the media's own title.

    uv run scripts/MediaURLFinder/media_url_finder.py --url="..." --output_format=urls | yt-dlp -a -
"""

import json
import sys
from urllib.parse import urlparse
import fire
from wdoc.utils.loaders.online_media import find_online_media
import yt_dlp as youtube_dl

ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}

OUTPUT_FORMATS = ("metadata", "urls", "json")

# subtitle tracks ride the same network requests as the media they belong to,
# so the browser reports them alongside the video. They are never what the
# caller is after, and yt-dlp pulls the ones it wants by itself anyway.
IGNORED_EXTENSIONS = (".srt", ".vtt")


def is_ignored(link: str) -> bool:
    """True for a link this script should pretend it never saw."""
    # the extension lives in the path, a query string must not fool the check
    return urlparse(link).path.lower().endswith(IGNORED_EXTENSIONS)


def ordered_links(out: dict) -> list:
    """Flatten the discovery buckets into one list, deduplicated, order kept."""
    return [
        link
        for link in dict.fromkeys(link for v in out.values() for link in v)
        if not is_ignored(link)
    ]


def probe(link: str) -> dict:
    """Ask yt-dlp about a link, as a JSON serializable dict.

    A link yt-dlp cannot handle yields an `error` entry instead of raising, as
    a media request is not necessarily something yt-dlp knows what to do with.
    """
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            return ydl.sanitize_info(ydl.extract_info(link, download=False))
        except Exception as err:
            return {"error": str(err)}


def estimate_size(info: dict) -> int | None:
    """Best guess at the byte size of what yt-dlp would download, or None.

    yt-dlp only knows an exact `filesize` for a progressive file. An HLS or
    DASH manifest advertises none, so fall back to duration * bitrate, and to
    the largest listed format when the top level says nothing at all. This is
    a ranking aid, not an accurate figure.
    """

    def one(fmt: dict) -> int | None:
        for key in ("filesize", "filesize_approx"):
            val = fmt.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
        # tbr is in kbit/s, duration in seconds
        duration = fmt.get("duration") or info.get("duration")
        tbr = fmt.get("tbr")
        if duration and tbr:
            return int(duration * tbr * 1000 / 8)
        return None

    # what yt-dlp would actually pull when it has to mux video and audio
    requested = info.get("requested_formats") or []
    if requested:
        parts = [one(fmt) for fmt in requested if isinstance(fmt, dict)]
        if all(part is not None for part in parts) and parts:
            return sum(parts)

    sizes = [one(info)]
    sizes += [one(fmt) for fmt in info.get("formats") or [] if isinstance(fmt, dict)]
    sizes = [size for size in sizes if size is not None]
    return max(sizes) if sizes else None


def main(
    url: str,
    output_format: str = "metadata",
    urls_only: bool = False,
    **kwargs,
) -> str:
    if urls_only:
        # kept as an alias of --output_format=urls
        output_format = "urls"
    assert output_format in OUTPUT_FORMATS, (
        f"Invalid output_format '{output_format}', expected one of {OUTPUT_FORMATS}"
    )

    out = find_online_media(url=url, **kwargs)
    # not a bucket of links, and it would be iterated character by character
    page_title = out.pop("page_title", "")

    links = ordered_links(out)
    if not links:
        print("No media links found")
        sys.exit(1)

    if output_format == "urls":
        return "\n".join(links)

    if output_format == "json":
        entries = []
        for link in links:
            info = probe(link)
            entry = {
                "url": link,
                "title": info.get("title"),
                # repeated on every entry so a `jq` one-liner that selects a
                # single entry still has it, without nesting the whole list
                "page_title": page_title,
                "filesize": estimate_size(info),
                "duration": info.get("duration"),
                "ext": info.get("ext"),
            }
            if "error" in info:
                entry["error"] = info["error"]
            entries.append(entry)
        return json.dumps(entries)

    d = {"page_title": page_title}

    for k, v in out.items():
        kept = [link for link in v if not is_ignored(link)]
        # skip the buckets that matched nothing instead of cluttering the output
        if not kept:
            continue
        d[k] = {link: probe(link) for link in kept}

    return json.dumps(d)


if __name__ == "__main__":
    fire.Fire(main)
