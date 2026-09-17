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

Three output formats are available via --output_format:
  - metadata (default): the full yt-dlp metadata, grouped by discovery bucket
  - urls: one media URL per line, ready to pipe into `yt-dlp -a -`
  - json: an ordered JSON list of {"url": ..., "title": ...}, one per link

    uv run scripts/MediaURLFinder/media_url_finder.py --url="..." --output_format=urls | yt-dlp -a -
"""

import json
import sys
import fire
from wdoc.utils.loaders.online_media import find_online_media
import yt_dlp as youtube_dl

ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}

OUTPUT_FORMATS = ("metadata", "urls", "json")


def ordered_links(out: dict) -> list:
    """Flatten the discovery buckets into one list, deduplicated, order kept."""
    return list(dict.fromkeys(link for v in out.values() for link in v))


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
    if not any(v for v in out.values()):
        print("No media links found")
        sys.exit(1)

    if output_format == "urls":
        return "\n".join(ordered_links(out))

    if output_format == "json":
        entries = []
        for link in ordered_links(out):
            info = probe(link)
            entry = {"url": link, "title": info.get("title")}
            if "error" in info:
                entry["error"] = info["error"]
            entries.append(entry)
        return json.dumps(entries)

    d = {}

    for k, v in out.items():
        # skip the buckets that matched nothing instead of cluttering the output
        if not v:
            continue
        d[k] = {link: probe(link) for link in v}

    return json.dumps(d)


if __name__ == "__main__":
    fire.Fire(main)
