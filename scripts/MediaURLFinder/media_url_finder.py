"""
This script utilizes the `find_online_media` function from `wdoc.utils.loaders`
to automatically discover URLs for video and audio content embedded within a
given webpage URL. It serves as a fallback mechanism when `yt-dlp` is unable
to directly identify the media links on the page.
"""

import json
import sys
import fire
from wdoc.utils.loaders.online_media import find_online_media
import yt_dlp as youtube_dl

ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}


def main(url: str, **kwargs) -> str:
    out = find_online_media(url=url, **kwargs)
    if not any(v for v in out.values()):
        print("No media links found")
        sys.exit(1)

    d = {}

    for k, v in out.items():
        # skip the buckets that matched nothing instead of cluttering the output
        if not v:
            continue
        d[k] = {}
        for link in v:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(link, download=False)
                    d[k][link] = ydl.sanitize_info(info)
                except Exception as err:
                    # a media request is not necessarily something yt-dlp can handle
                    d[k][link] = {"error": str(err)}

    return json.dumps(d)


if __name__ == "__main__":
    fire.Fire(main)
