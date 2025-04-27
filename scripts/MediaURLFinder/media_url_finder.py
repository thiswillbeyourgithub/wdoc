"""
This script utilizes the `find_online_media` function from `wdoc.utils.loaders`
to automatically discover URLs for video and audio content embedded within a
given webpage URL. It serves as a fallback mechanism when `yt-dlp` is unable
to directly identify the media links on the page.
"""

import json
import sys
import fire
from wdoc.utils.loaders import find_online_media
import yt_dlp as youtube_dl

ydl_opts = {"dump_single_json": True, "simulate": True}


def main(url: str, **kwargs) -> str:
    out = find_online_media(url=url, **kwargs)
    if not any(v for v in out.values()):
        print("No media links found")
        sys.exit(1)

    d = {k: [] for k in out.keys()}

    for k, v in out.items():
        d[k] = {}
        for link in v:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                j = ydl.download([url])
                d[k][url] = j

    # Remove unused
    # keys = d.keys()
    # for k in keys:
    #     if not d[k]:
    #         del d[k]

    return json.dumps(d)


if __name__ == "__main__":
    fire.Fire(main)
