# Media URL Finder

This script helps find direct URLs for video and audio media embedded within a webpage.

## Purpose

Sometimes, tools like `yt-dlp` might not directly find the media URLs on a complex webpage. This script uses the `find_online_media` function from [wdoc](https://github.com/thiswillbeyourgithub/wdoc) as a fallback. `find_online_media` uses `playwright` to load the page in a headless browser and intercepts network requests to identify potential media URLs based on regex patterns.

After finding potential URLs, the script uses `yt-dlp` to fetch metadata for each found link and returns the results as a JSON object.

## Usage

The script carries inline script metadata (PEP 723), so `uv` installs its dependencies on the fly and you never have to install wdoc yourself:

```bash
uv run --refresh scripts/MediaURLFinder/media_url_finder.py --url="<URL_OF_THE_WEBPAGE>" [OPTIONS]
```

`--refresh` bypasses the `uv` cache so the latest released wdoc is resolved on every run. Drop it if you prefer the cached (faster) resolution.

It also runs straight from GitHub, without cloning the repo:

```bash
uv run --refresh https://raw.githubusercontent.com/thiswillbeyourgithub/wdoc/main/scripts/MediaURLFinder/media_url_finder.py --url="<URL_OF_THE_WEBPAGE>"
```

The file is executable and its shebang is `#!/usr/bin/env -S uv run --script`, so `./scripts/MediaURLFinder/media_url_finder.py --url="..."` works too.

If you already have wdoc, `fire` and `yt-dlp` installed in your environment, the plain interpreter still works:

```bash
python scripts/MediaURLFinder/media_url_finder.py --url="<URL_OF_THE_WEBPAGE>" [OPTIONS]
```

Note: `find_online_media` drives a real browser, so you need the playwright browsers installed once (`playwright install firefox`, or chromium).

Replace `<URL_OF_THE_WEBPAGE>` with the actual URL you want to scan.

**Optional arguments:**

*   `--output_format`: one of `metadata` (the default), `urls` or `json`.
    *   `metadata`: the full `yt-dlp` metadata for every link, grouped by the bucket it was discovered in. Buckets that matched nothing are omitted, and a link `yt-dlp` cannot handle is reported as `{"error": "..."}` instead of aborting the run.
    *   `urls`: one media URL per line, deduplicated, discovery order preserved. Made to be piped into `yt-dlp -a -` (see below).
    *   `json`: an ordered JSON list of `{"url": ..., "title": ...}`, one entry per deduplicated link, with an extra `"error"` key (and a `null` title) when `yt-dlp` could not read the link. This is the format to parse with `jq`.
*   `--urls_only`: kept as an alias of `--output_format=urls`.

You can also pass any argument accepted by `wdoc.utils.loaders.find_online_media`, such as:

*   `--online_media_url_regex`: Custom regex to match media URLs.
*   `--online_media_resourcetype_regex`: Custom regex to match resource types (e.g., 'media', 'video').
*   `--headless=False`: Run the browser in non-headless mode (useful for debugging).

**Example:**

```bash
uv run scripts/MediaURLFinder/media_url_finder.py --url="https://example.com/page_with_embedded_video"
```

This will output a JSON string containing the media URLs found and their metadata fetched by `yt-dlp`. If no media links are found, it will print a message and exit with code 1.

**Downloading directly with `yt-dlp`:**

```bash
uv run scripts/MediaURLFinder/media_url_finder.py --url="https://example.com/page_with_embedded_video" --output_format=urls | yt-dlp -a -
```

**Picking a link by title with `jq`:**

```bash
uv run scripts/MediaURLFinder/media_url_finder.py --url="https://example.com/page_with_embedded_video" --output_format=json \
    | jq -r '.[0].url' \
    | xargs -r yt-dlp --referer "https://example.com/page_with_embedded_video"
```

Passing the page as `--referer` matters more often than not: scraped media URLs are usually served by a CDN that rejects a request coming from nowhere.

*(This README was generated with the help of [aider.chat](https://github.com/Aider-AI/aider/issues), then updated with Claude Code)*
