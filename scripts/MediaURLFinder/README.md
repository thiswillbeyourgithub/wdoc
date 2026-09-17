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

*   `--urls_only`: Print the discovered media URLs one per line (deduplicated) instead of the JSON metadata. Made to be piped into `yt-dlp` (see below).

You can also pass any argument accepted by `wdoc.utils.loaders.find_online_media`, such as:

*   `--online_media_url_regex`: Custom regex to match media URLs.
*   `--online_media_resourcetype_regex`: Custom regex to match resource types (e.g., 'media', 'video').
*   `--headless=False`: Run the browser in non-headless mode (useful for debugging).

**Example:**

```bash
uv run scripts/MediaURLFinder/media_url_finder.py --url="https://example.com/page_with_embedded_video"
```

This will output a JSON string containing the media URLs found and their metadata fetched by `yt-dlp`. Buckets that matched nothing are omitted, and a link that `yt-dlp` cannot handle is reported as `{"error": "..."}` instead of aborting the run. If no media links are found, it will print a message and exit.

**Downloading directly with `yt-dlp`:**

```bash
uv run scripts/MediaURLFinder/media_url_finder.py --url="https://example.com/page_with_embedded_video" --urls_only | yt-dlp -a -
```

*(This README was generated with the help of [aider.chat](https://github.com/Aider-AI/aider/issues), then updated with Claude Code)*
