# Media URL Finder

This script helps find direct URLs for video and audio media embedded within a webpage.

## Purpose

Sometimes, tools like `yt-dlp` might not directly find the media URLs on a complex webpage. This script uses the `find_online_media` function from [wdoc](https://github.com/thiswillbeyourgithub/wdoc) as a fallback. `find_online_media` uses `playwright` to load the page in a headless browser and intercepts network requests to identify potential media URLs based on regex patterns.

After finding potential URLs, the script uses `yt-dlp` to fetch metadata for each found link and returns the results as a JSON object.

Subtitle tracks (`.srt`, `.vtt`) are dropped from every output format. A player fetches them over the same network requests as the video, so the browser reports them alongside it, but they are never what the caller is after and `yt-dlp` fetches the subtitles it was asked for by itself. The check looks at the URL path only, so a query string neither smuggles one through nor causes a false positive.

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
    *   `metadata`: the full `yt-dlp` metadata for every link, grouped by the bucket it was discovered in, plus the browser tab title under `page_title`. Buckets that matched nothing are omitted, and a link `yt-dlp` cannot handle is reported as `{"error": "..."}` instead of aborting the run.
    *   `urls`: one media URL per line, deduplicated, discovery order preserved. Made to be piped into `yt-dlp -a -` (see below).
    *   `json`: a JSON list of `{"url", "title", "page_title", "filesize", "duration", "ext"}`, one entry per deduplicated link, in discovery order, with an extra `"error"` key (and null fields) when `yt-dlp` could not read the link. This is the format to parse with `jq`.
        *   `page_title` is the browser tab title, repeated on every entry so a `jq` expression that selects one entry still carries it. A scraped media URL usually has an opaque CDN filename, so the tab title is generally the better name for the downloaded file.
        *   `filesize` is in bytes and is a **ranking aid, not an accurate figure**. `yt-dlp` only knows an exact size for a progressive file; for an HLS or DASH manifest the script falls back to `duration * bitrate`, and to the largest listed format when the top level advertises nothing. It is `null` when even that is impossible.
        *   It exists because the first media URL a page loads is very often a pre-roll ad or a trailer, not the feature. Sorting on it is what picks the real video: `jq 'sort_by(.filesize // -1) | reverse'`.
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

**Picking the biggest link with `jq`:**

```bash
page="https://example.com/page_with_embedded_video"
found="$(uv run scripts/MediaURLFinder/media_url_finder.py --url="$page" --output_format=json \
    | jq -r 'sort_by(.filesize // -1) | reverse | .[0]')"
yt-dlp --referer "$page" \
    -o "$(jq -r '.page_title' <<<"$found" | sed -e 's/%/%%/g' -e 's#/#_#g').%(ext)s" \
    "$(jq -r '.url' <<<"$found")"
```

Note the `sed`: a title lands in a `yt-dlp` output template, where a `/` would spawn directories and a `%` is read as a field, so `%(title)s` inside a title really does get substituted. Leaving the extension as `%(ext)s` is what makes the file end up with whatever `yt-dlp` actually wrote, after any muxing.

Passing the page as `--referer` matters more often than not: scraped media URLs are usually served by a CDN that rejects a request coming from nowhere.

*(This README was generated with the help of [aider.chat](https://github.com/Aider-AI/aider/issues), then updated with Claude Code)*
