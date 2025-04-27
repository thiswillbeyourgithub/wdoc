# Media URL Finder

This script helps find direct URLs for video and audio media embedded within a webpage.

## Purpose

Sometimes, tools like `yt-dlp` might not directly find the media URLs on a complex webpage. This script uses the `find_online_media` function from [wdoc](https://github.com/thiswillbeyourgithub/wdoc) as a fallback. `find_online_media` uses `playwright` to load the page in a headless browser and intercepts network requests to identify potential media URLs based on regex patterns.

After finding potential URLs, the script uses `yt-dlp` to fetch metadata for each found link and returns the results as a JSON object.

## Usage

```bash
python scripts/MediaURLFinder/media_url_finder.py --url="<URL_OF_THE_WEBPAGE>" [OPTIONS]
```

Replace `<URL_OF_THE_WEBPAGE>` with the actual URL you want to scan.

**Optional arguments:**

You can pass additional arguments accepted by `wdoc.utils.loaders.find_online_media`, such as:

*   `--online_media_url_regex`: Custom regex to match media URLs.
*   `--online_media_resourcetype_regex`: Custom regex to match resource types (e.g., 'media', 'video').
*   `--headless=False`: Run the browser in non-headless mode (useful for debugging).

**Example:**

```bash
python scripts/MediaURLFinder/media_url_finder.py --url="https://example.com/page_with_embedded_video"
```

This will output a JSON string containing the media URLs found and their metadata fetched by `yt-dlp`. If no media links are found, it will print a message and exit.

*(This README was generated with the help of [aider.chat](https://github.com/Aider-AI/aider/issues))*
