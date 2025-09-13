import re
import time
from pathlib import Path

import ffmpeg
import playwright
import pydub
import uuid6
import yt_dlp as youtube_dl
from beartype.typing import List, Literal, Optional
from langchain.docstore.document import Document
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.loaders.local_audio import load_local_audio
from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import doc_loaders_cache, file_hasher, optional_strip_unexp_args


def find_online_media(
    url: str,
    online_media_url_regex: Optional[str] = None,
    online_media_resourcetype_regex: Optional[str] = None,
    headless: bool = True,
) -> dict:

    def check_browser_installation(browser_type: str, crash: bool = False) -> bool:
        try:
            with playwright.sync_api.sync_playwright() as p:
                browser = getattr(p, browser_type).launch()
                browser.close()
            return True
        except Exception as err:
            if crash:
                raise
            if "p" in locals():
                logger.warning(str(p))
            logger.warning(str(err))
            return False

    # the media request will be stored in this dict
    video_urls = {
        "url_regex": [],
        "resourcetype_regex": [],
        "media": [],
        "mpeg": [],
        "mp4": [],
        "mp3": [],
        "m3u": [],
    }
    if online_media_url_regex:
        online_media_url_regex = re.compile(online_media_url_regex)
    if online_media_resourcetype_regex:
        online_media_resourcetype_regex = re.compile(online_media_resourcetype_regex)
    nonmedia_urls = []

    def request_filter(req) -> None:
        if online_media_url_regex is not None and online_media_url_regex.match(req.url):
            video_urls["url_regex"].append(req.url)
        elif (
            online_media_resourcetype_regex is not None
            and online_media_resourcetype_regex.match(req.resource_type)
        ):
            video_urls["resourcetype_regex"].append(req.url)
        elif req.resource_type == "media":
            video_urls["media"].append(req.url)
        elif "media" in req.resource_type:
            video_urls["media"].append(req.url)
        elif "mpeg" in req.resource_type:
            video_urls["mpeg"].append(req.url)
        elif "m3u" in req.resource_type or ".m3u" in req.url:
            video_urls["m3u"].append(req.url)
        elif "mp3" in req.resource_type or ".mp3" in req.url:
            video_urls["mp3"].append(req.url)
        elif "mp4" in req.resource_type or ".mp4" in req.url:
            video_urls["mp4"].append(req.url)
        else:
            nonmedia_urls.append(req.url)

    if check_browser_installation("firefox"):
        installed = "firefox"
    elif check_browser_installation("chromium"):
        installed = "chromium"
    else:
        logger.warning(
            "Couldn't launch either firefox or chromium using playwright. "
            "Maybe try running 'playwright install'? Retrying to load them on "
            "purpose to make us crash and display the actual error."
        )
        check_browser_installation("firefox", crash=True)
        check_browser_installation("chromium", crash=True)
        raise Exception("We should have crashed earlier?!")

    with playwright.sync_api.sync_playwright() as p:
        browser = getattr(p, installed).launch(headless=headless)

        context = browser.new_context(
            java_script_enabled=True,
            geolocation={
                "latitude": 38.8954381,
                "longitude": -77.0312812,
            },
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            ignore_https_errors=True,
        )
        page = context.new_page()

        # start logging requests
        page.on("request", lambda request: request_filter(request))
        browser.on("request", lambda request: request_filter(request))
        context.on("request", lambda request: request_filter(request))

        # load page
        page.goto(url)
        try:
            page.wait_for_load_state("networkidle")
        except Exception as e:
            logger.debug(
                f"Ignoring exception on wait_for_load_state('networkidle'): {e}"
            )

        # Scroll the page to trigger lazy-loaded content
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)  # Wait for X seconds after scrolling
        page.evaluate("window.scrollTo(0, 0)")

        # Try to click on video play buttons using various selectors
        play_button_selectors = [
            '[class*="play-button"]',
            '[class*="play_button"]',
            '[class*="playbutton"]',
            '[class*="playback"]',
            '[class*="play-back"]',
            '[class*="play_back"]',
            '[class*="play"]',
            '[aria-label="Play"]',
            ".ytp-play-button",
            ".play-button",
            '[aria-label="播放"]',
            "div.avp-icon.avp-icon-playback",
        ]
        for selector in play_button_selectors:
            try:
                # Try clicking directly first (for specific selectors)
                page.click(selector, timeout=200)
                logger.debug(f"Clicked element matching selector: {selector}")
                page.wait_for_timeout(1000)  # Wait after click
                continue  # Move to next selector if successful
            except Exception:
                # If direct click fails or selector is general (like class*), try querying all
                try:
                    playback_elements = page.query_selector_all(selector)
                    for element in playback_elements:
                        if not element.is_visible() or not element.is_enabled():
                            continue
                        logger.debug(f"Found clickable element via query: {element}")
                        try:
                            element.click(timeout=500)
                            logger.debug(
                                f"Clicked element: {element.evaluate('el => el.outerHTML')}"
                            )
                            page.wait_for_timeout(1000)  # Wait after click
                            # Don't break here, maybe multiple elements match a general selector
                        except Exception as click_err:
                            logger.debug(
                                f"Failed to click element {element}: {click_err}"
                            )
                except Exception as query_err:
                    logger.debug(
                        f"Failed to query or click elements for selector {selector}: {query_err}"
                    )

        if not any(v for v in video_urls.values()):
            # Wait a bit more for any video to start loading if no media URLs found yet
            page.wait_for_timeout(10000)

        browser.close()

    # deduplicate urls
    for k, v in video_urls.items():
        video_urls[k] = list(set(v))

    return video_urls


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_online_media(
    path: str,
    audio_backend: Literal["whisper", "deepgram"],
    loaders_temp_dir: Path,
    audio_unsilence: bool = True,
    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,
    deepgram_kwargs: Optional[dict] = None,
    online_media_url_regex: Optional[str] = None,
    online_media_resourcetype_regex: Optional[str] = None,
) -> List[Document]:

    urls_to_try = [path]
    extra_media = find_online_media(
        url=path,
        online_media_url_regex=online_media_url_regex,
        online_media_resourcetype_regex=online_media_resourcetype_regex,
    )
    for k in [
        "url_regex",
        "resourcetype_regex",
        "media",
        "mpeg",
        "mp4",
        "mp3",
        "m3u",
    ]:
        urls_to_try.extend(extra_media[k])
    urls_to_try = list(set(urls_to_try))
    logger.info(f"Found {len(urls_to_try)} urls to try to get the media:")
    for u in urls_to_try:
        logger.info(f"  - {u}")

    def dl_audio_from_url(trial: int, url: str) -> Path:
        file_name = (
            loaders_temp_dir / f"online_media_{uuid6.uuid6()}"
        )  # without extension!
        ydl_opts = {
            # 'format': 'bestaudio/best',
            "format": "bestaudio/best",
            # 'force_generic_extractor': True,
            # 'default_search': 'auto',
            # 'match_filter': lambda x: None,
            "hls_prefer_native": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            # with extension
            "outtmpl": f"{file_name.absolute().resolve()}.%(ext)s",
            "verbose": env.WDOC_VERBOSE,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        candidate = []
        for f in loaders_temp_dir.iterdir():
            if file_name.name in f.name:
                candidate.append(f)
        assert len(candidate), f"Audio file of {url} failed to download?"
        assert (
            len(candidate) == 1
        ), f"Multiple audio file found for video: '{candidate}'"
        audio_file = candidate[0].absolute()
        return audio_file

    audio_file = None
    good_url = None
    for iurl, url in enumerate(urls_to_try):
        try:
            audio_file = dl_audio_from_url(trial=iurl, url=url)
            good_url = url
            break
        except Exception as err:
            logger.warning(
                f"Failed #{iurl+1}/{len(urls_to_try)} to download a media from url '{url}': '{err}'"
            )

    assert audio_file is not None, f"Failed to find suitable media for url '{path}'"

    audio_hash = file_hasher({"path": str(Path(audio_file).absolute())})
    audio_path = loaders_temp_dir / f"audio_from_video_{uuid6.uuid6()}.mp3"
    assert not audio_path.exists()

    # extract audio from video (sometimes instead of just the audio the whole video is downloaded)
    try:
        logger.info(
            f"Exporting audio from {audio_file} to {audio_path} (this can take some time)"
        )
        t = time.time()
        ffmpeg.input(
            audio_file,
        ).output(str(audio_path.resolve().absolute())).run()
        logger.info(f"Done extracting audio in {time.time()-t:.2f}s")
    except Exception as err:
        logger.warning(
            f"Error when getting audio from video using ffmpeg. Retrying with pydub. Error: '{err}'"
        )

        try:
            logger.debug(f"Audio path: '{audio_path}'")
            # don't delete it as some users might need it
            # Path(audio_path).unlink(missing_ok=True)
            audio = pydub.AudioSegment.from_file(audio_file)
            # extract audio from video
            logger.info(
                f"Extracting audio from {audio_file} to {audio_path} (this can take some time)"
            )
            t = time.time()
            audio.export(audio_path, format="mp3")
            logger.info(f"Done extracting audio in {time.time()-t:.2f}s")
        except Exception as err:
            raise Exception(
                f"Error when getting audio from video using ffmpeg: '{err}'"
            )

    assert Path(audio_path).exists(), f"FileNotFound: {audio_path}"

    # now need the hash from the mp3, not video
    audio_hash = file_hasher({"path": audio_path})

    sub_loaders_temp_dir = loaders_temp_dir / "local_audio"
    sub_loaders_temp_dir.mkdir()
    parsed_audio = load_local_audio(
        path=audio_path,
        loaders_temp_dir=sub_loaders_temp_dir,
        file_hash=audio_hash,
        audio_backend=audio_backend,
        whisper_lang=whisper_lang,
        whisper_prompt=whisper_prompt,
        deepgram_kwargs=deepgram_kwargs,
        audio_unsilence=audio_unsilence,
    )

    for ipa, pa in enumerate(parsed_audio):
        parsed_audio[ipa].metadata["online_media_url"] = str(good_url)

    return parsed_audio
