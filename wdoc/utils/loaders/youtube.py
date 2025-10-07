import json
import re
from pathlib import Path

import ftfy
import requests
import uuid6
import yt_dlp as youtube_dl
from beartype.typing import List, Literal, Optional
from langchain.docstore.document import Document
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.loaders.local_audio import (
    transcribe_audio_deepgram,
    transcribe_audio_whisper,
)
from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.loaders.shared_audio import (
    convert_verbose_json_to_timestamped_text,
    process_vtt_content_for_llm,
    seconds_to_timecode,
)
from wdoc.utils.misc import doc_loaders_cache, file_hasher, optional_strip_unexp_args

yt_link_regex = re.compile("youtube.*watch")

# to check that a youtube link is valid
emptyline_regex = re.compile(r"^\s*$", re.MULTILINE)
emptyline2_regex = re.compile(r"\n\n+", re.MULTILINE)
linebreak_before_letter = re.compile(
    r"\n([a-záéíóúü])", re.MULTILINE
)  # match any linebreak that is followed by a lowercase letter


@debug_return_empty
@optional_strip_unexp_args
def load_youtube(
    path: str,
    loaders_temp_dir: Path,
    youtube_language: Optional[str] = None,
    youtube_translation: Optional[str] = None,
    youtube_audio_backend: Literal["youtube", "whisper", "deepgram"] = "youtube",
    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,
    deepgram_kwargs: Optional[dict] = None,
) -> List[Document]:
    assert youtube_audio_backend in [
        "youtube",
        "whisper",
        "deepgram",
    ], f"Invalid value for youtube_audio_backend. Must be either youtube, whisper or deepgram, not '{youtube_audio_backend}'"

    if "\\" in path:
        logger.warning(f"Removed backslash found in '{path}'")
        path = path.replace("\\", "")

    if not yt_link_regex.search(path):
        logger.info(f"Not a youtube link but trying anyway: '{path}'")

    if youtube_audio_backend == "youtube":
        logger.info(f"Using youtube.com loader: '{path}'")
        logger.warning(
            "You are using the youtube_audio_backend `youtube`, which depends on the actual youtube subtitles. For better result, it is recommended to use the `whisper` or `deepgram` audio backends."
        )
        try:
            docs = cached_yt_loader(
                path=path,
                add_video_info=True,
                language=(
                    [youtube_language] if youtube_language else ["en", "en-US", "en-UK"]
                ),
                translation=youtube_translation if youtube_translation else None,
            )
        except Exception as err:
            raise Exception(
                f"Error when using yt-dlp. Keep in mind that youtube frequently changed its backend so upgrading yt-dlp to its latest version can often fix issues. Original error was: '{err}'"
            ) from err
    else:
        logger.info(f"Downloading audio from url: '{path}'")
        file_name = (
            loaders_temp_dir / f"youtube_audio_{uuid6.uuid6()}"
        )  # without extension!
        ydl_opts = {
            "format": "bestaudio/best",
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
            ydl.download([path])
        candidate = []
        for f in loaders_temp_dir.iterdir():
            if file_name.name in f.name:
                candidate.append(f)
        assert len(candidate), f"Audio file of {path} failed to download?"
        assert (
            len(candidate) == 1
        ), f"Multiple audio file found for video: '{candidate}'"
        audio_file = str(candidate[0].absolute())
        audio_hash = file_hasher({"path": audio_file})

        if youtube_audio_backend == "whisper":
            content = transcribe_audio_whisper(
                audio_path=audio_file,
                audio_hash=audio_hash,
                language=whisper_lang,
                prompt=whisper_prompt,
            )

            timestamped_text = convert_verbose_json_to_timestamped_text(content)

            docs = [
                Document(
                    page_content=timestamped_text,
                    metadata={
                        "source": "youtube_whisper",
                    },
                )
            ]
            if "duration" in content:
                docs[-1].metadata["duration"] = content["duration"]
            if "language" in content:
                docs[-1].metadata["language"] = content["language"]
            elif whisper_lang:
                docs[-1].metadata["language"] = whisper_lang

        elif youtube_audio_backend == "deepgram":
            content = transcribe_audio_deepgram(
                audio_path=audio_file,
                audio_hash=audio_hash,
                deepgram_kwargs=deepgram_kwargs,
            )
            assert (
                len(content["results"]["channels"]) == 1
            ), "unexpected deepgram output"
            assert (
                len(content["results"]["channels"][0]["alternatives"]) == 1
            ), "unexpected deepgram output"
            text = content["results"]["channels"][0]["alternatives"][0]["paragraphs"][
                "transcript"
            ].strip()
            assert text, "Empty text from deepgram transcription"

            docs = [
                Document(
                    page_content=text,
                    metadata={
                        "source": "youtube_deepgram",
                    },
                )
            ]
            docs[-1].metadata.update(content["metadata"])
            docs[-1].metadata["deepgram_kwargs"] = deepgram_kwargs

        else:
            raise ValueError(youtube_audio_backend)

        for f in Path(audio_file).parent.iterdir():
            if str(file_name.name) in f.stem:
                f.unlink()
        assert not Path(audio_file).exists()

    return docs


@doc_loaders_cache.cache
def cached_yt_loader(
    path: str, add_video_info: bool, language: List[str], translation: Optional[str]
) -> List[Document]:
    logger.debug(f"Not using cache for youtube {path}")

    options = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": language,
        "skip_download": True,
        "subtitlesformat": "vtt",
        "allsubtitles": True,
        "extract_flat": False,
    }
    if translation is None:
        translation = []
    else:
        translation = [translation]

    with youtube_dl.YoutubeDL(options) as ydl:
        # First check available subs
        info = ydl.extract_info(path, download=False)

        title = info.get("fulltitle", None)

        # Check both manual and auto subs
        good_subs = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})

        if not good_subs and not auto_subs:
            raise Exception(
                f"No subtitles found for youtube video entitled '{title}' at link '{path}'"
            )

        sub = None
        for subs in [good_subs, auto_subs]:
            if sub is not None:
                break
            for lang in language + translation:
                if lang in subs.keys():
                    sub_url = [s for s in subs[lang] if s["ext"] == "vtt"][0]["url"]
                    sub = requests.get(sub_url).content
                    sub = ftfy.fix_text(sub.decode()).strip()
                    if not sub:
                        continue
                    break
        if not sub:
            available = list(set(list(good_subs.keys()) + list(auto_subs.keys())))
            raise Exception(
                f"Subtitles found but not for the languages '{language}' nor '{translation}' for youtube video entitled '{title}' at link '{path}'\nAvailable languages were: '{available}'"
            )

    # get metadata too
    meta = {"title": title, "author": info["channel"]}
    for k in [
        "description",
        "categories",
        "tags",
        "channel",
        "upload_date",
        "duration_string",
        "language",
    ]:
        if k in info and info[k]:
            meta["yt_" + k] = info[k]

    # the chapters, if present, are in seconds, while the vtt uses human readable timecodes so converting the chapters
    if "chapters" in info and info["chapters"]:
        chap = info["chapters"]

        for ich, ch in enumerate(chap):
            chap[ich]["start"] = seconds_to_timecode(chap[ich]["start_time"])
            chap[ich]["end"] = seconds_to_timecode(chap[ich]["end_time"])
            del chap[ich]["start_time"], chap[ich]["end_time"]

        meta["yt_chapters"] = json.dumps(chap, ensure_ascii=False)

    assert sub, "The found subtitles are empty. Try running that command again."

    content = process_vtt_content_for_llm(sub, remove_hour_prefix=True)

    docs = [
        Document(
            page_content=content,
            metadata=meta,
        )
    ]

    return docs


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache
def load_youtube_playlist(playlist_url: str) -> dict:
    with youtube_dl.YoutubeDL({"quiet": False}) as ydl:
        try:
            loaded = ydl.extract_info(playlist_url, download=False)
        except (
            KeyError,
            youtube_dl.utils.DownloadError,
            youtube_dl.utils.ExtractorError,
        ) as e:
            raise Exception(
                logger.warning(
                    f"ERROR: Youtube playlist link skipped because : error during information \
        extraction from {playlist_url} : {e}"
                )
            )
    return loaded
