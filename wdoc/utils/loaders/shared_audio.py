import os
import re
import time
from pathlib import Path

import joblib
from beartype.typing import List, Optional, Union
from loguru import logger

from wdoc.utils.env import env
from wdoc.utils.misc import doc_loaders_cache, file_hasher


def seconds_to_timecode(inp: Union[str, float, int]) -> str:
    "used for vtt subtitle conversion"
    second = float(inp)
    minute = second // 60
    second = second % 60
    hour = minute // 60
    minute = minute % 60
    hour, minute, second = int(hour), int(minute), int(second)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def timecode_to_second(inp: str) -> int:
    "turns a vtt timecode into seconds"
    hour, minute, second = map(int, inp.split(":"))
    return hour * 3600 + minute * 60 + second


def is_timecode(inp: Union[float, str]) -> bool:
    try:
        timecode_to_second(inp)
        return True
    except Exception:
        return False


def process_vtt_content_for_llm(
    vtt_content: str, remove_hour_prefix: bool = True
) -> str:
    """
    Process VTT content to make it more suitable for LLMs by reducing timecodes
    and removing unnecessary formatting.

    Args:
        vtt_content: The VTT content to process
        remove_hour_prefix: Whether to remove "00:" hour prefix if all content is under 99 minutes

    Returns:
        Processed text content optimized for LLM consumption
    """
    # Reduce greatly the number of token in the subtitles by removing some less important formatting
    lines = vtt_content.splitlines()
    timecode_pattern = re.compile(
        r"(?:\d{2}:\d{2}:\d{2}\.\d{3})|(?:<\d{2}:\d{2}:\d{2}\.\d{3}>)|(?:</?c>)"
    )
    latest_tc = -1  # store the timecode once every Xs
    newlines = []

    for li in lines:
        if " --> " in li:
            li = re.sub(r"\.\d+ -->.*", "", li).strip()

            # remove duplicate timecodes:
            tc = timecode_to_second(li)
            if tc - latest_tc < 15:
                li = ""
            else:
                latest_tc = tc
        else:
            li = timecode_pattern.sub("", li).strip()

        is_tc = is_timecode(li)

        # We need at least one line, but skeep the lines before the first timecode
        if not newlines:
            if is_tc:
                newlines.append(li)
            continue

        # Check no consecutive timecodes (for cached_yt_loader compatibility)
        elif len(newlines) >= 2:
            if is_tc and is_timecode(newlines[-1]):
                # Skip consecutive timecodes to avoid VTT format issues
                continue

        if is_tc:
            newlines.append(li + " ")
        elif is_timecode(newlines[-1]):
            newlines[-1] += " " + li.strip()
        elif li not in newlines[-1]:
            newlines[-1] += " " + li.strip() if newlines[-1].strip() else li.strip()

    newlines = [nl.strip() for nl in newlines]

    # If the total length is less than 99 minutes, we remove the hour mark
    if remove_hour_prefix and newlines and newlines[-1].startswith("00:"):
        newlines = [nl[3:] if nl.startswith("00:") else nl for nl in newlines]

    content = "\n".join(newlines)
    return content


def convert_verbose_json_to_timestamped_text(transcript: dict) -> str:
    # turn the json into vtt, then reuse the code used for youtube chapters
    buffer = ""
    for seg in transcript["segments"]:
        start = seconds_to_timecode(seg["start"])
        end = seconds_to_timecode(seg["end"])
        text = seg["text"]
        buffer += f"\n\n{start}.0 --> {end}\n{text}"

    buffer = buffer.strip()

    content = process_vtt_content_for_llm(buffer, remove_hour_prefix=False)
    return content


@doc_loaders_cache.cache(ignore=["audio_path"])
def transcribe_audio_deepgram(
    audio_path: Union[str, Path],
    audio_hash: str,
    deepgram_kwargs: Optional[dict] = None,
) -> dict:
    "Use whisper to transcribe an audio file"
    import httpx
    import deepgram

    logger.info(f"Calling deepgram to transcribe {audio_path}")
    assert (
        not env.WDOC_PRIVATE_MODE
    ), "Private mode detected, aborting before trying to use deepgram's API"
    assert (
        "DEEPGRAM_API_KEY" in os.environ
        and not os.environ["DEEPGRAM_API_KEY"]
        == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
    ), "No environment variable DEEPGRAM_API_KEY found"

    # client
    try:
        client = deepgram.DeepgramClient()
    except Exception as err:
        raise Exception(f"Error when creating deepgram client: '{err}'")

    # set options
    options = dict(
        # docs: https://playground.deepgram.com/?endpoint=listen&smart_format=true&language=en&model=nova-3
        model="nova-3",
        detect_language=True,
        # not all features below are available for all languages
        # intelligence
        summarize=False,
        topics=False,
        intents=False,
        sentiment=False,
        # transcription
        smart_format=True,
        punctuate=True,
        paragraphs=True,
        utterances=True,
        diarize=True,
        # redact=None,
        # replace=None,
        # search=None,
        # keywords=None,
        # filler_words=False,
    )
    if deepgram_kwargs is None:
        deepgram_kwargs = {}
    if "language" in deepgram_kwargs and deepgram_kwargs["language"]:
        del options["detect_language"]
    options.update(deepgram_kwargs)
    options = deepgram.PrerecordedOptions(**options)

    # load file
    with open(audio_path, "rb") as f:
        payload = {"buffer": f.read()}

    # get content
    t = time.time()
    content = client.listen.prerecorded.v("1").transcribe_file(
        payload,
        options,
        timeout=httpx.Timeout(300.0, connect=10.0),  # timeout for large files
    )
    logger.info(f"Done deepgram transcribing {audio_path} in {int(time.time()-t)}s")
    d = content.to_dict()
    return d


@doc_loaders_cache.cache(ignore=["audio_path"])
def transcribe_audio_whisper(
    audio_path: Union[Path, str],
    audio_hash: str,
    language: Optional[str],
    prompt: Optional[str],
) -> dict:
    "Use whisper to transcribe an audio file"
    import requests
    import litellm

    logger.info(f"Calling openai's whisper to transcribe {audio_path}")
    if env.WDOC_PRIVATE_MODE:
        assert (
            env.WDOC_WHISPER_ENDPOINT
        ), "WDOC_PRIVATE_MODE is set but no WDOC_WHISPER_ENDPOINT is set. Crashing as it seems like your private request would call a remote API"
        assert (
            not os.environ["WDOC_WHISPER_API_KEY"]
            == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
        ), "No environment variable WDOC_WHISPER_API_KEY found"
    else:
        assert (
            "OPENAI_API_KEY" in os.environ
            and not os.environ["OPENAI_API_KEY"]
            == "REDACTED_BECAUSE_WDOC_IN_PRIVATE_MODE"
        ), "No environment variable OPENAI_API_KEY found"

    try:
        t1 = time.time()
        with open(audio_path, "rb") as audio_file:
            # Prepare transcription arguments
            transcription_kwargs = {
                "model": env.WDOC_WHISPER_MODEL,
                "file": audio_file,
                "prompt": prompt,
                "language": language,
                "temperature": 0,
                "response_format": "verbose_json",
            }

            # Add custom endpoint and API key if provided
            if env.WDOC_WHISPER_ENDPOINT:
                transcription_kwargs["api_base"] = env.WDOC_WHISPER_ENDPOINT
                logger.debug(
                    f"Using custom whisper endpoint: {env.WDOC_WHISPER_ENDPOINT}"
                )

            if env.WDOC_WHISPER_API_KEY:
                transcription_kwargs["api_key"] = env.WDOC_WHISPER_API_KEY
                logger.debug("Using custom whisper API key")

            try:
                transcript = litellm.transcription(**transcription_kwargs).json()
            except Exception as litellm_err:
                logger.warning(
                    f"litellm.transcription failed with error: {litellm_err}. "
                    f"Falling back to direct requests call to whisper endpoint."
                )

                # Fallback to direct requests call
                if not env.WDOC_WHISPER_ENDPOINT:
                    raise Exception(
                        "litellm failed and no WDOC_WHISPER_ENDPOINT set for fallback"
                    ) from litellm_err

                # Prepare the multipart form data
                files = {"file": audio_file}
                data = {
                    "model": env.WDOC_WHISPER_MODEL,
                    "response_format": "verbose_json",
                    "temperature": 0,
                }

                if prompt:
                    data["prompt"] = prompt
                if language:
                    data["language"] = language

                headers = {}
                if env.WDOC_WHISPER_API_KEY:
                    headers["Authorization"] = f"Bearer {env.WDOC_WHISPER_API_KEY}"

                # Make the request
                endpoint_url = (
                    env.WDOC_WHISPER_ENDPOINT.rstrip("/") + "/v1/audio/transcriptions"
                )
                response = requests.post(
                    endpoint_url, files=files, data=data, headers=headers
                )
                response.raise_for_status()
                transcript = response.json()

        t2 = time.time()
        logger.info(f"Done transcribing {audio_path} in {int(t2-t1)}s")

    except Exception as e:
        if "Maximum content size limit" in str(e):
            audio_splits = split_too_large_audio(audio_path)

            # reconstitute appropriate durations
            transcripts = []

            if env.WDOC_WHISPER_PARALLEL_SPLITS:
                logger.info(f"Processing {len(audio_splits)} audio splits in parallel")

                def process_audio_split(f: Path) -> dict:
                    """Process a single audio split file."""
                    h = file_hasher({"path": f})
                    return transcribe_audio_whisper(
                        audio_path=f,
                        audio_hash=h,
                        language=language,
                        prompt=prompt,
                    )

                # Process splits in parallel using joblib
                transcripts = joblib.Parallel(
                    n_jobs=-1,
                    backend="threading",
                )(joblib.delayed(process_audio_split)(f) for f in audio_splits)
            else:
                logger.warning(
                    "Using sequential processing for whisper over audio splits"
                )

                for f in audio_splits:
                    h = file_hasher({"path": f})
                    temp = transcribe_audio_whisper(
                        audio_path=f,
                        audio_hash=h,
                        language=language,
                        prompt=prompt,
                    )
                    transcripts.append(temp)

            if len(transcripts) == 1:
                return transcripts[0]

            logger.info(f"Combining {len(transcripts)} audio splits into a single json")
            ref = transcripts.pop(0)
            if ref["words"] is not None:
                logger.warning(
                    "Warning: the transcript contains a 'words' output, which will be discarded as the combination of word timestamps is not yet supported."
                )
                ref["words"] = None
            for itrans, trans in enumerate(transcripts):
                assert trans["task"] == ref["task"]
                if trans["language"] != ref["language"]:
                    logger.warning(
                        f"Warning: the language of the reference split audio ({ref['language']}) is not the same as the language of the current split ({trans['language']})"
                    )
                if trans["words"] is not None:
                    logger.warning(
                        "Warning: the transcript contains a 'words' output, which will be discarded as the combination of word timestamps is not yet supported."
                    )
                    trans["words"] = None

                temp = trans["segments"]
                for it, t in enumerate(temp):
                    temp[it]["end"] += ref["duration"]
                    temp[it]["start"] += ref["duration"]

                ref["segments"].extend(temp)

                ref["duration"] += trans["duration"]
                ref["text"] += " [note: audio was split here] " + trans["text"]

            return ref

        else:
            raise
    return transcript


def split_too_large_audio(
    audio_path: Union[Path, str],
) -> List[Path]:
    """Whisper has a file size limit of about 25mb. If we hit that limit, we
    split the audio file into multiple 30 minute files, then combine the
    outputs
    """
    import ffmpeg

    audio_path = Path(audio_path)
    logger.info(
        f"Splitting large audio file '{audio_path}' into 30minute segment because it's too long for whisper"
    )
    split_folder = audio_path.parent / (audio_path.stem + "_splits")
    split_folder.mkdir(exist_ok=False)
    ext = audio_path.suffix

    ffmpeg.input(str(audio_path.absolute())).output(
        str((split_folder / f"split__%03d.{ext}").absolute()),
        c="copy",
        f="segment",
        segment_time=1600,  # 30 minute by default
    ).run()
    split_files = [f for f in split_folder.iterdir()]
    assert split_files
    return split_files
