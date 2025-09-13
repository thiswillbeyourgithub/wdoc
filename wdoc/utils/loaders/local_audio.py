import time
from pathlib import Path

import ffmpeg
import uuid6
from beartype.typing import List, Literal, Optional, Union
from langchain.docstore.document import Document
from loguru import logger

try:
    import torchaudio
except Exception as e:
    # torchaudio can be tricky to install to just in case let's avoid crashing wdoc entirely
    logger.warning(f"Failed to import torchaudio: '{e}'")

from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.loaders.shared_audio import (
    convert_verbose_json_to_timestamped_text,
    transcribe_audio_deepgram,
    transcribe_audio_whisper,
)
from wdoc.utils.misc import doc_loaders_cache, file_hasher, optional_strip_unexp_args

# unsilence audio
sox_effects = [
    ["norm"],  # normalize audio
    # isolate voice frequency
    # human speech for low male is about 100hz and high female about 17khz
    ["highpass", "-1", "100"],
    ["lowpass", "-1", "17000"],
    # -2 is for a steeper filtering: removes high frequency and very low ones
    ["highpass", "-2", "50"],
    ["lowpass", "-2", "18000"],
    ["norm"],  # normalize audio
    # max silence should be 3s
    ["silence", "-l", "1", "0", "1%", "-1", "3.0", "1%"],
    ["norm"],
]


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_local_audio(
    path: Union[str, Path],
    file_hash: str,
    audio_backend: Literal["whisper", "deepgram"],
    loaders_temp_dir: Path,
    audio_unsilence: bool = True,
    whisper_lang: Optional[str] = None,
    whisper_prompt: Optional[str] = None,
    deepgram_kwargs: Optional[dict] = None,
) -> List[Document]:
    path = Path(path)
    assert path.exists(), f"file not found: '{path}'"

    if audio_unsilence:
        logger.warning(f"Removing silence from audio file {path.name}")
        waveform, sample_rate = torchaudio.load(path)

        dur = waveform.shape[1] / sample_rate
        start = time.time()
        try:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                sox_effects,
            )
        except Exception as e:
            if "libsox.so" in str(e).lower():
                logger.exception(
                    "The error hints at not being able to find libsox.so, on linux this can be solved by installing libsox-dev"
                )
            logger.warning(
                f"Error when applying sox effects: '{e}'.\nRetrying to apply each filter individually."
            )
            for sef in sox_effects:
                nfailed = 0
                logger.info(f"Applying filter '{sef}'")
                try:
                    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                        waveform,
                        sample_rate,
                        [sef],
                    )
                except Exception as err:
                    logger.warning(f"Error when applying sox effects '{sef}': {err}")
                    nfailed += 1
                if nfailed == len(sox_effects):
                    raise Exception(
                        "All sox_effects failed, you should report this bug and turn off --audio_unsilence"
                    )
        elapsed = time.time() - start
        new_dur = waveform.shape[1] / sample_rate

        if new_dur == dur:
            logger.warning(
                f"Duration of audio has not changed when trying to remove silence, something probably went wrong. Duration: {new_dur}"
            )
            # will crash anyway at the folling line because the assert is strict

        assert new_dur < dur, (
            f"Failed to remove silence for {path.name}:\n"
            f"Original duration: {dur:.1f}\n"
            f"New duration: {new_dur:.1f}\n"
        )
        assert new_dur > 10, (
            f"Silence removal ended up with a suspiciously short audio for {path.name}:\n"
            f"Original duration: {dur:.1f}\n"
            f"New duration: {new_dur:.1f}\n"
        )
        logger.warning(
            f"Removed silence from {path.name}: {dur:.1f} -> {new_dur:.1f} in {elapsed:.1f}s"
        )

        unsilenced_path_wav = loaders_temp_dir / f"unsilenced_audio_{uuid6.uuid6()}.wav"
        unsilenced_path_ogg = loaders_temp_dir / f"unsilenced_audio_{uuid6.uuid6()}.ogg"
        assert not unsilenced_path_wav.exists()
        assert not unsilenced_path_ogg.exists()
        torchaudio.save(
            uri=str(unsilenced_path_wav.resolve().absolute()),
            src=waveform,
            sample_rate=sample_rate,
            format="wav",
        )
        # turn the .wav into .ogg
        ffmpeg.input(str(unsilenced_path_wav.resolve().absolute())).output(
            str(unsilenced_path_ogg.resolve().absolute())
        ).run()
        unsilenced_hash = file_hasher({"path": unsilenced_path_ogg})

        # old_path = path
        # old_hash = file_hash
        path = unsilenced_path_ogg
        file_hash = unsilenced_hash

    if audio_backend == "whisper":
        assert (
            deepgram_kwargs is None
        ), "Found kwargs for deepgram but selected whisper backend for local_audio"
        content = transcribe_audio_whisper(
            audio_path=path,
            audio_hash=file_hash,
            language=whisper_lang,
            prompt=whisper_prompt,
        )
        timestamped_text = convert_verbose_json_to_timestamped_text(content)
        docs = [
            Document(
                page_content=timestamped_text,
                metadata={
                    "source": str(Path(path)),
                },
            )
        ]
        if "duration" in content:
            docs[-1].metadata["duration"] = content["duration"]
        if "language" in content:
            docs[-1].metadata["language"] = content["language"]
        elif whisper_lang:
            docs[-1].metadata["language"] = whisper_lang

    elif audio_backend == "deepgram":
        assert (
            whisper_prompt is None and whisper_lang is None
        ), "Found args whisper_prompt or whisper_lang but selected deepgram backend for local_audio"
        content = transcribe_audio_deepgram(
            audio_path=path,
            audio_hash=file_hash,
            deepgram_kwargs=deepgram_kwargs,
        )
        assert len(content["results"]["channels"]) == 1, "unexpected deepgram output"
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
                    "source": "local_audio_deepgram",
                },
            )
        ]
        docs[-1].metadata.update(content["metadata"])
        docs[-1].metadata["deepgram_kwargs"] = deepgram_kwargs

    else:
        raise ValueError(
            f"Invalid audio backend: must be either 'deepgram' or 'whisper'. Not '{audio_backend}'"
        )

    return docs
