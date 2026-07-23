# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline helpers for the direct client (S2S-to-LipSync audio/video adapters)."""

from collections.abc import Iterator

from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from client.common.audio import DATA_CHUNK_SIZE
from common.audio_utils import create_wav_header
from common.audio_utils import is_wav_file
from common.base_utils import logger
from common.source_sink.grpc.audio import AudioSinkSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator


def audio_iterator_from_s2s_response_with_format(
    response_iter: Iterator[SpeechToSpeechResponse],
    audio_format: str = "mp3",
    output_sink: AudioSinkSimulator | None = None,
) -> Iterator[LipsyncInputData]:
    """Create an audio iterator and determine the audio format from the first response.

    For WAV format, a synthetic header is emitted before the first
    audio-data chunk **only when the data is raw PCM** (no existing
    header).  If the first chunk already starts with a RIFF header
    the synthetic header is skipped to avoid a duplicate header with
    a wrong sample-rate.
    For MP3 format, sends the data directly without any header.

    Args:
        response_iter (Iterator[SpeechToSpeechResponse]): Iterator of
            S2S responses containing audio data.
        audio_format (str): Expected audio format (``"mp3"`` or ``"wav"``).
            Defaults to ``"mp3"``.
        output_sink (AudioSinkSimulator | None): Optional audio sink to
            write output to. Defaults to ``None``.

    Yields:
        LipsyncInputData: Audio data chunks for the LipSync service.

    Examples:
        >>> gen = audio_iterator_from_s2s_response_with_format(
        ...     response_iter=s2s_responses,
        ...     audio_format="mp3",
        ... )  # doctest: +SKIP
    """
    # We need to peek at the first response to determine format
    # Since we can't peek at an iterator, we'll create a custom iterator
    first_response = True
    for response in response_iter:
        try:
            # Handle keep-alive responses
            if response.HasField("keepalive"):
                logger.debug("s2s | received keep-alive from S2S, skipping")
                continue

            if first_response:
                audio_format_from_s2s = (
                    response.audio_format.lower() if response.audio_format else "mp3"
                )
                if audio_format_from_s2s != audio_format:
                    logger.warning(
                        f"Audio format from S2S service is "
                        f"{audio_format_from_s2s}, but expected "
                        f"{audio_format}. Continuing with detected "
                        f"format."
                    )
                    audio_format = audio_format_from_s2s
                first_response = False
                # For WAV: only prepend a synthetic header when the
                # data is raw PCM (no header). Some backends stream a
                # complete WAV file whose first bytes are already a
                # RIFF header — adding a second header with a wrong
                # sample-rate breaks playback.
                data_already_has_header = response.audio_data[:4] == b"RIFF"
                if audio_format_from_s2s == "wav" and not data_already_has_header:
                    wav_header = create_wav_header(
                        n_channels=response.audio_num_channels or 1,
                        sample_width=2,  # 16-bit PCM
                        frame_rate=response.audio_sample_rate or 16000,
                        n_frames=0,
                    )
                    yield LipsyncInputData(audio_file_data=wav_header)
                elif data_already_has_header:
                    logger.warning(
                        "s2s | first chunk already contains a WAV header, skipping synthetic header"
                    )

            # Write to output sink if provided
            if output_sink:
                output_sink.write(wave_bytes=response.audio_data)

            # Send audio data chunk
            yield LipsyncInputData(audio_file_data=response.audio_data)

        except (AttributeError, ValueError, OSError) as exc:
            # Recoverable per-chunk issues (malformed response, I/O hiccup):
            # log and skip just this chunk. TypeError/RuntimeError are left to
            # the generic handler below since they usually signal real bugs.
            logger.warning(f"Skipping malformed S2S response chunk: {exc}")
            continue
        except Exception:
            # Unexpected failure: surface it instead of silently dropping output.
            logger.exception("Unexpected error processing S2S response")
            raise


def video_iterator_from_source(
    source_iterator: Iterator[bytes],
) -> Iterator[LipsyncInputData]:
    """Wrap raw video byte chunks into LipsyncInputData messages.

    Args:
        source_iterator (Iterator[bytes]): Iterator of raw video byte
            chunks.

    Yields:
        LipsyncInputData: Wrapped video data for the LipSync service.

    Examples:
        >>> gen = video_iterator_from_source(
        ...     source_iterator=video_chunks,
        ... )  # doctest: +SKIP
    """
    for chunk in source_iterator:
        yield LipsyncInputData(video_file_data=chunk)


def audio_iterator_from_file(
    file_path: str,
    chunk_size_secs: float = 1.0,
    chunk_size_bytes: int = DATA_CHUNK_SIZE,
) -> Iterator[LipsyncInputData]:
    """Yield ``LipsyncInputData`` audio chunks from a pre-translated file.

    For real WAV files, uses ``AudioSourceSimulator`` for proper header
    generation and duration-based chunking.  For MP3 files (including
    those with a ``.wav`` extension), falls back to fixed-size byte
    reads so the raw MP3 data reaches LipSync unmodified.

    Args:
        file_path (str): Path to a pre-translated WAV or MP3 audio
            file.
        chunk_size_secs (float): Chunk duration in seconds (WAV only).
            Defaults to ``1.0``.
        chunk_size_bytes (int): Bytes per chunk (MP3 fallback).
            Defaults to 64 KB.

    Yields:
        LipsyncInputData: Wrapped audio data chunks for the LipSync
            service.

    Examples:
        >>> gen = audio_iterator_from_file(
        ...     file_path="translated.wav",
        ...     chunk_size_secs=1.0,
        ... )  # doctest: +SKIP
    """
    if is_wav_file(file_path):
        source = AudioSourceSimulator(file_path=file_path)
        for chunk in source.read(chunk_duration_secs=chunk_size_secs):
            yield LipsyncInputData(audio_file_data=chunk)
        if source.is_open():
            source.close()
    else:
        # MP3 or other non-WAV format — stream raw bytes
        with open(file_path, "rb") as f:
            while True:
                data = f.read(chunk_size_bytes)
                if not data:
                    break
                yield LipsyncInputData(audio_file_data=data)


def background_audio_iterator_from_file(
    file_path: str,
    chunk_size: int = DATA_CHUNK_SIZE,
) -> Iterator[LipsyncInputData]:
    """Yield ``LipsyncInputData`` background audio chunks from a file.

    Reads the file in fixed-size byte chunks and wraps each in a
    ``LipsyncInputData`` with ``background_audio_file_data`` set.

    Args:
        file_path (str): Path to a WAV or MP3 background audio file.
        chunk_size (int): Bytes per chunk. Defaults to 64 KB.

    Yields:
        LipsyncInputData: Wrapped background audio data chunks.

    Examples:
        >>> gen = background_audio_iterator_from_file(
        ...     file_path="bg_audio.wav",
        ... )  # doctest: +SKIP
    """
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield LipsyncInputData(background_audio_file_data=data)
