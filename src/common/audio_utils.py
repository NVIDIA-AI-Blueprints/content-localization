# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio file I/O utilities (download, WAV writing, header generation)."""

import io
import os
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from common.base_utils import logger

_AUDIO_MIME_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def is_wav_file(file_path: str | os.PathLike) -> bool:
    """Return True if the file starts with a RIFF (WAV) magic header.

    Reads the first 4 bytes only — does not validate the full WAV structure.
    Use this instead of checking file extensions, which can be misleading
    (e.g. ElevenLabs sometimes returns MP3 data with a ``.wav`` extension).

    Args:
        file_path (str | os.PathLike): Path to the audio file.

    Returns:
        bool: ``True`` if the file starts with ``RIFF``.

    Examples:
        >>> is_wav_file("audio.wav")  # doctest: +SKIP
        True
        >>> is_wav_file("audio.mp3")  # doctest: +SKIP
        False
    """
    with open(file_path, "rb") as f:
        return f.read(4) == b"RIFF"


def audio_mime_type(file_path: str | os.PathLike) -> str:
    """Return the MIME type for an audio file based on its extension.

    Used when uploading files to external APIs (ElevenLabs, CambAI) that
    require a ``Content-Type`` header. Falls back to ``"audio/wav"`` for
    unrecognised extensions.

    Args:
        file_path (str | os.PathLike): Path to the audio file.

    Returns:
        str: MIME type string (e.g. ``"audio/mpeg"`` for MP3).

    Examples:
        >>> audio_mime_type("clip.mp3")
        'audio/mpeg'
        >>> audio_mime_type("clip.wav")
        'audio/wav'
        >>> audio_mime_type("clip.unknown")
        'audio/wav'
    """
    return _AUDIO_MIME_TYPES.get(Path(file_path).suffix.lower(), "audio/wav")


def download_audio_file_from_iterator(
    chunks: Iterator[Any],
    file_path: Path,
) -> Path:
    """Write audio data from an iterator of chunks to a file.

    Args:
        chunks (Iterator[Any]): Iterator of audio data chunks. Can be
            raw bytes or objects with an ``audio_data`` attribute.
        file_path (Path): The file path to write the audio data to.

    Returns:
        Path: The path to the file that was written.

    Examples:
        >>> from pathlib import Path
        >>> p = download_audio_file_from_iterator(
        ...     chunks=iter([b"data"]),
        ...     file_path=Path("/tmp/audio.raw"),
        ... )  # doctest: +SKIP
    """
    logger.debug(f"Writing audio data to file: {file_path}")
    total_size = 0
    with open(file_path, "wb") as f:
        for chunk in chunks:
            if hasattr(chunk, "audio_data"):
                f.write(chunk.audio_data)
                total_size += len(chunk.audio_data)
            else:
                f.write(chunk)
                total_size += len(chunk)
                logger.debug(f"Wrote bytes chunk of {len(chunk)} bytes to {file_path}")
    logger.debug(f"Total size written to {file_path}: {total_size} bytes")
    return file_path


def write_wav_iterator_to_file(
    chunks: Iterator[Any],
    file_path: Path,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> Path:
    """Write audio data to a properly formatted WAV file with headers.

    Args:
        chunks (Iterator[Any]): Iterator of audio data chunks. Can be
            raw bytes or objects with an ``audio_data`` attribute.
        file_path (Path): The file path to write the WAV file to.
        sample_rate (int): Sample rate in Hz. Defaults to ``16000``.
        sample_width (int): Sample width in bytes. Defaults to ``2``
            (16-bit PCM).
        channels (int): Number of audio channels. Defaults to ``1``
            (mono).

    Returns:
        Path: The path to the WAV file that was written.

    Examples:
        >>> from pathlib import Path
        >>> p = write_wav_iterator_to_file(
        ...     chunks=iter([b"\\x00" * 320]),
        ...     file_path=Path("/tmp/audio.wav"),
        ... )  # doctest: +SKIP
    """
    logger.debug(f"Writing WAV audio data to file: {file_path}")
    total_size = 0

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)

        for chunk in chunks:
            if hasattr(chunk, "audio_data"):
                wav_file.writeframes(chunk.audio_data)
                total_size += len(chunk.audio_data)
            else:
                wav_file.writeframes(chunk)
                total_size += len(chunk)

    logger.debug(f"Total size written to {file_path}: {total_size} bytes")
    return file_path


def create_wav_header(
    n_channels: int,
    sample_width: int,
    frame_rate: int,
    n_frames: int,
) -> bytes:
    """Create a WAV file header from audio parameters.

    Args:
        n_channels (int): The number of channels in the WAV file.
        sample_width (int): The sample width in bytes, usually ``2``
            for 16-bit PCM.
        frame_rate (int): The frame rate in Hz (sample rate).
        n_frames (int): The number of frames in the WAV file.

    Returns:
        bytes: The WAV file header bytes.

    Examples:
        >>> header = create_wav_header(
        ...     n_channels=1,
        ...     sample_width=2,
        ...     frame_rate=16000,
        ...     n_frames=0,
        ... )
        >>> header[:4]
        b'RIFF'
    """
    buffer = io.BytesIO()
    with wave.open(buffer, mode="wb") as wf:
        wf.setnchannels(nchannels=n_channels)
        wf.setsampwidth(sampwidth=sample_width)
        wf.setframerate(framerate=frame_rate)
        wf.setnframes(nframes=n_frames)
    return buffer.getvalue()
