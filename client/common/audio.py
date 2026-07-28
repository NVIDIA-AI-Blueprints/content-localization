# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared audio helpers for the client applications.

Centralizes the audio-codec mapping, the streaming chunk size, and the
audio-source factory so every client selects codecs and input simulators
the same way.
"""

from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV

from common.audio_utils import is_wav_file
from common.source_sink.file import FileSourceSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator

# Size of each raw byte chunk when streaming media files over gRPC.
DATA_CHUNK_SIZE = 64 * 1024  # bytes

# Maps CLI codec names to the shared audio.v1 AudioCodec enum values.
AUDIO_CODEC_CONFIGS = {
    "mp3": AUDIO_CODEC_MP3,
    "wav": AUDIO_CODEC_WAV,
}


def detect_audio_codec(file_path: str) -> int:
    """Detect the audio codec of a file by sniffing its content.

    Uses the same RIFF-header sniff as :func:`create_audio_source`, so
    the reported codec always matches the simulator chosen for the file
    (extensions are unreliable — e.g. MP3 data inside a ``.wav`` name).

    Args:
        file_path (str): Path to the input audio file (WAV or MP3).

    Returns:
        int: ``AUDIO_CODEC_WAV`` for RIFF content, ``AUDIO_CODEC_MP3``
            for any other content.

    Examples:
        >>> detect_audio_codec(file_path="audio.wav")  # doctest: +SKIP
        1
    """
    return AUDIO_CODEC_WAV if is_wav_file(file_path) else AUDIO_CODEC_MP3


def create_audio_source(file_path: str) -> AudioSourceSimulator | FileSourceSimulator:
    """Create the appropriate audio source simulator for a file.

    Selects the simulator by sniffing the file content rather than the
    file extension: a genuine WAV file (RIFF header) gets an
    ``AudioSourceSimulator`` for duration-based chunking, while any
    other content (e.g. MP3 data — even inside a ``.wav`` filename)
    gets a ``FileSourceSimulator`` that streams raw bytes unmodified.

    Args:
        file_path (str): Path to the input audio file (WAV or MP3).

    Returns:
        AudioSourceSimulator | FileSourceSimulator: WAV-aware simulator
            for RIFF content, raw byte-stream simulator otherwise.

    Examples:
        >>> source = create_audio_source(file_path="audio.wav")  # doctest: +SKIP
        >>> source.is_open()  # doctest: +SKIP
        True
    """
    if detect_audio_codec(file_path=file_path) == AUDIO_CODEC_WAV:
        return AudioSourceSimulator(file_path=file_path)
    return FileSourceSimulator(file_path=file_path)
