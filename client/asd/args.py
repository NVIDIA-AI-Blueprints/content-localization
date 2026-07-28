# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the ASD (Active Speaker Detection) client.

Provides reusable ``add_asd_config_args_to_parser`` and ``asd_config_from_args``
so that any client (ASD, Direct, Controller) can share the same ASD config
arguments and build an ``ActiveSpeakerDetectionConfig`` identically.
"""

import argparse

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    AUDIO_SOURCE_CONFIG_SEPARATE_STREAM,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    AUDIO_SOURCE_CONFIG_UNSPECIFIED,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.video.v1.video_pb2 import VIDEO_CODEC_H264
from nvidia.ai4m.video.v1.video_pb2 import VideoConfig

from client.diarization_args import add_diarization_args

KB = 1024

_AUDIO_CODEC_MAP = {"WAV": AUDIO_CODEC_WAV, "MP3": AUDIO_CODEC_MP3}

_AUDIO_SOURCE_CONFIG_MAP = {
    "unspecified": AUDIO_SOURCE_CONFIG_UNSPECIFIED,
    "separate_stream": AUDIO_SOURCE_CONFIG_SEPARATE_STREAM,
    "embedded_in_video": AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO,
}


def add_asd_config_args_to_parser(
    parser: argparse.ArgumentParser | None = None,
    default_audio_source: str = "unspecified",
) -> argparse.ArgumentParser:
    """Add ASD config arguments to an argument parser.

    Args:
        parser (argparse.ArgumentParser | None): Existing parser to extend.
            Creates a new one if ``None``.
        default_audio_source (str): Default value for
            ``--asd-audio-source-config``. Use ``"separate_stream"``
            for clients that always send audio as a dedicated gRPC
            stream (default: ``"unspecified"``).

    Returns:
        argparse.ArgumentParser: The parser with ASD config arguments added.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> add_asd_config_args_to_parser(parser)  # returns parser with --asd-* args
    """
    if parser is None:
        parser = argparse.ArgumentParser("Active Speaker Detection arguments.")
    parser.add_argument(
        "--asd-input-audio-codec",
        type=str,
        default="WAV",
        choices=["WAV", "MP3"],
        help="Audio codec for ASD input (default: WAV)",
    )
    parser.add_argument(
        "--asd-input-video-codec",
        type=str,
        default=None,
        choices=["H264"],
        help="Video codec for ASD input (optional, default: unspecified)",
    )
    parser.add_argument(
        "--asd-audio-source-config",
        type=str,
        default=default_audio_source,
        choices=list(_AUDIO_SOURCE_CONFIG_MAP.keys()),
        help=(
            "Where audio originates: separate stream or embedded in video "
            f"(default: {default_audio_source})"
        ),
    )
    parser.add_argument(
        "--asd-speaker-detection-threshold",
        type=float,
        default=0.5986,
        help="Speaker detection confidence threshold in (0, 1) (default: 0.5986)",
    )
    return parser


def asd_config_from_args(args: argparse.Namespace) -> ActiveSpeakerDetectionConfig:
    """Build an ``ActiveSpeakerDetectionConfig`` from parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed argument namespace with
            ``asd_input_audio_codec``, ``asd_input_video_codec``,
            ``asd_audio_source_config``, and
            ``asd_speaker_detection_threshold``.

    Returns:
        ActiveSpeakerDetectionConfig: Populated protobuf config message.

    Examples:
        >>> args = argparse.Namespace(
        ...     asd_input_audio_codec="WAV",
        ...     asd_input_video_codec=None,
        ...     asd_audio_source_config="unspecified",
        ...     asd_speaker_detection_threshold=0.5986,
        ... )
        >>> cfg = asd_config_from_args(args)
        >>> cfg.input_audio_config.encoding == AUDIO_CODEC_WAV
        True
    """
    codec_enum = _AUDIO_CODEC_MAP[args.asd_input_audio_codec.upper()]
    audio_source = _AUDIO_SOURCE_CONFIG_MAP[args.asd_audio_source_config]
    config = ActiveSpeakerDetectionConfig(
        input_audio_config=AudioConfig(encoding=codec_enum),
        audio_source_config=audio_source,
        speaker_detection_threshold=args.asd_speaker_detection_threshold,
    )
    if args.asd_input_video_codec:
        config.input_video_config.CopyFrom(VideoConfig(codec=VIDEO_CODEC_H264))
    return config


def argsfactory() -> argparse.ArgumentParser:
    """Factory function for creating an ArgumentParser instance.

    Returns:
        argparse.ArgumentParser: An ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Active Speaker Detection (ASD) client")
    parser.add_argument(
        "--asd-server",
        type=str,
        default="localhost:50055",
        help="Port of the ASD gRPC service (default: localhost:50055)",
    )
    parser.add_argument(
        "--input-mp4",
        type=str,
        default="assets/sample_video_streamable.mp4",
        help="Path to input video file, MP4. Streamable MP4 is recommended for best "
        "performance but not required; convert using the script provided at: "
        "scripts/misc/convert_to_streamable_mp4.sh "
        "(default: assets/sample_video_streamable.mp4)",
    )
    parser.add_argument(
        "--input-audio",
        type=str,
        default="assets/sample_audio.wav",
        help="Path to input audio file (WAV format). "
        "Required for the new ASD NIM which needs both audio and video input. "
        "(default: assets/sample_audio.wav)",
    )
    parser.add_argument(
        "--chunk-size-video-bytes",
        type=int,
        default=1024 * KB,
        help="Chunk size for streaming video (default: 1024 KB)",
    )
    parser.add_argument(
        "--chunk-size-audio-secs",
        type=float,
        default=1.0,
        help="Chunk size for streaming audio in seconds (default: 1.0)",
    )
    add_diarization_args(parser=parser)
    parser.add_argument(
        "--output-speaker-info",
        type=str,
        default="assets/asd_speaker_info.csv",
        help="Path to output speaker info CSV file. (default: assets/asd_speaker_info.csv)",
    )
    add_asd_config_args_to_parser(parser)
    return parser
