# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the LipSync client.

Provides reusable ``add_lipsync_config_args_to_parser`` and ``lipsync_config_from_args``
so that any client (LipSync, Direct, Controller) can share the same LipSync config
arguments and build a ``LipsyncConfig`` identically.
"""

import argparse
import json
import os

from nvidia.ai4m.lipsync.v1.lipsync_pb2 import BackgroundAudioConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.video.v1.video_pb2 import LossyEncoding
from nvidia.ai4m.video.v1.video_pb2 import VideoEncoding

from client.common.audio import AUDIO_CODEC_CONFIGS
from client.lipsync.constants import DEFAULT_AUDIO_PATH
from client.lipsync.constants import DEFAULT_BITRATE_MBPS
from client.lipsync.constants import DEFAULT_IDR_INTERVAL
from client.lipsync.constants import DEFAULT_VIDEO_PATH
from client.lipsync.constants import EXTEND_AUDIO_CONFIGS
from client.lipsync.constants import EXTEND_VIDEO_CONFIGS
from client.lipsync.encoding import create_custom_encoding_params


def add_lipsync_config_args_to_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Add LipSync config arguments to an argument parser.

    Args:
        parser (argparse.ArgumentParser | None): Existing parser to extend.
            Creates a new one if ``None``.

    Returns:
        argparse.ArgumentParser: The parser with LipSync config arguments added.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> add_lipsync_config_args_to_parser(parser)  # adds --lipsync-* args
    """
    if parser is None:
        parser = argparse.ArgumentParser("LipSync Parser.")
    parser.add_argument(
        "--lipsync-input-audio-codec",
        type=str,
        default=None,
        choices=["WAV", "MP3"],
        help="Audio codec for LipSync input. Omit to auto-detect for file input.",
    )
    parser.add_argument(
        "--lipsync-extend-audio",
        type=str,
        default="unspecified",
        choices=list(EXTEND_AUDIO_CONFIGS.keys()),
        help="How to handle video longer than audio (default: unspecified)",
    )
    parser.add_argument(
        "--lipsync-extend-video",
        type=str,
        default="unspecified",
        choices=list(EXTEND_VIDEO_CONFIGS.keys()),
        help="How to handle audio longer than video (default: unspecified)",
    )
    parser.add_argument(
        "--lipsync-output-bitrate-mbps",
        type=int,
        default=DEFAULT_BITRATE_MBPS,
        help=f"Output video bitrate in Mbps (default: {DEFAULT_BITRATE_MBPS})",
    )
    parser.add_argument(
        "--lipsync-output-idr-interval",
        type=int,
        default=DEFAULT_IDR_INTERVAL,
        help=f"Output video IDR (keyframe) interval (default: {DEFAULT_IDR_INTERVAL})",
    )
    parser.add_argument(
        "--lipsync-head-movement-speed",
        type=int,
        default=None,
        help="Head movement speed: 0 for static/slow, 1 for fast (optional)",
    )
    parser.add_argument(
        "--lipsync-output-audio-codec",
        type=str,
        default=None,
        choices=["WAV", "MP3"],
        help="Output audio codec (optional)",
    )
    parser.add_argument(
        "--lipsync-is-speaker-info-provided",
        action="store_true",
        default=False,
        help="Whether speaker bounding boxes are provided (from ASD). Set when ASD is enabled.",
    )
    parser.add_argument(
        "--lipsync-background-audio-volume",
        type=float,
        default=1.0,
        help="Background audio volume for mixing (0.0-1.0). 0 = muted, 1 = full volume",
    )
    parser.add_argument(
        "--lipsync-background-audio-codec",
        type=str,
        default=None,
        choices=["WAV", "MP3"],
        help="Background audio codec. Auto-detected from file extension if omitted (optional)",
    )
    parser.add_argument(
        "--lipsync-lossless",
        action="store_true",
        default=False,
        help="Enable lossless video encoding (overrides bitrate/IDR settings)",
    )
    parser.add_argument(
        "--lipsync-custom-encoding-params",
        type=str,
        default=None,
        help="Custom encoding parameters in JSON format (overrides bitrate/IDR settings)",
    )
    return parser


def _build_video_encoding(args: argparse.Namespace) -> VideoEncoding:
    """Build a ``VideoEncoding`` from parsed CLI arguments.

    Encoding priority: lossless > custom-encoding-params > lossy (default).

    Args:
        args (argparse.Namespace): Parsed arguments with
            ``lipsync_lossless``, ``lipsync_custom_encoding_params``,
            ``lipsync_output_bitrate_mbps``, and ``lipsync_output_idr_interval``.

    Returns:
        VideoEncoding: Protobuf message from video.v1.

    Examples:
        >>> args = argparse.Namespace(
        ...     lipsync_lossless=False,
        ...     lipsync_custom_encoding_params=None,
        ...     lipsync_output_bitrate_mbps=20,
        ...     lipsync_output_idr_interval=8,
        ... )
        >>> enc = _build_video_encoding(args)
    """
    if getattr(args, "lipsync_lossless", False):
        return VideoEncoding(lossless=True)

    custom_json = getattr(args, "lipsync_custom_encoding_params", None)
    if custom_json:
        params = json.loads(custom_json)
        return VideoEncoding(
            custom_encoding=create_custom_encoding_params(params=params),
        )

    return VideoEncoding(
        lossy=LossyEncoding(
            bitrate_mbps=args.lipsync_output_bitrate_mbps,
            idr_interval=args.lipsync_output_idr_interval,
        )
    )


def lipsync_config_from_args(args: argparse.Namespace) -> LipsyncConfig:
    """Build a ``LipsyncConfig`` from parsed CLI arguments.

    Encoding priority: lossless > custom-encoding-params > lossy (default).

    Args:
        args (argparse.Namespace): Parsed argument namespace with
            ``lipsync_input_audio_codec``, ``lipsync_extend_audio``,
            ``lipsync_extend_video``, ``lipsync_output_bitrate_mbps``,
            ``lipsync_output_idr_interval``, ``lipsync_head_movement_speed``,
            ``lipsync_output_audio_codec``, ``lipsync_is_speaker_info_provided``,
            ``lipsync_lossless``, and ``lipsync_custom_encoding_params``.

    Returns:
        LipsyncConfig: Populated protobuf config message.

    Examples:
        >>> args = argparse.Namespace(
        ...     lipsync_input_audio_codec=None,
        ...     lipsync_extend_audio="unspecified",
        ...     lipsync_extend_video="unspecified",
        ...     lipsync_output_bitrate_mbps=20,
        ...     lipsync_output_idr_interval=8,
        ...     lipsync_head_movement_speed=None,
        ...     lipsync_output_audio_codec=None,
        ...     lipsync_is_speaker_info_provided=False,
        ...     lipsync_lossless=False,
        ...     lipsync_custom_encoding_params=None,
        ... )
        >>> cfg = lipsync_config_from_args(args)
    """
    output_video_encoding = _build_video_encoding(args)
    input_audio_codec = getattr(args, "lipsync_input_audio_codec", None) or "MP3"
    config = LipsyncConfig(
        input_audio_codec=AUDIO_CODEC_CONFIGS[input_audio_codec.lower()],
        extend_audio=EXTEND_AUDIO_CONFIGS[args.lipsync_extend_audio],
        extend_video=EXTEND_VIDEO_CONFIGS[args.lipsync_extend_video],
        output_video_encoding=output_video_encoding,
        is_speaker_info_provided=args.lipsync_is_speaker_info_provided,
    )
    if args.lipsync_head_movement_speed is not None:
        config.head_movement_speed = args.lipsync_head_movement_speed
    if args.lipsync_output_audio_codec:
        config.output_audio_codec = AUDIO_CODEC_CONFIGS[args.lipsync_output_audio_codec.lower()]

    # Build BackgroundAudioConfig when a background audio file is provided
    bg_audio_path = getattr(args, "background_audio_input", None)
    if bg_audio_path:
        bg_config = _build_background_audio_config(args=args, file_path=bg_audio_path)
        config.background_audio_config.CopyFrom(bg_config)

    return config


def _build_background_audio_config(
    args: argparse.Namespace,
    file_path: str,
) -> BackgroundAudioConfig:
    """Build a ``BackgroundAudioConfig`` from CLI args and the audio file path.

    Auto-detects the codec from the file extension when
    ``--lipsync-background-audio-codec`` is not explicitly set.

    Args:
        args (argparse.Namespace): Parsed arguments with optional
            ``lipsync_background_audio_codec`` and
            ``lipsync_background_audio_volume``.
        file_path (str): Path to the background audio file, used for
            codec auto-detection.

    Returns:
        BackgroundAudioConfig: Populated protobuf config message.

    Examples:
        >>> args = argparse.Namespace(
        ...     lipsync_background_audio_codec=None,
        ...     lipsync_background_audio_volume=0.5,
        ... )
        >>> cfg = _build_background_audio_config(args, "bg.wav")
        >>> cfg.is_background_audio_provided
        True
    """
    bg_config = BackgroundAudioConfig(is_background_audio_provided=True)

    # Determine codec: explicit flag or auto-detect from file extension
    explicit_codec = getattr(args, "lipsync_background_audio_codec", None)
    if explicit_codec:
        bg_config.audio_codec = AUDIO_CODEC_CONFIGS[explicit_codec.lower()]
    else:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext in AUDIO_CODEC_CONFIGS:
            bg_config.audio_codec = AUDIO_CODEC_CONFIGS[ext]

    # Set volume only when explicitly provided
    volume = getattr(args, "lipsync_background_audio_volume", None)
    if volume is not None:
        bg_config.audio_volume = volume

    return bg_config


def argsfactory(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Create and configure argument parser.

    The I/O flags are aligned with the other clients
    (``--lipsync-server``, ``--input-mp4``, ``--input-audio``,
    ``--output-mp4``); the original flag names (``--target``,
    ``--video-input``, ``--audio-input``, ``--output``) remain available
    as deprecated aliases.

    Args:
        parser (argparse.ArgumentParser | None): Existing parser to
            extend. Creates a new one if ``None``.

    Returns:
        argparse.ArgumentParser: Configured parser instance.

    Examples:
        >>> parser = argsfactory()
        >>> args = parser.parse_args(["--lipsync-server", "localhost:50054"])
        >>> args.lipsync_server
        'localhost:50054'
    """

    class SmartFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        """Custom formatter combining raw description and default value help text."""

    if parser is None:
        parser = argparse.ArgumentParser(
            description="Run LipSync inference with input video and audio files",
            formatter_class=lambda prog: SmartFormatter(prog, max_help_position=60),
        )

    # SSL configuration arguments
    parser.add_argument(
        "--ssl-mode",
        type=str,
        choices=["DISABLED", "MTLS", "TLS"],
        default="DISABLED",
        help="SSL mode for secure communication",
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default="../ssl_key/ssl_key_client.pem",
        help="Path to SSL private key",
    )
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default="../ssl_key/ssl_cert_client.pem",
        help="Path to SSL certificate chain",
    )
    parser.add_argument(
        "--ssl-root-cert",
        type=str,
        default="../ssl_key/ssl_ca_cert.pem",
        help="Path to SSL root certificate",
    )
    parser.add_argument(
        "--lipsync-server",
        "--target",
        dest="lipsync_server",
        type=str,
        default="127.0.0.1:50054",
        help="IP:port of the LipSync gRPC service (--target is a deprecated alias)",
    )

    # Input file arguments
    parser.add_argument(
        "--input-mp4",
        "--video-input",
        dest="input_mp4",
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help="Path to the input video file (--video-input is a deprecated alias)",
    )
    parser.add_argument(
        "--input-audio",
        "--audio-input",
        dest="input_audio",
        type=str,
        default=DEFAULT_AUDIO_PATH,
        help="Path to the input audio file (--audio-input is a deprecated alias)",
    )
    parser.add_argument(
        "--speaker-info-input",
        type=str,
        default=None,
        help="Path to the speaker info CSV file",
    )

    parser.add_argument(
        "--background-audio-input",
        type=str,
        default=None,
        help="Path to background audio file (WAV or MP3) for mixing with the output (optional)",
    )

    # LipSync config args (shared with other clients)
    add_lipsync_config_args_to_parser(parser)

    # Output arguments
    parser.add_argument(
        "--output-mp4",
        "--output",
        dest="output_mp4",
        type=str,
        default="outputs/lipsync_output.mp4",
        help="Path for the output video file (--output is a deprecated alias)",
    )

    return parser
