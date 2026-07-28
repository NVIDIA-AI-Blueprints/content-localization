# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the controller client."""

import argparse

from client.asd.args import add_asd_config_args_to_parser
from client.diarization_args import add_diarization_args
from client.lipsync.args import add_lipsync_config_args_to_parser
from client.s2s.args import add_s2s_config_args_to_parser

KB = 1024
MB = 1024 * KB


def argsfactory() -> argparse.ArgumentParser:
    """Factory function for creating an ArgumentParser instance.

    Registers controller-specific arguments (server address, I/O paths,
    chunk sizes) and delegates NIM-specific config arguments to the
    reusable helpers from the S2S, ASD, and LipSync client modules.

    Returns:
        argparse.ArgumentParser: An ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Controller client")
    parser.add_argument(
        "--controller-server",
        type=str,
        default="localhost:50056",
        help="Address and port of the controller gRPC service (default: localhost:50056)",
    )
    parser.add_argument(
        "--input-audio",
        type=str,
        default="assets/sample_audio.wav",
        help="Path to input audio file (default: assets/sample_audio.wav)",
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
    # SSL configuration for the controller channel, mirroring the
    # standalone NIM clients' --ssl-mode surface.
    parser.add_argument(
        "--ssl-mode",
        type=str,
        choices=["DISABLED", "TLS", "MTLS"],
        default="DISABLED",
        help="Channel security for the controller connection (default: DISABLED)",
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default=None,
        help="Path to the client private key PEM (required for MTLS)",
    )
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default=None,
        help="Path to the client certificate chain PEM (required for MTLS)",
    )
    parser.add_argument(
        "--ssl-root-cert",
        type=str,
        default=None,
        help="Path to the root certificate PEM used to verify the controller "
        "(required for TLS and MTLS)",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Correlation id stamped on every request message and echoed by the "
        "controller in its responses. Omit to generate a UUID4 per run.",
    )
    parser.add_argument(
        "--chunk-size-audio-secs",
        type=float,
        default=1,
        help="Chunk size for streaming audio in seconds (default: 1)",
    )
    parser.add_argument(
        "--chunk-size-video-bytes",
        type=int,
        default=1 * MB,
        help="Chunk size for streaming video (default: 1 MB)",
    )
    parser.add_argument(
        "--output-mp4",
        type=str,
        default="outputs/controller_output.mp4",
        help="Path to output video file, mp4 only. Default: outputs/controller_output.mp4",
    )
    add_diarization_args(parser=parser)

    parser.add_argument(
        "--bypass-asd",
        action="store_true",
        default=False,
        help="Bypass ASD (Active Speaker Detection) service. "
        "Auto-enabled when no --diarization-file is provided.",
    )
    parser.add_argument(
        "--background-audio-input",
        type=str,
        default=None,
        help="Path to background audio file (WAV or MP3) for mixing "
        "with the LipSync output (optional)",
    )
    parser.add_argument(
        "--translated-audio",
        type=str,
        default=None,
        help="Path to pre-translated audio file (WAV or MP3). When "
        "provided, S2S is bypassed and this audio is sent directly "
        "to LipSync (optional)",
    )
    parser.add_argument(
        "--diarization-rows-per-chunk",
        type=int,
        default=10,
        help="Number of diarization segment rows to send per chunk. "
        "Use -1 to send all segments in a single message "
        "(default: 10)",
    )

    # Delegate NIM-specific config args to per-client modules
    add_s2s_config_args_to_parser(parser)
    add_asd_config_args_to_parser(parser, default_audio_source="separate_stream")
    add_lipsync_config_args_to_parser(parser)

    return parser
