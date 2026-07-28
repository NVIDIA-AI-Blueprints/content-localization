# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared diarization CLI argument registration for the client apps.

The ASD, Direct, and Controller clients all accept the same trio of
diarization flags (``--diarization-file``, ``--diarization-format``,
``--diarization-chunked-per-segment``). Centralizing their registration
here keeps the help text and choices identical across every client.
"""

import argparse

# The accepted diarization file formats are defined once, next to the parsers
# in client.common.diarization, so a new format only has to be added in one place.
from client.common.diarization import VALID_DIARIZATION_FORMATS


def add_diarization_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared diarization arguments on a client parser.

    Adds ``--diarization-file``, ``--diarization-format``, and
    ``--diarization-chunked-per-segment`` with identical help text and
    choices for every client that consumes diarization input.

    Args:
        parser (argparse.ArgumentParser): The parser to register the
            diarization arguments on. Mutated in place.

    Returns:
        None

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> add_diarization_args(parser=parser)
        >>> args = parser.parse_args(["--diarization-format", "camb"])
        >>> args.diarization_format
        'camb'
    """
    parser.add_argument(
        "--diarization-file",
        type=str,
        default=None,
        help="Path to diarization file for speaker segments. "
        "Supports flat ASD format, ElevenLabs STT/Scribe JSON, "
        "ElevenLabs Dubbing Transcript API JSON, ElevenLabs Studio CSV, "
        "or Camb AI transcription JSON. "
        "Use --diarization-format to select the parser. "
        "Generate with: scripts/elevenlabs/diarize.py, "
        "scripts/elevenlabs/s2s_infer.py, or scripts/camb/diarize.py. "
        "Note: scripts/camb/s2s_infer.py also emits a 'camb'-format JSON, "
        "but the transcript is in the target language only.",
    )
    parser.add_argument(
        "--diarization-format",
        type=str,
        default="elevenlabs-scribe",
        choices=VALID_DIARIZATION_FORMATS,
        help="Format of the diarization file. "
        "'elevenlabs-scribe' for ElevenLabs STT (Scribe) JSON; "
        "'elevenlabs-dubbing-api' for ElevenLabs Dubbing Transcript API JSON; "
        "'camb' for Camb AI transcription/dubbing JSON "
        "(generate source-language diarization with scripts/camb/diarize.py; "
        "scripts/camb/s2s_infer.py emits target-language only) "
        "(default: elevenlabs-scribe). "
        "Note: the legacy value 'elevenlabs' was renamed to 'elevenlabs-scribe' — "
        "update any scripts/configs accordingly.",
    )
    parser.add_argument(
        "--diarization-chunked-per-segment",
        action="store_true",
        default=False,
        help="Emit one diarization chunk per source segment (e.g. per word) "
        "instead of merging consecutive same-speaker segments. Combined with "
        "--diarization-rows-per-chunk this streams diarization one unit at a "
        "time. By default, consecutive same-speaker segments are merged.",
    )
