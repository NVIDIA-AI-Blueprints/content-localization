#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate diarization data using Camb AI Transcription API.
# Outputs the native Camb AI transcription JSON response for direct reuse.
#
# Usage:
#   CAMB_API_KEY=<key> .venv/bin/python scripts/camb/diarize.py \
#       --input-file <path>.wav \
#       --output-file diarization.json

import argparse
import json
import os
import wave

from common.diarization.camb import extract_diarization_stats
from common.diarization.camb import get_transcription_result
from common.diarization.camb import submit_transcription
from common.diarization.camb import wait_for_transcription


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Camb AI diarization.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``input_file``: required
            - ``output_file``: ``"diarization.json"``
            - ``language_id``: ``1`` (English)

    Examples:
        >>> # python scripts/camb/diarize.py --input-file audio.wav
    """
    parser = argparse.ArgumentParser(
        description="Generate diarization data using Camb AI Transcription API. "
        "Outputs native Camb AI transcription JSON with word-level timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to audio file (WAV, MP3, etc.).",
    )
    parser.add_argument(
        "--output-file",
        default="diarization.json",
        help="Path to output JSON diarization file.",
    )
    parser.add_argument(
        "--language-id",
        type=int,
        default=1,
        help="Camb AI numeric language ID (1=English, 54=Spanish, etc.).",
    )
    return parser.parse_args()


def print_wav_info(wav_path: str) -> None:
    """Print WAV file metadata if the input is a WAV file.

    Silently skips non-WAV files.

    Args:
        wav_path (str): Path to the audio file.

    Returns:
        None.

    Examples:
        >>> print_wav_info("audio.wav")
        WAV: 16000 Hz, 1 ch, 160000 frames (10.00 s)
    """
    try:
        with wave.open(wav_path, "rb") as wav:
            nch = wav.getnchannels()
            framerate = wav.getframerate()
            nframes = wav.getnframes()
            total_sec = nframes / float(framerate)
            print(f"WAV: {framerate} Hz, {nch} ch, {nframes} frames ({total_sec:.2f} s)")
    except wave.Error:
        pass


def main() -> None:
    """Run Camb AI diarization and write native JSON output.

    Reads ``CAMB_API_KEY`` from the environment, sends the input
    audio file through the Camb AI Transcription API, and writes
    the native JSON response to disk.

    Returns:
        None.

    Raises:
        ValueError: If ``CAMB_API_KEY`` is not set.
        FileNotFoundError: If the input file does not exist.

    Examples:
        >>> # python scripts/camb/diarize.py --input-file audio.wav -o out.json
    """
    args = parse_args()

    api_key = os.getenv("CAMB_API_KEY")
    if not api_key:
        raise ValueError(
            "CAMB_API_KEY environment variable not set. "
            "Export it or pass via: CAMB_API_KEY=<key> python scripts/camb/diarize.py ..."
        )

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    headers = {"x-api-key": api_key}

    print_wav_info(args.input_file)

    file_size = os.path.getsize(args.input_file)
    print(f"Input file: {args.input_file} ({file_size:,} bytes)")
    print(f"Config: language_id={args.language_id}")

    print(f"Submitting {args.input_file} to Camb AI Transcription API...")
    task_id = submit_transcription(
        file_path=args.input_file,
        language_id=args.language_id,
        headers=headers,
    )
    print(f"Task submitted: task_id={task_id}")

    print("Polling for transcription completion...")
    run_id = wait_for_transcription(task_id=task_id, headers=headers)
    print(f"Transcription completed: run_id={run_id}")

    print("Fetching transcription result with word-level timestamps...")
    result = get_transcription_result(run_id=run_id, headers=headers)

    segment_count, speaker_count = extract_diarization_stats(result)
    if segment_count == 0:
        print("WARNING: No segments found in Camb AI response.")
    else:
        print(f"Generated {segment_count} segments across {speaker_count} speakers")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Native Camb AI diarization JSON written to {args.output_file}")


if __name__ == "__main__":
    main()
