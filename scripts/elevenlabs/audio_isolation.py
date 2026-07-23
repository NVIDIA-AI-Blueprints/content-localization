#!/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio Isolation script using ElevenLabs API.

Isolates voice from background noise in audio files.
Best for cleaning up speech recordings, podcasts, interviews, etc.

Reference: https://elevenlabs.io/docs/api-reference/audio-isolation
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def isolate_audio(
    input_file_path: Path,
    output_file_path: Path,
) -> Optional[Path]:
    """Isolate voice from background noise in an audio file.

    This removes background noise, music, and other non-speech sounds
    while preserving the voice/speech in the recording.

    Args:
        input_file_path: Path to the input audio file.
        output_file_path: Path to save the isolated audio.

    Returns:
        Path to the output file, or None if operation failed.
    """
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    print(f"Isolating audio from: {input_file_path}")

    with open(input_file_path, "rb") as audio_file:
        # Call the audio isolation API — response is a streaming iterator
        # that requires the input file to remain open during iteration
        response = client.audio_isolation.convert(audio=audio_file)

        with open(output_file_path, "wb") as output_file:
            for chunk in response:
                output_file.write(chunk)

    return output_file_path


def parse_args() -> argparse.Namespace:
    """Argparser for audio isolation script."""
    parser = argparse.ArgumentParser(
        description="Audio Isolation using ElevenLabs API - Remove background noise from speech",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        type=Path,
        help="Path to the input audio file (mp3, wav, etc.).",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=Path("isolated_audio.mp3"),
        help="Output file path for the isolated audio.",
    )

    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def main() -> None:
    """Main entry point for audio isolation."""
    start_time = time.time()
    args = parse_args()

    print("=" * 50)
    print("ElevenLabs Audio Isolation")
    print("=" * 50)

    result = isolate_audio(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
    )

    if result:
        print("=" * 50)
        print(f"Audio isolation successful! File saved at: {result}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
    else:
        print("Audio isolation failed.")


if __name__ == "__main__":
    main()
