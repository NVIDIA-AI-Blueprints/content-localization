#!/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stem Separation script using ElevenLabs API.

Separates an audio file into individual stems (vocals, drums, bass, etc.)
Reference: https://elevenlabs.io/docs/api-reference/music/separate-stems
"""

import argparse
import os
import tempfile
import time
import wave
import zipfile
from pathlib import Path
from typing import Literal
from typing import Optional

from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# Valid stem variation options
StemVariation = Literal["two_stems_v1", "six_stems_v1"]

# Valid output format options
OutputFormat = Literal[
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",
    "pcm_48000",
    "ulaw_8000",
    "alaw_8000",
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192",
]


def pcm_to_wav(pcm_path: Path, wav_path: Path, sample_rate: int = 16000) -> Path:
    """
    Convert raw PCM file to WAV format.

    Args:
        pcm_path: Path to the input PCM file.
        wav_path: Path for the output WAV file.
        sample_rate: Sample rate of the PCM audio.

    Returns:
        Path to the created WAV file.
    """
    with open(pcm_path, "rb") as pcm_file:
        pcm_data = pcm_file.read()

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit (2 bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return wav_path


def get_sample_rate_from_format(output_format: str) -> int:
    """Extract sample rate from output format string like 'pcm_16000'."""
    if output_format and output_format.startswith("pcm_"):
        return int(output_format.split("_")[1])
    return 16000  # Default


def separate_stems(
    input_file_path: Path,
    output_dir: Path,
    stem_variation: StemVariation = "two_stems_v1",
    output_format: Optional[OutputFormat] = None,
) -> Optional[Path]:
    """
    Separate an audio file into individual stems.

    Args:
        input_file_path: Path to the input audio file.
        output_dir: Directory to save the separated stems.
        stem_variation: The stem separation variant to use.
            - "two_stems_v1": Separates into vocals and accompaniment.
            - "six_stems_v1": Separates into vocals, drums, bass, guitar, piano, and other.
        output_format: Output format for the stems (e.g., "mp3_44100_128").

    Returns:
        Path to the output directory containing the separated stems,
        or None if operation failed.
    """
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Separating stems from: {input_file_path}")
    print(f"Using stem variation: {stem_variation}")
    if output_format:
        print(f"Output format: {output_format}")

    # The response is a ZIP archive containing the separated stems
    # Save and extract the ZIP file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip_file:
        temp_zip_path = temp_zip_file.name

    with open(input_file_path, "rb") as audio_file:
        # Call the stem separation API - must consume response while file is open
        response = client.music.separate_stems(
            file=audio_file,
            stem_variation_id=stem_variation,
            output_format=output_format,
        )

        # Write the streaming response to a temp file
        with open(temp_zip_path, "wb") as temp_zip:
            for chunk in response:
                temp_zip.write(chunk)

    # Extract the stems from the ZIP archive
    try:
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files = zip_ref.namelist()
    finally:
        # Clean up the temporary ZIP file
        os.unlink(temp_zip_path)

    # Convert PCM files to WAV
    sample_rate = get_sample_rate_from_format(output_format)
    wav_files = []
    for f in extracted_files:
        file_path = output_dir / f
        if file_path.suffix == ".pcm":
            wav_path = file_path.with_suffix(".wav")
            pcm_to_wav(file_path, wav_path, sample_rate)
            os.unlink(file_path)  # Remove original PCM file
            wav_files.append(wav_path.name)
        else:
            wav_files.append(f)

    print(f"Created {len(wav_files)} stem files:")
    for f in wav_files:
        print(f"  - {f}")

    return output_dir


def parse_args() -> argparse.Namespace:
    """
    Argparser for stem separation script.
    """
    parser = argparse.ArgumentParser(
        description="Stem Separation using ElevenLabs API",
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
        "--output-dir",
        "-o",
        type=Path,
        default=Path("stems_output"),
        help="Output directory to save the separated stems.",
    )
    parser.add_argument(
        "--stem-variation",
        "-s",
        choices=["two_stems_v1", "six_stems_v1"],
        default="two_stems_v1",
        help=(
            "Stem variation to use. "
            "'two_stems_v1' separates into vocals and accompaniment. "
            "'six_stems_v1' separates into vocals, drums, bass, guitar, piano, and other."
        ),
    )
    parser.add_argument(
        "--output-format",
        "-f",
        choices=[
            "mp3_22050_32",
            "mp3_24000_48",
            "mp3_44100_32",
            "mp3_44100_64",
            "mp3_44100_96",
            "mp3_44100_128",
            "mp3_44100_192",
            "pcm_8000",
            "pcm_16000",
            "pcm_22050",
            "pcm_24000",
            "pcm_32000",
            "pcm_44100",
            "pcm_48000",
            "ulaw_8000",
            "alaw_8000",
            "opus_48000_32",
            "opus_48000_64",
            "opus_48000_96",
            "opus_48000_128",
            "opus_48000_192",
        ],
        default="pcm_16000",
        help="Output format for the separated stems.",
    )

    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def main() -> None:
    """
    Main entry point for stem separation.
    """
    start_time = time.time()
    args = parse_args()

    print("=" * 50)
    print("ElevenLabs Stem Separation")
    print("=" * 50)

    result = separate_stems(
        input_file_path=args.input_file,
        output_dir=args.output_dir,
        stem_variation=args.stem_variation,
        output_format=args.output_format,
    )

    if result:
        print("=" * 50)
        print(f"Stem separation successful! Files saved at: {result}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
    else:
        print("Stem separation failed.")


if __name__ == "__main__":
    main()
