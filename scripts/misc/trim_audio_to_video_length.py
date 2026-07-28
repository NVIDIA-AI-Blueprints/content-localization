#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trim an audio file to match a video's duration.

Uses ffprobe to determine the video duration and ffmpeg to trim
the audio. If the audio is already shorter than or equal to the
video, it is copied as-is.

Examples:
    python scripts/misc/trim_audio_to_video_length.py \
        --video input.mp4 --audio background.wav --output trimmed.wav

    # Batch: trim every .wav in a directory to a single video's length
    for f in bg_audios/*.wav; do
        python scripts/misc/trim_audio_to_video_length.py \
            --video clip.mp4 --audio "$f" \
            --output "trimmed_$(basename "$f")"
    done
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_media_duration(filepath: Path) -> float:
    """Return the duration in seconds of a media file via ffprobe.

    Args:
        filepath: Path to the media file (video or audio).

    Returns:
        Duration in seconds as a float.

    Raises:
        RuntimeError: If ffprobe fails or returns no duration.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(filepath),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {filepath}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse duration from ffprobe output: {result.stdout!r}"
        ) from exc


def trim_audio(
    audio_path: Path,
    output_path: Path,
    duration_secs: float,
) -> Path:
    """Trim an audio file to a given duration using ffmpeg.

    If the audio is already shorter than or equal to ``duration_secs``
    the file is copied without re-encoding.

    Args:
        audio_path: Source audio file.
        output_path: Destination path for the trimmed audio.
        duration_secs: Maximum duration in seconds.

    Returns:
        The output path.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    audio_duration = get_media_duration(filepath=audio_path)

    if audio_duration <= duration_secs:
        # Audio is already short enough — copy without re-encoding
        print(f"Audio ({audio_duration:.2f}s) <= video ({duration_secs:.2f}s). Copying as-is.")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-c",
            "copy",
            str(output_path),
        ]
    else:
        print(f"Trimming audio from {audio_duration:.2f}s to {duration_secs:.2f}s")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-t",
            f"{duration_secs:.6f}",
            "-c",
            "copy",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    print(f"Wrote {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with video, audio, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Trim an audio file to match a video's duration.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video whose duration sets the trim length.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Input audio file to trim.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the trimmed audio.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: trim audio to video length."""
    args = parse_args()

    if not args.video.exists():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not args.audio.exists():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    video_duration = get_media_duration(filepath=args.video)
    print(f"Video duration: {video_duration:.2f}s")

    trim_audio(
        audio_path=args.audio,
        output_path=args.output,
        duration_secs=video_duration,
    )


if __name__ == "__main__":
    main()
