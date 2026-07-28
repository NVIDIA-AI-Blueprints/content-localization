# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video preprocessing: audio extraction and duration probing."""

import os
import subprocess

from client.common.paths import ensure_parent_dir


def get_video_duration(video_path: str) -> float:
    """Get duration of a video file in seconds using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails or returns invalid output.

    Examples:
        >>> dur = get_video_duration("sample.mp4")
        >>> dur > 0
        True
    """
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid duration from ffprobe for {video_path}: {result.stdout.strip()}"
        ) from exc


def get_video_stream_info(video_path: str) -> tuple[int, int, int]:
    """Get width, height, and frame count of a video's first stream via ffprobe.

    Uses ``-count_packets`` to count frames from the index without decoding,
    which is fast even for large files.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple[int, int, int]: ``(width, height, frame_count)``.

    Raises:
        RuntimeError: If ffprobe fails or returns unexpected output.

    Examples:
        >>> w, h, n = get_video_stream_info("sample.mp4")
        >>> w > 0 and h > 0 and n > 0
        True
    """
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,nb_read_packets",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    try:
        parts = result.stdout.strip().split(",")
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(
            f"Invalid stream info from ffprobe for {video_path}: {result.stdout.strip()!r}"
        ) from exc


def extract_audio(
    video_path: str,
    output_wav_path: str,
) -> None:
    """Extract audio from a video file as 16 kHz mono WAV.

    Uses ffmpeg to demux the audio track and re-encode it as
    16-bit PCM WAV at 16 kHz with a single channel.

    Args:
        video_path (str): Path to the input video file.
        output_wav_path (str): Path for the output WAV file.

    Raises:
        RuntimeError: If ffmpeg extraction fails.

    Examples:
        >>> extract_audio("input.mp4", "output.wav")
    """
    ensure_parent_dir(path=output_wav_path)
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_wav_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed for {video_path}: {result.stderr.strip()}")


def preprocess_video(
    video_path: str,
    output_dir: str,
) -> tuple[str, float, int, int, int]:
    """Extract audio from a video and probe its duration, resolution, and frame count.

    Creates a WAV file in ``{output_dir}/preprocessed/`` with
    the same stem as the input video.

    Args:
        video_path (str): Path to the input MP4 video.
        output_dir (str): Base directory for preprocessed files.

    Returns:
        tuple[str, float, int, int, int]:
            ``(audio_wav_path, duration_secs, width, height, frame_count)``.

    Raises:
        RuntimeError: If any preprocessing step fails.

    Examples:
        >>> wav, dur, w, h, n = preprocess_video("in.mp4", "out/")
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    prep_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    width, height, frame_count = get_video_stream_info(video_path)

    wav_path = os.path.join(prep_dir, f"{stem}.wav")
    extract_audio(
        video_path=video_path,
        output_wav_path=wav_path,
    )

    return wav_path, duration, width, height, frame_count
