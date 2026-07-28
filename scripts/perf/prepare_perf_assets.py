# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create benchmark clips and a perf manifest from one input video.

Examples:
    $ source .venv/bin/activate
    $ python scripts/perf/prepare_perf_assets.py \
        --input-video assets/sample_video_streamable.mp4 \
        --out-dir outputs/perf_segments/sample \
        --manifest outputs/perf_segments/sample/assets.manifest
"""

import argparse
import math
import re
import subprocess
from pathlib import Path

DEFAULT_DURATIONS = "10,20,30,60,120,300,600,full"
SECONDS_PER_MINUTE = 60


def _run(command: list[str]) -> None:
    """Run a subprocess command and raise on failure.

    Args:
        command (list[str]): Command and arguments.

    Examples:
        >>> _run(["true"])
    """
    subprocess.run(command, check=True)  # noqa: S603


def _safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for generated clip names.

    Args:
        path (Path): Source video path.

    Returns:
        str: Sanitized stem.

    Examples:
        >>> _safe_stem(Path("My Video.mp4"))
        'My_Video'
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")
    return stem or "asset"


def _duration_label(seconds: int | None) -> str:
    """Format a clip duration for manifest tags and filenames.

    Args:
        seconds (int | None): Duration in seconds, or ``None`` for full length.

    Returns:
        str: Label such as ``10s``, ``1min``, or ``full``.

    Examples:
        >>> _duration_label(60)
        '1min'
    """
    if seconds is None:
        return "full"
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds}s"
    if seconds % SECONDS_PER_MINUTE == 0:
        return f"{seconds // SECONDS_PER_MINUTE}min"
    minutes, secs = divmod(seconds, SECONDS_PER_MINUTE)
    return f"{minutes}m{secs:02d}s"


def _duration_suffix(seconds: int | None) -> str:
    """Return a sortable duration suffix for generated filenames.

    Args:
        seconds (int | None): Duration in seconds, or ``None`` for full length.

    Returns:
        str: Filename-safe suffix.

    Examples:
        >>> _duration_suffix(60)
        '0060s'
    """
    return "full" if seconds is None else f"{seconds:04d}s"


def _parse_durations(value: str) -> list[int | None]:
    """Parse a comma-separated duration list.

    Args:
        value (str): Comma-separated seconds plus optional ``full``.

    Returns:
        list[int | None]: Durations in seconds; ``None`` means full length.

    Raises:
        ValueError: If a duration token is invalid.

    Examples:
        >>> _parse_durations("10,60,full")
        [10, 60, None]
    """
    durations: list[int | None] = []
    for raw_token in value.split(","):
        duration_value = raw_token.strip().lower()
        if not duration_value:
            continue
        if duration_value == "full":
            durations.append(None)
            continue
        seconds = int(duration_value)
        if seconds <= 0:
            raise ValueError(f"Duration must be positive: {raw_token}")
        durations.append(seconds)
    if not durations:
        raise ValueError("At least one duration is required")
    return durations


def _probe_duration(path: Path) -> float | None:
    """Return video duration from ffprobe when available.

    Args:
        path (Path): Video file path.

    Returns:
        float | None: Duration in seconds, or ``None`` when ffprobe fails.

    Examples:
        >>> _probe_duration(Path("missing.mp4")) is None
        True
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _ffmpeg_clip_command(
    input_video: Path,
    output_video: Path,
    seconds: int | None,
    audio_codec: str,
    video_codec: str,
) -> list[str]:
    """Build the ffmpeg command for one generated clip.

    Args:
        input_video (Path): Source video.
        output_video (Path): Destination MP4.
        seconds (int | None): Clip length in seconds, or ``None`` for full.
        audio_codec (str): ffmpeg audio codec value.
        video_codec (str): ffmpeg video codec value.

    Returns:
        list[str]: Command arguments.

    Examples:
        >>> _ffmpeg_clip_command(Path("in.mp4"), Path("out.mp4"), 10, "copy", "copy")[:2]
        ['ffmpeg', '-y']
    """
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if seconds is not None:
        command += ["-t", str(seconds)]
    command += [
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        video_codec,
        "-c:a",
        audio_codec,
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    return command


def prepare_assets(  # noqa: PLR0913
    input_video: Path,
    out_dir: Path,
    manifest: Path,
    durations: list[int | None],
    prefix: str | None,
    audio_codec: str,
    video_codec: str,
) -> list[Path]:
    """Create benchmark clips and write a manifest.

    Args:
        input_video (Path): Source MP4 or other ffmpeg-readable video.
        out_dir (Path): Directory for generated clips.
        manifest (Path): Manifest path to write.
        durations (list[int | None]): Clip durations.
        prefix (str | None): Optional filename prefix.
        audio_codec (str): ffmpeg audio codec value.
        video_codec (str): ffmpeg video codec value.

    Returns:
        list[Path]: Generated clip paths.

    Raises:
        FileNotFoundError: If the input video is missing.

    Examples:
        >>> prepare_assets(Path("missing.mp4"), Path("out"), Path("m"), [], None, "copy", "copy")
        Traceback (most recent call last):
        ...
        FileNotFoundError: missing.mp4
    """
    if not input_video.is_file():
        raise FileNotFoundError(input_video)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    source_duration = _probe_duration(input_video)
    base = prefix or _safe_stem(input_video)
    generated: list[Path] = []
    manifest_lines = ["# tag\tpath"]

    for seconds in durations:
        if (
            seconds is not None
            and source_duration is not None
            and seconds > math.ceil(source_duration)
        ):
            continue
        label_seconds = round(source_duration) if seconds is None and source_duration else seconds
        label = _duration_label(label_seconds)
        suffix = _duration_suffix(seconds)
        output_video = out_dir / f"{base}_{suffix}.mp4"
        _run(
            _ffmpeg_clip_command(
                input_video=input_video,
                output_video=output_video,
                seconds=seconds,
                audio_codec=audio_codec,
                video_codec=video_codec,
            )
        )
        generated.append(output_video)
        manifest_lines.append(f"{label}\t{output_video}")

    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return generated


def main() -> None:
    """Parse arguments and create benchmark assets.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="Create CLBP perf benchmark clips.")
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--durations", default=DEFAULT_DURATIONS)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--audio-codec",
        default="copy",
        choices=["copy", "aac"],
        help="Use aac when the target machine needs AAC-normalized MP4 inputs.",
    )
    parser.add_argument("--video-codec", default="copy")
    args = parser.parse_args()

    durations = _parse_durations(args.durations)
    clips = prepare_assets(
        input_video=args.input_video,
        out_dir=args.out_dir,
        manifest=args.manifest,
        durations=durations,
        prefix=args.prefix,
        audio_codec=args.audio_codec,
        video_codec=args.video_codec,
    )
    print(f"Wrote {len(clips)} clip(s) to {args.out_dir}")
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
