#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the Direct client.

This test runs the actual Direct client and validates the complete pipeline:
1. Runs the direct client with sample inputs
2. Validates output video generation
3. Checks file formats and sizes
4. Verifies the direct service communication pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.direct.args import argsfactory
from common.health import check_service_health

pytestmark = pytest.mark.functional

# Live services this module exercises. The autouse conftest fixture probes
# each address and skips the whole module when any service is unreachable.
REQUIRED_SERVICES = (
    ("S2S", "localhost:50050"),
    ("LipSync", "localhost:50054"),
    ("ASD", "localhost:50055"),
)


def test_services_health() -> None:
    """Every service used by the direct client answers the gRPC health probe.

    Examples:
        >>> test_services_health()  # doctest: +SKIP
    """
    for service_name, address in REQUIRED_SERVICES:
        assert check_service_health(server=address) is True, (
            f"{service_name} service at {address} reported unhealthy"
        )


def test_input_files_exist() -> None:
    """The sample audio and video inputs used by the client exist.

    Examples:
        >>> test_input_files_exist()  # doctest: +SKIP
    """
    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.input_audio
    video_file = project_root / defaults.input_mp4

    assert audio_file.exists(), f"Audio file not found: {audio_file}"
    assert video_file.exists(), f"Video file not found: {video_file}"


def cleanup_previous_outputs() -> None:
    """Clean up any previous test outputs.

    Examples:
        >>> cleanup_previous_outputs()  # doctest: +SKIP
    """
    outputs_dir = Path(__file__).parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.glob("direct_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def _assert_valid_mp4(output_file: Path) -> None:
    """Assert that a file has a valid MP4 container signature.

    Args:
        output_file (Path): Path to the video file to validate.

    Examples:
        >>> _assert_valid_mp4(Path("out.mp4"))  # doctest: +SKIP
    """
    with open(output_file, "rb") as f:
        header = f.read(12)
    assert len(header) >= 8, f"Output video file too small: {len(header)} bytes"

    # Check for MP4 signature patterns
    is_valid_mp4 = (
        header[4:8] == b"ftyp"  # ftyp atom
        or header[4:8] == b"moov"  # moov atom
        or header[4:8] == b"mdat"  # mdat atom
    )
    assert is_valid_mp4, f"Output video file is not valid MP4, header: {header!r}"


def _assert_valid_audio_header(output_audio: Path, audio_format: str) -> None:
    """Assert that an audio file starts with a valid WAV or MP3 header.

    Args:
        output_audio (Path): Path to the audio file to validate.
        audio_format (str): Expected format, ``wav`` or ``mp3``.

    Examples:
        >>> _assert_valid_audio_header(Path("out.mp3"), "mp3")  # doctest: +SKIP
    """
    with open(output_audio, "rb") as f:
        header = f.read(10)
    assert len(header) >= 3, f"Output audio file too small: {len(header)} bytes"

    if audio_format == "wav":
        is_valid = header.startswith(b"RIFF")
    else:
        # MP3 files start with an ID3 tag or an MPEG frame-sync pattern
        # (11 set bits across the first two bytes).
        is_valid = header.startswith(b"ID3") or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
    assert is_valid, (
        f"Output audio file is not valid {audio_format.upper()}, header: {header[:10]!r}"
    )


# Argument-count suppression: every parameter is a pytest fixture and must
# be declared by name for injection.
def test_direct_client_comprehensive(  # noqa: PLR0913
    client_subprocess_env: dict[str, str],
    source_language: str | None,
    target_language: str | None,
    diarization_file: str | None,
    diarization_format: str | None,
    audio_format: str | None,
) -> None:
    """Run the Direct client end to end and validate the video and audio outputs.

    Args:
        source_language (str | None): Optional source language override from
            the ``--source-language`` option; falls back to client defaults.
        target_language (str | None): Optional target language override from
            the ``--target-language`` option; falls back to client defaults.
        diarization_file (str | None): Optional diarization JSON path from the
            ``--diarization-file`` option; falls back to the sample asset.
        diarization_format (str | None): Optional diarization format from the
            ``--diarization-format`` option; falls back to elevenlabs-scribe.
        audio_format (str | None): Optional output audio format from the
            ``--audio-format`` option; falls back to mp3.

    Examples:
        >>> # Invoked by pytest with fixture-resolved arguments.
        >>> test_direct_client_comprehensive(None, None, None, None, None)  # doctest: +SKIP
    """
    cleanup_previous_outputs()

    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "direct_comprehensive_output.mp4"

    defaults = argsfactory().parse_args([])
    resolved_source_language = source_language or defaults.source_language
    resolved_target_language = target_language or defaults.target_language
    resolved_audio_format = audio_format or "mp3"

    output_audio = outputs_dir / f"direct_comprehensive_audio.{resolved_audio_format}"
    audio_file = project_root / defaults.input_audio
    video_file = project_root / defaults.input_mp4
    resolved_diarization_file = diarization_file or str(
        project_root / "assets" / "diarization.json"
    )
    resolved_diarization_format = diarization_format or "elevenlabs-scribe"

    assert Path(resolved_diarization_file).exists(), (
        f"Diarization file not found: {resolved_diarization_file}"
    )

    # Build command with optimized chunk sizes for better streaming
    cmd = [
        sys.executable,
        "client/direct/app.py",
        "--s2s-server",
        "localhost:50050",
        "--lipsync-server",
        "localhost:50054",
        "--asd-server",
        "localhost:50055",
        "--input-audio",
        str(audio_file),
        "--input-mp4",
        str(video_file),
        "--output-mp4",
        str(output_file),
        "--output-audio",
        str(output_audio),
        "--source-language",
        resolved_source_language,
        "--target-language",
        resolved_target_language,
        "--chunk-size-audio-secs",
        "2.0",  # Optimized chunk size for better streaming
        "--chunk-size-video-bytes",
        "1048576",  # 1MB chunk size for optimal streaming
        "--diarization-file",
        resolved_diarization_file,
        "--diarization-format",
        resolved_diarization_format,
        "--diarization-chunked-per-segment",
    ]

    print(f"Running comprehensive Direct test: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        env=client_subprocess_env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
        check=False,
    )
    processing_time = time.time() - start_time

    assert result.returncode == 0, (
        f"Direct client failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    print(f"OK: Direct client completed successfully in {processing_time:.2f} seconds")

    # Validate video output
    assert output_file.exists(), f"Output video file not created: {output_file}"
    assert output_file.stat().st_size > 0, f"Output video file is empty: {output_file}"
    _assert_valid_mp4(output_file=output_file)

    # Validate audio output
    assert output_audio.exists(), f"Output audio file not created: {output_audio}"
    assert output_audio.stat().st_size > 0, f"Output audio file is empty: {output_audio}"
    _assert_valid_audio_header(output_audio=output_audio, audio_format=resolved_audio_format)

    # Check file sizes are reasonable.
    # LipSync re-encodes video at a low bitrate (default 3 Mbps) so the
    # output will be much smaller than the high-bitrate input (~20 Mbps).
    # S2S similarly re-encodes audio at a different bitrate.
    # Use absolute minimums instead of input-relative thresholds.
    output_video_size = output_file.stat().st_size
    output_audio_size = output_audio.stat().st_size
    min_video_bytes = 1 * 1024 * 1024  # 1 MB
    min_audio_bytes = 10 * 1024  # 10 KB

    assert output_video_size >= min_video_bytes, (
        f"Output video seems too small: "
        f"{output_video_size} bytes (minimum: {min_video_bytes} bytes)"
    )
    assert output_audio_size >= min_audio_bytes, (
        f"Output audio seems too small: "
        f"{output_audio_size} bytes (minimum: {min_audio_bytes} bytes)"
    )

    print(f"OK: Output video is valid MP4 ({output_video_size} bytes): {output_file}")
    print(
        f"OK: Output audio is valid {resolved_audio_format.upper()} "
        f"({output_audio_size} bytes): {output_audio}"
    )
