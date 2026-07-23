#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the Controller client.

This test runs the actual Controller client and validates the complete pipeline:
1. Runs the controller client with sample inputs
2. Validates output video generation
3. Checks file formats and sizes
4. Verifies the complete content localization pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.controller.args import argsfactory
from common.health import check_service_health

pytestmark = pytest.mark.functional

# Live services this module exercises. The autouse conftest fixture probes
# each address and skips the whole module when any service is unreachable.
REQUIRED_SERVICES = (("Controller", "localhost:50056"),)


def test_controller_service_health() -> None:
    """The Controller service answers the standard gRPC health probe.

    Examples:
        >>> test_controller_service_health()  # doctest: +SKIP
    """
    assert check_service_health(server="localhost:50056") is True


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
        for file in outputs_dir.glob("controller_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_controller_client_comprehensive(
    client_subprocess_env: dict[str, str],
    source_language: str | None,
    target_language: str | None,
    diarization_file: str | None,
    diarization_format: str | None,
) -> None:
    """Run the Controller client end to end and validate the dubbed MP4 output.

    Args:
        source_language (str | None): Optional source language override from
            the ``--source-language`` option; falls back to client defaults.
        target_language (str | None): Optional target language override from
            the ``--target-language`` option; falls back to client defaults.
        diarization_file (str | None): Optional diarization JSON path from the
            ``--diarization-file`` option; falls back to the sample asset.
        diarization_format (str | None): Optional diarization format from the
            ``--diarization-format`` option; falls back to elevenlabs-scribe.

    Examples:
        >>> # Invoked by pytest with fixture-resolved arguments.
        >>> test_controller_client_comprehensive(None, None, None, None)  # doctest: +SKIP
    """
    cleanup_previous_outputs()

    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "controller_comprehensive_output.mp4"

    defaults = argsfactory().parse_args([])
    resolved_source_language = source_language or defaults.source_language
    resolved_target_language = target_language or defaults.target_language
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
        "client/controller/app.py",
        "--controller-server",
        "localhost:50056",
        "--input-audio",
        str(audio_file),
        "--input-mp4",
        str(video_file),
        "--output-mp4",
        str(output_file),
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

    print(f"Running comprehensive Controller test: {' '.join(cmd)}")

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
        f"Controller client failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    print(f"OK: Controller client completed successfully in {processing_time:.2f} seconds")

    assert output_file.exists(), f"Output file not created: {output_file}"
    assert output_file.stat().st_size > 0, f"Output file is empty: {output_file}"

    # Validate MP4 format and content
    with open(output_file, "rb") as f:
        header = f.read(12)
    assert len(header) >= 8, f"Output file too small: {len(header)} bytes"

    # Check for MP4 signature patterns
    is_valid_mp4 = (
        header[4:8] == b"ftyp"  # ftyp atom
        or header[4:8] == b"moov"  # moov atom
        or header[4:8] == b"mdat"  # mdat atom
    )
    assert is_valid_mp4, f"Output file is not valid MP4, header: {header!r}"

    # Check file size is reasonable.
    # LipSync re-encodes at a low bitrate (default 3 Mbps) so the output
    # will be much smaller than the original high-bitrate input (~20 Mbps).
    # Use a 1 MB absolute minimum instead of an input-relative threshold.
    output_size = output_file.stat().st_size
    min_output_bytes = 1 * 1024 * 1024  # 1 MB
    assert output_size >= min_output_bytes, (
        f"Output file seems too small: {output_size} bytes (minimum: {min_output_bytes} bytes)"
    )

    print(f"OK: Output file is valid MP4 format ({output_size} bytes): {output_file}")
