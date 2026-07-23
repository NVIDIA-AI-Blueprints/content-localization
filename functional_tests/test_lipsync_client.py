#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the LipSync client.

This test runs the actual LipSync client and validates the lip-sync pipeline:
1. Runs the LipSync client with sample audio and video inputs
2. Validates output video generation
3. Checks file formats and sizes
4. Verifies lip-sync functionality
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.lipsync.args import argsfactory
from common.health import check_service_health

pytestmark = pytest.mark.functional

# Live services this module exercises. The autouse conftest fixture probes
# each address and skips the whole module when any service is unreachable.
REQUIRED_SERVICES = (("LipSync", "localhost:50054"),)


def test_lipsync_service_health() -> None:
    """The LipSync service answers the standard gRPC health probe.

    Examples:
        >>> test_lipsync_service_health()  # doctest: +SKIP
    """
    assert check_service_health(server="localhost:50054") is True


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
        for file in outputs_dir.glob("lipsync_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_lipsync_client_comprehensive(client_subprocess_env: dict[str, str]) -> None:
    """Run the LipSync client end to end and validate the MP4 output.

    Examples:
        >>> test_lipsync_client_comprehensive()  # doctest: +SKIP
    """
    cleanup_previous_outputs()

    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "lipsync_comprehensive_output.mp4"

    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.input_audio
    video_file = project_root / defaults.input_mp4

    # Build command with optimized parameters
    cmd = [
        sys.executable,
        "client/lipsync/app.py",
        "--lipsync-server",
        "localhost:50054",
        "--input-audio",
        str(audio_file),
        "--input-mp4",
        str(video_file),
        "--output-mp4",
        str(output_file),
    ]

    print(f"Running comprehensive LipSync test: {' '.join(cmd)}")

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
        f"LipSync client failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    print(f"OK: LipSync client completed successfully in {processing_time:.2f} seconds")

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

    # Check file size is reasonable (lip-sync output is often highly compressed)
    input_video_size = video_file.stat().st_size
    output_size = output_file.stat().st_size
    assert output_size >= input_video_size * 0.1, (
        f"Output file seems too small: {output_size} bytes (input: {input_video_size} bytes)"
    )

    print(f"OK: Output file is valid MP4 format ({output_size} bytes): {output_file}")
