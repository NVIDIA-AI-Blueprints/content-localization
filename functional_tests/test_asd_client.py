#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the ASD client.

This test runs the actual ASD client and validates the speaker detection pipeline:
1. Runs the ASD client with sample video input
2. Validates output generation
3. Checks file formats and sizes
4. Verifies speaker detection functionality
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.asd.args import argsfactory
from common.health import check_service_health

pytestmark = pytest.mark.functional

# Live services this module exercises. The autouse conftest fixture probes
# each address and skips the whole module when any service is unreachable.
REQUIRED_SERVICES = (("ASD", "localhost:50055"),)


def test_asd_service_health() -> None:
    """The ASD service answers the standard gRPC health probe.

    Examples:
        >>> test_asd_service_health()  # doctest: +SKIP
    """
    assert check_service_health(server="localhost:50055") is True


def test_input_files_exist() -> None:
    """The sample video and diarization inputs used by the client exist.

    Examples:
        >>> test_input_files_exist()  # doctest: +SKIP
    """
    defaults = argsfactory().parse_args([])
    video_file = project_root / defaults.input_mp4
    diarization_file = project_root / "assets" / "diarization.json"

    assert video_file.exists(), f"Video file not found: {video_file}"
    assert diarization_file.exists(), f"Diarization file not found: {diarization_file}"


def cleanup_previous_outputs() -> None:
    """Clean up any previous test outputs.

    Examples:
        >>> cleanup_previous_outputs()  # doctest: +SKIP
    """
    outputs_dir = Path(__file__).parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.glob("asd_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_asd_client_comprehensive(
    client_subprocess_env: dict[str, str],
    diarization_file: str | None,
    diarization_format: str | None,
) -> None:
    """Run the ASD client end to end and validate the speaker-info CSV output.

    Args:
        diarization_file (str | None): Optional diarization JSON path from the
            ``--diarization-file`` option; falls back to the sample asset.
        diarization_format (str | None): Optional diarization format from the
            ``--diarization-format`` option; falls back to elevenlabs-scribe.

    Examples:
        >>> # Invoked by pytest with fixture-resolved arguments.
        >>> test_asd_client_comprehensive(None, None)  # doctest: +SKIP
    """
    cleanup_previous_outputs()

    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "asd_comprehensive_output.csv"

    defaults = argsfactory().parse_args([])
    video_file = project_root / defaults.input_mp4
    resolved_diarization_file = diarization_file or str(
        project_root / "assets" / "diarization.json"
    )
    resolved_diarization_format = diarization_format or "elevenlabs-scribe"

    # Build command with optimized chunk sizes for better streaming
    cmd = [
        sys.executable,
        "client/asd/app.py",
        "--asd-server",
        "localhost:50055",
        "--input-mp4",
        str(video_file),
        "--output-speaker-info",
        str(output_file),
        "--chunk-size-video-bytes",
        "1048576",  # 1MB chunk size for optimal streaming
        "--diarization-file",
        resolved_diarization_file,
        "--diarization-format",
        resolved_diarization_format,
        "--diarization-chunked-per-segment",
    ]

    print(f"Running comprehensive ASD test: {' '.join(cmd)}")

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
        f"ASD client failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    print(f"OK: ASD client completed successfully in {processing_time:.2f} seconds")

    assert output_file.exists(), f"Output file not created: {output_file}"
    assert output_file.stat().st_size > 0, f"Output file is empty: {output_file}"

    # Validate CSV structure: a header row plus at least one data row of
    # speaker detection results.
    with open(output_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2, f"Output CSV has no data rows: {output_file}"

    header = lines[0].strip()
    assert header and "," in header, f"Output file has an invalid CSV header: {header!r}"

    data_line = lines[1].strip()
    assert data_line and "," in data_line, f"Output CSV data row is invalid: {data_line!r}"

    print(f"OK: Output file is valid CSV format with {len(lines)} lines")
    print(f"OK: Output file size: {output_file.stat().st_size} bytes")
