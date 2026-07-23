#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the S2S client.

This test runs the actual S2S client and validates the audio translation pipeline:
1. Runs the S2S client with sample audio input
2. Validates output audio generation
3. Checks file formats and sizes
4. Verifies audio translation functionality
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.s2s.args import argsfactory
from common.health import check_service_health

pytestmark = pytest.mark.functional

# Live services this module exercises. The autouse conftest fixture probes
# each address and skips the whole module when any service is unreachable.
REQUIRED_SERVICES = (("S2S", "localhost:50050"),)


def test_s2s_service_health() -> None:
    """The S2S service answers the standard gRPC health probe.

    Examples:
        >>> test_s2s_service_health()  # doctest: +SKIP
    """
    assert check_service_health(server="localhost:50050") is True


def test_input_files_exist() -> None:
    """The sample audio input used by the client exists.

    Examples:
        >>> test_input_files_exist()  # doctest: +SKIP
    """
    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.input_audio

    assert audio_file.exists(), f"Audio file not found: {audio_file}"


def cleanup_previous_outputs() -> None:
    """Clean up any previous test outputs.

    Examples:
        >>> cleanup_previous_outputs()  # doctest: +SKIP
    """
    outputs_dir = Path(__file__).parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.glob("s2s_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def _assert_valid_audio_header(output_file: Path, audio_format: str) -> None:
    """Assert that an audio file starts with a valid WAV or MP3 header.

    Args:
        output_file (Path): Path to the audio file to validate.
        audio_format (str): Expected format, ``wav`` or ``mp3``.

    Examples:
        >>> _assert_valid_audio_header(Path("out.mp3"), "mp3")  # doctest: +SKIP
    """
    with open(output_file, "rb") as f:
        header = f.read(10)
    assert len(header) >= 3, f"Output file too small: {len(header)} bytes"

    if audio_format == "wav":
        is_valid = header.startswith(b"RIFF")
    else:
        # MP3 files start with an ID3 tag or an MPEG frame-sync pattern
        # (11 set bits across the first two bytes).
        is_valid = header.startswith(b"ID3") or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
    assert is_valid, f"Output file is not valid {audio_format.upper()}, header: {header[:10]!r}"


def test_s2s_client_comprehensive(
    client_subprocess_env: dict[str, str],
    source_language: str | None,
    target_language: str | None,
    audio_format: str | None,
) -> None:
    """Run the S2S client end to end and validate the translated audio output.

    Args:
        source_language (str | None): Optional source language override from
            the ``--source-language`` option; falls back to client defaults.
        target_language (str | None): Optional target language override from
            the ``--target-language`` option; falls back to client defaults.
        audio_format (str | None): Optional output audio format from the
            ``--audio-format`` option; falls back to mp3.

    Examples:
        >>> # Invoked by pytest with fixture-resolved arguments.
        >>> test_s2s_client_comprehensive(None, None, None)  # doctest: +SKIP
    """
    cleanup_previous_outputs()

    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    defaults = argsfactory().parse_args([])
    resolved_source_language = source_language or defaults.source_language
    resolved_target_language = target_language or defaults.target_language
    resolved_audio_format = audio_format or "mp3"

    output_file = outputs_dir / f"s2s_comprehensive_output.{resolved_audio_format}"
    latency_plot = outputs_dir / "s2s_comprehensive_latency_plot.png"
    audio_file = project_root / defaults.input_audio

    # Build command with latency analysis and optimized chunk sizes
    cmd = [
        sys.executable,
        "client/s2s/app.py",
        "--s2s-server",
        "localhost:50050",
        "--input-audio",
        str(audio_file),
        "--output-audio",
        str(output_file),
        "--latency-plot",
        str(latency_plot),
        "--source-language",
        resolved_source_language,
        "--target-language",
        resolved_target_language,
        "--chunk-size-audio-secs",
        "1.0",  # Optimized chunk size for better streaming
    ]

    print(f"Running comprehensive S2S test: {' '.join(cmd)}")

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
        f"S2S client failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    print(f"OK: S2S client completed successfully in {processing_time:.2f} seconds")

    # Validate audio output
    assert output_file.exists(), f"Output audio file not created: {output_file}"
    assert output_file.stat().st_size > 0, f"Output audio file is empty: {output_file}"
    _assert_valid_audio_header(output_file=output_file, audio_format=resolved_audio_format)

    # Validate latency plot
    assert latency_plot.exists(), f"Latency plot not created: {latency_plot}"
    assert latency_plot.stat().st_size > 0, f"Latency plot is empty: {latency_plot}"

    with open(latency_plot, "rb") as f:
        png_header = f.read(8)
    assert png_header == b"\x89PNG\r\n\x1a\n", "Latency plot is not valid PNG format"

    print(
        f"OK: Output audio is valid {resolved_audio_format.upper()} "
        f"({output_file.stat().st_size} bytes): {output_file}"
    )
    print(f"OK: Latency plot created ({latency_plot.stat().st_size} bytes): {latency_plot}")
