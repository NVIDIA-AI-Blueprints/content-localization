#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolate foreground (speech/vocals) from background using Camb AI audio separation.

Uses Camb.ai's ``/audio-separation`` API: upload media, poll until ``SUCCESS``, fetch URLs from
``/audio-separation-result/{run_id}``, then download the **foreground** stem (and
optionally **background**).

Reference: https://docs.camb.ai/api-reference/endpoint/create-audio-separation

Examples:
    CAMB_API_KEY=<key> python scripts/camb/audio_isolation.py -i noisy.wav -o voice.wav

    python scripts/camb/audio_isolation.py -i clip.mp3 -o fg.wav --background-output bg.wav
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import requests

CAMB_API_BASE_URL = "https://client.camb.ai/apis"
CAMB_API_KEY_ENV = "CAMB_API_KEY"
_DEFAULT_MAX_ATTEMPTS = 120
_DEFAULT_POLL_INTERVAL = 10


def _headers() -> dict[str, str]:
    """Build Camb request headers with ``x-api-key``.

    Returns:
        dict[str, str]: Headers for authenticated API calls.

    Raises:
        ValueError: If ``CAMB_API_KEY`` is unset.

    Examples:
        >>> # _headers()  # doctest: +SKIP
    """
    key = os.environ.get(CAMB_API_KEY_ENV)
    if not key:
        raise ValueError(
            f"{CAMB_API_KEY_ENV} environment variable not set. "
            "Export it or load .env before running this script."
        )
    return {"x-api-key": key}


def submit_audio_separation(
    *,
    input_file_path: Path,
    headers: dict[str, str],
) -> str:
    """POST ``/audio-separation`` with multipart ``media_file``.

    Args:
        input_file_path (Path): Local AAC/FLAC/MP3/WAV file.
        headers (dict[str, str]): Must include ``x-api-key``.

    Returns:
        str: Camb ``task_id`` for status polling.

    Raises:
        FileNotFoundError: If *input_file_path* is missing.
        requests.HTTPError: If the API returns a non-2xx response.
        RuntimeError: If the JSON body omits ``task_id``.

    Examples:
        >>> # submit_audio_separation(input_file_path=Path("a.wav"), headers=h)  # doctest: +SKIP
    """
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    with input_file_path.open("rb") as audio_file:
        response = requests.post(
            f"{CAMB_API_BASE_URL}/audio-separation",
            headers=headers,
            files={"media_file": (input_file_path.name, audio_file)},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"Camb /audio-separation missing task_id: {data}")
    return task_id


def wait_for_separation_success(
    *,
    task_id: str,
    headers: dict[str, str],
    max_attempts: int,
    poll_interval_seconds: int,
) -> int:
    """Poll ``GET /audio-separation/{task_id}`` until ``SUCCESS``; return ``run_id``.

    Args:
        task_id (str): Task id from ``submit_audio_separation``.
        headers (dict[str, str]): Auth headers.
        max_attempts (int): Max poll iterations.
        poll_interval_seconds (int): Sleep between polls.

    Returns:
        int: ``run_id`` for ``/audio-separation-result/{run_id}``.

    Raises:
        requests.HTTPError: On HTTP errors.
        RuntimeError: On terminal failure states or missing ``run_id``.
        TimeoutError: If *max_attempts* is exceeded.

    Examples:
        >>> # wait_for_separation_success(task_id="t", headers=h, ...)  # doctest: +SKIP
    """
    for attempt in range(max_attempts):
        response = requests.get(
            f"{CAMB_API_BASE_URL}/audio-separation/{task_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        status = str(payload.get("status", "")).upper()
        print(f"  Poll {attempt + 1}: status={status}")

        if status == "SUCCESS":
            run_id = payload.get("run_id")
            if not isinstance(run_id, int):
                raise RuntimeError(f"Camb audio-separation SUCCESS but missing run_id: {payload}")
            return run_id
        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
            message = payload.get("message")
            raise RuntimeError(f"Camb audio-separation failed: status={status}, message={message}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Camb audio-separation timed out after {max_attempts} polls "
        f"(interval={poll_interval_seconds}s)."
    )


def fetch_separation_urls(
    *,
    run_id: int,
    headers: dict[str, str],
) -> tuple[str, str]:
    """GET ``/audio-separation-result/{run_id}`` and return foreground/background URLs.

    Args:
        run_id (int): Run id from a successful separation task.
        headers (dict[str, str]): Auth headers.

    Returns:
        tuple[str, str]: ``(foreground_audio_url, background_audio_url)``.

    Raises:
        requests.HTTPError: On HTTP errors.
        RuntimeError: If either URL is missing.

    Examples:
        >>> # fetch_separation_urls(run_id=1, headers=h)  # doctest: +SKIP
    """
    response = requests.get(
        f"{CAMB_API_BASE_URL}/audio-separation-result/{run_id}",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    fg = data.get("foreground_audio_url")
    bg = data.get("background_audio_url")
    if not isinstance(fg, str) or not fg:
        raise RuntimeError(f"Camb separation-result missing foreground_audio_url: {data}")
    if not isinstance(bg, str) or not bg:
        raise RuntimeError(f"Camb separation-result missing background_audio_url: {data}")
    return fg, bg


def _download_audio_url(
    *,
    audio_url: str,
    output_path: Path,
) -> Path:
    """Stream-download *audio_url* to *output_path*.

    Args:
        audio_url (str): HTTPS URL from Camb result payload.
        output_path (Path): Destination file.

    Returns:
        Path: *output_path* after write.

    Raises:
        requests.HTTPError: If download is non-2xx.

    Examples:
        >>> # _download_audio_url(...)  # doctest: +SKIP
    """
    response = requests.get(audio_url, stream=True, timeout=120)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return output_path


def isolate_audio_camb(
    *,
    input_file_path: Path,
    output_file_path: Path,
    background_output_path: Path | None,
    max_attempts: int,
    poll_interval_seconds: int,
) -> tuple[Path, Path | None]:
    """Run Camb separation and save foreground (and optional background) stems.

    Foreground is the closest analogue to ElevenLabs "isolated voice" output.

    Args:
        input_file_path (Path): Input AAC/FLAC/MP3/WAV.
        output_file_path (Path): Where to write the **foreground** stem.
        background_output_path (Path | None): If set, also write **background** stem.
        max_attempts (int): Status poll limit.
        poll_interval_seconds (int): Seconds between polls.

    Returns:
        tuple[Path, Path | None]: ``(foreground_path, background_path_or_none)``.

    Raises:
        ValueError: If ``CAMB_API_KEY`` is missing.
        FileNotFoundError: If the input file is missing.
        requests.HTTPError: On Camb or download HTTP errors.
        RuntimeError: On API contract violations.
        TimeoutError: If polling times out.

    Examples:
        >>> # isolate_audio_camb(input_file_path=Path("in.wav"), ...)  # doctest: +SKIP
    """
    headers = _headers()
    print(f"Submitting Camb audio separation: {input_file_path}")
    task_id = submit_audio_separation(input_file_path=input_file_path, headers=headers)
    print(f"  task_id={task_id}")
    print("Polling for completion...")
    run_id = wait_for_separation_success(
        task_id=task_id,
        headers=headers,
        max_attempts=max_attempts,
        poll_interval_seconds=poll_interval_seconds,
    )
    print(f"  run_id={run_id}")
    fg_url, bg_url = fetch_separation_urls(run_id=run_id, headers=headers)
    print(f"Downloading foreground → {output_file_path}")
    _download_audio_url(audio_url=fg_url, output_path=output_file_path)
    bg_written: Path | None = None
    if background_output_path is not None:
        print(f"Downloading background → {background_output_path}")
        _download_audio_url(audio_url=bg_url, output_path=background_output_path)
        bg_written = background_output_path
    return output_file_path, bg_written


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed CLI values.

    Examples:
        >>> # parse_args()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(
        description=(
            "Camb AI audio separation — foreground (speech/vocals) vs background, "
            "similar to ElevenLabs audio isolation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        type=Path,
        help="Input audio file (AAC, FLAC, MP3, or WAV).",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=Path("camb_isolated_foreground.wav"),
        help="Output path for the foreground (isolated) stem.",
    )
    parser.add_argument(
        "--background-output",
        type=Path,
        default=None,
        help="If set, also save the background stem to this path.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=_DEFAULT_MAX_ATTEMPTS,
        help="Max status polls while waiting for separation.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=_DEFAULT_POLL_INTERVAL,
        help="Seconds between status polls.",
    )
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    if args.background_output is not None:
        args.background_output = args.background_output.expanduser()
    return args


def main() -> None:
    """CLI entry: run Camb separation and write foreground (and optional background)."""
    start = time.time()
    args = parse_args()

    print("=" * 50)
    print("Camb AI audio separation (foreground / background)")
    print("=" * 50)

    fg_path, bg_path = isolate_audio_camb(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        background_output_path=args.background_output,
        max_attempts=args.max_attempts,
        poll_interval_seconds=args.poll_interval_seconds,
    )

    print("=" * 50)
    print(f"Foreground saved: {fg_path}")
    if bg_path is not None:
        print(f"Background saved: {bg_path}")
    print(f"Time taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
