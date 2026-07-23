# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camb AI Transcription API wrappers for diarization generation.

Provides the HTTP submit / poll / fetch / stats primitives used by both
the CLI in ``scripts/camb/diarize.py`` and the batch-processing client
in ``client/batch_processing/diarization.py``. Lives under ``src/`` so
that ``scripts/`` and ``client/`` consumers depend on a stable library
rather than on each other.
"""

import os
import time

import requests

from common.base_utils import logger

CAMB_API_BASE_URL = "https://client.camb.ai/apis"
DEFAULT_POLL_INTERVAL = 10
DEFAULT_MAX_ATTEMPTS = 120


def submit_transcription(
    file_path: str,
    language_id: int,
    headers: dict[str, str],
) -> str:
    """Submit a transcription request to Camb AI.

    Args:
        file_path (str): Path to the audio file to transcribe.
        language_id (int): Camb AI numeric language ID.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        str: Task ID for polling.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.
        RuntimeError: If the response does not contain a task_id.

    Examples:
        >>> submit_transcription("audio.wav", 1, {"x-api-key": "k"})
        'task_123'
    """
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{CAMB_API_BASE_URL}/transcribe",
            headers=headers,
            files={"media_file": (os.path.basename(file_path), f)},
            data={"language": str(language_id)},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Camb AI /transcribe response missing task_id: {data}")
    return str(task_id)


def wait_for_transcription(
    task_id: str,
    headers: dict[str, str],
) -> int:
    """Poll Camb AI transcription status until SUCCESS.

    Args:
        task_id (str): Task ID from ``submit_transcription``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        int: Run ID when task reaches SUCCESS.

    Raises:
        requests.HTTPError: If the status endpoint returns a non-2xx response.
        RuntimeError: If Camb AI returns a terminal error status.
        TimeoutError: If polling exceeds max attempts.

    Examples:
        >>> wait_for_transcription("task_123", {"x-api-key": "k"})
        42
    """
    for attempt in range(DEFAULT_MAX_ATTEMPTS):
        try:
            response = requests.get(
                f"{CAMB_API_BASE_URL}/transcribe/{task_id}",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            # A single network blip should not abort the whole poll; retry
            # until attempts are exhausted, then surface a timeout.
            logger.warning(
                f"Camb AI transcription poll attempt {attempt + 1} failed with "
                f"transient error: {exc}"
            )
            if attempt < DEFAULT_MAX_ATTEMPTS - 1:
                time.sleep(DEFAULT_POLL_INTERVAL)
                continue
            raise TimeoutError(
                f"Camb AI transcription timed out after {DEFAULT_MAX_ATTEMPTS} attempts "
                f"(interval={DEFAULT_POLL_INTERVAL}s)."
            ) from exc
        status = str(data.get("status", "")).upper()
        logger.info(f"Camb AI transcription poll attempt {attempt + 1}: status={status}")

        if status == "SUCCESS":
            run_id = data.get("run_id")
            if not isinstance(run_id, int):
                raise RuntimeError(f"Camb AI status missing run_id on SUCCESS: {data}")
            return run_id
        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
            message = data.get("message")
            raise RuntimeError(f"Camb AI transcription failed: status={status}, message={message}")

        # Skip the sleep on the final iteration so TimeoutError is raised
        # immediately rather than after one extra DEFAULT_POLL_INTERVAL wait.
        if attempt < DEFAULT_MAX_ATTEMPTS - 1:
            time.sleep(DEFAULT_POLL_INTERVAL)

    raise TimeoutError(
        f"Camb AI transcription timed out after {DEFAULT_MAX_ATTEMPTS} attempts "
        f"(interval={DEFAULT_POLL_INTERVAL}s)."
    )


def get_transcription_result(
    run_id: int,
    headers: dict[str, str],
) -> list[dict]:
    """Fetch the transcription result with word-level timestamps.

    The Camb AI API wraps the segment list inside
    ``{"transcript": [...]}``.  This function unwraps it and returns
    the inner list directly.

    Args:
        run_id (int): Run ID from ``wait_for_transcription``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        list[dict]: List of transcription segments with start, end, text, speaker.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.
        RuntimeError: If the response payload is not a list and does not contain
            a ``transcript`` list (i.e. the response shape is unexpected).

    Examples:
        >>> get_transcription_result(42, {"x-api-key": "k"})
        [{'start': 0.0, 'end': 1.5, 'text': 'hello', 'speaker': 'SPEAKER_0'}]
    """
    response = requests.get(
        f"{CAMB_API_BASE_URL}/transcription-result/{run_id}",
        headers=headers,
        params={"word_level_timestamps": "true"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Camb AI usually wraps segments inside {"transcript": [...]}, but has
    # been observed to return the bare list directly. Normalize to list[dict]
    # so downstream consumers (extract_diarization_stats, JSON dump) get a
    # predictable shape.
    if isinstance(data, dict):
        transcript = data.get("transcript")
        if isinstance(transcript, list):
            return transcript
        raise RuntimeError(f"Camb AI transcription payload missing list 'transcript' key: {data}")
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Unexpected Camb AI transcription payload type: {type(data).__name__}")


def extract_diarization_stats(data: list[dict]) -> tuple[int, int]:
    """Extract segment and speaker counts from Camb AI transcription result.

    Args:
        data (list[dict]): List of transcription segments.

    Returns:
        tuple[int, int]: Tuple of (segment_count, unique_speaker_count).

    Examples:
        >>> extract_diarization_stats([])
        (0, 0)
        >>> extract_diarization_stats(
        ...     [
        ...         {"start": 0, "end": 1, "text": "hi", "speaker": "Speaker 1"},
        ...         {"start": 1, "end": 2, "text": "bye", "speaker": "Speaker 2"},
        ...     ]
        ... )
        (2, 2)
    """
    speakers: set[str] = set()
    for seg in data:
        speaker = seg.get("speaker")
        if speaker is not None:
            speakers.add(str(speaker))
    return len(data), len(speakers)
