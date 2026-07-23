# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101,PLR2004

"""Unit tests for scripts.camb.audio_isolation."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from scripts.camb.audio_isolation import fetch_separation_urls
from scripts.camb.audio_isolation import submit_audio_separation
from scripts.camb.audio_isolation import wait_for_separation_success

pytestmark = pytest.mark.unit

HEADERS = {"x-api-key": "k"}


def test_submit_audio_separation_success(tmp_path: Path) -> None:
    """submit_audio_separation returns task_id from JSON."""
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFFfake")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"task_id": "task-abc"}
    mock_resp.raise_for_status = MagicMock()

    with patch("scripts.camb.audio_isolation.requests.post", return_value=mock_resp) as post:
        tid = submit_audio_separation(input_file_path=wav, headers=HEADERS)

    assert tid == "task-abc"
    post.assert_called_once()


def test_submit_audio_separation_missing_task_id(tmp_path: Path) -> None:
    """Missing task_id raises RuntimeError."""
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"x")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("scripts.camb.audio_isolation.requests.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="missing task_id"),
    ):
        submit_audio_separation(input_file_path=wav, headers=HEADERS)


def test_wait_for_separation_success_returns_run_id() -> None:
    """SUCCESS payload with run_id returns int."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "SUCCESS", "run_id": 42}
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("scripts.camb.audio_isolation.requests.get", return_value=mock_resp),
        patch("scripts.camb.audio_isolation.time.sleep"),
    ):
        rid = wait_for_separation_success(
            task_id="t1",
            headers=HEADERS,
            max_attempts=5,
            poll_interval_seconds=0,
        )
    assert rid == 42


def test_wait_for_separation_terminal_error() -> None:
    """ERROR status raises RuntimeError."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "ERROR", "message": "bad"}
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("scripts.camb.audio_isolation.requests.get", return_value=mock_resp),
        pytest.raises(RuntimeError, match="failed"),
    ):
        wait_for_separation_success(
            task_id="t1",
            headers=HEADERS,
            max_attempts=3,
            poll_interval_seconds=0,
        )


def test_fetch_separation_urls() -> None:
    """fetch_separation_urls returns foreground and background URLs."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "foreground_audio_url": "https://fg",
        "background_audio_url": "https://bg",
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("scripts.camb.audio_isolation.requests.get", return_value=mock_resp):
        fg, bg = fetch_separation_urls(run_id=1, headers=HEADERS)

    assert fg == "https://fg"
    assert bg == "https://bg"


def test_fetch_separation_urls_missing_foreground() -> None:
    """Missing foreground URL raises RuntimeError."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"background_audio_url": "https://bg"}
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("scripts.camb.audio_isolation.requests.get", return_value=mock_resp),
        pytest.raises(RuntimeError, match="foreground"),
    ):
        fetch_separation_urls(run_id=1, headers=HEADERS)
