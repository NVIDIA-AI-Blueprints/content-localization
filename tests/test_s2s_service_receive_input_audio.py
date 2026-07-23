# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import time
import wave
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from s2s_service.service import S2SService

pytestmark = pytest.mark.unit


class _DummyService(S2SService):
    """Minimal concrete implementation for testing."""

    def infer(  # type: ignore[override]
        self,
        request_iterator,
        context,
        request_id: str,
    ) -> Iterator:
        return iter(())

    def validate_audio_format(self, value: str) -> bool:  # type: ignore[override]
        return True


class _Req:
    def __init__(self, audio_data: bytes) -> None:
        self.audio_data = audio_data


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x02" * 10)
    return buf.getvalue()


def test_download_input_audio_streams_via_buffer_thread(tmp_path) -> None:
    wav_bytes = _make_wav_bytes()

    def req_iter():
        yield _Req(wav_bytes[: len(wav_bytes) // 2])
        time.sleep(0.01)
        yield _Req(wav_bytes[len(wav_bytes) // 2 :])

    svc = _DummyService()
    context = MagicMock()

    input_path = svc.download_input_audio(
        request_iterator=req_iter(),
        context=context,
        request_id="test-req",
    )

    with open(input_path, "rb") as f:
        assert f.read() == wav_bytes

    os.remove(input_path)
