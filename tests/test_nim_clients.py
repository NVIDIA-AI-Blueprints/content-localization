#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIM client classes."""

import unittest
from collections.abc import Iterator
from unittest.mock import MagicMock

import grpc
import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionResult,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import ActiveSpeakerDetectionClient
from common.nims import LipsyncClient
from common.nims import SpeechToSpeechClient

pytestmark = pytest.mark.unit


class _FakeHandle:
    def __init__(self, responses: list[object]) -> None:
        self.stub = None
        self._responses = responses
        self.connect_called = 0

    def is_healthy(self) -> bool:
        return True

    def connect(self) -> None:
        self.stub = object()
        self.connect_called += 1

    def get_response_iterator(self, request_iterator: Iterator[object]) -> Iterator[object]:
        _ = list(request_iterator)
        return iter(self._responses)


class _MidStreamFailureHandle(_FakeHandle):
    """Handle whose response stream raises after yielding its responses."""

    def get_response_iterator(self, request_iterator: Iterator[object]) -> Iterator[object]:
        _ = list(request_iterator)

        def _iter() -> Iterator[object]:
            yield from self._responses
            raise RuntimeError("NIM stream died")

        return _iter()


class _RaisingAbortContext:
    """Context stub whose abort raises, matching real gRPC semantics."""

    def __init__(self) -> None:
        self.abort_calls: list[tuple[object, str]] = []

    def abort(self, code: object, details: str) -> None:
        self.abort_calls.append((code, details))
        raise RuntimeError(f"Aborted: {code}")


class TestNimClients(unittest.TestCase):
    """Unit tests for NIM client buffer behavior."""

    def _run_client(self, client_cls: type, responses: list[object]) -> list[object]:
        handle = _FakeHandle(responses=responses)
        client = client_cls(handle)
        context = MagicMock(spec=grpc.ServicerContext)
        output_buffer: Buffer[object] = Buffer()

        def request_iter() -> Iterator[object]:
            yield object()
            yield object()

        client(
            request_iterator=request_iter(),
            output_buffer=output_buffer,
            context=context,
            request_id="r1",
        )
        self.assertEqual(handle.connect_called, 1)
        self.assertTrue(output_buffer.done)
        return list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))

    def _run_client_with_stub(self, client_cls: type, responses: list[object]) -> list[object]:
        handle = _FakeHandle(responses=responses)
        handle.stub = object()
        client = client_cls(handle)
        context = MagicMock(spec=grpc.ServicerContext)
        output_buffer: Buffer[object] = Buffer()

        def request_iter() -> Iterator[object]:
            yield object()

        client(
            request_iterator=request_iter(),
            output_buffer=output_buffer,
            context=context,
            request_id="r2",
        )
        self.assertEqual(handle.connect_called, 0)
        self.assertTrue(output_buffer.done)
        return list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))

    def test_s2s_client_streams_responses(self) -> None:
        """Speech-to-Speech client streams all responses into buffer."""
        responses = [
            SpeechToSpeechResponse(audio_data=b"chunk1"),
            SpeechToSpeechResponse(audio_data=b"chunk2"),
            SpeechToSpeechResponse(audio_data=b"chunk3"),
        ]
        buffered = self._run_client(SpeechToSpeechClient, responses)
        self.assertEqual(len(buffered), 3)

    def test_asd_client_streams_responses(self) -> None:
        """Active speaker detection client streams all responses into buffer."""
        responses = [
            DetectActiveSpeakerResponse(
                active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
            ),
            DetectActiveSpeakerResponse(
                active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=1)
            ),
        ]
        buffered = self._run_client(ActiveSpeakerDetectionClient, responses)
        self.assertEqual(len(buffered), 2)

    def test_lipsync_client_streams_responses(self) -> None:
        """LipSync client streams all responses into buffer."""
        responses = [LipsyncResponse(video_file_data=b"frame")]
        buffered = self._run_client(LipsyncClient, responses)
        self.assertEqual(len(buffered), 1)

    def test_clients_skip_connect_when_stub_exists(self) -> None:
        """Clients do not reconnect when a stub already exists."""
        responses = [
            SpeechToSpeechResponse(audio_data=b"chunk1"),
            SpeechToSpeechResponse(audio_data=b"chunk2"),
        ]
        buffered = self._run_client_with_stub(SpeechToSpeechClient, responses)
        self.assertEqual(len(buffered), 2)

    def test_mid_stream_failure_aborts_once_with_original_details(self) -> None:
        """A mid-stream NIM failure aborts the context exactly once.

        The abort details carry the originating error so callers see the root
        cause of the stream failure.
        """
        cases = [
            (SpeechToSpeechClient, SpeechToSpeechResponse(audio_data=b"chunk1")),
            (
                ActiveSpeakerDetectionClient,
                DetectActiveSpeakerResponse(
                    active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
                ),
            ),
            (LipsyncClient, LipsyncResponse(video_file_data=b"frame")),
        ]
        for client_cls, response in cases:
            with self.subTest(client=client_cls.__name__):
                handle = _MidStreamFailureHandle(responses=[response])
                client = client_cls(handle)
                context = _RaisingAbortContext()
                output_buffer: Buffer[object] = Buffer()

                with self.assertRaises(RuntimeError):
                    client(
                        request_iterator=iter([object()]),
                        output_buffer=output_buffer,
                        context=context,
                        request_id="r-fail",
                    )

                self.assertTrue(output_buffer.done)
                self.assertEqual(len(context.abort_calls), 1)
                _, details = context.abort_calls[0]
                self.assertIn("NIM stream died", details)
                self.assertNotIn("Aborted:", details)

    def test_clients_filter_keepalive_responses(self) -> None:
        """S2S/ASD clients drop keep-alive; the LipSync client passes them through."""
        s2s_keepalive = SpeechToSpeechResponse()
        s2s_keepalive.keepalive.SetInParent()
        s2s_audio = SpeechToSpeechResponse(audio_data=b"audio")
        s2s_buffered = self._run_client(SpeechToSpeechClient, [s2s_keepalive, s2s_audio])
        self.assertEqual(len(s2s_buffered), 1)
        self.assertTrue(s2s_buffered[0].HasField("audio_data"))

        asd_keepalive = DetectActiveSpeakerResponse()
        asd_keepalive.keepalive.SetInParent()
        asd_result = DetectActiveSpeakerResponse(
            active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
        )
        asd_buffered = self._run_client(ActiveSpeakerDetectionClient, [asd_keepalive, asd_result])
        self.assertEqual(len(asd_buffered), 1)
        self.assertEqual(asd_buffered[0].active_speaker_detection_result.frame_id, 0)

        # LipSync keepalives flow through so the controller can forward them
        # to its own client while LipSync waits for input.
        lipsync_keepalive = LipsyncResponse()
        lipsync_keepalive.keepalive.SetInParent()
        lipsync_video = LipsyncResponse(video_file_data=b"video")
        lipsync_buffered = self._run_client(LipsyncClient, [lipsync_keepalive, lipsync_video])
        self.assertEqual(len(lipsync_buffered), 2)
        self.assertTrue(lipsync_buffered[0].HasField("keepalive"))
        self.assertEqual(lipsync_buffered[1].video_file_data, b"video")


if __name__ == "__main__":
    unittest.main()
