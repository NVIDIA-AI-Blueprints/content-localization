# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the common client abstractions."""

import unittest
from unittest.mock import MagicMock

import grpc
import pytest

from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.clients import Client

pytestmark = pytest.mark.unit


class _DummyClient(Client[int, int]):
    """Concrete client for testing that echoes requests into the output buffer."""

    def _impl(
        self,
        request_iterator,
        output_buffer,
        context,
        request_id,
        **kwargs,
    ) -> None:
        for item in request_iterator:
            output_buffer.put(item)


class _ErrorClient(Client[int, int]):
    """Concrete client for testing error handling."""

    def _impl(
        self,
        request_iterator,
        output_buffer,
        context,
        request_id,
        **kwargs,
    ) -> None:  # pragma: no cover - exercised indirectly
        raise RuntimeError("boom")


class _RaisingAbortContext:
    """Context stub whose abort raises, matching real gRPC semantics."""

    def __init__(self) -> None:
        self.abort_calls: list[tuple[object, str]] = []

    def abort(self, code: object, details: str) -> None:
        self.abort_calls.append((code, details))
        raise RuntimeError(f"Aborted: {code}")


class TestClient(unittest.TestCase):
    """Unit tests for the Client base class."""

    def test_call_streams_responses_into_buffer_and_marks_done(self) -> None:
        """__call__ drains requests through _impl into output buffer and sets done."""
        request_iter = iter([1, 2, 3])
        output_buffer: Buffer[int] = Buffer()
        handle = MagicMock()
        handle.is_healthy.return_value = True
        context = MagicMock(spec=grpc.ServicerContext)

        client = _DummyClient(handle)
        client(
            request_iterator=request_iter,
            output_buffer=output_buffer,
            context=context,
            request_id="req-1",
        )

        self.assertTrue(output_buffer.done)
        self.assertEqual(
            list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01)),
            [1, 2, 3],
        )
        handle.is_healthy.assert_called_once()
        context.abort.assert_not_called()

    def test_call_handles_impl_exception_and_marks_done(self) -> None:
        """Exceptions propagate through context.abort and still mark buffer done."""
        request_iter = iter([1])
        output_buffer: Buffer[int] = Buffer()
        handle = MagicMock()
        handle.is_healthy.return_value = True
        context = MagicMock(spec=grpc.ServicerContext)

        client = _ErrorClient(handle)
        client(
            request_iterator=request_iter,
            output_buffer=output_buffer,
            context=context,
            request_id="req-err",
        )

        self.assertTrue(output_buffer.done)
        context.abort.assert_called_once()

    def test_health_check_failure_marks_done_and_aborts_unavailable(self) -> None:
        """A failing health check marks the buffer done and aborts with UNAVAILABLE.

        ``output_buffer.done`` must be set on every exit path so downstream
        consumers observe stream completion.
        """
        output_buffer: Buffer[int] = Buffer()
        handle = MagicMock()
        handle.is_healthy.side_effect = ConnectionError("NIM down")
        context = MagicMock(spec=grpc.ServicerContext)

        client = _DummyClient(handle)
        client(
            request_iterator=iter([1]),
            output_buffer=output_buffer,
            context=context,
            request_id="req-unhealthy",
        )

        self.assertTrue(output_buffer.done)
        context.abort.assert_called_once()
        code, details = context.abort.call_args.args
        self.assertEqual(code, grpc.StatusCode.UNAVAILABLE)
        self.assertIn("NIM down", details)

    def test_health_check_failure_with_raising_context_still_marks_done(self) -> None:
        """With a raising abort (real gRPC semantics), done is still guaranteed."""
        output_buffer: Buffer[int] = Buffer()
        handle = MagicMock()
        handle.is_healthy.side_effect = ConnectionError("NIM down")
        context = _RaisingAbortContext()

        client = _DummyClient(handle)
        with self.assertRaises(RuntimeError):
            client(
                request_iterator=iter([1]),
                output_buffer=output_buffer,
                context=context,
                request_id="req-unhealthy",
            )

        self.assertTrue(output_buffer.done)
        self.assertEqual(len(context.abort_calls), 1)

    def test_impl_exception_aborts_once_with_original_details(self) -> None:
        """The context is aborted exactly once, with the originating error details."""
        output_buffer: Buffer[int] = Buffer()
        handle = MagicMock()
        handle.is_healthy.return_value = True
        context = _RaisingAbortContext()

        client = _ErrorClient(handle)
        with self.assertRaises(RuntimeError):
            client(
                request_iterator=iter([1]),
                output_buffer=output_buffer,
                context=context,
                request_id="req-err",
            )

        self.assertTrue(output_buffer.done)
        self.assertEqual(len(context.abort_calls), 1)
        _, details = context.abort_calls[0]
        self.assertIn("RuntimeError: boom", details)
        self.assertNotIn("Aborted:", details)

    def test_buffer_request_generator_yields_until_exhausted(self) -> None:
        """RequestIteratorFromBuffer drains a Buffer and stops when done."""
        buffer: Buffer[int] = Buffer()
        buffer.put(1)
        buffer.put(2)
        buffer.done = True

        iterator = RequestIteratorFromBuffer(buffer, poll_timeout=0.01)

        self.assertEqual(list(iterator), [1, 2])

    def test_request_iterator_returns_empty_when_done_and_no_items(self) -> None:
        """Returns immediately when buffer is done with no items."""
        buffer: Buffer[int] = Buffer()
        buffer.done = True

        iterator = RequestIteratorFromBuffer(buffer, poll_timeout=0.01)

        self.assertEqual(list(iterator), [])

    def test_request_iterator_respects_consumer_id(self) -> None:
        """Reads from the specified consumer queue in multi-queue buffers."""
        buffer: Buffer[int] = Buffer(num_queues=2)
        buffer.put(7)
        buffer.done = True

        consumer0_iter = RequestIteratorFromBuffer(buffer, consumer_id=0, poll_timeout=0.01)
        consumer1_iter = RequestIteratorFromBuffer(buffer, consumer_id=1, poll_timeout=0.01)

        self.assertEqual(list(consumer0_iter), [7])
        self.assertEqual(list(consumer1_iter), [7])
