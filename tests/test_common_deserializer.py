# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the abstract Deserializer base class."""

import time
import unittest

import pytest

from common.buffers import Buffer
from common.deserializer import Deserializer

pytestmark = pytest.mark.unit


class _IntDeserializer(Deserializer[int]):
    """Concrete deserializer for testing that routes ints into a buffer."""

    def __init__(self, request_iterator):
        super().__init__(request_iterator)
        self.buf: Buffer[int] = Buffer(num_queues=1)
        self.distribute_calls: list[int] = []

    def _distribute(self, request: int) -> None:
        self.distribute_calls.append(request)
        self.buf.put(request)

    def _on_complete(self) -> None:
        self.buf.done = True


class TestDeserializerLifecycle(unittest.TestCase):
    """Tests for start/stop/join and the background thread."""

    def test_consumes_all_items_and_marks_done(self) -> None:
        """All items from the iterator are distributed and buffer is marked done."""
        items = list(range(5))
        ds = _IntDeserializer(iter(items))
        ds.start(request_id="test-1")
        ds.join(timeout=5.0)

        self.assertEqual(ds.distribute_calls, items)
        self.assertTrue(ds.buf.done)

    def test_empty_iterator(self) -> None:
        """An empty iterator still triggers _on_complete."""
        ds = _IntDeserializer(iter([]))
        ds.start(request_id="test-empty")
        ds.join(timeout=5.0)

        self.assertEqual(ds.distribute_calls, [])
        self.assertTrue(ds.buf.done)

    def test_stop_mid_stream(self) -> None:
        """stop() interrupts an in-progress iterator."""

        def slow_iterator():
            for i in range(1000):
                time.sleep(0.01)
                yield i

        ds = _IntDeserializer(slow_iterator())
        ds.start(request_id="test-stop")
        time.sleep(0.05)  # Let a few items through
        ds.stop(timeout=2.0)

        # Some items should have been processed, but not all 1000
        self.assertTrue(len(ds.distribute_calls) < 1000)
        # _on_complete should still have fired
        self.assertTrue(ds.buf.done)

    def test_on_complete_fires_on_exception(self) -> None:
        """_on_complete fires even when the iterator raises."""

        def bad_iterator():
            yield 1
            raise RuntimeError("boom")

        ds = _IntDeserializer(bad_iterator())
        ds.start(request_id="test-exc")
        ds.join(timeout=5.0)

        # At least the first item was processed
        self.assertIn(1, ds.distribute_calls)
        self.assertTrue(ds.buf.done)

    def test_thread_is_daemon(self) -> None:
        """The background thread is a daemon thread."""
        ds = _IntDeserializer(iter([1]))
        ds.start(request_id="test-daemon")
        self.assertTrue(ds._thread.daemon)
        ds.join(timeout=5.0)

    def test_thread_name_contains_request_id(self) -> None:
        """The thread name includes the request_id."""
        ds = _IntDeserializer(iter([]))
        ds.start(request_id="my-req")
        self.assertIn("my-req", ds._thread.name)
        ds.join(timeout=5.0)
