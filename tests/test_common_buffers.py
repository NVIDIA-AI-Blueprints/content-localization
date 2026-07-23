# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the common buffers module."""

import queue
import threading
import unittest

import pytest

from common.buffers import Buffer

pytestmark = pytest.mark.unit


class TestBuffer(unittest.TestCase):
    """Unit tests covering Buffer producer/consumer behavior."""

    def test_single_queue_roundtrip(self) -> None:
        """Items put into a single queue are received in order."""
        buf: Buffer[int] = Buffer()
        items = list(range(5))

        for item in items:
            buf.put(item)

        received = [buf.get() for _ in items]

        self.assertEqual(received, items)
        self.assertTrue(buf.empty())

    def test_multi_queue_put_creates_copies(self) -> None:
        """Multi-queue put uses copy_func to duplicate items."""
        copy_calls: list[int] = []

        def copy_func(item: dict[str, int]) -> dict[str, int]:
            copy_calls.append(1)
            return {"value": item["value"]}

        buf: Buffer[dict[str, int]] = Buffer(num_queues=2, copy_func=copy_func)
        buf.put({"value": 7})

        first = buf.get(0)
        second = buf.get(1)

        self.assertEqual(first["value"], 7)
        self.assertEqual(second["value"], 7)
        self.assertIsNot(first, second)
        self.assertEqual(len(copy_calls), 1)

    def test_consumer_bounds_check(self) -> None:
        """Invalid consumer id raises IndexError."""
        buf: Buffer[int] = Buffer()
        with self.assertRaises(IndexError):
            buf.get(consumer_id=1, timeout=0.01)

    def test_full_and_qsize_for_bounded_queue(self) -> None:
        """Bounded queue reports full and correct size."""
        buf: Buffer[str] = Buffer(max_size=1)
        buf.put("a")

        self.assertTrue(buf.full())
        self.assertEqual(buf.qsize(), 1)

        with self.assertRaises(queue.Full):
            buf.put("b", timeout=0.01)

    def test_done_and_is_exhausted(self) -> None:
        """is_exhausted reflects done + empty state."""
        buf: Buffer[int] = Buffer()
        self.assertFalse(buf.is_exhausted())
        buf.done = True
        self.assertTrue(buf.is_exhausted())

    def test_bounded_multi_queue_partial_fanout_on_full(self) -> None:
        """A full queue interrupts fan-out after earlier queues received the item.

        Pins the documented non-atomic ``put`` semantics: with multiple
        bounded queues, queues before the full one keep the item while the
        full queue (and any after it) never receive it.
        """
        buf: Buffer[str] = Buffer(num_queues=2, max_size=1)

        # Fill both queues, then drain only queue 0 so queue 1 stays full.
        buf.put("seed")
        self.assertEqual(buf.get(0), "seed")
        self.assertTrue(buf.full(1))

        with self.assertRaises(queue.Full):
            buf.put("next", timeout=0.01)

        # Queue 0 received "next" before the fan-out hit the full queue 1.
        self.assertEqual(buf.qsize(0), 1)
        self.assertEqual(buf.get(0), "next")

        # Queue 1 still holds only the original item; "next" was never
        # delivered, so a retry would duplicate "next" on queue 0 only.
        self.assertEqual(buf.qsize(1), 1)
        self.assertEqual(buf.get(1), "seed")
        self.assertTrue(buf.empty(1))

    def test_multi_producer_stress_counts_and_copy_isolation(self) -> None:
        """Concurrent producers fan out every item to each queue with copies.

        Several producer threads put into one multi-queue buffer while a
        consumer drains each queue. Every queue must receive every produced
        item exactly once, and queue 1 must receive copies (never the
        producers' original objects, which go to queue 0).
        """
        buf: Buffer[dict[str, int]] = Buffer(num_queues=2)
        num_producers = 4
        items_per_producer = 50
        total_items = num_producers * items_per_producer
        produced_ids: set[int] = set()
        produced_lock = threading.Lock()
        consumed: dict[int, list[dict[str, int]]] = {0: [], 1: []}

        def producer(producer_id: int) -> None:
            for value in range(items_per_producer):
                item = {"producer": producer_id, "value": value}
                with produced_lock:
                    produced_ids.add(id(item))
                buf.put(item)

        def consumer(consumer_id: int) -> None:
            while True:
                try:
                    consumed[consumer_id].append(buf.get(consumer_id, timeout=0.05))
                except queue.Empty:
                    if buf.is_exhausted(consumer_id):
                        break

        producers = [
            threading.Thread(target=producer, args=(producer_id,))
            for producer_id in range(num_producers)
        ]
        consumers = [
            threading.Thread(target=consumer, args=(consumer_id,)) for consumer_id in (0, 1)
        ]

        # Bounded joins keep a deadlock regression from hanging the test run.
        for thread in producers + consumers:
            thread.start()
        for thread in producers:
            thread.join(timeout=30.0)
            self.assertFalse(thread.is_alive())
        # Only signal done after every producer finished, otherwise consumers
        # could observe done + momentarily-empty queues and exit early.
        buf.done = True
        for thread in consumers:
            thread.join(timeout=30.0)
            self.assertFalse(thread.is_alive())

        # Each queue received every produced item exactly once.
        expected = sorted(
            (producer_id, value)
            for producer_id in range(num_producers)
            for value in range(items_per_producer)
        )
        for consumer_id in (0, 1):
            self.assertEqual(len(consumed[consumer_id]), total_items)
            received = sorted((item["producer"], item["value"]) for item in consumed[consumer_id])
            self.assertEqual(received, expected)

        # Copy isolation: queue 0 got the producers' originals, queue 1 got
        # copies created by copy_func — never a shared object.
        self.assertTrue(all(id(item) in produced_ids for item in consumed[0]))
        self.assertFalse(any(id(item) in produced_ids for item in consumed[1]))

    def test_multithreaded_producer_consumer(self) -> None:
        """Concurrent producers/consumers preserve ordering and duplication."""
        buf: Buffer[dict[str, int]] = Buffer(num_queues=2)
        total_items = 200
        produced = [{"value": i} for i in range(total_items)]
        consumed: dict[int, list[dict[str, int]]] = {0: [], 1: []}

        def producer() -> None:
            for item in produced:
                buf.put(item)
            buf.done = True

        def consumer(consumer_id: int) -> None:
            while True:
                try:
                    value = buf.get(consumer_id, timeout=0.05)
                    consumed[consumer_id].append(value)
                except queue.Empty:
                    if buf.is_exhausted(consumer_id):
                        break

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer, args=(0,)),
            threading.Thread(target=consumer, args=(1,)),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(consumed[0], produced)
        self.assertEqual(consumed[1], produced)
        for idx in range(total_items):
            self.assertIsNot(consumed[0][idx], consumed[1][idx])
        self.assertTrue(buf.empty(0))
        self.assertTrue(buf.empty(1))

    def test_put_and_get_counts(self) -> None:
        """put_count counts items; get_count counts gets across all queues."""
        buf: Buffer[str] = Buffer(num_queues=2)
        self.assertEqual(buf.put_count, 0)
        self.assertEqual(buf.get_count, 0)

        for value in ("a", "b", "c"):
            buf.put(value)
        self.assertEqual(buf.put_count, 3)

        for consumer_id in (0, 1):
            for _ in range(3):
                buf.get(consumer_id)
        # A fully consumed multi-queue buffer sees one get per queue per item.
        self.assertEqual(buf.get_count, buf.put_count * buf.num_queues)

    def test_counts_survive_draining(self) -> None:
        """Counters record arrivals and consumption after queues are empty."""
        buf: Buffer[int] = Buffer(num_queues=1)
        buf.put(1)
        buf.put(2)
        _ = buf.get(0)
        _ = buf.get(0)

        self.assertTrue(buf.empty(0))
        self.assertEqual(buf.put_count, 2)
        self.assertEqual(buf.get_count, 2)
