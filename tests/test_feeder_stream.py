# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FeederSource and FeederStream."""

import time
import unittest
from collections.abc import Iterator

import pytest

from common.feeder_stream import FeederSource
from common.feeder_stream import FeederStream


def _slow_iterator(items: list[int], delay: float = 0.05) -> Iterator[int]:
    """Yield items with a delay between each to simulate a slow source."""
    for item in items:
        time.sleep(delay)
        yield item


def _failing_iterator(items: list[int], fail_after: int = 3) -> Iterator[int]:
    """Yield items then raise after ``fail_after`` items."""
    for i, item in enumerate(items):
        if i >= fail_after:
            raise RuntimeError(f"Simulated failure after {fail_after} items")
        yield item


@pytest.mark.unit
class TestFeederStreamSingleSource(unittest.TestCase):
    """Basic single-source FeederStream behavior."""

    def test_single_source_yields_all_items(self) -> None:
        """All items from a single source are yielded and counted."""
        items = [10, 20, 30, 40, 50]
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="nums",
                    iterator=iter(items),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start()
        result = list(stream)

        self.assertEqual(sorted(result), sorted(items))
        self.assertEqual(stream.chunk_counts, {"nums": 5})
        self.assertEqual(stream.errors, {})

    def test_empty_source_terminates_immediately(self) -> None:
        """A stream with an empty source terminates with zero items."""
        stream: FeederStream[str] = FeederStream(
            sources=[
                FeederSource(
                    name="empty",
                    iterator=iter([]),
                    transform=str,
                ),
            ],
        )
        stream.start()
        result = list(stream)

        self.assertEqual(result, [])
        self.assertEqual(stream.chunk_counts, {"empty": 0})

    def test_transform_is_applied(self) -> None:
        """The transform callable is applied to every item."""
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="doubled",
                    iterator=iter([1, 2, 3]),
                    transform=lambda x: x * 2,
                ),
            ],
        )
        stream.start()
        result = sorted(stream)

        self.assertEqual(result, [2, 4, 6])


@pytest.mark.unit
class TestFeederStreamMultiSource(unittest.TestCase):
    """Multi-source merging behavior."""

    def test_multiple_sources_all_items_yielded(self) -> None:
        """Items from all sources are present in the output."""
        stream: FeederStream[str] = FeederStream(
            sources=[
                FeederSource(
                    name="letters",
                    iterator=iter(["a", "b", "c"]),
                    transform=lambda x: x,
                ),
                FeederSource(
                    name="digits",
                    iterator=iter(["1", "2"]),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start()
        result = sorted(stream)

        self.assertEqual(result, ["1", "2", "a", "b", "c"])
        counts = stream.chunk_counts
        self.assertEqual(counts["letters"], 3)
        self.assertEqual(counts["digits"], 2)

    def test_uneven_source_lengths(self) -> None:
        """Sources with different lengths all drain fully."""
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="short",
                    iterator=iter([1]),
                    transform=lambda x: x,
                ),
                FeederSource(
                    name="long",
                    iterator=iter([10, 20, 30, 40, 50]),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start()
        result = sorted(stream)

        self.assertEqual(result, [1, 10, 20, 30, 40, 50])
        self.assertEqual(stream.chunk_counts["short"], 1)
        self.assertEqual(stream.chunk_counts["long"], 5)

    def test_slow_source_does_not_block_fast_source(self) -> None:
        """A slow source does not prevent fast source items from arriving."""
        fast_items = list(range(10))
        slow_items = list(range(100, 103))
        timestamps: list[float] = []

        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="fast",
                    iterator=iter(fast_items),
                    transform=lambda x: x,
                ),
                FeederSource(
                    name="slow",
                    iterator=_slow_iterator(slow_items, delay=0.1),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start()
        for _item in stream:
            timestamps.append(time.monotonic())

        # All 13 items should be yielded
        self.assertEqual(stream.chunk_counts["fast"], 10)
        self.assertEqual(stream.chunk_counts["slow"], 3)

        # Fast items should arrive well before all slow items finish.
        # The first 10 timestamps (fast) should cluster early; with
        # zip_longest they'd be spread across 0.3s. Check that at least
        # 8 of the first 10 items arrive within the first 0.15s.
        t0 = timestamps[0]
        early_count = sum(1 for t in timestamps[:10] if t - t0 < 0.15)
        self.assertGreaterEqual(
            early_count,
            8,
            f"Expected fast items to arrive quickly, but only {early_count}/10 "
            f"arrived within 0.15s",
        )

    def test_all_empty_sources(self) -> None:
        """Multiple empty sources terminate cleanly."""
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(name="a", iterator=iter([]), transform=lambda x: x),
                FeederSource(name="b", iterator=iter([]), transform=lambda x: x),
                FeederSource(name="c", iterator=iter([]), transform=lambda x: x),
            ],
        )
        stream.start()
        result = list(stream)

        self.assertEqual(result, [])
        self.assertEqual(stream.chunk_counts, {"a": 0, "b": 0, "c": 0})


@pytest.mark.unit
class TestFeederStreamErrorHandling(unittest.TestCase):
    """Tests for error handling in feeder threads."""

    def test_exception_does_not_crash_other_feeders(self) -> None:
        """An exception in one feeder doesn't prevent others from completing."""
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="failing",
                    iterator=_failing_iterator(list(range(10)), fail_after=3),
                    transform=lambda x: x,
                ),
                FeederSource(
                    name="healthy",
                    iterator=iter(list(range(100, 110))),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start()
        result = list(stream)

        # Healthy source should complete fully
        self.assertEqual(stream.chunk_counts["healthy"], 10)
        # Failing source should have produced 3 items before crashing
        self.assertEqual(stream.chunk_counts["failing"], 3)
        # All 13 items should appear
        self.assertEqual(len(result), 13)
        # Error should be captured
        self.assertIn("failing", stream.errors)
        self.assertIsInstance(stream.errors["failing"], RuntimeError)
        # Healthy source should have no error
        self.assertNotIn("healthy", stream.errors)


@pytest.mark.unit
class TestFeederStreamLifecycle(unittest.TestCase):
    """Tests for start/stop lifecycle."""

    def test_stop_terminates_feeders_mid_stream(self) -> None:
        """Calling stop() terminates feeders before they finish."""

        def infinite_iterator() -> Iterator[int]:
            """Yield integers forever with small delays."""
            i = 0
            while True:
                yield i
                i += 1
                time.sleep(0.01)

        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="infinite",
                    iterator=infinite_iterator(),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start()

        # Let it run briefly
        time.sleep(0.1)
        stream.stop(timeout=2.0)

        # Should have produced some items but not an infinite amount
        count = stream.chunk_counts["infinite"]
        self.assertGreater(count, 0)
        self.assertLess(count, 1000)

        # All threads should be done
        for thread in stream._threads:
            self.assertFalse(thread.is_alive())

    def test_start_with_request_id_names_threads(self) -> None:
        """Thread names include the request_id."""
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="video",
                    iterator=iter([1]),
                    transform=lambda x: x,
                ),
            ],
        )
        stream.start(request_id="req-42")

        self.assertEqual(len(stream._threads), 1)
        self.assertEqual(stream._threads[0].name, "Feeder-req-42-video")

        # Drain to clean up
        list(stream)


@pytest.mark.unit
class TestFeederStreamBackpressure(unittest.TestCase):
    """Tests for backpressure support."""

    def test_backpressure_delay_slows_feeders(self) -> None:
        """A backpressure delay causes feeders to take longer."""
        items = list(range(5))

        start = time.monotonic()
        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="src",
                    iterator=iter(items),
                    transform=lambda x: x,
                ),
            ],
            backpressure_delay=0.05,
        )
        stream.start()
        list(stream)
        elapsed = time.monotonic() - start

        # 5 items * 0.05s delay = ~0.25s minimum
        self.assertGreaterEqual(
            elapsed,
            0.2,
            f"Expected backpressure delay, but finished in {elapsed:.3f}s",
        )

    def test_bounded_buffer_provides_backpressure(self) -> None:
        """A bounded buffer causes fast producers to block on put."""
        items = list(range(20))

        stream: FeederStream[int] = FeederStream(
            sources=[
                FeederSource(
                    name="fast",
                    iterator=iter(items),
                    transform=lambda x: x,
                ),
            ],
            max_buffer_size=2,
        )
        stream.start()

        # Consume slowly to let backpressure build
        result = []
        for item in stream:
            result.append(item)
            time.sleep(0.01)

        self.assertEqual(sorted(result), sorted(items))
        self.assertEqual(stream.chunk_counts["fast"], 20)


@pytest.mark.unit
class TestFeederSourceDefaultTransform(unittest.TestCase):
    """FeederSource forwards items unchanged when no transform is set."""

    def test_default_transform_forwards_items(self) -> None:
        source = FeederSource(name="ready", iterator=iter([1, 2, 3]))
        self.assertEqual(list(source.get()), [1, 2, 3])

    def test_explicit_transform_still_applies(self) -> None:
        source = FeederSource(name="x10", iterator=iter([1, 2]), transform=lambda x: x * 10)
        self.assertEqual(list(source.get()), [10, 20])
