# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe buffer implementations for producer-consumer patterns.

This module provides buffer classes for use in multi-threaded
environments where producers and consumers run on different threads.

Key Features:
- Thread-safe put/get operations
- Support for multiple consumer queues with copy-on-put semantics
- Done signaling for clean producer-consumer coordination
- Configurable timeouts for non-blocking operations
"""

import os
import queue
import threading
from collections.abc import Callable
from collections.abc import Iterator
from copy import deepcopy
from typing import Generic
from typing import TypeVar

from common.base_utils import logger

T = TypeVar("T")
ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT")

# Default polling cadence for RequestIteratorFromBuffer. Controls how
# often the iterator checks for new items or buffer exhaustion.
BUFFER_POLL_TIMEOUT: float = float(os.environ.get("BUFFER_POLL_TIMEOUT", "0.1"))


class Buffer(Generic[T]):
    """Thread-safe producer-consumer buffer with multi-queue fan-out.

    This class provides a thread-safe buffer implementation that supports:
    - Single producer putting items into the buffer
    - Multiple consumers, each with their own queue receiving copies of items
    - Done signaling to indicate producer has finished
    - Configurable buffer sizes and timeouts

    The buffer uses copy-on-put semantics when multiple queues are configured:
    queue 0 receives the producer's original object and every other queue
    receives its own copy created via the provided copy function. Producers
    must therefore not mutate an item after ``put()`` — the mutation would be
    visible to consumer 0 only.

    Type Parameters:
        T: The type of items stored in the buffer.

    Attributes:
        done: Property indicating whether the producer has finished.

    Example:
        >>> from copy import deepcopy
        >>> buffer = Buffer(
        ...     num_queues=2,
        ...     copy_func=deepcopy,
        ... )
        >>> buffer.put("some_item")  # Original to queue 0, copy to queue 1
        >>> item_asd = buffer.get(0)
        >>> item_lipsync = buffer.get(1)
        >>> buffer.done = True  # Signal producer finished
    """

    def __init__(
        self,
        num_queues: int = 1,
        max_size: int | None = None,
        copy_func: Callable[..., T] = deepcopy,
    ) -> None:
        """Initialize the buffer with the specified configuration.

        Args:
            num_queues: Number of consumer queues to create. Each consumer
                gets their own queue. Defaults to 1.
            max_size: Maximum size for each queue. If None, queues are
                unbounded. Defaults to None.
            copy_func (Callable[..., T]): Function to create copies of items
                for multi-queue scenarios. Required if num_queues > 1.
                Should perform deep copy if items are mutable.
                Defaults to deepcopy.

        Raises:
            ValueError: If num_queues < 1.
        """
        if num_queues < 1:
            raise ValueError("num_queues must be at least 1")

        self._copy_func = copy_func
        self._max_size = max_size

        # Initialize queues - use maxsize=0 for unbounded
        queue_maxsize = max_size if max_size is not None else 0
        self._queues: list[queue.Queue[T]] = [
            queue.Queue(maxsize=queue_maxsize) for _ in range(num_queues)
        ]

        # Done state with lock for thread-safe access
        self._done = False
        self._done_lock = threading.Lock()

        # Item counters guarded by a dedicated stats lock so producers and
        # consumers can report throughput without contending on the done flag.
        self._stats_lock = threading.Lock()
        self._put_count = 0
        self._get_count = 0
        logger.debug(f"Buffer initialized: num_queues={num_queues}, max_size={max_size}")

    @property
    def done(self) -> bool:
        """Check if the producer has signaled completion.

        Thread-safe property that indicates whether the producer has finished
        putting items into the buffer.

        Returns:
            True if producer has signaled done, False otherwise.
        """
        with self._done_lock:
            return self._done

    @done.setter
    def done(self, value: bool) -> None:
        """Set the done state to signal producer completion.

        Thread-safe setter that allows the producer to signal that it has
        finished putting items into the buffer. Consumers can check this
        state to know when to stop waiting for new items.

        Args:
            value: True to signal producer is done, False to reset.

        """
        with self._done_lock:
            self._done = value

    @property
    def num_queues(self) -> int:
        """Return the number of consumer queues."""
        return len(self._queues)

    @property
    def put_count(self) -> int:
        """Number of items successfully put into the buffer.

        Counts ``put()`` calls that delivered to every queue — items, not
        per-queue fan-out copies. Records arrivals independently of
        consumption, so it remains meaningful after consumers have drained
        the queues.

        Returns:
            int: Total items accepted by ``put()``.

        Examples:
            >>> buffer = Buffer(num_queues=2)
            >>> buffer.put("item")
            >>> buffer.put_count
            1
        """
        with self._stats_lock:
            return self._put_count

    @property
    def get_count(self) -> int:
        """Number of items successfully returned by ``get()``.

        Counts across all consumer queues, so a fully consumed multi-queue
        buffer reports one get per queue per item
        (``put_count * num_queues``).

        Returns:
            int: Total items returned by ``get()``.

        Examples:
            >>> buffer = Buffer(num_queues=1)
            >>> buffer.put("item")
            >>> _ = buffer.get(0)
            >>> buffer.get_count
            1
        """
        with self._stats_lock:
            return self._get_count

    def _validate_consumer(self, consumer_id: int) -> None:
        """Validate that consumer_id refers to an existing queue."""
        if not 0 <= consumer_id < len(self._queues):
            raise IndexError(f"Consumer ID {consumer_id} out of range (0-{len(self._queues) - 1})")

    def put(self, item: T, timeout: float | None = None) -> None:
        """Put an item into the buffer for all consumers.

        When multiple queues are configured, queue 0 receives the original
        item and every other queue receives a copy created via ``copy_func``.
        Callers must not mutate the item after ``put()`` — consumer 0 shares
        the caller's object.

        Args:
            item: The item to put into the buffer.
            timeout: Maximum time to wait if queue is full. None means wait
                indefinitely. Defaults to None.

        Raises:
            queue.Full: If timeout expires before item can be added. With
                multiple bounded queues the fan-out is not atomic: queues
                before the full one have already received the item, so a
                retry would deliver duplicates to those queues.
        """
        if len(self._queues) == 1:
            if timeout is None:
                self._queues[0].put(item)
            else:
                self._queues[0].put(item, timeout=timeout)
            with self._stats_lock:
                self._put_count += 1
            return

        for idx, _queue in enumerate(self._queues):
            copy_item = item if idx == 0 else self._copy_func(item)
            if timeout is None:
                _queue.put(copy_item)
            else:
                _queue.put(copy_item, timeout=timeout)
        with self._stats_lock:
            self._put_count += 1

    def get(self, consumer_id: int = 0, timeout: float | None = None) -> T:
        """Get an item from the specified consumer's queue.

        Args:
            consumer_id: The ID of the consumer queue to get from. Defaults to 0.
            timeout: Maximum time to wait if queue is empty. None means wait
                indefinitely. Defaults to None.

        Returns:
            The next item from the consumer's queue.

        Raises:
            IndexError: If consumer_id is not valid.
            queue.Empty: If timeout expires before an item is available.
        """
        self._validate_consumer(consumer_id)
        consumer_queue = self._queues[consumer_id]
        if timeout is None:
            item = consumer_queue.get()
        else:
            item = consumer_queue.get(timeout=timeout)
        with self._stats_lock:
            self._get_count += 1
        return item

    def qsize(self, consumer_id: int = 0) -> int:
        """Return the approximate size of the specified consumer's queue.

        Args:
            consumer_id: The ID of the consumer queue. Defaults to 0.

        Returns:
            Approximate number of items in the queue.

        Raises:
            IndexError: If consumer_id is not valid.
        """
        self._validate_consumer(consumer_id)
        return self._queues[consumer_id].qsize()

    def empty(self, consumer_id: int = 0) -> bool:
        """Check if the specified consumer's queue is empty.

        Args:
            consumer_id: The ID of the consumer queue. Defaults to 0.

        Returns:
            True if queue is empty, False otherwise.

        Raises:
            IndexError: If consumer_id is not valid.
        """
        self._validate_consumer(consumer_id)
        return self._queues[consumer_id].empty()

    def full(self, consumer_id: int = 0) -> bool:
        """Check if the specified consumer's queue is full.

        Only meaningful for bounded queues (max_size is set).

        Args:
            consumer_id: The ID of the consumer queue. Defaults to 0.

        Returns:
            True if queue is full, False otherwise.

        Raises:
            IndexError: If consumer_id is not valid.
        """
        self._validate_consumer(consumer_id)
        return self._queues[consumer_id].full()

    def is_exhausted(self, consumer_id: int = 0) -> bool:
        """Check if buffer is exhausted for a specific consumer.

        Buffer is exhausted when producer is done AND the consumer's queue
        is empty.

        Args:
            consumer_id: The ID of the consumer queue. Defaults to 0.

        Returns:
            True if done and queue is empty, False otherwise.

        Raises:
            IndexError: If consumer_id is not valid.
        """
        exhausted = self.done and self.empty(consumer_id)
        if exhausted:
            logger.debug(f"Buffer exhausted for consumer_id={consumer_id}")
        return exhausted


class RequestIteratorFromBuffer(Iterator[ReqT]):
    """Iterator that drains a ``Buffer`` until it is exhausted.
    Waits for items to be available in the buffer if it is still pending but empty.
    """

    def __init__(
        self,
        buffer: Buffer[ReqT],
        consumer_id: int = 0,
        poll_timeout: float = BUFFER_POLL_TIMEOUT,
    ) -> None:
        """Create an iterator backed by a buffer consumer queue.

        Args:
            buffer: Buffer to read requests from.
            consumer_id: Which consumer queue to pull from. Defaults to 0.
            poll_timeout: Timeout passed to ``Buffer.get`` for polling
                cadence. Defaults to :data:`BUFFER_POLL_TIMEOUT`
                (env ``BUFFER_POLL_TIMEOUT``, default ``0.1``).
        """
        self._buffer = buffer
        self._consumer_id = consumer_id
        self._poll_timeout = poll_timeout
        self._generator = self._make_generator()
        logger.debug(
            f"RequestIteratorFromBuffer initialized: consumer_id={consumer_id}, "
            f"poll_timeout={poll_timeout}"
        )

    def __iter__(self) -> Iterator[ReqT]:
        """Return self as the iterator."""
        return self

    def __next__(self) -> ReqT:
        """Return next item from the underlying buffer generator."""
        return next(self._generator)

    def _make_generator(self) -> Iterator[ReqT]:
        while True:
            try:
                yield self._buffer.get(consumer_id=self._consumer_id, timeout=self._poll_timeout)
            except queue.Empty:
                if self._buffer.is_exhausted(self._consumer_id):
                    break
                continue
