# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent stream-merging primitives.

This module merges multiple input iterators into a single output stream:
each :class:`FeederSource` drains its iterator on a dedicated daemon
thread and puts transformed items into a shared
:class:`~common.buffers.Buffer`, which :class:`FeederStream` exposes as
one merged iterator.

Key Features:
- One daemon thread per source, so a slow source never stalls the others
- Per-source transforms applied before items enter the shared buffer
- Per-source chunk counts and error capture for diagnostics
- Optional bounded buffering and backpressure delay for rate-limiting
"""

import queue
import threading
import time
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Generic
from typing import TypeVar

from common.base_utils import logger
from common.buffers import BUFFER_POLL_TIMEOUT
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class FeederSource(Generic[InT, OutT]):
    """A single input stream to be merged by :class:`FeederStream`.

    Each source owns its iterator and its transform logic. The
    :meth:`get` method yields transformed output items one at a
    time, keeping the transform responsibility inside the source
    and the merge plumbing inside :class:`FeederStream`.

    Type Parameters:
        InT: The type of items produced by the source iterator.
        OutT: The type of items placed into the shared buffer.

    Examples:
        >>> source = FeederSource(
        ...     name="video",
        ...     iterator=iter([1, 2, 3]),
        ...     transform=lambda x: x * 10,
        ... )
        >>> next(source.get())
        10
    """

    def __init__(
        self,
        name: str,
        iterator: Iterator[InT],
        transform: Callable[[InT], OutT] | None = None,
    ) -> None:
        """Create a feeder source.

        Args:
            name: Human-readable label used in thread names and
                chunk count reporting (e.g. ``"video"``, ``"audio"``).
            iterator: The upstream iterator to drain. May block on
                ``__next__``; that blocking is isolated to the
                feeder thread.
            transform: Callable that converts an ``InT`` item into
                an ``OutT`` item for the shared buffer. ``None``
                (default) forwards items unchanged, for iterators
                that already yield buffer-ready items.
        """
        self.name = name
        self._iterator = iterator
        self._transform = transform

    def get(self) -> Iterator[OutT]:
        """Yield transformed items from the underlying iterator.

        Each call to the upstream iterator's ``__next__`` is
        followed by the ``transform`` callable when one is set,
        producing one output item at a time.

        Yields:
            Transformed ``OutT`` items (the original items when no
            transform is configured).
        """
        for item in self._iterator:
            yield item if self._transform is None else self._transform(item)


class FeederStream(Generic[OutT]):
    """Concurrently drains multiple input iterators into a shared buffer.

    Each :class:`FeederSource` gets its own daemon thread that consumes
    its iterator, applies the source's ``transform``, and puts the
    result into a single shared :class:`~common.buffers.Buffer`. A
    :class:`~common.buffers.RequestIteratorFromBuffer` provides the
    merged output iterator, yielding items as soon as *any* source
    produces them.

    This eliminates the head-of-line blocking caused by
    :func:`itertools.zip_longest`, where a slow source stalls all
    other sources even when they have data ready.

    Type Parameters:
        OutT: The output item type placed into the shared buffer.

    Attributes:
        chunk_counts: Per-source item counts (thread-safe snapshot).
        errors: Per-source exceptions captured from failed feeders.

    Examples:
        >>> stream = FeederStream(
        ...     sources=[
        ...         FeederSource(
        ...             name="evens",
        ...             iterator=iter([2, 4, 6]),
        ...             transform=str,
        ...         ),
        ...         FeederSource(
        ...             name="odds",
        ...             iterator=iter([1, 3, 5]),
        ...             transform=str,
        ...         ),
        ...     ],
        ... )
        >>> stream.start()
        >>> sorted(stream) == ["1", "2", "3", "4", "5", "6"]
        True
    """

    def __init__(
        self,
        sources: Sequence[FeederSource[InT, OutT]],
        *,
        max_buffer_size: int | None = None,
        poll_timeout: float = BUFFER_POLL_TIMEOUT,
        backpressure_delay: float = 0.0,
    ) -> None:
        """Initialise the feeder stream.

        Args:
            sources: Sequence of :class:`FeederSource` descriptors.
                Each will be consumed on its own daemon thread once
                :meth:`start` is called.
            max_buffer_size: Maximum size for the shared buffer queue.
                ``None`` means unbounded. Use a bounded size for
                natural backpressure on fast producers.
            poll_timeout: Polling cadence for the consumer iterator.
                Defaults to :data:`~common.buffers.BUFFER_POLL_TIMEOUT`.
            backpressure_delay: Seconds to sleep after each ``put()``
                in every feeder thread. Provides rate-limiting when
                downstream gRPC servers cannot absorb data as fast
                as it is produced. ``0.0`` disables the delay.
        """
        self._sources = list(sources)
        self._backpressure_delay = backpressure_delay
        self._stop_event = threading.Event()

        self._buffer: Buffer[OutT] = Buffer(
            num_queues=1,
            max_size=max_buffer_size,
        )
        self._consumer = RequestIteratorFromBuffer(
            buffer=self._buffer,
            consumer_id=0,
            poll_timeout=poll_timeout,
        )

        # Thread bookkeeping
        self._threads: list[threading.Thread] = []
        self._active_count = len(self._sources)
        self._active_lock = threading.Lock()

        # With no sources, no feeder thread will ever mark the buffer done,
        # so the consumer iterator would poll forever. Terminate it now.
        if self._active_count == 0:
            self._buffer.done = True

        # Per-source metrics
        self._chunk_counts: dict[str, int] = {s.name: 0 for s in self._sources}
        self._errors: dict[str, BaseException] = {}
        self._metrics_lock = threading.Lock()

        logger.debug(
            f"FeederStream initialized: "
            f"sources={[s.name for s in self._sources]}, "
            f"max_buffer_size={max_buffer_size}, "
            f"backpressure_delay={backpressure_delay}"
        )

    # -- public properties -----------------------------------------------

    @property
    def chunk_counts(self) -> dict[str, int]:
        """Return a snapshot of per-source chunk counts.

        Returns:
            Dictionary mapping source name to the number of items
            that source has put into the shared buffer so far.
        """
        with self._metrics_lock:
            return dict(self._chunk_counts)

    @property
    def errors(self) -> dict[str, BaseException]:
        """Return per-source exceptions from failed feeders.

        Returns:
            Dictionary mapping source name to the exception raised
            during iteration. Empty if all feeders succeeded.
        """
        with self._metrics_lock:
            return dict(self._errors)

    def raise_on_error(self) -> None:
        """Raise the first captured feeder exception, if any.

        Re-raises the exception from the first failed feeder so
        callers can propagate it upstream (e.g. into a gRPC
        ``context.abort``).

        Raises:
            BaseException: The first feeder error encountered.
        """
        errors = self.errors
        if errors:
            first_name = next(iter(errors))
            raise errors[first_name]

    # -- lifecycle -------------------------------------------------------

    def start(self, request_id: str = "") -> None:
        """Launch one daemon thread per source.

        Args:
            request_id: Optional correlation id embedded in thread
                names for log tracing.
        """
        for source in self._sources:
            thread = threading.Thread(
                target=self._feeder_worker,
                args=(source,),
                daemon=True,
                name=f"Feeder-{request_id}-{source.name}",
            )
            thread.start()
            self._threads.append(thread)
        logger.debug(f"FeederStream started {len(self._threads)} feeder threads")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal all feeder threads to stop and wait for them.

        Args:
            timeout: Maximum seconds to wait for each thread to
                finish. Defaults to ``5.0``.
        """
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(f"Feeder thread {thread.name} did not stop within {timeout}s")
        logger.debug("FeederStream stopped")

    # -- iterator protocol -----------------------------------------------

    def __iter__(self) -> Iterator[OutT]:
        """Return self as the iterator.

        Returns:
            This ``FeederStream`` instance.
        """
        return self

    def __next__(self) -> OutT:
        """Return the next merged item from any source.

        Returns:
            The next item from the shared buffer.

        Raises:
            StopIteration: When all sources are exhausted.
        """
        return next(self._consumer)

    # -- internal --------------------------------------------------------

    def _feeder_worker(self, source: FeederSource[InT, OutT]) -> None:
        """Drain one source into the shared buffer.

        Runs on a dedicated daemon thread. Calls
        :meth:`FeederSource.get` to obtain transformed items and
        puts each into the shared buffer. Decrements
        ``_active_count`` on exit; the last thread to finish marks
        the buffer as done.

        Args:
            source: The :class:`FeederSource` to drain.
        """
        try:
            for out_item in source.get():
                if self._stop_event.is_set():
                    logger.debug(f"Feeder {source.name}: stop event set, exiting")
                    break

                # A bounded buffer's put() blocks when full; poll with a
                # timeout so a stop request is observed promptly instead of
                # hanging until downstream frees space.
                while not self._stop_event.is_set():
                    try:
                        self._buffer.put(item=out_item, timeout=BUFFER_POLL_TIMEOUT)
                        break
                    except queue.Full:
                        continue
                if self._stop_event.is_set():
                    logger.debug(f"Feeder {source.name}: stop event set during put, exiting")
                    break

                with self._metrics_lock:
                    self._chunk_counts[source.name] += 1

                if self._backpressure_delay > 0:
                    time.sleep(self._backpressure_delay)

        except Exception as exc:
            logger.error(f"Feeder {source.name} failed: {exc}", exc_info=True)
            with self._metrics_lock:
                self._errors[source.name] = exc
        finally:
            with self._active_lock:
                self._active_count -= 1
                last_feeder = self._active_count == 0

            if last_feeder:
                self._buffer.done = True
                logger.debug("FeederStream: all feeders finished")

            logger.debug(
                f"Feeder {source.name} exited: chunks={self._chunk_counts.get(source.name, 0)}"
            )
