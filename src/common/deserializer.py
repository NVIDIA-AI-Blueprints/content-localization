# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract deserializer for consuming an iterator on a background thread.

A ``Deserializer`` reads items from a request iterator and distributes
them into one or more :class:`~common.buffers.Buffer` instances. The
iteration happens on a dedicated daemon thread so that the caller is
never blocked by upstream I/O.

Subclasses implement two hooks:

* ``_distribute(item)`` -- route a single item to the correct buffer(s).
* ``_on_complete()``   -- mark every output buffer as *done*.

Example::

    class MyDeserializer(Deserializer[int]):
        def __init__(self, it):
            super().__init__(it)
            self.buf = Buffer(num_queues=1)

        def _distribute(self, item):
            self.buf.put(item)

        def _on_complete(self):
            self.buf.done = True


    ds = MyDeserializer(iter(range(10)))
    ds.start(request_id="demo")
    ds.join()
    assert ds.buf.done
"""

import threading
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Generic
from typing import TypeVar

from common.base_utils import logger

T = TypeVar("T")


class Deserializer(ABC, Generic[T]):
    """Consumes an iterator on a background thread, distributing items
    into one or more output :class:`~common.buffers.Buffer` instances.

    Type Parameters:
        T: The type of items consumed from the iterator.
    """

    def __init__(self, request_iterator: Iterator[T]) -> None:
        """Initialise the deserializer.

        Args:
            request_iterator: The upstream iterator to consume. May block
                on ``__next__``; that blocking is isolated to the
                background thread.
        """
        self._request_iterator = request_iterator
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- subclass hooks --------------------------------------------------

    @abstractmethod
    def _distribute(self, request: T) -> None:
        """Route a single item to the correct output buffer(s).

        Implementations should call ``buffer.put(item)`` for every
        buffer that needs the item.  Deep-copying (when required) is
        the responsibility of this method or the buffer's
        ``copy_func``.

        Args:
            request: The item to distribute.
        """

    @abstractmethod
    def _on_complete(self) -> None:
        """Signal completion on every output buffer.

        Called exactly once, after the iterator is exhausted **or**
        after an early stop.  Implementations must set
        ``buffer.done = True`` on every output buffer.
        """

    # -- thread lifecycle ------------------------------------------------

    def _worker(self) -> None:
        """Main loop executed on the background thread."""
        try:
            for request in self._request_iterator:
                if self._stop_event.is_set():
                    logger.debug("Deserializer stop event set, exiting loop")
                    break
                self._distribute(request)
        except Exception:
            logger.error("Deserializer error", exc_info=True)
        finally:
            self._on_complete()
            logger.debug("Deserializer worker finished")

    def start(self, request_id: str = "") -> None:
        """Start the background consumption thread.

        Args:
            request_id: Optional correlation id used in the thread name.
        """
        self._thread = threading.Thread(
            target=self._worker,
            name=f"Deserializer-{request_id}",
            daemon=True,
        )
        self._thread.start()
        logger.debug(f"Deserializer thread started: {self._thread.name}")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background thread to stop and wait for it.

        Args:
            timeout: Maximum seconds to wait for the thread to finish.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Deserializer thread did not stop within timeout")
        logger.debug("Deserializer stopped")

    def join(self, timeout: float | None = None) -> None:
        """Block until the background thread finishes.

        Args:
            timeout: Maximum seconds to wait.  ``None`` waits forever.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
