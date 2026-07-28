# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background-thread helpers for standalone client applications.

The shared ``common.clients.Client`` reports failures by aborting its gRPC
context. Standalone CLI clients run the client on a background thread with a
``LocalContext`` whose ``abort`` raises inside that thread; ``ClientWorker``
captures the exception and re-raises it on the main thread so the process exit
code reflects the failure.
"""

import threading
from collections.abc import Callable

from common.base_utils import logger


class ClientWorker:
    """Daemon worker thread that propagates its failure to the calling thread.

    Wraps a zero-argument callable (typically a closure invoking a
    ``common.clients.Client``) on a daemon thread and stores any exception it
    raises, so the calling thread can re-raise it after ``join`` and exit with
    a non-zero status.

    Examples:
        >>> worker = ClientWorker(target=lambda: None, name="s2s-client")
        >>> worker.start()
        >>> worker.join_and_raise()
    """

    def __init__(self, target: Callable[[], None], name: str) -> None:
        """Initialize the worker without starting it.

        Args:
            target (Callable[[], None]): Zero-argument callable to run on the
                worker thread.
            name (str): Thread name, used in logs.

        Returns:
            None.

        Examples:
            >>> worker = ClientWorker(target=lambda: None, name="s2s-client")
        """
        self._target = target
        self._error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)

    def _run(self) -> None:
        """Run the target, storing any exception for ``join_and_raise``.

        Returns:
            None.

        Examples:
            Invoked automatically on the worker thread by ``start``; any
            exception raised by the target is stored for ``join_and_raise``.

            >>> worker = ClientWorker(target=lambda: None, name="w")
            >>> worker._run()
        """
        try:
            self._target()
        except Exception as exc:
            logger.error(f"Worker thread {self.thread.name} failed: {exc}")
            self._error = exc

    def start(self) -> None:
        """Start the worker thread.

        Returns:
            None.

        Examples:
            >>> worker = ClientWorker(target=lambda: None, name="w")
            >>> worker.start()
        """
        self.thread.start()

    def join_and_raise(self, timeout: float | None = None) -> None:
        """Join the worker, then re-raise the exception it captured, if any.

        Args:
            timeout (float | None): Maximum seconds to wait for the thread;
                ``None`` (default) waits indefinitely.

        Returns:
            None: When the worker completed without error.

        Raises:
            TimeoutError: If the worker is still running when *timeout*
                expires.
            Exception: The exception captured on the worker thread, re-raised
                on the caller's thread so CLI entry points exit non-zero.

        Examples:
            >>> worker = ClientWorker(target=lambda: None, name="w")
            >>> worker.start()
            >>> worker.join_and_raise()
        """
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError(
                f"Worker thread {self.thread.name} did not complete within {timeout}s"
            )
        if self._error is not None:
            raise self._error
