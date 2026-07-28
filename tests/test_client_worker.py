# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ClientWorker error-propagating thread wrapper."""

import unittest

import pytest

from client.common.worker import ClientWorker
from common.context import LocalContext
from common.context import LocalContextAbortError

pytestmark = pytest.mark.unit


class TestClientWorker(unittest.TestCase):
    """Unit tests for ClientWorker."""

    def test_successful_target_joins_cleanly(self) -> None:
        """A worker whose target succeeds joins without raising."""
        ran: list[bool] = []

        worker = ClientWorker(target=lambda: ran.append(True), name="ok-worker")
        worker.start()
        worker.join_and_raise(timeout=5.0)

        self.assertEqual(ran, [True])
        self.assertFalse(worker.thread.is_alive())

    def test_failing_target_reraises_on_main_thread(self) -> None:
        """The worker's exception re-raises in join_and_raise on the caller thread."""

        def _fail() -> None:
            raise ValueError("worker exploded")

        worker = ClientWorker(target=_fail, name="fail-worker")
        worker.start()

        with self.assertRaises(ValueError) as ctx:
            worker.join_and_raise(timeout=5.0)
        self.assertIn("worker exploded", str(ctx.exception))

    def test_local_context_abort_reraises_on_main_thread(self) -> None:
        """A LocalContext abort inside the worker surfaces to the calling thread.

        The shared Client aborts its LocalContext on failure, which raises
        inside the worker thread; the wrapper re-raises it on the main thread
        so the CLI process exit code reflects the failure.
        """

        def _abort() -> None:
            LocalContext().abort("INTERNAL", "NIM failed mid-stream")

        worker = ClientWorker(target=_abort, name="abort-worker")
        worker.start()

        with self.assertRaises(LocalContextAbortError) as ctx:
            worker.join_and_raise(timeout=5.0)
        self.assertIn("NIM failed mid-stream", str(ctx.exception))
