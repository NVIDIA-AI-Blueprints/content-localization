# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for S2S entrypoint argument wiring."""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from s2s_service import entrypoint
from s2s_service.camb_utils.dubbing import CambDubbingService
from s2s_service.el_utils.dubbing import ELDubbingService

pytestmark = pytest.mark.unit


def _base_argv() -> list[str]:
    return [
        "s2s-entrypoint",
        "el_dubbing",
        "--service-uri",
        "speech-to-speech:50050",
        "--max-concurrency",
        "1",
        "--concurrency-mode",
        "threading",
        "--threads-per-process",
        "1",
    ]


class TestS2SEntrypoint(unittest.TestCase):
    """Regression tests for S2S startup arg wiring."""

    def _run_main(self, argv: list[str]) -> MagicMock:
        """Run entrypoint.main with a mocked service and the real parser."""
        with patch("s2s_service.entrypoint.ELDubbingService") as mock_service_cls:
            # Keep the production argument surface (vendored SSL args
            # included) while mocking the service construction itself.
            mock_service_cls.argsfactory.side_effect = ELDubbingService.argsfactory
            mock_service_cls.return_value = MagicMock()
            with patch("sys.argv", argv):
                entrypoint.main()
            return mock_service_cls.return_value

    def test_main_forwards_server_ssl_args_to_serve(self) -> None:
        """--use-ssl and --ssl_server_* args reach the service's serve()."""
        argv = [
            *_base_argv(),
            "--use-ssl",
            "--ssl_server_key_path",
            "/certs/s2s.key",
            "--ssl_server_cert_path",
            "/certs/s2s.pem",
            "--ssl_root_cert_path",
            "/certs/root.pem",
        ]
        service = self._run_main(argv=argv)

        serve_kwargs = service.serve.call_args.kwargs
        self.assertTrue(serve_kwargs["use_ssl"])
        self.assertEqual(serve_kwargs["ssl_server_key_path"], "/certs/s2s.key")
        self.assertEqual(serve_kwargs["ssl_server_cert_path"], "/certs/s2s.pem")
        self.assertEqual(serve_kwargs["ssl_root_cert_path"], "/certs/root.pem")

    def test_main_forwards_server_ssl_args_for_camb_dubbing(self) -> None:
        """The camb_dubbing branch forwards the same SSL surface."""
        argv = [
            "s2s-entrypoint",
            "camb_dubbing",
            *_base_argv()[2:],
            "--use-ssl",
            "--ssl_server_key_path",
            "/certs/s2s.key",
            "--ssl_server_cert_path",
            "/certs/s2s.pem",
        ]
        with patch("s2s_service.entrypoint.CambDubbingService") as mock_service_cls:
            mock_service_cls.argsfactory.side_effect = CambDubbingService.argsfactory
            mock_service_cls.return_value = MagicMock()
            with patch("sys.argv", argv):
                entrypoint.main()

        serve_kwargs = mock_service_cls.return_value.serve.call_args.kwargs
        self.assertTrue(serve_kwargs["use_ssl"])
        self.assertEqual(serve_kwargs["ssl_server_key_path"], "/certs/s2s.key")

    def test_main_defaults_to_plaintext_serving(self) -> None:
        """Without --use-ssl the service serves plaintext."""
        service = self._run_main(argv=_base_argv())

        serve_kwargs = service.serve.call_args.kwargs
        self.assertFalse(serve_kwargs["use_ssl"])
        self.assertIsNone(serve_kwargs["ssl_server_key_path"])
        self.assertIsNone(serve_kwargs["ssl_server_cert_path"])
        self.assertIsNone(serve_kwargs["ssl_root_cert_path"])
