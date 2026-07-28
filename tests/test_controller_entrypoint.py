# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for controller entrypoint argument wiring."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from controller_service import entrypoint
from controller_service.service import ControllerService

pytestmark = pytest.mark.unit


def _base_argv() -> list[str]:
    return [
        "controller-entrypoint",
        "--service-uri",
        "controller:50056",
        "--max-concurrency",
        "1",
        "--concurrency-mode",
        "threading",
        "--threads-per-process",
        "1",
        "--s2s-server",
        "speech-to-speech:50050",
        "--lipsync-server",
        "lipsync:50054",
    ]


@pytest.mark.unit
class TestControllerEntrypoint(unittest.TestCase):
    """Regression tests for controller startup arg wiring."""

    def test_argsfactory_does_not_expose_service_mode(self) -> None:
        """Controller args are push-only and do not require --service-mode."""
        parser = ControllerService.argsfactory()
        all_options = {
            option for action in parser._actions for option in getattr(action, "option_strings", [])
        }
        self.assertNotIn("--service-mode", all_options)

    @patch("controller_service.entrypoint.ControllerService")
    @patch("controller_service.entrypoint.GRPCServiceHandle.from_string")
    @patch("controller_service.entrypoint.SpeechToSpeechHandle")
    @patch("controller_service.entrypoint.LipsyncHandle")
    @patch("controller_service.entrypoint.ActiveSpeakerDetectionHandle")
    def test_main_wires_controller_without_service_mode(
        self,
        mock_asd_handle_ctor: MagicMock,
        mock_lipsync_handle_ctor: MagicMock,
        mock_s2s_handle_ctor: MagicMock,
        mock_from_string: MagicMock,
        mock_controller_service: MagicMock,
    ) -> None:
        """Entrypoint builds ControllerService without obsolete service_mode."""
        # from_string call order: service_uri, lipsync, s2s, asd
        # (mock parse_args returns truthy attrs, so all branches execute,
        # including the credential path — hence create_channel_credentials
        # is patched below)
        mock_from_string.side_effect = [
            SimpleNamespace(host="controller", port=50056),
            SimpleNamespace(host="lipsync", port=50054),
            SimpleNamespace(host="speech-to-speech", port=50050),
            SimpleNamespace(host="asd", port=50055),
        ]
        mock_s2s_handle_ctor.return_value = MagicMock()
        mock_lipsync_handle_ctor.return_value = MagicMock()
        mock_asd_handle_ctor.return_value = MagicMock()
        mock_controller_service.return_value = MagicMock()

        argv = _base_argv()
        with (
            patch("sys.argv", argv),
            patch("controller_service.entrypoint.create_channel_credentials"),
        ):
            entrypoint.main()

        kwargs = mock_controller_service.call_args.kwargs
        self.assertNotIn("service_mode", kwargs)
        mock_controller_service.return_value.serve.assert_called_once()

    def test_argsfactory_supports_asd_server(self) -> None:
        """Parser accepts explicit ASD endpoint."""
        parser = ControllerService.argsfactory()
        args = parser.parse_args([*_base_argv()[1:], "--asd-server", "asd:50055"])

        self.assertEqual(args.asd_server, "asd:50055")

    def test_argsfactory_message_size_default_carries_media_chunks(self) -> None:
        """The message-size default accommodates the 1 MiB media chunks.

        The downstream NIM channels enforce this limit, so it must exceed
        the largest chunk the clients send by default.
        """
        parser = ControllerService.argsfactory()
        args = parser.parse_args(_base_argv()[1:])

        self.assertEqual(args.message_size, 1024 * 1024 * 4)

    def test_argsfactory_ssl_defaults_are_plaintext(self) -> None:
        """Server SSL is off and NIM channels are plaintext by default."""
        # Drop any ambient controller SSL values so the assertion exercises
        # the built-in defaults regardless of the host environment.
        ssl_env_prefixes = (
            "CONTROLLER_NIM_SSL_",
            "CONTROLLER_S2S_SSL_",
            "CONTROLLER_ASD_SSL_",
            "CONTROLLER_LIPSYNC_SSL_",
        )
        clean_env = {
            key: value for key, value in os.environ.items() if not key.startswith(ssl_env_prefixes)
        }
        with patch.dict("os.environ", clean_env, clear=True):
            parser = ControllerService.argsfactory()
        args = parser.parse_args(_base_argv()[1:])

        self.assertFalse(args.use_ssl)
        self.assertIsNone(args.ssl_server_key_path)
        self.assertIsNone(args.ssl_server_cert_path)
        self.assertIsNone(args.ssl_root_cert_path)
        self.assertEqual(args.ssl_mode, "DISABLED")
        self.assertIsNone(args.ssl_key)
        self.assertIsNone(args.ssl_cert)
        self.assertIsNone(args.ssl_root_cert)
        # Per-hop overrides default to None, meaning inherit --ssl-mode.
        self.assertIsNone(args.s2s_ssl_mode)
        self.assertIsNone(args.asd_ssl_mode)
        self.assertIsNone(args.lipsync_ssl_mode)

    def test_argsfactory_nim_ssl_mode_env_default(self) -> None:
        """CONTROLLER_NIM_SSL_* env vars drive the NIM channel TLS defaults."""
        env = {
            "CONTROLLER_NIM_SSL_MODE": "TLS",
            "CONTROLLER_NIM_SSL_ROOT_CERT": "/certs/root.pem",
        }
        with patch.dict("os.environ", env):
            parser = ControllerService.argsfactory()
        args = parser.parse_args(_base_argv()[1:])

        self.assertEqual(args.ssl_mode, "TLS")
        self.assertEqual(args.ssl_root_cert, "/certs/root.pem")

    def test_argsfactory_per_service_ssl_mode_env_default(self) -> None:
        """CONTROLLER_<SERVICE>_SSL_MODE env vars drive the per-hop overrides."""
        env = {
            "CONTROLLER_S2S_SSL_MODE": "DISABLED",
            "CONTROLLER_ASD_SSL_MODE": "MTLS",
        }
        with patch.dict("os.environ", env):
            parser = ControllerService.argsfactory()
        args = parser.parse_args(_base_argv()[1:])

        self.assertEqual(args.s2s_ssl_mode, "DISABLED")
        self.assertEqual(args.asd_ssl_mode, "MTLS")

    def test_argsfactory_rejects_invalid_ssl_mode_env_value(self) -> None:
        """argparse skips choices-validation for defaults, so env is checked."""
        with (
            patch.dict("os.environ", {"CONTROLLER_S2S_SSL_MODE": "PLAINTEXT"}),
            self.assertRaises(ValueError),
        ):
            ControllerService.argsfactory()

    def _run_main(
        self,
        argv: list[str],
        mock_controller_service: MagicMock,
        mock_from_string: MagicMock,
    ) -> None:
        """Run entrypoint.main with a real parser and mocked service ctor."""
        # Use the real argsfactory so argv is parsed with the production
        # argument surface (vendored SSL args included).
        mock_controller_service.argsfactory.side_effect = ControllerService.argsfactory
        mock_controller_service.return_value = MagicMock()
        mock_from_string.side_effect = [
            SimpleNamespace(host="controller", port=50056),
            SimpleNamespace(host="lipsync", port=50054),
            SimpleNamespace(host="speech-to-speech", port=50050),
        ]
        with patch("sys.argv", argv):
            entrypoint.main()

    @patch("controller_service.entrypoint.ControllerService")
    @patch("controller_service.entrypoint.GRPCServiceHandle.from_string")
    @patch("controller_service.entrypoint.SpeechToSpeechHandle")
    @patch("controller_service.entrypoint.LipsyncHandle")
    def test_main_forwards_server_ssl_args_to_serve(
        self,
        mock_lipsync_handle_ctor: MagicMock,
        mock_s2s_handle_ctor: MagicMock,
        mock_from_string: MagicMock,
        mock_controller_service: MagicMock,
    ) -> None:
        """--use-ssl and --ssl_server_* args reach ControllerService.serve."""
        mock_s2s_handle_ctor.return_value = MagicMock()
        mock_lipsync_handle_ctor.return_value = MagicMock()

        argv = [
            *_base_argv(),
            "--use-ssl",
            "--ssl_server_key_path",
            "/certs/server.key",
            "--ssl_server_cert_path",
            "/certs/server.pem",
            "--ssl_root_cert_path",
            "/certs/root.pem",
        ]
        self._run_main(
            argv=argv,
            mock_controller_service=mock_controller_service,
            mock_from_string=mock_from_string,
        )

        serve_kwargs = mock_controller_service.return_value.serve.call_args.kwargs
        self.assertTrue(serve_kwargs["use_ssl"])
        self.assertEqual(serve_kwargs["ssl_server_key_path"], "/certs/server.key")
        self.assertEqual(serve_kwargs["ssl_server_cert_path"], "/certs/server.pem")
        self.assertEqual(serve_kwargs["ssl_root_cert_path"], "/certs/root.pem")

    @patch("controller_service.entrypoint.create_channel_credentials")
    @patch("controller_service.entrypoint.ControllerService")
    @patch("controller_service.entrypoint.GRPCServiceHandle.from_string")
    @patch("controller_service.entrypoint.SpeechToSpeechHandle")
    @patch("controller_service.entrypoint.LipsyncHandle")
    def test_main_passes_nim_credentials_to_handles(
        self,
        mock_lipsync_handle_ctor: MagicMock,
        mock_s2s_handle_ctor: MagicMock,
        mock_from_string: MagicMock,
        mock_controller_service: MagicMock,
        mock_create_credentials: MagicMock,
    ) -> None:
        """--ssl-mode TLS builds per-hop credentials for every NIM handle."""
        sentinel_credentials = MagicMock()
        mock_create_credentials.return_value = sentinel_credentials
        mock_s2s_handle_ctor.return_value = MagicMock()
        mock_lipsync_handle_ctor.return_value = MagicMock()

        argv = [*_base_argv(), "--ssl-mode", "TLS", "--ssl-root-cert", "/certs/root.pem"]
        self._run_main(
            argv=argv,
            mock_controller_service=mock_controller_service,
            mock_from_string=mock_from_string,
        )

        # One credentials build per configured hop (LipSync and S2S; no ASD
        # in the base argv), each inheriting the global TLS mode.
        self.assertEqual(mock_create_credentials.call_count, 2)
        for call in mock_create_credentials.call_args_list:
            self.assertEqual(call.kwargs["ssl_mode"], "TLS")
        lipsync_kwargs = mock_lipsync_handle_ctor.call_args.kwargs
        s2s_kwargs = mock_s2s_handle_ctor.call_args.kwargs
        self.assertIs(lipsync_kwargs["channel_credentials"], sentinel_credentials)
        self.assertIs(s2s_kwargs["channel_credentials"], sentinel_credentials)

    @patch("controller_service.entrypoint.create_channel_credentials")
    @patch("controller_service.entrypoint.ControllerService")
    @patch("controller_service.entrypoint.GRPCServiceHandle.from_string")
    @patch("controller_service.entrypoint.SpeechToSpeechHandle")
    @patch("controller_service.entrypoint.LipsyncHandle")
    def test_main_s2s_ssl_mode_override_keeps_s2s_plaintext(
        self,
        mock_lipsync_handle_ctor: MagicMock,
        mock_s2s_handle_ctor: MagicMock,
        mock_from_string: MagicMock,
        mock_controller_service: MagicMock,
        mock_create_credentials: MagicMock,
    ) -> None:
        """--s2s-ssl-mode DISABLED keeps S2S plaintext while NIMs use TLS."""
        sentinel_credentials = MagicMock()
        mock_create_credentials.return_value = sentinel_credentials
        mock_s2s_handle_ctor.return_value = MagicMock()
        mock_lipsync_handle_ctor.return_value = MagicMock()

        argv = [
            *_base_argv(),
            "--ssl-mode",
            "TLS",
            "--ssl-root-cert",
            "/certs/root.pem",
            "--s2s-ssl-mode",
            "DISABLED",
        ]
        self._run_main(
            argv=argv,
            mock_controller_service=mock_controller_service,
            mock_from_string=mock_from_string,
        )

        # Only the LipSync hop builds credentials; the S2S hop stays
        # plaintext despite the global TLS mode.
        self.assertEqual(mock_create_credentials.call_count, 1)
        lipsync_kwargs = mock_lipsync_handle_ctor.call_args.kwargs
        s2s_kwargs = mock_s2s_handle_ctor.call_args.kwargs
        self.assertIs(lipsync_kwargs["channel_credentials"], sentinel_credentials)
        self.assertIsNone(s2s_kwargs["channel_credentials"])

    @patch("controller_service.entrypoint.create_channel_credentials")
    @patch("controller_service.entrypoint.ControllerService")
    @patch("controller_service.entrypoint.GRPCServiceHandle.from_string")
    @patch("controller_service.entrypoint.SpeechToSpeechHandle")
    @patch("controller_service.entrypoint.LipsyncHandle")
    def test_main_defaults_to_plaintext_nim_channels(
        self,
        mock_lipsync_handle_ctor: MagicMock,
        mock_s2s_handle_ctor: MagicMock,
        mock_from_string: MagicMock,
        mock_controller_service: MagicMock,
        mock_create_credentials: MagicMock,
    ) -> None:
        """Without --ssl-mode, NIM handles get no credentials and SSL is off."""
        mock_s2s_handle_ctor.return_value = MagicMock()
        mock_lipsync_handle_ctor.return_value = MagicMock()

        self._run_main(
            argv=_base_argv(),
            mock_controller_service=mock_controller_service,
            mock_from_string=mock_from_string,
        )

        mock_create_credentials.assert_not_called()
        lipsync_kwargs = mock_lipsync_handle_ctor.call_args.kwargs
        self.assertIsNone(lipsync_kwargs["channel_credentials"])
        serve_kwargs = mock_controller_service.return_value.serve.call_args.kwargs
        self.assertFalse(serve_kwargs["use_ssl"])
