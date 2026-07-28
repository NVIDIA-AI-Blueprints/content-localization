# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.grpc_service module."""

import argparse
import os
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from ai4m_base_utils.config import AI4M_DEFAULT_SERVICE_GRPC_URI

from common.base_utils import AI4M_DEFAULT_MESSAGE_SIZE
from common.base_utils import GRPCServiceBase
from common.base_utils import ServiceConfigurationError

pytestmark = pytest.mark.unit


# Concrete implementation for testing the abstract base class
class ConcreteGRPCService(GRPCServiceBase):
    """Concrete implementation of GRPCServiceBase for testing."""

    def add_servicer_to_server(self, server):
        """Mock implementation of abstract method."""
        self._servicer_added = True


class TestGRPCServiceBase:
    """Test GRPCServiceBase functionality."""

    def test_message_size_property(self):
        """Test message size getter and setter."""
        service = ConcreteGRPCService()
        assert service.message_size == AI4M_DEFAULT_MESSAGE_SIZE
        service.message_size = 1024
        assert service.message_size == 1024

    def test_initialization_with_custom_message_size(self):
        """Test initialization with custom message size."""
        custom_size = 2048
        service = ConcreteGRPCService(message_size=custom_size)
        assert service.message_size == custom_size

    def test_argsfactory_creates_parser(self):
        """Test that argsfactory creates argument parser with expected arguments."""
        parser = GRPCServiceBase.argsfactory()
        args = parser.parse_args([])
        assert hasattr(args, "service_uri")
        assert hasattr(args, "max_concurrency")
        assert hasattr(args, "message_size")
        assert hasattr(args, "concurrency_mode")
        assert hasattr(args, "threads_per_process")
        assert args.service_uri == AI4M_DEFAULT_SERVICE_GRPC_URI
        assert args.message_size == AI4M_DEFAULT_MESSAGE_SIZE
        assert args.concurrency_mode == "multiprocessing"

    def test_argsfactory_with_existing_parser(self):
        """Test argsfactory with existing argument parser."""
        existing_parser = argparse.ArgumentParser()
        existing_parser.add_argument("--custom-arg", default="test")
        enhanced_parser = GRPCServiceBase.argsfactory(existing_parser)
        args = enhanced_parser.parse_args(["--custom-arg", "modified"])
        assert args.custom_arg == "modified"
        assert hasattr(args, "service_uri")

    def test_parse_service_uri_valid_formats(self):
        """Test parsing various valid service URI formats."""
        service = ConcreteGRPCService()
        host, port = service._parse_service_uri("localhost:8001")
        assert host == "localhost"
        assert port == 8001

        host, port = service._parse_service_uri("http://0.0.0.0:9000")
        assert host == "0.0.0.0"
        assert port == 9000

        host, port = service._parse_service_uri("[::1]:8001")
        assert host == "[::1]"
        assert port == 8001

    def test_parse_service_uri_invalid_formats(self):
        """Test parsing invalid service URI formats."""
        service = ConcreteGRPCService()
        with pytest.raises(ServiceConfigurationError, match="Invalid service URI format"):
            service._parse_service_uri("localhost")

        with pytest.raises(ServiceConfigurationError, match="Invalid port"):
            service._parse_service_uri("localhost:invalid")

    def test_parse_service_uri_none_uses_default(self):
        """Test that None URI uses default."""
        service = ConcreteGRPCService()
        host, port = service._parse_service_uri(None)
        expected_host, expected_port = service._parse_service_uri(AI4M_DEFAULT_SERVICE_GRPC_URI)
        assert host == expected_host
        assert port == expected_port

    def test_invalid_concurrency_mode(self):
        """Test that invalid concurrency mode raises error."""
        service = ConcreteGRPCService()
        with pytest.raises(ServiceConfigurationError, match="Invalid concurrency mode"):
            service.serve(service_uri="localhost:8001", concurrency_mode="invalid_mode")

    def test_serve_threading_mode(self):
        """Test serving in threading mode."""
        service = ConcreteGRPCService()
        with patch.object(service, "_serve_threading") as mock_serve:
            service.serve(
                service_uri="localhost:8001",
                max_concurrency=3,
                concurrency_mode="threading",
            )
        mock_serve.assert_called_once_with(
            service_uri="localhost:8001",
            max_concurrency=3,
            use_ssl=False,
            ssl_server_key_path=None,
            ssl_server_cert_path=None,
            ssl_root_cert_path=None,
        )

    def test_serve_updates_message_size(self):
        """Test that serve method updates message size if provided."""
        service = ConcreteGRPCService()
        original_size = service.message_size
        new_size = original_size * 2
        with patch.object(service, "_serve_threading"):
            service.serve(
                service_uri="localhost:8001",
                message_size=new_size,
                concurrency_mode="threading",
            )
        assert service.message_size == new_size


class TestServiceConfiguration:
    """Test service configuration and argument parsing."""

    def test_concurrency_mode_environment_variable(self):
        """Test that concurrency mode respects environment variable."""
        with patch.dict(os.environ, {"GRPC_CONCURRENCY_MODE": "threading"}):
            parser = GRPCServiceBase.argsfactory()
            args = parser.parse_args([])
            assert args.concurrency_mode == "threading"

    def test_threads_per_process_environment_variable(self):
        """Test that threads per process respects environment variable."""
        with patch.dict(os.environ, {"GRPC_THREADS_PER_PROCESS": "4"}):
            parser = GRPCServiceBase.argsfactory()
            args = parser.parse_args([])
            assert args.threads_per_process == 4

    def test_command_line_argument_parsing(self):
        """Test parsing command line arguments."""
        parser = GRPCServiceBase.argsfactory()
        args = parser.parse_args(
            [
                "--service-uri",
                "0.0.0.0:9000",
                "--max-concurrency",
                "10",
                "--message-size",
                "128000",
                "--concurrency-mode",
                "threading",
                "--threads-per-process",
                "3",
            ]
        )
        assert args.service_uri == "0.0.0.0:9000"
        assert args.max_concurrency == 10
        assert args.message_size == 128000
        assert args.concurrency_mode == "threading"
        assert args.threads_per_process == 3


class TestThreadingMode:
    """Test threading mode functionality."""

    @patch("ai4m_base_utils.grpc_service.grpc.server")
    @patch("ai4m_base_utils.grpc_service.add_HealthServicer_to_server")
    @patch("ai4m_base_utils.auth.Auth.configure_ssl_credentials")
    def test_serve_threading_ssl_setup(self, mock_ssl_config, mock_health, mock_grpc_server):
        """Test threading mode SSL credential setup."""
        service = ConcreteGRPCService()
        mock_server = Mock()
        mock_grpc_server.return_value = mock_server
        mock_credentials = Mock()
        mock_ssl_config.return_value = mock_credentials

        service._serve_threading(
            service_uri="localhost:8001",
            max_concurrency=5,
            use_ssl=True,
            ssl_server_key_path="/key.pem",
            ssl_server_cert_path="/cert.pem",
            ssl_root_cert_path=None,
        )
        mock_ssl_config.assert_called_once_with(
            ssl_server_key_path="/key.pem",
            ssl_server_cert_path="/cert.pem",
            ssl_root_cert_path=None,
            use_ssl=True,
        )
        mock_server.add_secure_port.assert_called_once_with("localhost:8001", mock_credentials)

    @patch("ai4m_base_utils.grpc_service.grpc.server")
    def test_serve_threading_server_start_failure(self, mock_grpc_server):
        """Test threading mode server start failure."""
        service = ConcreteGRPCService()
        mock_server = Mock()
        mock_server.start.side_effect = Exception("Failed to bind port")
        mock_grpc_server.return_value = mock_server

        with pytest.raises(ServiceConfigurationError, match="Failed to start server"):
            service._serve_threading(
                service_uri="localhost:8001",
                max_concurrency=5,
                use_ssl=False,
                ssl_server_key_path=None,
                ssl_server_cert_path=None,
                ssl_root_cert_path=None,
            )


class TestMultiprocessingMode:
    """Test multiprocessing mode functionality."""

    def test_serve_multiprocess_with_specific_port(self):
        """Test multiprocess mode with specific port."""
        service = ConcreteGRPCService()
        with (
            patch.object(service, "_parse_service_uri", return_value=("localhost", 8001)),
            patch.object(service, "_start_multiprocess_workers") as mock_start_workers,
        ):
            service._serve_multiprocess(
                service_uri="localhost:8001",
                max_concurrency=3,
                use_ssl=False,
                ssl_server_key_path=None,
                ssl_server_cert_path=None,
                ssl_root_cert_path=None,
                threads_per_process=2,
            )
            mock_start_workers.assert_called_once_with(
                bind_address="localhost:8001",
                process_count=3,
                use_ssl=False,
                ssl_server_key_path=None,
                ssl_server_cert_path=None,
                ssl_root_cert_path=None,
                threads_per_process=2,
            )

    @patch("ai4m_base_utils.grpc_service.multiprocessing.Process")
    def test_start_multiprocess_workers(self, mock_process):
        """Test starting multiprocess workers."""
        service = ConcreteGRPCService()
        mock_worker1 = Mock()
        mock_worker1.pid = 1001
        mock_worker2 = Mock()
        mock_worker2.pid = 1002
        mock_process.side_effect = [mock_worker1, mock_worker2]

        with patch.object(mock_worker1, "join"), patch.object(mock_worker2, "join"):
            service._start_multiprocess_workers(
                bind_address="localhost:8001",
                process_count=2,
                use_ssl=False,
                ssl_server_key_path=None,
                ssl_server_cert_path=None,
                ssl_root_cert_path=None,
                threads_per_process=1,
            )
        assert mock_process.call_count == 2
        mock_worker1.start.assert_called_once()
        mock_worker2.start.assert_called_once()

    @patch("ai4m_base_utils.grpc_service.grpc.server")
    @patch("ai4m_base_utils.grpc_service.add_HealthServicer_to_server")
    @patch("ai4m_base_utils.grpc_service._wait_forever")
    def test_run_server_worker_success(self, mock_wait, mock_health, mock_grpc_server):
        """Test successful server worker execution."""
        service = ConcreteGRPCService()
        mock_server = Mock()
        mock_grpc_server.return_value = mock_server
        mock_wait.return_value = None

        service._run_server_worker(
            bind_address="localhost:8001",
            use_ssl=False,
            ssl_server_key_path=None,
            ssl_server_cert_path=None,
            ssl_root_cert_path=None,
            threads_per_process=1,
        )
        mock_server.start.assert_called_once()
        mock_server.add_insecure_port.assert_called_once_with("localhost:8001")
        mock_wait.assert_called_once_with(mock_server)

    @patch("ai4m_base_utils.grpc_service.grpc.server")
    def test_run_server_worker_start_failure(self, mock_grpc_server):
        """Test server worker start failure."""
        service = ConcreteGRPCService()
        mock_server = Mock()
        mock_server.start.side_effect = Exception("Port already in use")
        mock_grpc_server.return_value = mock_server

        with pytest.raises(ServiceConfigurationError, match="Worker failed to start server"):
            service._run_server_worker(
                bind_address="localhost:8001",
                use_ssl=False,
                ssl_server_key_path=None,
                ssl_server_cert_path=None,
                ssl_root_cert_path=None,
            )


class TestPortReservation:
    """Test port reservation functionality."""

    @patch("socket.socket")
    def test_reserve_port_success(self, mock_socket_class):
        """Test successful port reservation."""
        from ai4m_base_utils.grpc_service import _reserve_port

        mock_socket = Mock()
        mock_socket.getsockname.return_value = ("", 8080)
        mock_socket.getsockopt.return_value = 1
        mock_socket_class.return_value = mock_socket

        with _reserve_port() as port:
            assert port == 8080
        mock_socket.setsockopt.assert_called()
        mock_socket.bind.assert_called_with(("", 0))
        mock_socket.close.assert_called_once()

    @patch("socket.socket")
    def test_reserve_port_reuseport_failure(self, mock_socket_class):
        """Test port reservation when SO_REUSEPORT fails."""
        from ai4m_base_utils.grpc_service import _reserve_port

        mock_socket = Mock()
        mock_socket.getsockopt.return_value = 0
        mock_socket_class.return_value = mock_socket

        with pytest.raises(RuntimeError, match="Failed to set SO_REUSEPORT"):
            with _reserve_port() as port:
                pass


class TestWaitForever:
    """Test server wait functionality."""

    @patch("time.sleep", side_effect=KeyboardInterrupt())
    def test_wait_forever_keyboard_interrupt(self, mock_sleep):
        """Test that wait_forever handles KeyboardInterrupt."""
        from ai4m_base_utils.grpc_service import _wait_forever

        mock_server = Mock()
        _wait_forever(mock_server)
        mock_server.stop.assert_called_once_with(None)
        mock_sleep.assert_called()
