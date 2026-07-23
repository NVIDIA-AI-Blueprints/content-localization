# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common shared utilities"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import grpc as real_grpc
import pytest

from common.handles import GRPCServiceHandle

pytestmark = pytest.mark.unit


class TestGRPCServiceHandle(unittest.TestCase):
    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_is_healthy_success(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.return_value.status = 1  # SERVING
        mock_stub.return_value = mock_instance
        handle = GRPCServiceHandle("localhost", 50051)
        self.assertTrue(handle.is_healthy())

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_is_healthy_failure(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.return_value.status = 2  # NOT_SERVING
        mock_stub.return_value = mock_instance
        handle = GRPCServiceHandle("localhost", 50051)
        with self.assertRaises(ConnectionError):
            handle.is_healthy()

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_is_healthy_grpc_error(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.side_effect = real_grpc.RpcError("fail")
        mock_stub.return_value = mock_instance
        handle = GRPCServiceHandle("localhost", 50051)
        with self.assertRaises(ConnectionError):
            handle.is_healthy()

    def test_from_string(self):
        handle = GRPCServiceHandle.from_string("localhost:50051")
        self.assertEqual(handle.host, "localhost")
        self.assertEqual(handle.port, 50051)

    def test_from_string_ipv6(self):
        # The address must split on the last colon so IPv6 literals parse
        handle = GRPCServiceHandle.from_string("[::1]:50051")
        self.assertEqual(handle.host, "[::1]")
        self.assertEqual(handle.port, 50051)
        self.assertEqual(str(handle), "[::1]:50051")

    def test_from_string_with_channel_credentials(self):
        credentials = MagicMock()
        handle = GRPCServiceHandle.from_string(
            "localhost:50051",
            channel_credentials=credentials,
        )
        self.assertIs(handle.channel_credentials, credentials)

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_call_success(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.return_value.status = 1  # SERVING
        mock_stub.return_value = mock_instance
        handle = GRPCServiceHandle("localhost", 50051)
        # __call__ should return host:port and trigger health check
        self.assertEqual(handle(), "localhost:50051")

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_call_failure(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.side_effect = real_grpc.RpcError("fail")
        mock_stub.return_value = mock_instance
        handle = GRPCServiceHandle("localhost", 50051)
        with self.assertRaises(ConnectionError):
            handle()


if __name__ == "__main__":
    unittest.main()
