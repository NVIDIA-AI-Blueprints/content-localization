# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.auth module."""

import argparse
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from common.base_utils import Auth
from common.base_utils import SSLConfigurationError

pytestmark = pytest.mark.unit


class TestAuth:
    """Test authentication utilities."""

    def test_validate_server_config_no_ssl(self):
        """Test server config validation with SSL disabled."""
        Auth.validate_server_config(use_ssl=False)

    def test_validate_server_config_ssl_missing_key(self):
        """Test that missing SSL key raises error."""
        with pytest.raises(SSLConfigurationError, match="SSL key file must be specified"):
            Auth.validate_server_config(use_ssl=True, ssl_cert_path="/path/to/cert.pem")

    def test_validate_server_config_ssl_missing_cert(self):
        """Test that missing SSL cert raises error."""
        with pytest.raises(SSLConfigurationError, match="SSL certificate file must be specified"):
            Auth.validate_server_config(use_ssl=True, ssl_key_path="/path/to/key.pem")

    def test_is_file_accessible_existing_file(self, temp_dir):
        """Test file accessibility check with existing file."""
        test_file = os.path.join(temp_dir, "test.pem")
        with open(test_file, "w") as f:
            f.write("test content")
        assert Auth._is_file_accessible(test_file) is True

    def test_is_file_accessible_nonexistent_file(self):
        """Test file accessibility check with nonexistent file."""
        assert Auth._is_file_accessible("/nonexistent/file.pem") is False

    def test_argsfactory_adds_ssl_arguments(self):
        """Test that argsfactory adds SSL-related arguments."""
        parser = argparse.ArgumentParser()
        enhanced_parser = Auth.argsfactory(parser)
        args = enhanced_parser.parse_args([])
        assert hasattr(args, "use_ssl")
        assert hasattr(args, "ssl_server_key_path")
        assert hasattr(args, "ssl_server_cert_path")
        assert hasattr(args, "ssl_root_cert_path")
        assert args.use_ssl is False
        assert args.ssl_server_key_path is None

    @patch("ai4m_base_utils.auth.Auth._is_file_accessible", return_value=True)
    @patch("ai4m_base_utils.file_utils.FileUtils.read_file_bytes")
    @patch("grpc.ssl_server_credentials")
    def test_configure_ssl_credentials_success(
        self, mock_ssl_creds, mock_read_bytes, mock_accessible
    ):
        """Test successful SSL credentials configuration."""
        mock_read_bytes.side_effect = [
            b"key_content",
            b"cert_content",
            b"root_content",
        ]
        mock_creds_object = MagicMock()
        mock_ssl_creds.return_value = mock_creds_object
        result = Auth.configure_ssl_credentials(
            ssl_server_key_path="/path/to/key.pem",
            ssl_server_cert_path="/path/to/cert.pem",
            ssl_root_cert_path="/path/to/root.pem",
            use_ssl=True,
        )
        assert mock_read_bytes.call_count == 3
        mock_ssl_creds.assert_called_once()
        assert result == mock_creds_object

    @patch("ai4m_base_utils.auth.Auth._is_file_accessible", return_value=False)
    def test_configure_ssl_credentials_inaccessible_file(self, mock_accessible):
        """Test SSL configuration with inaccessible files."""
        with pytest.raises(SSLConfigurationError, match="not accessible"):
            Auth.configure_ssl_credentials(
                ssl_server_key_path="/bad/key.pem",
                ssl_server_cert_path="/bad/cert.pem",
                ssl_root_cert_path=None,
                use_ssl=True,
            )

    @patch("ai4m_base_utils.auth.Auth._is_file_accessible", return_value=True)
    @patch("ai4m_base_utils.file_utils.FileUtils.read_file_bytes")
    @patch("grpc.ssl_channel_credentials")
    def test_get_client_ssl_credentials(self, mock_ssl_creds, mock_read_bytes, mock_accessible):
        """Test client SSL credentials creation."""
        mock_read_bytes.side_effect = [b"key", b"cert", b"root"]
        mock_creds = MagicMock()
        mock_ssl_creds.return_value = mock_creds
        result = Auth.get_client_ssl_credentials(
            server_key_path="/key.pem",
            server_cert_path="/cert.pem",
            root_cert_path="/root.pem",
        )
        mock_ssl_creds.assert_called_once_with(
            root_certificates=b"root",
            private_key=b"key",
            certificate_chain=b"cert",
        )
        assert result == mock_creds
