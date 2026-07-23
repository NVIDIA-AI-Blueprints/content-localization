# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.error_utils module."""

import pytest

from common.base_utils import FileSizeError
from common.base_utils import ServiceConfigurationError
from common.base_utils import SSLConfigurationError

pytestmark = pytest.mark.unit


class TestFileSizeError:
    """Test FileSizeError functionality."""

    def test_file_size_error_creation(self):
        """Test basic FileSizeError creation."""
        error = FileSizeError("File too large")
        assert str(error) == "File too large"
        assert isinstance(error, Exception)


class TestSSLConfigurationError:
    """Test SSLConfigurationError functionality."""

    def test_ssl_config_error_creation(self):
        """Test basic SSLConfigurationError creation."""
        error = SSLConfigurationError("SSL setup failed")
        assert str(error) == "SSL setup failed"
        assert isinstance(error, Exception)


class TestServiceConfigurationError:
    """Test ServiceConfigurationError functionality."""

    def test_service_config_error_creation(self):
        """Test basic ServiceConfigurationError creation."""
        error = ServiceConfigurationError("Service config invalid")
        assert str(error) == "Service config invalid"
        assert isinstance(error, Exception)


class TestErrorExceptionHandling:
    """Test error handling in various scenarios."""

    def test_error_can_be_caught_as_exception(self):
        """Test that custom errors can be caught as general exceptions."""
        with pytest.raises(Exception):
            raise FileSizeError("Test")

        with pytest.raises(Exception):
            raise SSLConfigurationError("Test")

        with pytest.raises(Exception):
            raise ServiceConfigurationError("Test")

    def test_error_specific_exception_handling(self):
        """Test catching specific error types."""
        with pytest.raises(FileSizeError):
            raise FileSizeError("File too large")

        with pytest.raises(SSLConfigurationError):
            raise SSLConfigurationError("SSL failed")

        with pytest.raises(ServiceConfigurationError):
            raise ServiceConfigurationError("Config failed")
