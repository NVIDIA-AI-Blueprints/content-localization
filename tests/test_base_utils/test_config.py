# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.config module."""

import importlib
from unittest.mock import patch

import pytest
from ai4m_base_utils.config import AI4M_DEFAULT_MAX_CONCURRENCY
from ai4m_base_utils.config import AI4M_DEFAULT_MESSAGE_SIZE
from ai4m_base_utils.config import AI4M_DEFAULT_SERVICE_GRPC_HOST
from ai4m_base_utils.config import AI4M_DEFAULT_SERVICE_GRPC_PORT
from ai4m_base_utils.config import AI4M_DEFAULT_SERVICE_GRPC_URI
from ai4m_base_utils.config import AI4M_MAX_INPUT_FILE_SIZE_MB

pytestmark = pytest.mark.unit

# NOTE: This test imports directly from the vendored module because it tests
# config constants not re-exported via base_utils, and uses importlib.reload.


class TestConfig:
    """Test configuration constants and functions."""

    def test_default_constants(self):
        """Test that default constants have expected values."""
        assert AI4M_DEFAULT_SERVICE_GRPC_HOST == "0.0.0.0"
        assert AI4M_DEFAULT_SERVICE_GRPC_PORT == 8001
        assert AI4M_DEFAULT_SERVICE_GRPC_URI == "0.0.0.0:8001"
        assert AI4M_DEFAULT_MAX_CONCURRENCY == 5
        assert AI4M_DEFAULT_MESSAGE_SIZE == 64 * 1024
        assert AI4M_MAX_INPUT_FILE_SIZE_MB is None or isinstance(AI4M_MAX_INPUT_FILE_SIZE_MB, int)

    @patch.dict("os.environ", {"AI4M_MAX_INPUT_FILE_SIZE_MB": "512"})
    def test_file_size_limit_env_override(self):
        """Test that environment variable overrides work."""
        import ai4m_base_utils.config as config_module

        importlib.reload(config_module)
        assert config_module.AI4M_MAX_INPUT_FILE_SIZE_MB == 512

    @patch.dict("os.environ", {}, clear=True)
    def test_file_size_limit_no_env_var(self):
        """Test that no file size limit is set when env var is not provided."""
        import ai4m_base_utils.config as config_module

        importlib.reload(config_module)
        assert config_module.AI4M_MAX_INPUT_FILE_SIZE_MB is None
