# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the base_utils wrapper module.

Validates that the wrapper at ``src/common/base_utils/__init__.py`` correctly
re-exports all public symbols from the vendored ``ai4m_base_utils``
package, and that the vendored package's internal imports still resolve.
"""

import pytest


@pytest.mark.integration
class TestBaseUtilsReExports:
    """Verify all expected symbols are importable via ``from common.base_utils import ...``."""

    def test_logger_importable(self):
        """Logger singleton is importable and functional."""
        from common.base_utils import logger

        logger.info("integration test: logger works")

    def test_grpc_service_base_importable(self):
        """GRPCServiceBase can be imported and subclassed."""
        from common.base_utils import GRPCServiceBase

        class _TestService(GRPCServiceBase):
            def add_servicer_to_server(self, server):
                pass

        svc = _TestService()
        assert svc.message_size == 64 * 1024

    def test_config_constant_importable(self):
        """AI4M_DEFAULT_MESSAGE_SIZE is importable with the expected value."""
        from common.base_utils import AI4M_DEFAULT_MESSAGE_SIZE

        assert AI4M_DEFAULT_MESSAGE_SIZE == 64 * 1024

    def test_error_classes_importable(self):
        """All custom exception classes are importable and raisable."""
        from common.base_utils import FileSizeError
        from common.base_utils import ServiceConfigurationError
        from common.base_utils import SSLConfigurationError

        for exc_cls in (FileSizeError, SSLConfigurationError, ServiceConfigurationError):
            with pytest.raises(exc_cls):
                raise exc_cls("test")

    def test_auth_importable(self):
        """Auth utility class is importable."""
        from common.base_utils import Auth

        assert hasattr(Auth, "configure_ssl_credentials")

    def test_file_utils_importable(self):
        """FileUtils class is importable."""
        from common.base_utils import FileUtils

        assert hasattr(FileUtils, "read_file_bytes")

    def test_hooks_importable(self):
        """Hook base classes are importable."""
        from common.base_utils import BaseHooks

        hooks = BaseHooks()
        hooks.on_before_initialize()
        hooks.on_after_initialize()

    def test_all_exports_listed(self):
        """All re-exported names appear in __all__."""
        from common import base_utils

        expected = {
            "logger",
            "AI4M_DEFAULT_MESSAGE_SIZE",
            "GRPCServiceBase",
            "FileSizeError",
            "SSLConfigurationError",
            "ServiceConfigurationError",
            "Auth",
            "FileUtils",
            "BaseHooks",
            "CleanupHooks",
            "MonitoringHooks",
        }
        assert expected == set(base_utils.__all__)


@pytest.mark.integration
class TestVendoredInternalImports:
    """Verify the vendored ai4m_base_utils internal imports still resolve."""

    def test_vendored_config_importable(self):
        """Vendored config module is importable directly."""
        from ai4m_base_utils.config import AI4M_DEFAULT_MESSAGE_SIZE

        assert AI4M_DEFAULT_MESSAGE_SIZE == 64 * 1024

    def test_vendored_logger_importable(self):
        """Vendored logger module is importable directly."""
        from ai4m_base_utils.logger import logger

        assert logger is not None

    def test_vendored_grpc_service_importable(self):
        """Vendored grpc_service module is importable directly."""
        from ai4m_base_utils.grpc_service import GRPCServiceBase

        assert GRPCServiceBase is not None

    def test_wrapper_and_vendored_return_same_objects(self):
        """Wrapper re-exports are the exact same objects as vendored originals."""
        from ai4m_base_utils.logger import logger as vendored_logger

        from common.base_utils import logger as wrapper_logger

        assert wrapper_logger is vendored_logger

        from ai4m_base_utils.grpc_service import GRPCServiceBase as vendored_cls

        from common.base_utils import GRPCServiceBase as wrapper_cls

        assert wrapper_cls is vendored_cls
