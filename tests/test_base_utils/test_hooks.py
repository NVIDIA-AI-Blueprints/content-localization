# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.hooks module."""

import time

import pytest

from common.base_utils import BaseHooks
from common.base_utils import CleanupHooks
from common.base_utils import MonitoringHooks

pytestmark = pytest.mark.unit


class TestBaseHooks:
    """Test BaseHooks functionality."""

    @pytest.fixture
    def base_hooks(self):
        """Create BaseHooks instance for testing."""
        return BaseHooks()

    def test_initialization_timing(self, base_hooks):
        """Test initialization timing hooks."""
        base_hooks.on_before_initialize()
        assert hasattr(base_hooks, "_start_initialization_time")
        time.sleep(0.01)
        base_hooks.on_after_initialize()

    def test_execution_timing(self, base_hooks):
        """Test execution timing hooks."""
        base_hooks.on_before_execute()
        assert hasattr(base_hooks, "_start_execution_time")
        time.sleep(0.01)
        base_hooks.on_after_execute()

    def test_request_timing(self, base_hooks):
        """Test request timing hooks."""
        base_hooks.on_before_execute_request()
        assert hasattr(base_hooks, "_start_execute_request_time")
        time.sleep(0.01)
        base_hooks.on_after_execute_request()

    def test_finalization_timing(self, base_hooks):
        """Test finalization timing hooks."""
        base_hooks.on_before_finalize()
        assert hasattr(base_hooks, "_start_finalization_time")
        time.sleep(0.01)
        base_hooks.on_after_finalize()

    def test_serve_timing(self, base_hooks):
        """Test serve timing hooks."""
        base_hooks.on_before_serve()
        assert hasattr(base_hooks, "_start_serve_time")
        time.sleep(0.01)
        base_hooks.on_after_serve()

    def test_model_ready_check_timing(self, base_hooks):
        """Test model ready check timing."""
        model_name = "test_model"
        base_hooks.on_before_model_ready_check(model_name)
        assert hasattr(base_hooks, "_start_model_load_time")
        time.sleep(0.01)
        base_hooks.on_after_model_ready_check(model_name, success=True)
        base_hooks.on_after_model_ready_check(model_name, success=False)

    def test_inference_timing(self, base_hooks):
        """Test inference timing hooks."""
        model_name = "test_model"
        metadata = {"sequence_id": 123}
        base_hooks.on_before_infer(model_name, metadata)
        assert hasattr(base_hooks, "_start_infer_time")
        time.sleep(0.01)
        base_hooks.on_after_infer(model_name, success=True)
        error = Exception("Inference failed")
        base_hooks.on_after_infer(model_name, success=False, error=error)


class TestCleanupHooks:
    """Test CleanupHooks functionality."""

    @pytest.fixture
    def cleanup_hooks(self):
        """Create CleanupHooks instance for testing."""
        return CleanupHooks()

    def test_cleanup_methods_callable(self, cleanup_hooks):
        """Test that all cleanup methods are callable."""
        cleanup_hooks.cleanup_after_request()
        cleanup_hooks.cleanup_after_execute()
        cleanup_hooks.cleanup_model_resources("test_model")
        cleanup_hooks.cleanup_temporary_files("/tmp/test.txt")
        cleanup_hooks.cleanup_temporary_files(["/tmp/test1.txt", "/tmp/test2.txt"])
        cleanup_hooks.cleanup_before_shutdown()


class TestMonitoringHooks:
    """Test MonitoringHooks functionality."""

    @pytest.fixture
    def monitoring_hooks(self):
        """Create MonitoringHooks instance for testing."""
        return MonitoringHooks()

    def test_health_check(self, monitoring_hooks):
        """Test service health check."""
        result = monitoring_hooks.on_service_health_check()
        assert result is True

    def test_collect_metrics(self, monitoring_hooks):
        """Test metrics collection."""
        result = monitoring_hooks.collect_metrics()
        assert isinstance(result, dict)

    def test_log_performance(self, monitoring_hooks):
        """Test performance logging."""
        monitoring_hooks.log_performance("test_operation", 123.45)

    def test_on_error_with_context(self, monitoring_hooks):
        """Test error logging with context."""
        error = Exception("Test error")
        context = {"request_id": "123", "model": "test_model"}
        monitoring_hooks.on_error(error, context)

    def test_on_error_without_context(self, monitoring_hooks):
        """Test error logging without context."""
        error = Exception("Test error")
        monitoring_hooks.on_error(error)
