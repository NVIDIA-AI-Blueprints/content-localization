# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.logger module."""

import json
import logging
import os
from unittest.mock import patch

import pytest
from ai4m_base_utils.logger import JsonFormatter
from ai4m_base_utils.logger import LoggerAdapter
from ai4m_base_utils.logger import SafeFormatter

pytestmark = pytest.mark.unit


class TestSafeFormatter:
    """Test SafeFormatter functionality."""

    def test_safe_formatter_newline_replacement(self):
        """Test that SafeFormatter handles newlines properly."""
        formatter = SafeFormatter(fmt="%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Message with\nnewline and\rcarriage return",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert formatted is not None
        assert isinstance(formatted, str)

    def test_safe_formatter_with_context(self):
        """Test SafeFormatter with context data."""
        formatter = SafeFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.context = {"user_id": "123", "request_id": "abc"}
        formatted = formatter.format(record)
        assert "user_id=123" in formatted
        assert "request_id=abc" in formatted


class TestJsonFormatter:
    """Test JsonFormatter functionality."""

    def test_json_formatter_structure(self):
        """Test that JSON formatter produces valid JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["file_name"] == "test.py"
        assert parsed["line_number"] == 42
        assert "time" in parsed

    def test_json_formatter_with_context(self):
        """Test JSON formatter with context data."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        record.context = {"error_code": "E001", "user_id": "456"}
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["error_code"] == "E001"
        assert parsed["user_id"] == "456"

    def test_json_formatter_serialization_error(self):
        """Test JSON formatter handles non-serializable data."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.context = {"object": object()}
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["level"] == "INFO"

    def test_json_formatter_serialization_fallback(self):
        """Test JSON formatter fallback when serialization completely fails."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        with patch("json.dumps", side_effect=ValueError("Circular reference")):
            formatted = formatter.format(record)
        assert formatted is not None
        assert "Serialization error" in formatted
        assert "Circular reference" in formatted


class TestLoggerAdapter:
    """Test LoggerAdapter functionality."""

    @pytest.fixture
    def test_logger(self):
        """Create a test logger instance."""
        base = logging.getLogger("test_logger_adapter")
        base.handlers.clear()
        return LoggerAdapter(logger=base)

    def test_context_management(self, test_logger):
        """Test adding and removing context."""
        test_logger.add_context(user_id="123", session_id="abc")
        assert test_logger.extra["context"]["user_id"] == "123"
        assert test_logger.extra["context"]["session_id"] == "abc"

        test_logger.remove_context("user_id")
        assert "user_id" not in test_logger.extra["context"]
        assert "session_id" in test_logger.extra["context"]

        test_logger.clear_context()
        assert test_logger.extra["context"] == {}

    def test_logger_configuration(self, test_logger):
        """Test logger configuration."""
        test_logger._configure(level="DEBUG")
        assert test_logger._logger.level == logging.DEBUG

        test_logger._configure(level="ERROR")
        assert test_logger._logger.level == logging.ERROR

    def test_invalid_log_level(self, test_logger):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            test_logger._configure(level="INVALID")

    def test_file_logging_configuration(self, test_logger, temp_dir):
        """Test file logging configuration."""
        log_file = os.path.join(temp_dir, "test.log")
        test_logger._configure(log_file_path=log_file)
        file_handlers = [
            h for h in test_logger._logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        test_logger.info("Test message")
        assert os.path.exists(log_file)

    def test_logger_adapter_process_method(self, test_logger):
        """Test LoggerAdapter process method."""
        test_logger.add_context(user_id="123")
        msg, kwargs = test_logger.process("Test message", {})
        assert msg == "Test message"
        assert "extra" in kwargs
        assert "context" in kwargs["extra"]
        assert kwargs["extra"]["context"]["user_id"] == "123"

    def test_logger_adapter_process_with_existing_context(self, test_logger):
        """Test process method when kwargs already has context."""
        test_logger.add_context(logger_context="from_logger")
        msg, kwargs = test_logger.process(
            "Test message",
            {"extra": {"context": {"kwargs_context": "from_kwargs"}}},
        )
        context = kwargs["extra"]["context"]
        assert "logger_context" in context
        assert context["logger_context"] == "from_logger"

    def test_file_logging_directory_creation(self, test_logger, temp_dir):
        """Test that logger creates directory for log file if it doesn't exist."""
        log_dir = os.path.join(temp_dir, "logs", "subdir")
        log_file = os.path.join(log_dir, "test.log")
        assert not os.path.exists(log_dir)
        test_logger._configure(log_file_path=log_file)
        assert os.path.exists(log_dir)
