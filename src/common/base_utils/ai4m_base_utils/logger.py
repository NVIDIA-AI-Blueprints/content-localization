#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for gRPC services.

This module provides a global logger instance that can be configured either through
environment variables or programmatically:

Environment variables:
- AI4M_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- AI4M_LOG_JSON_ENABLED: Enable JSON logging (1 or 0)
- AI4M_LOG_FILE_PATH: Path to log file
- AI4M_LOG_DETAILED: Enable detailed logging with filename:function:line
  (1 or 0, default: True for DEBUG mode)

Usage:
    from ai4m_base_utils.logger import logger

    # Use default configuration (from environment variables)
    logger.info("Application starting")

    # Or configure programmatically
    logger.configure(level="DEBUG", use_json_logging=True, log_file_path="/path/to/logs.log")

    # Add context data to logs
    logger.add_context(user_id="12345", transaction_id="abc123")

    # Remove specific context keys
    logger.remove_context("user_id")

    # Clear all context
    logger.clear_context()
"""

import json
import logging
import os
import sys
from logging import LoggerAdapter as LoggerAdapterBase

from ai4m_base_utils.config import (
    AI4M_DEFAULT_LOG_LEVEL,
    AI4M_DEFAULT_LOG_DETAILED,
)

# Get configuration from environment variables
name = os.environ.get("AI4M_LOG_NAME", "AI4M BASE LOGGER")
log_level = os.environ.get("AI4M_LOG_LEVEL", AI4M_DEFAULT_LOG_LEVEL)
use_json_str = os.environ.get("AI4M_LOG_JSON_ENABLED", "False")
use_json = use_json_str.lower() in ("true", "1", "t", "y", "yes")
log_filepath = os.environ.get("AI4M_LOG_FILE_PATH", None)

# Get detailed logging configuration - defaults to True for DEBUG mode
detailed_logging_str = os.environ.get("AI4M_LOG_DETAILED", AI4M_DEFAULT_LOG_DETAILED)
if detailed_logging_str is False:
    # Default to True if log level is DEBUG
    use_detailed_logging = log_level.upper() == "DEBUG"
else:
    use_detailed_logging = detailed_logging_str.lower() in ("true", "1", "t", "y", "yes")

logger_base = logging.getLogger(name=name)


class SafeFormatter(logging.Formatter):
    """A formatter that safely handles newlines in log messages."""

    def __init__(self, *args, use_detailed_logging=True, **kwargs):
        self.use_detailed_logging = use_detailed_logging
        super().__init__(*args, **kwargs)

    def format(self, record):
        if isinstance(record.msg, str):
            record.message = record.msg.replace("\n", "\\n").replace("\r", "\\r")
            try:
                if record.args:
                    record.message = record.message % record.args
            except Exception:
                pass  # Use unformatted message.
        else:
            record.message = record.getMessage()

        # Format the base message
        result = super().format(record)

        # Add context data if present
        if hasattr(record, "context") and record.context:
            context_str = " ".join([f"{k}={v}" for k, v in record.context.items()])
            if context_str:
                result += f" [{context_str}]"

        return result


class JsonFormatter(SafeFormatter):
    """A formatter that outputs log records as JSON objects."""

    def __init__(self, *args, use_detailed_logging=True, **kwargs):
        super().__init__(*args, use_detailed_logging=use_detailed_logging, **kwargs)

    def format(self, record):
        if self.use_detailed_logging:
            log_record = {
                "level": record.levelname,
                "name": record.name,
                "time": self.formatTime(record),
                "file_name": record.filename,
                "function_name": record.funcName,
                "line_number": record.lineno,
                "PID": record.process,
                "message": record.getMessage(),
            }
        else:
            log_record = {
                "level": record.levelname,
                "time": self.formatTime(record),
                "PID": record.process,
                "message": record.getMessage(),
            }

        # Add context data if present
        if hasattr(record, "context") and record.context:
            # Create a safe copy of the context with properly serialized values
            safe_context = {}
            for k, v in record.context.items():
                try:
                    # Test if the value is JSON serializable
                    json.dumps({k: v})
                    safe_context[k] = v
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    safe_context[k] = str(v)
            log_record.update(safe_context)

        try:
            return json.dumps(log_record, separators=(",", ":"))  # Compact JSON.
        except (TypeError, ValueError) as e:
            # Handle serialization error
            fallback_message = (
                f'{{"time": "{log_record["time"]}", '
                f'"level": "{log_record["level"]}", '
                f'"message": "Serialization error: {e}, '
                f'original message: {log_record["message"]}"}}'
            )
            return fallback_message


class LoggerAdapter(LoggerAdapterBase):
    """Adapter that wraps the global logger to provide a consistent interface.

    Ensures that all logging goes through a single global logger instance.

    Configuration is read from environment variables by default:
    - AI4M_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - AI4M_LOG_JSON_ENABLED: Enable JSON logging (1 or 0)
    - AI4M_LOG_FILE_PATH: Path to log file
    - AI4M_LOG_DETAILED: Enable detailed logging with filename:function:line
      (1 or 0, default: True for DEBUG mode)
    """

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def __init__(
        self,
        extra=None,
        logger=logging.getLogger(name="AI4M BASE LOGGER"),
        log_level: str = "INFO",
        use_json_logging: bool = False,
        log_file_path: os.PathLike = None,
        use_detailed_logging: bool = None,
    ):
        # Initialize with empty context if extra is None
        if extra is None:
            extra = {"context": {}}
        # Ensure context exists in extra
        elif "context" not in extra:
            extra["context"] = {}

        super().__init__(logger=logger, extra=extra)

        # Store the detailed logging preference or default to DEBUG mode if not specified
        if use_detailed_logging is None:
            self.use_detailed_logging = log_level.upper() == "DEBUG"
        else:
            self.use_detailed_logging = use_detailed_logging

        # Initialize with configuration from environment
        self._configure(
            name=name,
            level=log_level,
            use_json_logging=use_json_logging,
            log_file_path=log_file_path,
            use_detailed_logging=self.use_detailed_logging,
        )

    def add_context(self, **kwargs) -> None:
        """Add context data to all subsequent log messages.

        Args:
            **kwargs: Key-value pairs to add to the context

        Examples:
            logger.add_context(user_id="12345", transaction_id="abc123")
        """
        self.extra["context"].update(kwargs)

    def remove_context(self, *args) -> None:
        """Remove specific keys from the context.

        Args:
            *args: Keys to remove from the context

        Examples:
            logger.remove_context("user_id", "transaction_id")
        """
        for key in args:
            if key in self.extra["context"]:
                del self.extra["context"][key]

    def clear_context(self) -> None:
        """Clear all context data.

        Examples:
            logger.clear_context()
        """
        self.extra["context"] = {}

    def process(self, msg, kwargs):
        """Process the logging message and keyword arguments passed in to a logging call.

        Adds context data to the record.
        """
        msg, kwargs = super().process(msg, kwargs)

        # Ensure context is passed to the record
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        if "context" not in kwargs["extra"]:
            kwargs["extra"]["context"] = self.extra["context"]
        else:
            # Merge contexts if both exist
            kwargs["extra"]["context"].update(self.extra["context"])

        return msg, kwargs

    def _configure(
        self,
        name: str = name,
        level: str = "INFO",
        use_json_logging: bool = False,
        log_file_path: os.PathLike = None,
        use_detailed_logging: bool = True,
    ) -> None:
        """Configure the logger with the given settings.

        If a parameter is None, it will not change the current setting.

        Args:
            level (str, optional): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            use_json_logging (bool, optional): Whether to use JSON logging format
            log_file_path (os.PathLike, optional): Path to log file
            use_detailed_logging (bool, optional): Whether to include
                filename:function:line in logs
        """
        # Set the log level
        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(
                f"Invalid log level: {level}, supported levels are: "
                f"DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        self._logger.setLevel(level=level)

        # Clear any existing handlers
        if self._logger.handlers:
            self._logger.handlers.clear()

        # Create formatter based on JSON preference
        if use_json_logging:
            formatter = JsonFormatter(
                datefmt="%Y-%m-%d %H:%M:%S", use_detailed_logging=use_detailed_logging
            )
            formatter.default_msec_format = "%s.%03d"  # Use dot (.) instead of comma (,)
        else:
            # Create format string based on detailed logging preference
            if use_detailed_logging:
                fmt_string = (
                    "[%(levelname)s %(asctime)s.%(msecs)03d "
                    "%(filename)s:%(funcName)s:%(lineno)d PID:%(process)d] %(message)s"
                )
            else:
                fmt_string = (
                    "[%(levelname)s %(name)s %(asctime)s.%(msecs)03d "
                    "PID:%(process)d] %(message)s"
                )

            formatter = SafeFormatter(
                fmt=fmt_string,
                datefmt="%Y-%m-%d %H:%M:%S",
                use_detailed_logging=use_detailed_logging,
            )

        # Add console handler
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file_path:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(p=log_file_path)
                if log_dir and not os.path.exists(path=log_dir):
                    os.makedirs(name=log_dir, exist_ok=True)

                # Create a new file handler using the same formatter
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(fmt=formatter)
                self._logger.addHandler(hdlr=file_handler)
                self._logger.info(msg=f"Logging to file: {log_file_path}")
            except (IOError, OSError) as e:
                self._logger.error(msg=f"Failed to set up file logging: {e}")

        # Log configuration info
        if use_json_logging:
            self._logger.debug(msg="JSON logging enabled")
        else:
            self._logger.debug(msg="Standard logging enabled")


# Global logger instance
logger = LoggerAdapter(
    logger=logger_base,
    log_level=log_level,
    use_json_logging=use_json,
    log_file_path=log_filepath,
    use_detailed_logging=use_detailed_logging,
)
