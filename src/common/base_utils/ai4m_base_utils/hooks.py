#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contains the base hooks classes for AI4M services."""

import time
from typing import Any, Dict

from ai4m_base_utils.logger import logger


class BaseHooks:
    """General hooks for service lifecycle events.

    These already come with some basic debug logs. These are not essential
    for any mode, so they can be completely overwritten.
    """

    # pylint: disable=attribute-defined-outside-init
    def on_before_initialize(self) -> None:
        """on_before_initialize - run this before initialize contents run."""
        self._start_initialization_time = time.time()
        logger.debug("Service initializing.")

    def on_after_initialize(self) -> None:
        """on_after_initialize - run this after initialize contents run."""
        _total_initialization_time = time.time() - self._start_initialization_time
        logger.debug(f"Service initialized in {_total_initialization_time:.4f}s.")

    def on_before_execute(self) -> None:
        """on_before_execute - run this before execute contents run."""
        logger.debug("Service execution begins.")
        self._start_execution_time = time.time()

    def on_after_execute(self) -> None:
        """on_after_execute - run this after execute contents run."""
        _total_execution_time = time.time() - self._start_execution_time
        logger.debug(f"Service execution completed in {_total_execution_time:.4f}s.")

    def on_before_execute_request(self) -> None:
        """on_before_execute_request - run this before execute_request contents run."""
        logger.debug("Service execution begins for request.")
        self._start_execute_request_time = time.time()

    def on_after_execute_request(self) -> None:
        """on_after_execute_request - run this after execute_request contents run."""
        _total_execute_request_time = time.time() - self._start_execute_request_time
        logger.debug(
            f"Service request execution completed in {_total_execute_request_time:.4f}s."
        )

    def on_before_finalize(self) -> None:
        """on_before_finalize - run this before execute contents run."""
        self._start_finalization_time = time.time()
        logger.debug("Service finalization begins.")

    def on_after_finalize(self) -> None:
        """on_after_finalize - run this after execute contents run."""
        _total_finalization_time = time.time() - self._start_finalization_time
        logger.debug(f"Service finalized in {_total_finalization_time:.4f}s.")

    # pylint: disable=attribute-defined-outside-init
    def on_before_serve(self) -> None:
        """on_before_serve - run this before serve contents run."""
        self._start_serve_time = time.time()
        logger.debug("Service serving.")

    def on_after_serve(self) -> None:
        """on_after_serve - run this after serve contents run."""
        _total_serve_time = time.time() - self._start_serve_time
        logger.debug(f"Total service uptime {_total_serve_time:.4f}s.")

    def on_before_model_ready_check(self, model_name: str) -> None:
        """on_before_model_ready_check - run this before checking if a model is ready.

        Args:
            model_name (str): Name of the model to check
        """
        self._start_model_load_time = time.time()
        logger.debug(f"Checking if model is ready: {model_name}")

    def on_after_model_ready_check(self, model_name: str, success: bool) -> None:
        """on_after_model_ready_check - run this after checking if a model is ready.

        Args:
            model_name (str): Name of the model that was checked
            success (bool): Whether model is ready
        """
        _total_load_time = time.time() - self._start_model_load_time
        if success:
            logger.debug(f"Model {model_name} is ready, check took {_total_load_time:.4f}s.")
        else:
            logger.debug(
                f"Model {model_name} is not ready, check took {_total_load_time:.4f}s."
            )

    def on_before_infer(
        self, model_name: str, metadata: Dict[str, Any] | None = None
    ) -> None:
        """on_before_infer - run this before inference.

        Args:
            model_name (str): Name of the model being used for inference
            metadata (Dict[str, Any] | None): Additional metadata about the inference request
        """
        self._start_infer_time = time.time()
        if metadata:
            logger.debug(f"Starting inference with model {model_name}, metadata: {metadata}")
        else:
            logger.debug(f"Starting inference with model {model_name}")

    def on_after_infer(
        self, model_name: str, success: bool, error: Exception | None = None
    ) -> None:
        """on_after_infer - run this after inference.

        Args:
            model_name (str): Name of the model used for inference
            success (bool): Whether inference was successful
            error (Exception | None): Exception if inference failed
        """
        _total_infer_time = time.time() - self._start_infer_time
        if success:
            logger.debug(
                f"Inference with model {model_name} completed in {_total_infer_time:.4f}s."
            )
        else:
            logger.debug(
                f"Inference with model {model_name} failed "
                f"after {_total_infer_time:.4f}s: {error}"
            )


class CleanupHooks:
    """Customizable hooks to do cleanup inside the service.

    These are not essential for any mode, so they can be completely overwritten.
    """

    def cleanup_after_request(self) -> None:
        """cleanup_after_request Use this hook to do any cleanup after every request."""

    def cleanup_after_execute(self) -> None:
        """cleanup_after_execute Use this hook to do any cleanup after every execution."""

    def cleanup_model_resources(self, model_name: str) -> None:
        """cleanup_model_resources Use this hook to cleanup resources for a model.

        Args:
            model_name (str): Name of the model whose resources should be cleaned up
        """

    def cleanup_temporary_files(self, file_paths: str | list) -> None:
        """cleanup_temporary_files Use this hook to cleanup temporary files.

        Args:
            file_paths (str | list): Path or list of paths to files to clean up
        """

    def cleanup_before_shutdown(self) -> None:
        """cleanup_before_shutdown Use this hook for final cleanup before shutdown."""


class MonitoringHooks:
    """Hooks for monitoring service performance and health."""

    def on_service_health_check(self) -> bool:
        """on_service_health_check - Called during health check requests.

        Returns:
            bool: True if service is healthy, False otherwise
        """
        return True

    def collect_metrics(self) -> Dict[str, Any]:
        """collect_metrics - Collect service metrics.

        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        return {}

    def log_performance(self, operation: str, duration_ms: float) -> None:
        """log_performance - Log performance metrics.

        Args:
            operation (str): Name of the operation being measured
            duration_ms (float): Duration in milliseconds
        """
        logger.debug(f"Performance: {operation} took {duration_ms:.2f}ms")

    def on_error(
        self, error: Exception, context: Dict[str, Any] | None = None
    ) -> None:
        """on_error - Called when an error occurs.

        Args:
            error (Exception): The error that occurred
            context (Dict[str, Any] | None): Additional context about where the error occurred
        """
        if context:
            logger.error(f"Error occurred: {error}, context: {context}")
        else:
            logger.error(f"Error occurred: {error}")
