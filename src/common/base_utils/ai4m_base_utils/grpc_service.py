#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for streaming gRPC service."""

import argparse
import time
import os
import sys

import contextlib
import datetime
import multiprocessing
import socket
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Tuple, List


from grpc_health.v1 import health
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server

import grpc
from ai4m_base_utils.logger import logger
from ai4m_base_utils.error_utils import (
    ServiceConfigurationError,
)
from ai4m_base_utils.auth import Auth
from ai4m_base_utils.config import (
    AI4M_DEFAULT_SERVICE_GRPC_URI,
    AI4M_DEFAULT_MAX_CONCURRENCY,
    AI4M_DEFAULT_MESSAGE_SIZE,
)

from ai4m_base_utils.hooks import (
    BaseHooks,
    CleanupHooks,
    MonitoringHooks,
)

# Multiprocessing constants
_ONE_DAY = datetime.timedelta(days=1)


@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", 0))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _wait_forever(server):
    """Wait for server termination."""
    try:
        while True:
            time.sleep(_ONE_DAY.total_seconds())
    except KeyboardInterrupt:
        server.stop(None)


class GRPCServiceBase(ABC, BaseHooks, CleanupHooks, MonitoringHooks):
    """Abstract base class for transactional and streaming gRPC service.

    This class provides a foundation for building gRPC services with configurable
    message size, logging, SSL options, and concurrency modes. It implements core
    functionalities for serving gRPC endpoints and includes extensive hook interfaces
    for monitoring, lifecycle management, and error handling.

    Features:
      - Abstract interface requiring concrete implementations to define servicers
      - Command-line argument parsing for service configuration
      - SSL/TLS support for secure communication
      - Lifecycle hooks for initialization, monitoring, and cleanup
      - Configurable concurrency modes: multi-threading or multi-processing
      - Configurable message size options
      - Environment variable support for configuration

    Concurrency Modes:
      - Threading: Single process with multiple threads (original behavior)
      - Multiprocessing: Multiple processes with configurable threads per process

    Environment Variables:
      - GRPC_CONCURRENCY_MODE: Set to 'threading' or 'multiprocessing'
        (default: 'multiprocessing')
      - GRPC_THREADS_PER_PROCESS: Number of threads per process in multiprocessing
        mode (default: 1)

    Inherits from:
      - ABC: Makes this an abstract base class requiring implementation
      - BaseHooks: Provides lifecycle hook interfaces
      - CleanupHooks: Provides resource cleanup mechanisms
      - MonitoringHooks: Provides performance and health monitoring

    Attributes:
        _message_size (int): Maximum size for gRPC messages
    """

    @property
    def message_size(self) -> int:
        """Get the maximum message size for gRPC communication."""
        return self._message_size

    @message_size.setter
    def message_size(self, value: int) -> None:
        self._message_size = value

    def __init__(
        self,
        message_size: int = AI4M_DEFAULT_MESSAGE_SIZE,
        interceptors: List[grpc.ServerInterceptor] = None,
    ) -> None:
        self._message_size = message_size
        self._interceptors = interceptors

    @abstractmethod
    def add_servicer_to_server(self, server: grpc.Server) -> None:
        """Hook function that the derived class needs to implement.

        The derived class function should connect a servicer object to provided
        gRPC server object.

        This function is called by the base class in run_service(...) function,
        while setting up the gRPC service.

        Args:
            server (grpc.Server): A grpc server object.
        """

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Parse the command line arguments.

        Returns: parsed command line arguments namespace.
        """
        # Server configuration
        if parser is None:
            parser = argparse.ArgumentParser(description="AI4M Base Parser")

        parser.add_argument(
            "--service-uri",
            type=str,
            help=f"URI to bind the service, default {AI4M_DEFAULT_SERVICE_GRPC_URI}",
            default=AI4M_DEFAULT_SERVICE_GRPC_URI,
        )
        # Add SSL arguments through Auth utility
        parser = Auth.argsfactory(parser=parser)

        parser.add_argument(
            "--max-concurrency",
            type=int,
            help=(
                f"Maximum number of concurrent connections, default "
                f"{AI4M_DEFAULT_MAX_CONCURRENCY}"
            ),
            default=AI4M_DEFAULT_MAX_CONCURRENCY,
        )

        # Add message size command line argument
        parser.add_argument(
            "--message-size",
            type=int,
            help=(
                f"Maximum size for gRPC messages in bytes, "
                f"default {AI4M_DEFAULT_MESSAGE_SIZE}"
            ),
            default=AI4M_DEFAULT_MESSAGE_SIZE,
        )

        # Add concurrency mode selection
        parser.add_argument(
            "--concurrency-mode",
            type=str,
            choices=["threading", "multiprocessing"],
            help=(
                "Concurrency mode: 'threading' for multi-threading or 'multiprocessing' "
                "for multi-processing. Can also be set via GRPC_CONCURRENCY_MODE "
                "environment variable. Default is 'multiprocessing'."
            ),
            default=os.getenv("GRPC_CONCURRENCY_MODE", "multiprocessing"),
        )

        # Add threads per process for multiprocessing mode
        parser.add_argument(
            "--threads-per-process",
            type=int,
            help="Number of threads per process in multiprocessing mode. "
            "Can also be set via GRPC_THREADS_PER_PROCESS environment variable. "
            "Default is 1.",
            default=int(os.getenv("GRPC_THREADS_PER_PROCESS", "2")),
        )

        return parser

    def serve(
        self,
        service_uri: str,
        max_concurrency: int = AI4M_DEFAULT_MAX_CONCURRENCY,
        use_ssl: bool = False,
        ssl_server_key_path: os.PathLike = None,
        ssl_server_cert_path: os.PathLike = None,
        ssl_root_cert_path: os.PathLike = None,
        message_size: int = None,
        concurrency_mode: str = None,
        threads_per_process: int = None,
    ) -> None:
        """Run the gRPC service with the specified configuration.

        This method configures and starts a gRPC server with the provided parameters.
        It sets up both the feature service (via add_servicer_to_server) and health
        service, configures SSL if enabled, and starts listening on the specified URI.

        Args:
            service_uri (str): URI where the gRPC service will listen
                (e.g., "localhost:50051")
            max_concurrency (int): Maximum number of concurrent RPCs the server
                can handle
            use_ssl (bool): Whether to enable SSL/TLS encryption for the server
            ssl_server_key_path (os.PathLike): Path to the server's private key file
                (required if use_ssl=True)
            ssl_server_cert_path (os.PathLike): Path to the server's certificate file
                (required if use_ssl=True)
            ssl_root_cert_path (os.PathLike): Path to the root certificate for mTLS
                authentication (optional)
            message_size (int): Maximum size for gRPC messages in bytes
            concurrency_mode (str): Concurrency mode: 'threading' or 'multiprocessing'.
                Default is 'multiprocessing'
            threads_per_process (int): Number of threads per process in multiprocessing
                mode. Default is 1

        Raises:
            ServiceConfigurationError: If there are issues starting the server
        """
        # Update message_size if provided
        if message_size is not None:
            self.message_size = message_size

        # Choose the appropriate serving mode
        if concurrency_mode == "threading":
            logger.info("Using threading mode for gRPC service")
            self._serve_threading(
                service_uri=service_uri,
                max_concurrency=max_concurrency,
                use_ssl=use_ssl,
                ssl_server_key_path=ssl_server_key_path,
                ssl_server_cert_path=ssl_server_cert_path,
                ssl_root_cert_path=ssl_root_cert_path,
            )
        elif concurrency_mode == "multiprocessing":
            logger.info("Using multiprocessing mode for gRPC service")
            self._serve_multiprocess(
                service_uri=service_uri,
                max_concurrency=max_concurrency,
                use_ssl=use_ssl,
                ssl_server_key_path=ssl_server_key_path,
                ssl_server_cert_path=ssl_server_cert_path,
                ssl_root_cert_path=ssl_root_cert_path,
                threads_per_process=threads_per_process,
            )
        else:
            raise ServiceConfigurationError(
                f"Invalid concurrency mode: {concurrency_mode}. "
                f"Must be 'threading' or 'multiprocessing'."
            )

    def _serve_threading(
        self,
        service_uri: str,
        max_concurrency: int,
        use_ssl: bool,
        ssl_server_key_path: os.PathLike,
        ssl_server_cert_path: os.PathLike,
        ssl_root_cert_path: os.PathLike,
    ) -> None:
        """Run the gRPC service in threading mode (single-process multi-threading)."""
        logger.info(f"Starting threading gRPC service with {max_concurrency} threads")

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_concurrency),
            interceptors=self._interceptors,
        )

        # Add servicers
        self.add_servicer_to_server(server)  # feature service
        add_HealthServicer_to_server(health.HealthServicer(), server)  # health service

        # Configure server credentials and start listening
        if use_ssl:
            creds = Auth.configure_ssl_credentials(
                ssl_server_key_path=ssl_server_key_path,
                ssl_server_cert_path=ssl_server_cert_path,
                ssl_root_cert_path=ssl_root_cert_path,
                use_ssl=use_ssl,
            )
            server.add_secure_port(service_uri, creds)
        else:
            logger.info("Using Insecure Server Credentials")
            server.add_insecure_port(service_uri)

        try:
            server.start()
            logger.info(f"Listening to {service_uri}")
            server.wait_for_termination()
        except Exception as e:
            raise ServiceConfigurationError(f"Failed to start server: {str(e)}") from e

    def _parse_service_uri(self, service_uri: str = None) -> Tuple[str, int]:
        """Parse service URI and extract host and port.

        Args:
            service_uri (str, optional): URI to parse. If None, uses default from config.

        Returns:
            Tuple[str, int]: Host and port extracted from the URI.

        Raises:
            ServiceConfigurationError: If URI format is invalid.
        """
        # Use default service URI if none provided
        if service_uri is None or service_uri == "":
            service_uri = AI4M_DEFAULT_SERVICE_GRPC_URI
            logger.info(f"Using default service URI: {service_uri}")

        # Extract host and port from service_uri
        if "://" in service_uri:
            # Handle URIs like "http://localhost:8001" - extract just the host:port part
            host_port = service_uri.split("://")[1]
        else:
            # Handle URIs like "localhost:8001" or "0.0.0.0:8001"
            host_port = service_uri

        # Parse host and port
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError as exc:
                raise ServiceConfigurationError(
                    f"Invalid port in service URI: {service_uri}"
                ) from exc
        else:
            raise ServiceConfigurationError(
                f"Invalid service URI format: {service_uri}"
            )

        return host, port

    def _serve_multiprocess(
        self,
        service_uri: str,
        max_concurrency: int,
        use_ssl: bool,
        ssl_server_key_path: os.PathLike,
        ssl_server_cert_path: os.PathLike,
        ssl_root_cert_path: os.PathLike,
        threads_per_process: int = 1,
    ) -> None:
        """Run the gRPC service in multiprocess mode."""
        logger.info(
            f"Starting multiprocess gRPC service with {max_concurrency} processes "
            f"and {threads_per_process} threads per process"
        )

        # Parse service URI to extract host and port
        host, port = self._parse_service_uri(service_uri)

        # Use the specified port or reserve a new one
        if port != 0:
            bind_address = f"{host}:{port}"
            self._start_multiprocess_workers(
                bind_address=bind_address,
                process_count=max_concurrency,
                use_ssl=use_ssl,
                ssl_server_key_path=ssl_server_key_path,
                ssl_server_cert_path=ssl_server_cert_path,
                ssl_root_cert_path=ssl_root_cert_path,
                threads_per_process=threads_per_process,
            )
        else:
            # Reserve a port for dynamic allocation
            with _reserve_port() as reserved_port:
                bind_address = f"{host}:{reserved_port}"
                logger.info(f"Reserved port {reserved_port} for multiprocess service")
                self._start_multiprocess_workers(
                    bind_address=bind_address,
                    process_count=max_concurrency,
                    use_ssl=use_ssl,
                    ssl_server_key_path=ssl_server_key_path,
                    ssl_server_cert_path=ssl_server_cert_path,
                    ssl_root_cert_path=ssl_root_cert_path,
                    threads_per_process=threads_per_process,
                )

    def _start_multiprocess_workers(
        self,
        bind_address: str,
        process_count: int,
        use_ssl: bool,
        ssl_server_key_path: os.PathLike,
        ssl_server_cert_path: os.PathLike,
        ssl_root_cert_path: os.PathLike,
        threads_per_process: int = 1,
    ) -> None:
        """Start multiple worker processes."""
        logger.info(
            f"Binding to '{bind_address}' with {process_count} processes "
            f"and {threads_per_process} threads per process"
        )
        sys.stdout.flush()

        workers = []
        for i in range(process_count):
            # NOTE: It is imperative that the worker subprocesses be forked before
            # any gRPC servers start up. See
            # https://github.com/grpc/grpc/issues/16001 for more details.
            worker = multiprocessing.Process(
                target=self._run_server_worker,
                args=(
                    bind_address,
                    use_ssl,
                    ssl_server_key_path,
                    ssl_server_cert_path,
                    ssl_root_cert_path,
                    threads_per_process,
                ),
            )
            worker.start()
            workers.append(worker)
            logger.info(
                f"Started worker process {i+1}/{process_count} with PID {worker.pid}"
            )

        # Wait for all workers to complete
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            logger.info("Shutting down multiprocess service...")
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

    def _run_server_worker(
        self,
        bind_address: str,
        use_ssl: bool,
        ssl_server_key_path: os.PathLike,
        ssl_server_cert_path: os.PathLike,
        ssl_root_cert_path: os.PathLike,
        threads_per_process: int = 1,
    ) -> None:
        """Start a server in a subprocess."""
        logger.info(
            f"[PID {os.getpid()}] Starting new server worker "
            f"with {threads_per_process} threads."
        )

        # Enable SO_REUSEPORT for multiprocess support and set message size limits

        # Use configurable number of threads per process
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=threads_per_process),
            interceptors=self._interceptors,
        )

        # Add servicers
        self.add_servicer_to_server(server)  # feature service
        add_HealthServicer_to_server(health.HealthServicer(), server)  # health service

        # Configure server credentials and start listening
        if use_ssl:
            creds = Auth.configure_ssl_credentials(
                ssl_server_key_path=ssl_server_key_path,
                ssl_server_cert_path=ssl_server_cert_path,
                ssl_root_cert_path=ssl_root_cert_path,
                use_ssl=use_ssl,
            )
            server.add_secure_port(bind_address, creds)
        else:
            server.add_insecure_port(bind_address)

        try:
            server.start()
            logger.info(f"[PID {os.getpid()}] Worker listening on {bind_address}")
            _wait_forever(server)
        except Exception as e:
            logger.error(f"[PID {os.getpid()}] Worker failed to start: {str(e)}")
            raise ServiceConfigurationError(
                f"Worker failed to start server: {str(e)}"
            ) from e
