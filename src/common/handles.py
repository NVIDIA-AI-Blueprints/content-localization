# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side handles for gRPC service endpoints, with health-check support."""

import os

import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from common.base_utils import logger

# Shared timeout for health checks. Prevents indefinite hangs when a
# downstream service is unreachable.
HEALTH_CHECK_TIMEOUT: float = float(os.environ.get("HEALTH_CHECK_TIMEOUT", "5.0"))


class GRPCServiceHandle:
    """A client-side handle for a remote gRPC service endpoint."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the GRPCServiceHandle.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.
            health_check_service (str): The gRPC health check service
                name. Defaults to ``""``.
            channel_credentials (grpc.ChannelCredentials | None):
                Optional credentials for secure channels.
                Defaults to ``None``.

        Examples:
            >>> handle = GRPCServiceHandle(host="localhost", port=50051)
            >>> str(handle)
            'localhost:50051'
        """
        self.host = host
        self.port = port
        self.health_check_service = health_check_service
        self.channel_credentials = channel_credentials

    @classmethod
    def from_string(
        cls,
        url: str,
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> "GRPCServiceHandle":
        """Create a GRPCServiceHandle from a string.

        The address is split on the last colon so bracketed IPv6
        literals such as ``'[::1]:50051'`` parse correctly.

        Args:
            url (str): The URL string in the format ``'host:port'``.
            channel_credentials (grpc.ChannelCredentials | None):
                Optional credentials for secure channels.
                Defaults to ``None``.

        Returns:
            GRPCServiceHandle: A new GRPCServiceHandle instance.

        Examples:
            >>> handle = GRPCServiceHandle.from_string("localhost:50051")
            >>> handle.port
            50051
            >>> GRPCServiceHandle.from_string("[::1]:50051").host
            '[::1]'
        """
        host, port = url.rsplit(":", maxsplit=1)
        return cls(host=host, port=int(port), channel_credentials=channel_credentials)

    def is_healthy(self) -> bool:
        """Check if the remote gRPC service is healthy.

        Uses the standard gRPC health checking protocol.

        Returns:
            bool: ``True`` if the service is healthy.

        Raises:
            ConnectionError: If the service is not healthy.

        Examples:
            >>> handle = GRPCServiceHandle(host="localhost", port=50051)
            >>> handle.is_healthy()  # doctest: +SKIP
            True
        """
        address = f"{self.host}:{self.port}"
        if self.channel_credentials is not None:
            channel = grpc.secure_channel(address, self.channel_credentials)
        else:
            channel = grpc.insecure_channel(address)
        stub = health_pb2_grpc.HealthStub(channel)
        try:
            logger.debug(f"Checking gRPC health: {address} (service='{self.health_check_service}')")
            response = stub.Check(
                health_pb2.HealthCheckRequest(
                    service=self.health_check_service,
                ),
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                logger.debug(f"gRPC health check passed for {address}")
                return True
            else:
                logger.error(f"gRPC health check failed for {address}: status={response.status}")
                raise ConnectionError(
                    f"gRPC service at {address} not healthy: status={response.status}"
                )
        except grpc.RpcError as e:
            logger.error(f"gRPC health check failed for {address}: {e!s}")
            raise ConnectionError(f"gRPC service at {address} health check failed: {e!s}") from e
        finally:
            # Always close the probe channel so it doesn't linger as an open
            # connection to the downstream service.
            channel.close()

    def __call__(self) -> str:
        """Return the endpoint address after performing a health check.

        Returns:
            str: The service address in ``'host:port'`` format.

        Raises:
            ConnectionError: If the health check fails.

        Examples:
            >>> handle = GRPCServiceHandle(host="localhost", port=50051)
            >>> handle()  # doctest: +SKIP
            'localhost:50051'
        """
        _ = self.is_healthy()
        return f"{self.host}:{self.port}"

    def __str__(self) -> str:
        """Return the endpoint address as a string.

        Returns:
            str: The service address in ``'host:port'`` format.

        Examples:
            >>> str(GRPCServiceHandle(host="localhost", port=50051))
            'localhost:50051'
        """
        return f"{self.host}:{self.port}"
