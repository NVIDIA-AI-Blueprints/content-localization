# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC inference service handle abstractions."""

import copy
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any
from typing import Self

import grpc

from common.base_utils import logger
from common.handles import GRPCServiceHandle


def message_size_channel_options(message_size: int) -> list[tuple[str, int]]:
    """Build gRPC channel options that set the maximum message size.

    Args:
        message_size (int): Maximum message size in bytes applied to both
            sending and receiving directions.

    Returns:
        list[tuple[str, int]]: Channel options suitable for
            ``grpc.insecure_channel``/``grpc.secure_channel``.

    Examples:
        >>> message_size_channel_options(message_size=4194304)
        [('grpc.max_receive_message_length', 4194304), ('grpc.max_send_message_length', 4194304)]
    """
    return [
        ("grpc.max_receive_message_length", message_size),
        ("grpc.max_send_message_length", message_size),
    ]


class GRPCInferenceHandle(GRPCServiceHandle, ABC):
    """Handle abstractions for NIMs and other gRPC model services.

    Manages the outbound channel and stub for one remote inference service.
    Subclasses implement :meth:`get_response_iterator` to call the specific
    RPC exposed by that service.
    """

    def __init__(
        self,
        host: str,
        port: int,
        stub_class: Any,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the GRPCInferenceHandle.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.
            stub_class (Any): The stub class to use.
            health_check_service (str): The health check service to use.
                Defaults to ``""``.
            channel_credentials (grpc.ChannelCredentials | None):
                Optional credentials for secure channels.
                Defaults to ``None``.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            channel_credentials=channel_credentials,
        )

        self.stub_class = stub_class
        self.stub = None
        self.channel = None
        logger.debug(
            f"GRPCInferenceHandle initialized: host={host}, port={port}, "
            f"stub_class={stub_class.__name__}, health_check_service={health_check_service}"
        )

    def clone(self) -> Self:
        """Create a disconnected handle to the same service endpoint.

        The clone shares the endpoint configuration (host, port, stub class,
        credentials) but has no channel or stub, so callers can open and close
        per-request channels without affecting other handles to the same
        service.

        Returns:
            Self: A new handle of the same type with ``channel`` and ``stub``
                unset.

        Examples:
            >>> request_handle = handle.clone()  # doctest: +SKIP
            >>> request_handle.connect()  # doctest: +SKIP
        """
        cloned = copy.copy(self)
        cloned.channel = None
        cloned.stub = None
        return cloned

    def connect(
        self,
        channel_options: list | None = None,
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Open a channel to the remote service and create the stub.

        Args:
            channel_options (list): The channel options.
            channel_credentials (grpc.ChannelCredentials | None): Optional credentials for
                secure channels. Defaults to None.
        """
        logger.debug(f"Opening channel to {self.host}:{self.port}")
        if channel_options is None:
            channel_options = []
        credentials = channel_credentials or self.channel_credentials
        if credentials is not None:
            self.channel = grpc.secure_channel(
                target=f"{self.host}:{self.port}",
                credentials=credentials,
                options=channel_options,
            )
        else:
            self.channel = grpc.insecure_channel(
                target=f"{self.host}:{self.port}", options=channel_options
            )
        self.stub = self.stub_class(self.channel)
        logger.debug(f"Channel opened with stub: {self.stub_class.__name__}")

    def close(self) -> None:
        """Close the gRPC channel and drop the stub.

        Explicitly terminating the channel ends the connection to the
        downstream service immediately, instead of waiting for garbage
        collection. This frees single-concurrency NIMs (e.g. LipSync) so a
        subsequent health check or request is not blocked by a lingering
        connection. Safe to call repeatedly; :meth:`connect` lazily
        recreates the channel on next use.

        Examples:
            >>> handle.connect()  # doctest: +SKIP
            >>> handle.close()  # doctest: +SKIP
        """
        if self.channel is not None:
            logger.debug(f"Closing channel to {self.host}:{self.port}")
            self.channel.close()
        self.channel = None
        self.stub = None

    @abstractmethod
    def get_response_iterator(self, request_iterator: Iterator[Any]) -> Iterator[Any]:
        """Get a response iterator from the NIM.

        Args:
            request_iterator (Iterator[Any]): The request iterator.
        """
