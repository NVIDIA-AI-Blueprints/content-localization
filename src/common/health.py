# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service health-check helpers.

Client applications call :func:`check_service_health` before streaming
requests so connection problems surface as a clear ``ConnectionError``
instead of a mid-stream RPC failure. The probe delegates to
:meth:`common.handles.GRPCServiceHandle.is_healthy`, the single
implementation of the standard gRPC health-checking protocol.
"""

import grpc

from common.handles import GRPCServiceHandle


def check_service_health(
    server: str,
    channel_credentials: grpc.ChannelCredentials | None = None,
) -> bool:
    """Check gRPC health using the standard gRPC health checking protocol.

    Args:
        server (str): The gRPC server address (``host:port``).
        channel_credentials (grpc.ChannelCredentials | None): Optional
            credentials for probing a TLS-secured endpoint. When ``None``
            an insecure channel is used. Defaults to ``None``.

    Returns:
        bool: ``True`` if the service is healthy.

    Raises:
        ConnectionError: If the service is unreachable or not serving.

    Examples:
        >>> check_service_health(server="localhost:50051")  # doctest: +SKIP
        True
    """
    handle = GRPCServiceHandle.from_string(
        url=server,
        channel_credentials=channel_credentials,
    )
    return handle.is_healthy()
