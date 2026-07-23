# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed pipeline exceptions and their gRPC status mapping.

Pipeline internals raise these exceptions instead of calling
``context.abort`` directly; the gRPC servicer maps the exception to the most
specific status code via :func:`grpc_status_for` and aborts exactly once.
This keeps status codes accurate end to end — an inner layer's specific
status can never be overwritten by an outer catch-all.
"""

import grpc


class PipelinePreconditionError(RuntimeError):
    """Raised when a request's service preconditions cannot be satisfied.

    Examples include a bypass flag requiring a downstream service that is not
    configured. Maps to ``FAILED_PRECONDITION``.
    """


class PipelineInputError(ValueError):
    """Raised when the request stream violates the API contract.

    Examples include a missing required config message or payloads sent in a
    mode that does not consume them. Maps to ``INVALID_ARGUMENT``.
    """


def grpc_status_for(error: BaseException) -> grpc.StatusCode:
    """Map an exception to the most specific gRPC status code.

    Args:
        error (BaseException): The exception raised by the pipeline.

    Returns:
        grpc.StatusCode: ``FAILED_PRECONDITION`` for
            :class:`PipelinePreconditionError`, ``INVALID_ARGUMENT`` for
            :class:`PipelineInputError` and other ``ValueError``s,
            ``UNAVAILABLE`` for ``ConnectionError``, ``NOT_FOUND`` for
            ``FileNotFoundError``, the original code for ``grpc.RpcError``,
            and ``INTERNAL`` otherwise.

    Examples:
        >>> grpc_status_for(error=PipelineInputError("missing config"))
        <StatusCode.INVALID_ARGUMENT: (3, 'invalid argument')>
    """
    if isinstance(error, grpc.RpcError):
        code = error.code() if callable(getattr(error, "code", None)) else None
        if isinstance(code, grpc.StatusCode):
            return code
    if isinstance(error, PipelinePreconditionError):
        return grpc.StatusCode.FAILED_PRECONDITION
    if isinstance(error, ConnectionError):
        return grpc.StatusCode.UNAVAILABLE
    if isinstance(error, FileNotFoundError):
        return grpc.StatusCode.NOT_FOUND
    if isinstance(error, ValueError):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL
