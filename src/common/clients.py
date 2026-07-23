# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client abstractions for controller services.

Example::

    from common.buffers import Buffer
    from common.clients import Client
    from common.service import GRPCInferenceHandle


    # Upstream code populates the request buffer; the client will set done=True when finished
    request_buffer: Buffer[int] = Buffer()
    output_buffer: Buffer[str] = Buffer()
    handle = GRPCInferenceHandle(...)  # concrete handle instance

    client = Client(handle)
    client(
        request_iterator=request_buffer,
        output_buffer=output_buffer,
        context=...,
        request_id="req-1",
    )
"""

import traceback
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any
from typing import Generic
from typing import TypeVar

import grpc

from common.base_utils import logger
from common.buffers import Buffer
from common.service import GRPCInferenceHandle

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT")


class Client(ABC, Generic[ReqT, RespT]):
    """Abstract client that manages requests and an output buffer.

    Each client maintains its own thread-safe output buffer that was created by
    the caller. The request generation strategy is also provided by a callable
    so that one client can consume from another client's output buffer or from
    other iterators without coupling that logic to the client itself. The
    request iterator must be thread-safe. If the request iterator is an output
    buffer of another client, it will automatically be thread-safe.
    """

    def __init__(
        self,
        handle: GRPCInferenceHandle,
    ) -> None:
        """Initialize a client that streams service responses into an output buffer.

        Args:
            handle: GRPCInferenceHandle used to create the response iterator.
        """
        self.handle = handle
        logger.debug(f"Client initialized with handle: {handle}")

    def is_healthy(self) -> bool:
        """Check if the remote inference service is healthy.

        Returns:
            bool: True when the service is healthy. This method never returns False —
                an unhealthy service raises instead.

        Raises:
            ConnectionError: If the service is not healthy.
        """
        logger.debug(f"Checking health of service at {self.handle}")
        try:
            self.handle.is_healthy()
        except ConnectionError as e:
            logger.error(f"Service at {self.handle} is not healthy: {e}\n" + traceback.format_exc())
            raise e
        logger.debug(f"Service at {self.handle} is healthy")
        return True

    def __call__(
        self,
        request_iterator: Iterator[ReqT],
        output_buffer: Buffer[RespT],
        context: grpc.ServicerContext,
        request_id: str,
        **kwargs: Any,
    ) -> None:
        """Run the client by streaming responses into the output buffer.

        The health check runs inside the try/finally so that ``output_buffer.done``
        is set on every exit path and downstream consumers always observe stream
        completion.

        Args:
            request_iterator: Inbound requests (buffer, generator, or gRPC iterator).
            output_buffer: Buffer receiving service responses. Client doesn't own buffer.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            **kwargs: Additional keyword arguments forwarded to ``_impl``.

        Returns:
            None. ``output_buffer`` is populated in place.
        """
        logger.debug(f"Client __call__ invoked: request_id={request_id}")
        try:
            # is_healthy() raises ConnectionError on failure (never returns False).
            self.is_healthy()
            logger.debug(f"Starting _impl for request_id={request_id}")
            self._impl(
                request_iterator=request_iterator,
                output_buffer=output_buffer,
                context=context,
                request_id=request_id,
                **kwargs,
            )
            logger.debug(f"Completed _impl for request_id={request_id}")
        except ConnectionError as e:
            tb = traceback.format_exc()
            logger.error(f"Service at {self.handle} is not healthy: {e}\n{tb}")
            context.abort(grpc.StatusCode.UNAVAILABLE, f"{type(e).__name__}: {e}\n{tb}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error running client: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        finally:
            output_buffer.done = True
            logger.debug(f"Client __call__ finished: request_id={request_id}")

    @abstractmethod
    def _impl(
        self,
        request_iterator: Iterator[ReqT],
        output_buffer: Buffer[RespT],
        context: grpc.ServicerContext,
        request_id: str,
        **kwargs: Any,
    ) -> None:
        """Implement client-specific logic to produce responses.

        Exceptions raised here propagate to ``__call__``, which logs, aborts the
        context exactly once, and marks the output buffer done. Subclasses must not
        call ``context.abort`` themselves; ``__call__`` owns error reporting.

        Args:
            request_iterator: Inbound requests to consume.
            output_buffer: Destination buffer for responses.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            **kwargs: Additional keyword arguments.

        Returns:
            None. Subclasses must place results into ``output_buffer``; ``__call__``
            owns setting the buffer ``done``.
        """
        raise NotImplementedError
