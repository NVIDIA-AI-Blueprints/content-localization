# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC servicer for the Controller Service.

``ControllerServiceServicer`` is the gRPC boundary of the content
localization pipeline: it receives the client request stream, delegates
orchestration to ``ControllerService.infer``, streams responses back, and
acts as the pipeline's single abort point that maps exceptions to specific
gRPC status codes.
"""

import traceback
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import grpc
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse
from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerServicer

from common.base_utils import logger
from common.errors import grpc_status_for

if TYPE_CHECKING:
    from controller_service.service import ControllerService


class ControllerServiceServicer(ContentLocalizationControllerServicer):
    """gRPC servicer implementation for the Controller Service.

    This class implements the gRPC servicer interface for the Controller Service,
    handling client communication, request processing, and response streaming.
    It acts as the bridge between gRPC clients and the main ControllerService
    implementation.

    Responsibilities
    ================

    - **Client Communication**: Handles gRPC request/response streaming
    - **Request Routing**: Routes client requests to the main service implementation
    - **Context Management**: Manages gRPC context and metadata
    - **Error Propagation**: Propagates errors from the service to clients
    - **Request Tracking**: Tracks request IDs and client information

    Key Methods
    ===========

    StreamContentLocalization
    -------------------------
    The main gRPC method that handles content localization requests:
    - Accepts streaming requests from clients
    - Generates unique request IDs for tracking
    - Delegates processing to the main ControllerService
    - Streams responses back to clients
    - Handles errors and propagates them appropriately

    Request Lifecycle
    =================

    1. **Request Reception**: Receives streaming requests from gRPC clients
    2. **Request ID Generation**: Generates unique UUID for request tracking
    3. **Service Delegation**: Delegates processing to ControllerService.infer()
    4. **Response Streaming**: Streams processed responses back to client
    5. **Error Handling**: Catches and propagates any processing errors

    Error Handling
    ==============

    The servicer provides robust error handling:
    - Catches exceptions from the main service implementation
    - Converts exceptions to appropriate gRPC status codes
    - Provides detailed error information to clients
    - Ensures proper cleanup on errors

    Thread Safety
    =============

    The servicer is designed to be thread-safe:
    - Each request gets a unique request ID
    - Request handling is isolated per request
    - Proper error isolation between concurrent requests

    Usage
    =====

    The servicer is typically used by the gRPC server framework:

    .. code-block:: python

        # Server setup
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = ControllerServiceServicer(controller_service)
        add_ContentLocalizationControllerServicer_to_server(servicer, server)
        server.add_insecure_port("[::]:50056")
        server.start()

    Client Interaction
    ==================

    Clients interact with this servicer through the gRPC interface:

    .. code-block:: python

        # Client usage
        channel = grpc.insecure_channel("localhost:50056")
        stub = ContentLocalizationControllerStub(channel)

        # Stream requests
        responses = stub.StreamContentLocalization(request_iterator)
        for response in responses:
            # Process responses
            pass
    """

    def __init__(self, service: "ControllerService") -> None:
        """Initialize the controller service servicer.

        Args:
            service (ControllerService): The controller service instance that provides for an
                content localization.
        """
        super().__init__()
        self.service = service

    def StreamContentLocalization(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[ContentLocalizationResponse]:
        """Run the end-to-end content-localization pipeline for one request stream.

        This is the Controller's primary streaming RPC. It assigns a request id,
        delegates the multi-threaded orchestration to ``self.service.infer`` (S2S,
        ASD, and LipSync), and streams the resulting video chunks back to the
        client. This method is the pipeline's single abort point: exceptions
        raised anywhere in the handler-thread pipeline are mapped to the most
        specific gRPC status via :func:`common.errors.grpc_status_for` and
        reported once via ``context.abort`` with the full traceback.

        Args:
            request_iterator (Iterator[ContentLocalizationRequest]): Inbound
                stream of content-localization requests (config, audio, video,
                diarization, and optional translated/background audio).
            context (grpc.ServicerContext): The gRPC servicer context.

        Yields:
            ContentLocalizationResponse: Dubbed and lip-synced video chunks.

        Raises:
            grpc.RpcError: Aborted via ``context.abort`` on any processing error.
        """
        # Assign a correlation id for this request.
        logger.debug("Creating request id.")
        request_id = str(uuid.uuid4())
        peer = context.peer() if hasattr(context, "peer") else "unknown"
        logger.debug(f"Request received | id={request_id} | peer={peer}")

        # It is the responsibility of the infer method to handle content localization
        # and yield chunks of video in the ContentLocalizationResponse format.
        try:
            logger.debug("Running content localization call.")
            yield from self.service.infer(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error processing request {request_id}: {e}\n{tb}")
            context.abort(grpc_status_for(e), f"{type(e).__name__}: {e}\n{tb}")
        logger.debug(f"Request completed | id={request_id}")
