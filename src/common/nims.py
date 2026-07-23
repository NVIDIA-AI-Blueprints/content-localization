# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIMs and tools for interacting with NIMs."""

from collections.abc import Iterator
from typing import Any

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2_grpc import (
    ActiveSpeakerDetectionServiceStub,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.lipsync.v1.lipsync_pb2_grpc import LipSyncServiceStub
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse
from nvidia.ai4m.s2s.v1.s2s_pb2_grpc import SpeechToSpeechStub

from common.base_utils import logger
from common.buffers import Buffer
from common.clients import Client
from common.service import GRPCInferenceHandle


class SpeechToSpeechHandle(GRPCInferenceHandle):
    """Speech to Speech NIM service handle."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the SpeechToSpeechHandle.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.
            health_check_service (str): The health check service to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=SpeechToSpeechStub,
            channel_credentials=channel_credentials,
        )

    def get_response_iterator(
        self, request_iterator: Iterator[SpeechToSpeechRequest]
    ) -> Iterator[SpeechToSpeechResponse]:
        """Get a response iterator from the Speech to Speech service.

        Args:
            request_iterator (Iterator[Any]): The request iterator.

        """
        return self.stub.StreamSpeechToSpeech(request_iterator)


class ActiveSpeakerDetectionHandle(GRPCInferenceHandle):
    """Active Speaker Detection NIM service handle."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the ActiveSpeakerDetectionHandle.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.
            health_check_service (str): The health check service to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=ActiveSpeakerDetectionServiceStub,
            channel_credentials=channel_credentials,
        )

    def get_response_iterator(
        self, request_iterator: Iterator[DetectActiveSpeakerRequest]
    ) -> Iterator[DetectActiveSpeakerResponse]:
        """Get a response iterator from the Active Speaker Detection service.

        Args:
            request_iterator (Iterator[Any]): The request iterator.
        """
        return self.stub.DetectActiveSpeaker(request_iterator)


class LipsyncHandle(GRPCInferenceHandle):
    """Lipsync NIM service handle."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the LipsyncHandle.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.
            health_check_service (str): The health check service to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=LipSyncServiceStub,
            channel_credentials=channel_credentials,
        )

    def get_response_iterator(
        self, request_iterator: Iterator[LipsyncRequest]
    ) -> Iterator[LipsyncResponse]:
        """Get a response iterator from the LipSync service.

        Args:
            request_iterator (Iterator[Any]): The request iterator.
        """
        return self.stub.Lipsync(request_iterator)


class SpeechToSpeechClient(Client[SpeechToSpeechRequest, SpeechToSpeechResponse]):
    """Client that streams non-keepalive S2S responses into an output buffer."""

    def _impl(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        output_buffer: Buffer[SpeechToSpeechResponse],
        context: grpc.ServicerContext,
        request_id: str,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Starting SpeechToSpeech client for request_id={request_id}")
        if self.handle.stub is None:
            self.handle.connect()
        response_iterator = self.handle.get_response_iterator(request_iterator=request_iterator)
        # Errors propagate to Client.__call__, which owns logging and the single
        # context abort.
        for response in response_iterator:
            if response.HasField("keepalive"):
                logger.debug("SpeechToSpeech client: skipping keep-alive response")
                continue
            output_buffer.put(response)


class ActiveSpeakerDetectionClient(Client[DetectActiveSpeakerRequest, DetectActiveSpeakerResponse]):
    """Client that streams ASD detection results into an output buffer."""

    def _impl(
        self,
        request_iterator: Iterator[DetectActiveSpeakerRequest],
        output_buffer: Buffer[DetectActiveSpeakerResponse],
        context: grpc.ServicerContext,
        request_id: str,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Starting ActiveSpeakerDetection client for request_id={request_id}")
        if self.handle.stub is None:
            self.handle.connect()
        response_iterator = self.handle.get_response_iterator(request_iterator=request_iterator)
        result_count = 0
        keepalive_count = 0
        config_count = 0
        # Errors propagate to Client.__call__, which owns logging and the single
        # context abort; the finally logs the count summary even when the stream
        # ends early.
        try:
            for response in response_iterator:
                if response.HasField("keepalive"):
                    keepalive_count += 1
                    continue
                if response.HasField("config"):
                    config_count += 1
                    continue
                result_count += 1
                output_buffer.put(response)
        finally:
            logger.info(
                f"ASD client finished: results={result_count},"
                f" keepalives={keepalive_count}, configs={config_count}"
            )


class LipsyncClient(Client[LipsyncRequest, LipsyncResponse]):
    """Client that streams LipSync responses into an output buffer.

    Keepalive responses are passed through so downstream consumers can
    forward them and keep their own streams alive while LipSync waits for
    input (for example during a long-running dubbing job).
    """

    def _impl(
        self,
        request_iterator: Iterator[LipsyncRequest],
        output_buffer: Buffer[LipsyncResponse],
        context: grpc.ServicerContext,
        request_id: str,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Starting LipSync client for request_id={request_id}")
        if self.handle.stub is None:
            self.handle.connect()
        response_iterator = self.handle.get_response_iterator(request_iterator=request_iterator)
        # Errors propagate to Client.__call__, which owns logging and the single
        # context abort.
        for response in response_iterator:
            output_buffer.put(response)
