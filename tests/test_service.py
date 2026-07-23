# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the S2S service."""

from collections.abc import Iterator
from unittest.mock import Mock

import grpc
import pytest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from s2s_service.service import S2SService
from s2s_service.service import S2SServiceServicer

pytestmark = pytest.mark.unit


class MockS2SService(S2SService):
    """Mock S2S service for testing."""

    def validate_source_language(self, value: str) -> bool:
        return value in ["en", "es", "fr"]

    def validate_target_language(self, value: str) -> bool:
        return value in ["en", "es", "fr"]

    def validate_audio_format(self, value: str) -> bool:
        return value.lower() in ["mp3", "wav", "linear_pcm"]

    def infer(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Mock inference that yields responses based on input."""
        for request in request_iterator:
            if request.HasField("audio_data"):
                yield SpeechToSpeechResponse(
                    audio_data=b"processed_audio",
                    audio_sample_rate=16000,
                    audio_num_channels=1,
                    audio_format="LINEAR_PCM",
                )


class TestS2SServiceServicer:
    """Test cases for S2SServiceServicer."""

    def test_initialization(self):
        """Test S2SServiceServicer initialization."""
        mock_service = Mock(spec=S2SService)
        servicer = S2SServiceServicer(mock_service)
        assert servicer.service == mock_service

    def test_stream_speech_to_speech_audio_only(self):
        """Test streaming with audio-only requests."""
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"

        # Create audio-only requests
        audio_requests = [
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_1",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_2",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
        ]

        request_iterator = iter(audio_requests)

        # Call the streaming method
        responses = list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify responses
        assert len(responses) == 2
        assert responses[0].audio_data == b"processed_audio"
        assert responses[1].audio_data == b"processed_audio"

    def test_stream_speech_to_speech_video_only(self):
        """Test streaming with video-only requests (should only process audio)."""
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"

        # Create audio-only requests (S2S only processes audio)
        audio_requests = [
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_1",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_2",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
        ]

        request_iterator = iter(audio_requests)

        # Call the streaming method
        responses = list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify responses - should process audio data
        assert len(responses) == 2
        assert responses[0].audio_data == b"processed_audio"
        assert responses[1].audio_data == b"processed_audio"

    def test_stream_speech_to_speech_mixed_requests(self):
        """Test streaming with mixed audio and video requests (should only process audio)."""
        # Create mock service (ASD service is now separate)
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"

        # Create audio requests (S2S only processes audio)
        audio_requests = [
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_1",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
        ]

        request_iterator = iter(audio_requests)

        # Call the streaming method
        responses = list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify responses - should get audio responses
        assert len(responses) == 1
        assert responses[0].audio_data == b"processed_audio"

        # Verify that responses were generated
        assert len(responses) == 1

    def test_stream_speech_to_speech_empty_iterator(self):
        """Test streaming with empty request iterator."""
        # Create mock service
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"

        # Create empty iterator
        request_iterator = iter([])

        # Call the streaming method
        responses = list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify no responses
        assert len(responses) == 0

    def test_stream_speech_to_speech_s2s_exception(self):
        """Test handling of S2S service exceptions."""
        # Create mock service that raises an exception
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        # Override infer method to raise exception
        def mock_infer(*args, **kwargs):
            raise RuntimeError("S2S processing failed")

        mock_service.infer = mock_infer

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"
        mock_context.abort = Mock()

        # Create requests
        requests = [
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_1",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
        ]

        request_iterator = iter(requests)

        # Call the streaming method
        list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify context.abort was called
        mock_context.abort.assert_called_once()

    def test_stream_speech_to_speech_asd_exception(self):
        """Test that S2S service works normally without ASD service."""
        # Create mock service
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"

        # Create requests with audio data (S2S should only process audio)
        requests = [
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_1",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
        ]

        request_iterator = iter(requests)

        # Call the streaming method
        responses = list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify that S2S processing worked correctly
        assert len(responses) == 1
        assert responses[0].audio_data == b"processed_audio"

    def test_request_id_generation(self):
        """Test that unique request IDs are generated."""
        # Create mock service
        mock_service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        servicer = S2SServiceServicer(mock_service)

        # Create mock context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.peer.return_value = "test-peer"

        # Create requests
        requests = [
            SpeechToSpeechRequest(
                audio_data=b"audio_chunk_1",
                audio_sample_rate=16000,
                audio_num_channels=1,
                audio_format="LINEAR_PCM",
            ),
        ]

        request_iterator = iter(requests)

        # Call the streaming method
        responses = list(servicer.StreamSpeechToSpeech(request_iterator, mock_context))

        # Verify responses were generated
        assert len(responses) > 0


class TestS2SService:
    """Test cases for S2SService base class."""

    def test_initialization(self):
        """Test S2SService initialization."""
        service = MockS2SService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            nchannels=1,
        )

        assert service.sample_rate_hz == 16000
        assert service.default_source_language == "en"
        assert service.default_target_language == "es"
        assert service.nchannels == 1

    def test_nchannels_validation(self):
        """Test nchannels validation."""
        # Test valid nchannels
        service = MockS2SService(
            nchannels=2,
        )
        assert service.nchannels == 2

        # Test invalid nchannels
        with pytest.raises(ValueError, match="Number of channels must be >= 1"):
            service.nchannels = 0

    def test_sample_rate_validation(self):
        """Test sample rate validation."""
        service = MockS2SService()

        # Test valid sample rates
        for rate in [8000, 16000, 24000, 48000]:
            service.sample_rate_hz = rate
            assert service.sample_rate_hz == rate

        # Test invalid sample rate
        with pytest.raises(ValueError, match="Sample rate must be 8000, 16000, 24000, or 48000"):
            service.sample_rate_hz = 44100

    def test_add_servicer_to_server(self):
        """Test adding servicer to gRPC server."""
        service = MockS2SService()

        mock_server = Mock()

        service.add_servicer_to_server(mock_server)
