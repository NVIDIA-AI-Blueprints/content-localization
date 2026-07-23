# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for request generators module."""

import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from client.controller.request_generators import create_controller_request_generator
from common.source_sink.grpc.audio import simulated_audio_chunk_generator
from common.source_sink.grpc.video import VideoSourceSimulator
from common.source_sink.grpc.video import simulated_asd_video_chunk_generator


@pytest.mark.unit
class TestSimulatedAudioChunkGenerator(unittest.TestCase):
    """Test cases for simulated_audio_chunk_generator function."""

    @patch("common.source_sink.grpc.audio.AudioSourceSimulator")
    def test_generator_yields_requests(self, mock_simulator_class: MagicMock) -> None:
        """Test that generator yields SpeechToSpeechRequest objects."""
        # Mock the simulator
        mock_simulator = MagicMock()
        mock_simulator.frame_rate = 16000
        mock_simulator.n_channels = 1
        mock_simulator.read.return_value = [b"audio_chunk_1", b"audio_chunk_2"]
        mock_simulator_class.return_value = mock_simulator

        # Test the generator
        requests = list(simulated_audio_chunk_generator(mock_simulator, chunk_size_secs=0.128))

        # Should yield 2 requests
        self.assertEqual(len(requests), 2)

        # Check first request
        self.assertEqual(requests[0].audio_data, b"audio_chunk_1")
        self.assertEqual(requests[0].audio_sample_rate, 16000)
        self.assertEqual(requests[0].audio_num_channels, 1)
        self.assertEqual(requests[0].audio_format, "LINEAR_PCM")

        # Check second request
        self.assertEqual(requests[1].audio_data, b"audio_chunk_2")
        self.assertEqual(requests[1].audio_sample_rate, 16000)
        self.assertEqual(requests[1].audio_num_channels, 1)
        self.assertEqual(requests[1].audio_format, "LINEAR_PCM")

    @patch("common.source_sink.grpc.audio.AudioSourceSimulator")
    def test_generator_with_different_chunk_size(self, mock_simulator_class: MagicMock) -> None:
        """Test generator with different chunk size."""
        mock_simulator = MagicMock()
        mock_simulator.frame_rate = 44100
        mock_simulator.n_channels = 2
        mock_simulator.read.return_value = [b"audio_chunk"]
        mock_simulator_class.return_value = mock_simulator

        requests = list(simulated_audio_chunk_generator(mock_simulator, chunk_size_secs=0.5))

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].audio_sample_rate, 44100)
        self.assertEqual(requests[0].audio_num_channels, 2)

    @patch("common.source_sink.grpc.audio.AudioSourceSimulator")
    def test_generator_empty_simulator(self, mock_simulator_class: MagicMock) -> None:
        """Test generator with empty simulator."""
        mock_simulator = MagicMock()
        mock_simulator.frame_rate = 16000
        mock_simulator.n_channels = 1
        mock_simulator.read.return_value = []
        mock_simulator_class.return_value = mock_simulator

        requests = list(simulated_audio_chunk_generator(mock_simulator, chunk_size_secs=0.128))

        self.assertEqual(len(requests), 0)


@pytest.mark.unit
class TestSimulatedAsdVideoChunkGenerator(unittest.TestCase):
    """Test cases for simulated_asd_video_chunk_generator function."""

    def test_generator_yields_video_requests(self) -> None:
        """Test that generator yields DetectActiveSpeakerRequest objects from file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Create test video data
            video_data = b"video_chunk_1" * 100
            tmp_file.write(video_data)
            tmp_file.flush()

            try:
                # Test with default chunk size
                simulator = VideoSourceSimulator(tmp_file.name)
                requests = list(simulated_asd_video_chunk_generator(simulator, chunk_size=64))

                # Should generate DetectActiveSpeakerRequest objects
                self.assertGreater(len(requests), 0)
                for req in requests:
                    self.assertIsInstance(req, DetectActiveSpeakerRequest)
                    self.assertNotEqual(req.data.video_data, b"")

            finally:
                Path(tmp_file.name).unlink()

    def test_generator_with_custom_chunk_size(self) -> None:
        """Test generator with custom chunk size."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            video_data = b"video_chunk_1" * 50
            tmp_file.write(video_data)
            tmp_file.flush()

            try:
                # Test with larger chunk size
                simulator = VideoSourceSimulator(tmp_file.name)
                requests = list(simulated_asd_video_chunk_generator(simulator, chunk_size=200))

                # Should generate DetectActiveSpeakerRequest objects
                self.assertGreater(len(requests), 0)
                for req in requests:
                    self.assertIsInstance(req, DetectActiveSpeakerRequest)
                    self.assertNotEqual(req.data.video_data, b"")

            finally:
                Path(tmp_file.name).unlink()

    def test_generator_empty_file(self) -> None:
        """Test generator with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                simulator = VideoSourceSimulator(tmp_file.name)
                requests = list(simulated_asd_video_chunk_generator(simulator))
                self.assertEqual(len(requests), 0)
            finally:
                Path(tmp_file.name).unlink()

    def test_generator_file_not_found(self) -> None:
        """Test generator with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            VideoSourceSimulator("nonexistent_file.mp4")


@pytest.mark.unit
class TestControllerRequestGenerator(unittest.TestCase):
    """Test cases for controller ContentLocalization request generation."""

    def test_emits_each_config_once_before_data(self) -> None:
        """ASD and LipSync configs are emitted once, before streamed data."""
        audio_source = MagicMock()
        video_source = MagicMock()
        asd_config = ActiveSpeakerDetectionConfig(
            input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV),
        )
        lipsync_config = LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3)
        s2s_config = SpeechToSpeechConfig(source_language="en", target_language="de")

        with (
            patch(
                "client.controller.request_generators.simulated_audio_chunk_generator_raw",
                return_value=iter([b"audio"]),
            ),
            patch(
                "client.controller.request_generators.simulated_video_chunk_generator_raw",
                return_value=iter([b"video"]),
            ),
        ):
            requests = list(
                create_controller_request_generator(
                    audio_source=audio_source,
                    video_source=video_source,
                    chunk_size_audio_secs=0.25,
                    chunk_size_video_bytes=1024,
                    s2s_config=s2s_config,
                    asd_config=asd_config,
                    lipsync_config=lipsync_config,
                )
            )

        self.assertEqual(
            [self._config_field(request) for request in requests[:4]],
            [
                "controller_config",
                "asd_config",
                "lipsync_config",
                "s2s_config",
            ],
        )
        self.assertEqual(
            sum(request.HasField("controller_config") for request in requests),
            1,
        )
        self.assertEqual(sum(request.HasField("asd_config") for request in requests), 1)
        self.assertEqual(sum(request.HasField("lipsync_config") for request in requests), 1)
        self.assertEqual(sum(request.HasField("s2s_config") for request in requests), 1)
        self.assertEqual(sum(bool(request.audio_data) for request in requests), 1)
        self.assertEqual(sum(bool(request.video_file_data) for request in requests), 1)

    def test_controller_config_carries_declared_input_audio_codec(self) -> None:
        """The input_audio_codec argument is sent as input_audio_config."""
        requests = self._generate_config_only_requests(input_audio_codec=AUDIO_CODEC_MP3)

        controller_config = requests[0].controller_config
        self.assertTrue(controller_config.HasField("input_audio_config"))
        self.assertEqual(controller_config.input_audio_config.encoding, AUDIO_CODEC_MP3)

    def test_controller_config_defaults_to_wav_input_audio_codec(self) -> None:
        """Omitting input_audio_codec keeps the WAV default for old callers."""
        requests = self._generate_config_only_requests()

        controller_config = requests[0].controller_config
        self.assertTrue(controller_config.HasField("input_audio_config"))
        self.assertEqual(controller_config.input_audio_config.encoding, AUDIO_CODEC_WAV)

    def test_every_message_carries_the_supplied_request_id(self) -> None:
        """Config and data messages are all stamped with the given id."""
        with (
            patch(
                "client.controller.request_generators.simulated_audio_chunk_generator_raw",
                return_value=iter([b"audio"]),
            ),
            patch(
                "client.controller.request_generators.simulated_video_chunk_generator_raw",
                return_value=iter([b"video"]),
            ),
        ):
            requests = list(
                create_controller_request_generator(
                    audio_source=MagicMock(),
                    video_source=MagicMock(),
                    chunk_size_audio_secs=0.25,
                    chunk_size_video_bytes=1024,
                    s2s_config=SpeechToSpeechConfig(source_language="en", target_language="de"),
                    asd_config=None,
                    lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
                    request_id="req-fixed-id",
                )
            )

        self.assertGreater(len(requests), 2)  # configs plus audio and video data
        for request in requests:
            self.assertEqual(request.request_id, "req-fixed-id")

    def test_default_request_id_is_a_uuid_shared_by_all_messages(self) -> None:
        """Omitting request_id generates one UUID used on every message."""
        requests = self._generate_config_only_requests()

        request_ids = {request.request_id for request in requests}
        self.assertEqual(len(request_ids), 1)
        # Raises ValueError if the generated id is not a valid UUID.
        uuid.UUID(request_ids.pop())

    @staticmethod
    def _generate_config_only_requests(**kwargs: int) -> list[ContentLocalizationRequest]:
        """Run the generator with stubbed data streams and return its requests."""
        with (
            patch(
                "client.controller.request_generators.simulated_audio_chunk_generator_raw",
                return_value=iter([]),
            ),
            patch(
                "client.controller.request_generators.simulated_video_chunk_generator_raw",
                return_value=iter([]),
            ),
        ):
            return list(
                create_controller_request_generator(
                    audio_source=MagicMock(),
                    video_source=MagicMock(),
                    chunk_size_audio_secs=0.25,
                    chunk_size_video_bytes=1024,
                    s2s_config=SpeechToSpeechConfig(source_language="en", target_language="de"),
                    asd_config=None,
                    lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
                    **kwargs,
                )
            )

    @staticmethod
    def _config_field(request: ContentLocalizationRequest) -> str | None:
        for field in ("controller_config", "asd_config", "lipsync_config", "s2s_config"):
            if request.HasField(field):
                return field
        return None


if __name__ == "__main__":
    pytest.main([__file__])
