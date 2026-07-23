# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for client main module."""

from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest

from client.direct.app import main


def _configure_mock_client(mock_client, responses, consume_request_iterator=False):
    client_instance = MagicMock()

    def _run_client(*args, **kwargs):
        request_iterator = kwargs.get("request_iterator")
        output_buffer = kwargs.get("output_buffer")
        if request_iterator is None and args:
            request_iterator = args[0]
        if output_buffer is None and len(args) > 1:
            output_buffer = args[1]
        if consume_request_iterator and request_iterator is not None:
            list(request_iterator)
        for response in responses:
            output_buffer.put(response)
        output_buffer.done = True

    client_instance.side_effect = _run_client
    mock_client.return_value = client_instance
    return client_instance


@pytest.mark.unit
class TestMain:
    """Test cases for main function."""

    def setup_method(self):
        """Set up test environment by creating necessary directories."""
        import os

        # Create outputs directory for tests
        os.makedirs("outputs", exist_ok=True)

    @patch("client.direct.app.VideoSinkSimulator")
    @patch("client.direct.app.VideoSourceSimulator")
    @patch("common.source_sink.grpc.video.simulated_video_chunk_generator_raw")
    @patch("client.direct.app.AudioSinkSimulator")
    @patch("client.direct.app.AudioSourceSimulator")
    @patch("client.direct.app.simulated_audio_chunk_generator")
    @patch("client.direct.app.LipsyncClient")
    @patch("client.direct.app.ActiveSpeakerDetectionClient")
    @patch("client.direct.app.SpeechToSpeechClient")
    @patch("client.direct.app.check_service_health")
    @patch("client.direct.app.load_diarization_info")
    def test_main_happy_path(
        self,
        mock_load_diarization,
        mock_health,
        mock_s2s_client,
        mock_asd_client,
        mock_lipsync_client,
        mock_audio_gen,
        mock_source_sim,
        mock_sink_sim,
        mock_video_gen,
        mock_video_source_sim,
        mock_video_sink_sim,
        tmp_path,
    ):
        """Test main function with successful execution."""
        # Mock diarization so bypass_asd stays False and ASD runs
        mock_load_diarization.return_value = MagicMock(segments=[])

        # Mock health check
        mock_health.return_value = True

        # Mock AudioSourceSimulator
        mock_source_instance = MagicMock()
        mock_source_instance.frame_rate = 16000
        mock_source_instance.sample_width = 2
        mock_source_instance.n_channels = 1
        mock_source_instance.n_frames = 44100
        mock_source_instance.header = b"mock_header"
        mock_source_instance.ledger = {0: 1000.0, 1: 1001.0, 2: 1002.0}
        mock_source_sim.return_value = mock_source_instance

        # Mock AudioSinkSimulator

        mock_sink_instance = MagicMock()
        mock_sink_instance.ledger = {0: 1000.5, 1: 1001.5, 2: 1002.5}
        mock_sink_sim.return_value = mock_sink_instance

        # Mock VideoSourceSimulator
        mock_video_source_instance = MagicMock()
        mock_video_source_sim.return_value = mock_video_source_instance

        # Mock VideoSinkSimulator
        mock_video_sink_instance = MagicMock()
        mock_video_sink_sim.return_value = mock_video_sink_instance

        # Mock S2S responses
        mock_s2s_response1 = MagicMock()
        mock_s2s_response1.audio_data = b"audio1"
        mock_s2s_response1.audio_format = "mp3"
        mock_s2s_response1.HasField.return_value = False
        mock_s2s_response2 = MagicMock()
        mock_s2s_response2.audio_data = b"audio2"
        mock_s2s_response2.audio_format = "mp3"
        mock_s2s_response2.HasField.return_value = False
        _configure_mock_client(
            mock_s2s_client,
            [mock_s2s_response1, mock_s2s_response2],
            consume_request_iterator=True,
        )

        # Mock ASD responses
        mock_asd_response = MagicMock()
        mock_asd_response.active_speaker_detection_result.speaker_data = []
        mock_asd_response.confidences = []
        _configure_mock_client(mock_asd_client, [mock_asd_response])

        # Mock LipSync responses
        mock_lipsync_response = MagicMock()
        mock_lipsync_response.audio_file_data = b"audio1"
        mock_lipsync_response.video_file_data = b"video1"
        _configure_mock_client(mock_lipsync_client, [mock_lipsync_response])

        # Mock generators
        mock_audio_gen.return_value = [b"audio_chunk"]
        mock_video_gen.return_value = [b"video_chunk"]

        # Run main. The diarization file must exist because
        # DirectPipelineConfig.validate_io checks it at startup.
        import sys
        from unittest.mock import patch as patch_sys_argv

        diarization_file = tmp_path / "diar.json"
        diarization_file.write_text("[]")
        with patch_sys_argv.object(
            sys,
            "argv",
            ["client.py", "--diarization-file", str(diarization_file)],
        ):
            main()

        # Diarization provided → bypass_asd=False → ASD runs → audio opened twice.
        if mock_source_sim.call_count != 2:
            raise AssertionError(
                f"Expected AudioSourceSimulator to be called 2 times, got {mock_source_sim.call_count}"
            )
        if mock_source_sim.call_args_list != [
            call(file_path="assets/sample_audio.wav"),
            call(file_path="assets/sample_audio.wav"),
        ]:
            raise AssertionError(
                "Expected AudioSourceSimulator to be called twice with sample audio path"
            )

        # Check that AudioSinkSimulator was created with correct parameters
        mock_sink_sim.assert_called_once_with(
            frame_rate=mock_source_instance.frame_rate,
            sample_width=mock_source_instance.sample_width,
            n_channels=mock_source_instance.n_channels,
            n_frames=mock_source_instance.n_frames,
            file_path="outputs/sample_audio_output.mp3",
            chunk_duration_secs=1,
            audio_format="mp3",
        )

        # Check that generator was called with correct parameters
        mock_audio_gen.assert_called_once_with(simulator=mock_source_instance, chunk_size_secs=1)

        # Check that sink write method was called for the response with audio data
        # Note: Current implementation doesn't write audio directly, only processes LipSync
        # So we don't expect the write method to be called
        # mock_sink_instance.write.assert_called_once_with(b"audio1")

    @patch("client.direct.app.check_service_health")
    def test_main_health_check_failure(self, mock_health):
        """Test main function when health check fails."""
        # Mock health check to raise an exception
        mock_health.side_effect = ConnectionError("Service not healthy")

        # Mock sys.argv to avoid argument parsing errors
        with patch("sys.argv", ["client.py"]):
            # Run main and expect it to exit with the exception
            with pytest.raises(ConnectionError, match="Service not healthy"):
                main()

    @patch("client.direct.app.AudioSourceSimulator")
    @patch("client.direct.app.check_service_health")
    def test_main_source_simulator_failure(self, mock_health, mock_source_sim):
        """Test main function when AudioSourceSimulator fails."""
        # Mock health check to pass
        mock_health.return_value = True

        # Mock AudioSourceSimulator to raise an exception
        mock_source_sim.side_effect = FileNotFoundError("Input file not found")

        # Mock sys.argv to avoid argument parsing errors
        with patch("sys.argv", ["client.py"]):
            # Run main and expect it to exit with the exception
            with pytest.raises(FileNotFoundError, match="Input file not found"):
                main()

    @patch("client.direct.app.AudioSinkSimulator")
    @patch("client.direct.app.AudioSourceSimulator")
    @patch("client.direct.app.check_service_health")
    def test_main_sink_simulator_failure(self, mock_health, mock_source_sim, mock_sink_sim):
        """Test main function when AudioSinkSimulator fails."""
        # Mock health check to pass
        mock_health.return_value = True

        # Mock AudioSourceSimulator to succeed
        mock_source_instance = MagicMock()
        mock_source_instance.frame_rate = 16000
        mock_source_instance.sample_width = 2
        mock_source_instance.n_channels = 1
        mock_source_instance.n_frames = 44100
        mock_source_sim.return_value = mock_source_instance

        # Mock AudioSinkSimulator to raise an exception
        mock_sink_sim.side_effect = FileNotFoundError("Output directory not found")

        # Mock sys.argv to avoid argument parsing errors
        with patch("sys.argv", ["client.py"]):
            # Run main and expect it to exit with the exception
            with pytest.raises(FileNotFoundError, match="Output directory not found"):
                main()

    @patch("client.direct.app.AudioSinkSimulator")
    @patch("client.direct.app.AudioSourceSimulator")
    @patch("client.direct.app.simulated_audio_chunk_generator")
    @patch("client.direct.app.LipsyncClient")
    @patch("client.direct.app.ActiveSpeakerDetectionClient")
    @patch("client.direct.app.SpeechToSpeechClient")
    @patch("client.direct.app.check_service_health")
    def test_main_grpc_failure(
        self,
        mock_health,
        mock_s2s_client,
        mock_asd_client,
        mock_lipsync_client,
        mock_gen,
        mock_source_sim,
        mock_sink_sim,
    ):
        """Test main function when gRPC call fails."""
        # Mock health check to pass
        mock_health.return_value = True

        # Mock AudioSourceSimulator
        mock_source_instance = MagicMock()
        mock_source_instance.frame_rate = 16000
        mock_source_instance.sample_width = 2
        mock_source_instance.n_channels = 1
        mock_source_instance.n_frames = 44100
        mock_source_sim.return_value = mock_source_instance

        # Mock AudioSinkSimulator
        mock_sink_instance = MagicMock()
        mock_sink_sim.return_value = mock_sink_instance

        # Mock generator
        mock_gen.return_value = [b"chunk"]

        # Mock client to produce no responses and finish
        _configure_mock_client(mock_s2s_client, [], consume_request_iterator=True)
        _configure_mock_client(mock_asd_client, [])
        _configure_mock_client(mock_lipsync_client, [])

        # Mock sys.argv to avoid argument parsing errors
        with patch("sys.argv", ["client.py"]):
            main()

    @patch("client.direct.app.VideoSinkSimulator")
    @patch("client.direct.app.VideoSourceSimulator")
    @patch("common.source_sink.grpc.video.simulated_video_chunk_generator_raw")
    @patch("client.direct.app.AudioSinkSimulator")
    @patch("client.direct.app.AudioSourceSimulator")
    @patch("client.direct.app.simulated_audio_chunk_generator")
    @patch("client.direct.app.LipsyncClient")
    @patch("client.direct.app.ActiveSpeakerDetectionClient")
    @patch("client.direct.app.SpeechToSpeechClient")
    @patch("client.direct.app.check_service_health")
    @patch("client.direct.app.load_diarization_info")
    def test_main_with_custom_arguments(
        self,
        mock_load_diarization,
        mock_health,
        mock_s2s_client,
        mock_asd_client,
        mock_lipsync_client,
        mock_audio_gen,
        mock_source_sim,
        mock_sink_sim,
        mock_video_gen,
        mock_video_source_sim,
        mock_video_sink_sim,
        tmp_path,
    ):
        """Test main function with custom command line arguments."""
        # Mock diarization so bypass_asd stays False and ASD runs
        mock_load_diarization.return_value = MagicMock(segments=[])

        # Mock health check
        mock_health.return_value = True

        # Mock AudioSourceSimulator
        mock_source_instance = MagicMock()
        mock_source_instance.frame_rate = 44100
        mock_source_instance.sample_width = 2
        mock_source_instance.n_channels = 2
        mock_source_instance.n_frames = 88200
        mock_source_instance.header = b"mock_header"
        mock_source_instance.ledger = {0: 1000.0}
        mock_source_sim.return_value = mock_source_instance

        # Mock AudioSinkSimulator
        mock_sink_instance = MagicMock()
        mock_sink_instance.ledger = {0: 1000.5}
        mock_sink_sim.return_value = mock_sink_instance

        # Mock VideoSourceSimulator
        mock_video_source_instance = MagicMock()
        mock_video_source_sim.return_value = mock_video_source_instance

        # Mock VideoSinkSimulator
        mock_video_sink_instance = MagicMock()
        mock_video_sink_sim.return_value = mock_video_sink_instance

        # Mock S2S responses
        mock_s2s_response = MagicMock()
        mock_s2s_response.audio_data = b"audio1"
        mock_s2s_response.audio_format = "wav"
        mock_s2s_response.audio_num_channels = 2
        mock_s2s_response.audio_sample_rate = 44100
        mock_s2s_response.HasField.return_value = False
        _configure_mock_client(mock_s2s_client, [mock_s2s_response], consume_request_iterator=True)

        # Mock ASD responses
        mock_asd_response = MagicMock()
        mock_asd_response.active_speaker_detection_result.speaker_data = []
        mock_asd_response.confidences = []
        _configure_mock_client(mock_asd_client, [mock_asd_response])

        # Mock LipSync responses
        mock_lipsync_response = MagicMock()
        mock_lipsync_response.audio_file_data = b"audio1"
        mock_lipsync_response.video_file_data = b"video1"
        _configure_mock_client(mock_lipsync_client, [mock_lipsync_response])

        # Mock generators
        mock_audio_gen.return_value = [b"audio_chunk"]
        mock_video_gen.return_value = [b"video_chunk"]

        # Run main with custom arguments. Input and diarization files must
        # exist because DirectPipelineConfig.validate_io checks them.
        import sys
        from unittest.mock import patch as patch_sys_argv

        input_audio = tmp_path / "custom_input.wav"
        input_audio.write_bytes(b"RIFF")
        output_audio = tmp_path / "custom_output.wav"
        diarization_file = tmp_path / "diar.json"
        diarization_file.write_text("[]")
        with patch_sys_argv.object(
            sys,
            "argv",
            [
                "client.py",
                "--s2s-server",
                "custom-server:50050",
                "--input-audio",
                str(input_audio),
                "--output-audio",
                str(output_audio),
                "--chunk-size-audio-secs",
                "0.5",
                "--diarization-file",
                str(diarization_file),
            ],
        ):
            main()

        # Diarization provided → bypass_asd=False → ASD runs → audio opened twice.
        if mock_source_sim.call_count != 2:
            raise AssertionError(
                f"Expected AudioSourceSimulator to be called 2 times, got {mock_source_sim.call_count}"
            )
        if mock_source_sim.call_args_list != [
            call(file_path=str(input_audio)),
            call(file_path=str(input_audio)),
        ]:
            raise AssertionError(
                "Expected AudioSourceSimulator to be called twice with custom input path"
            )

        # Check that AudioSinkSimulator was created with custom parameters
        mock_sink_sim.assert_called_once_with(
            frame_rate=mock_source_instance.frame_rate,
            sample_width=mock_source_instance.sample_width,
            n_channels=mock_source_instance.n_channels,
            n_frames=mock_source_instance.n_frames,
            file_path=str(output_audio),
            chunk_duration_secs=0.5,
            audio_format="wav",
        )

        # Check that generator was called with custom chunk size
        mock_audio_gen.assert_called_once_with(simulator=mock_source_instance, chunk_size_secs=0.5)

        # Check that sink write method was called for the response with audio data
        # Note: Current implementation doesn't write audio directly, only processes LipSync
        # So we don't expect the write method to be called
        # mock_sink_instance.write.assert_called_once_with(b"audio1")

    @patch("client.direct.app.VideoSinkSimulator")
    @patch("client.direct.app.VideoSourceSimulator")
    @patch("common.source_sink.grpc.video.simulated_video_chunk_generator_raw")
    @patch("client.direct.app.AudioSinkSimulator")
    @patch("client.direct.app.AudioSourceSimulator")
    @patch("client.direct.app.simulated_audio_chunk_generator")
    @patch("client.direct.app.LipsyncClient")
    @patch("client.direct.app.ActiveSpeakerDetectionClient")
    @patch("client.direct.app.SpeechToSpeechClient")
    @patch("client.direct.app.check_service_health")
    def test_main_with_video_output(
        self,
        mock_health,
        mock_s2s_client,
        mock_asd_client,
        mock_lipsync_client,
        mock_audio_gen,
        mock_source_sim,
        mock_sink_sim,
        mock_video_gen,
        mock_video_source_sim,
        mock_video_sink_sim,
    ):
        """Test main function with a bare output video filename."""
        # Mock health check
        mock_health.return_value = True

        # Mock AudioSourceSimulator
        mock_source_instance = MagicMock()
        mock_source_instance.frame_rate = 16000
        mock_source_instance.sample_width = 2
        mock_source_instance.n_channels = 1
        mock_source_instance.n_frames = 44100
        mock_source_instance.header = b"mock_header"
        mock_source_instance.ledger = {0: 1000.0}
        mock_source_sim.return_value = mock_source_instance

        # Mock AudioSinkSimulator
        mock_sink_instance = MagicMock()
        mock_sink_instance.ledger = {0: 1000.5}
        mock_sink_sim.return_value = mock_sink_instance

        # Mock VideoSourceSimulator
        mock_video_source_instance = MagicMock()
        mock_video_source_sim.return_value = mock_video_source_instance

        # Mock VideoSinkSimulator
        mock_video_sink_instance = MagicMock()
        mock_video_sink_sim.return_value = mock_video_sink_instance

        # Mock S2S responses
        mock_s2s_response = MagicMock()
        mock_s2s_response.audio_data = b"audio1"
        mock_s2s_response.audio_format = "mp3"
        mock_s2s_response.HasField.return_value = False
        _configure_mock_client(mock_s2s_client, [mock_s2s_response], consume_request_iterator=True)

        # Mock ASD responses
        mock_asd_response = MagicMock()
        mock_asd_response.active_speaker_detection_result.speaker_data = []
        mock_asd_response.confidences = []
        _configure_mock_client(mock_asd_client, [mock_asd_response])

        # Mock LipSync responses
        mock_lipsync_response = MagicMock()
        mock_lipsync_response.audio_file_data = b"audio1"
        mock_lipsync_response.video_file_data = b"video1"
        _configure_mock_client(mock_lipsync_client, [mock_lipsync_response])

        # Mock generators
        mock_audio_gen.return_value = [b"audio_chunk"]
        mock_video_gen.return_value = [b"video_chunk"]

        # Run main with a bare output filename: validate_io must accept a
        # path with no directory component.
        import sys
        from unittest.mock import patch as patch_sys_argv

        with patch_sys_argv.object(
            sys,
            "argv",
            ["client.py", "--output-mp4", "output.mp4"],
        ):
            main()

        # The bare filename reaches the video sink unchanged.
        mock_video_sink_sim.assert_called_once_with(
            file_path="output.mp4",
            chunk_size=1024 * 1024,
        )


if __name__ == "__main__":
    pytest.main([__file__])
