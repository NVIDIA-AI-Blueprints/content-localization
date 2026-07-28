# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ElevenLabs Dubbing S2S service."""

import os
import tempfile
import unittest
import wave
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

pytestmark = pytest.mark.unit


class DummyContext:
    """Dummy gRPC context for testing."""

    def __init__(self) -> None:
        self.aborted = False
        self.abort_args = None

    def abort(self, code, msg) -> NoReturn:
        self.aborted = True
        self.abort_args = (code, msg)
        raise Exception(f"Aborted: {code}, {msg}")

    def peer(self) -> str:
        return "test-peer"


def create_test_wav_file(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a test WAV file.

    Args:
        duration_seconds: Duration of the audio file in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the temporary WAV file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    # Generate test audio (simple sine wave)
    frequency = 440  # Hz
    duration = duration_seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767
    audio_data = audio_data.astype(np.int16)

    # Write WAV file
    with wave.open(temp_file.name, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())

    return temp_file.name


class TestELDubbingUtilities(unittest.TestCase):
    """Test cases for ElevenLabs dubbing utility functions."""

    @patch("s2s_service.el_utils.dubbing.client")
    def test_download_dubbed_file(self, mock_client):
        """Test downloading dubbed file from ElevenLabs."""
        from s2s_service.el_utils.dubbing import download_dubbed_file

        # Mock the dubbing audio get method
        mock_dubbing = MagicMock()
        mock_audio = MagicMock()
        mock_audio.get.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        mock_dubbing.audio = mock_audio
        mock_client.dubbing = mock_dubbing

        # Test download
        dubbing_id = "test-dubbing-id"
        language_code = "es"

        chunks = list(download_dubbed_file(dubbing_id, language_code))

        # Verify chunks
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], b"chunk1")
        self.assertEqual(chunks[1], b"chunk2")
        self.assertEqual(chunks[2], b"chunk3")

        # Verify API was called correctly
        mock_audio.get.assert_called_once_with(dubbing_id, language_code)

    @patch("s2s_service.el_utils.dubbing.client")
    def test_download_dubbed_file_error(self, mock_client):
        """Test error handling in download_dubbed_file."""
        from s2s_service.el_utils.dubbing import download_dubbed_file

        # Mock the dubbing audio get method to raise an error
        mock_dubbing = MagicMock()
        mock_audio = MagicMock()
        mock_audio.get.side_effect = Exception("Download failed")
        mock_dubbing.audio = mock_audio
        mock_client.dubbing = mock_dubbing

        # Test download with error
        with self.assertRaises(Exception) as context:
            list(download_dubbed_file("test-id", "es"))

        self.assertIn("Download failed", str(context.exception))

    @patch("s2s_service.el_utils.dubbing.time.sleep")
    @patch("s2s_service.el_utils.dubbing.client")
    def test_wait_for_dubbing_completion_success(self, mock_client, mock_sleep):
        """Test waiting for dubbing completion - success case."""
        from s2s_service.el_utils.dubbing import wait_for_dubbing_completion

        # Mock the dubbing get method
        mock_dubbing = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.status = "dubbed"
        mock_dubbing.get.return_value = mock_metadata
        mock_client.dubbing = mock_dubbing

        # Test waiting
        result = wait_for_dubbing_completion("test-dubbing-id")

        # Verify result
        self.assertTrue(result)
        mock_dubbing.get.assert_called_with("test-dubbing-id")

    @patch("s2s_service.el_utils.dubbing.time.sleep")
    @patch("s2s_service.el_utils.dubbing.client")
    def test_wait_for_dubbing_completion_in_progress(self, mock_client, mock_sleep):
        """Test waiting for dubbing completion - in progress then success."""
        from s2s_service.el_utils.dubbing import wait_for_dubbing_completion

        # Mock the dubbing get method
        mock_dubbing = MagicMock()
        mock_metadata_dubbing = MagicMock()
        mock_metadata_dubbing.status = "dubbing"
        mock_metadata_done = MagicMock()
        mock_metadata_done.status = "dubbed"

        # Return "dubbing" twice, then "dubbed"
        mock_dubbing.get.side_effect = [
            mock_metadata_dubbing,
            mock_metadata_dubbing,
            mock_metadata_done,
        ]
        mock_client.dubbing = mock_dubbing

        # Test waiting
        result = wait_for_dubbing_completion("test-dubbing-id")

        # Verify result
        self.assertTrue(result)
        self.assertEqual(mock_dubbing.get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("s2s_service.el_utils.dubbing.time.sleep")
    @patch("s2s_service.el_utils.dubbing.client")
    def test_wait_for_dubbing_completion_failed(self, mock_client, mock_sleep):
        """Test waiting for dubbing completion - failed case."""
        from s2s_service.el_utils.dubbing import wait_for_dubbing_completion

        # Mock the dubbing get method
        mock_dubbing = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.status = "failed"
        mock_metadata.error = "Dubbing failed"
        mock_dubbing.get.return_value = mock_metadata
        mock_client.dubbing = mock_dubbing

        # Test waiting
        result = wait_for_dubbing_completion("test-dubbing-id")

        # Verify result
        self.assertFalse(result)

    @patch("s2s_service.el_utils.dubbing.time.sleep")
    @patch("s2s_service.el_utils.dubbing.client")
    def test_wait_for_dubbing_completion_timeout(self, mock_client, mock_sleep):
        """Test waiting for dubbing completion - timeout case."""
        from s2s_service.el_utils.dubbing import wait_for_dubbing_completion

        # Mock the dubbing get method to always return "dubbing"
        mock_dubbing = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.status = "dubbing"
        mock_dubbing.get.return_value = mock_metadata
        mock_client.dubbing = mock_dubbing

        # Test waiting (will timeout)
        result = wait_for_dubbing_completion("test-dubbing-id")

        # Verify result
        self.assertFalse(result)
        # Should have tried MAX_ATTEMPTS times
        self.assertEqual(mock_dubbing.get.call_count, 120)

    @patch("s2s_service.el_utils.dubbing.download_dubbed_file")
    @patch("s2s_service.el_utils.dubbing.wait_for_dubbing_completion")
    @patch("s2s_service.el_utils.dubbing.client")
    @patch("os.path.isfile", return_value=True)
    def test_create_dub_from_file_success(self, mock_isfile, mock_client, mock_wait, mock_download):
        """Test creating a dub from file - success case."""
        from s2s_service.el_utils.dubbing import create_dub_from_file

        # Create a test file
        test_file = Path(create_test_wav_file())

        try:
            # Mock the dubbing create method
            mock_dubbing = MagicMock()
            mock_response = MagicMock()
            mock_response.dubbing_id = "test-dubbing-id"
            mock_dubbing.create.return_value = mock_response
            mock_client.dubbing = mock_dubbing

            # Mock wait for completion
            mock_wait.return_value = True

            # Mock download
            mock_download.return_value = iter([b"chunk1", b"chunk2"])

            # Test create dub
            chunks = list(
                create_dub_from_file(test_file, source_language="en", target_language="es")
            )

            # Verify chunks
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0], b"chunk1")
            self.assertEqual(chunks[1], b"chunk2")

            # Verify API calls
            mock_dubbing.create.assert_called_once()
            mock_wait.assert_called_once_with(dubbing_id="test-dubbing-id")
            mock_download.assert_called_once_with(dubbing_id="test-dubbing-id", language_code="es")
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()

    @patch("s2s_service.el_utils.dubbing.wait_for_dubbing_completion")
    @patch("s2s_service.el_utils.dubbing.client")
    @patch("os.path.isfile", return_value=True)
    def test_create_dub_from_file_dubbing_failed(self, mock_isfile, mock_client, mock_wait):
        """Test creating a dub from file - dubbing failed case."""
        from s2s_service.el_utils.dubbing import create_dub_from_file

        # Create a test file
        test_file = Path(create_test_wav_file())

        try:
            # Mock the dubbing create method
            mock_dubbing = MagicMock()
            mock_response = MagicMock()
            mock_response.dubbing_id = "test-dubbing-id"
            mock_dubbing.create.return_value = mock_response
            mock_client.dubbing = mock_dubbing

            # Mock wait for completion to return False
            mock_wait.return_value = False

            # Test create dub (should raise exception)
            with self.assertRaises(Exception) as context:
                list(create_dub_from_file(test_file, source_language="en", target_language="es"))

            self.assertIn("Dubbing failed", str(context.exception))
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()

    @patch("os.path.isfile", return_value=False)
    def test_create_dub_from_file_not_found(self, mock_isfile):
        """Test creating a dub from file - file not found case."""
        from s2s_service.el_utils.dubbing import create_dub_from_file

        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            list(
                create_dub_from_file(
                    Path("/nonexistent/file.wav"), source_language="en", target_language="es"
                )
            )


@patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-api-key"})
class TestELDubbingService(unittest.TestCase):
    """Test cases for ElevenLabs Dubbing S2S service."""

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    def test_initialization(self, mock_elevenlabs, mock_client):
        """Test ELDubbingService initialization."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService(
            sample_rate_hz=16000,
            default_source_language="en",
            default_target_language="es",
            audio_format="mp3",
        )

        # Verify initialization
        self.assertEqual(service.sample_rate_hz, 16000)
        self.assertEqual(service.default_source_language, "en")
        self.assertEqual(service.default_target_language, "es")
        self.assertEqual(service.audio_format, "mp3")

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    def test_supported_languages(self, mock_elevenlabs, mock_client):
        """Test supported language validation."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService()

        # Test valid source languages
        self.assertTrue(service.validate_source_language("en"))
        self.assertTrue(service.validate_source_language("es"))
        self.assertTrue(service.validate_source_language("fr"))
        self.assertTrue(service.validate_source_language("auto"))

        # Test invalid source language
        self.assertFalse(service.validate_source_language("invalid"))

        # Test valid target languages (same as source)
        self.assertTrue(service.validate_target_language("en"))
        self.assertTrue(service.validate_target_language("es"))

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    def test_audio_format_validation(self, mock_elevenlabs, mock_client):
        """Test audio format validation."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService()

        # Test valid audio format
        self.assertTrue(service.validate_audio_format("mp3"))

        # Test invalid audio formats
        self.assertFalse(service.validate_audio_format("wav"))
        self.assertFalse(service.validate_audio_format("flac"))

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    @patch("s2s_service.el_utils.dubbing.create_dub_from_file")
    @patch("s2s_service.el_utils.dubbing.download_audio_file_from_iterator")
    @patch("os.remove")
    @patch("os.path.getsize", return_value=1024)
    def test_impl_success(
        self,
        mock_getsize,
        mock_remove,
        mock_download,
        mock_create_dub,
        mock_elevenlabs,
        mock_client,
    ):
        """Test _impl method - success case."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService()
        context = DummyContext()

        # Create a test MP3 file to simulate downloaded dubbing
        test_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        test_mp3.write(b"fake mp3 data" * 100)
        test_mp3.close()

        try:
            # Mock create_dub_from_file to return chunks
            mock_create_dub.return_value = iter([b"chunk1", b"chunk2"])

            # Mock download to return the test file path
            mock_download.return_value = test_mp3.name

            # Create test input file
            input_file = create_test_wav_file()

            # Call _impl
            responses = list(
                service._impl(
                    input_path=input_file,
                    request_id="test-request",
                    context=context,
                    source_language="en",
                    target_language="es",
                )
            )

            # Verify responses (should include audio chunks and keepalives)
            audio_responses = [r for r in responses if r.audio_data]
            self.assertGreater(len(audio_responses), 0)

            # Verify each audio response has correct format
            for response in audio_responses:
                self.assertEqual(response.audio_format, "mp3")
        finally:
            # Clean up
            if Path(test_mp3.name).exists():
                Path(test_mp3.name).unlink()

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_with_config(self, mock_download, mock_elevenlabs, mock_client):
        """Test infer method with configuration."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService(default_source_language="en", default_target_language="es")
        context = DummyContext()

        # Create test audio file
        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            # Create requests with config
            config = SpeechToSpeechConfig(source_language="fr", target_language="de")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            # Mock _impl to return responses
            with patch.object(service, "_impl") as mock_impl:
                mock_impl.return_value = iter([SpeechToSpeechResponse(audio_data=b"output")])

                # Call infer
                responses = list(service.infer(request_iterator(), context, "test-request"))

                # Verify responses
                self.assertEqual(len(responses), 1)
                self.assertEqual(responses[0].audio_data, b"output")

                # Verify _impl was called with correct parameters
                mock_impl.assert_called_once()
                call_kwargs = mock_impl.call_args[1]
                self.assertEqual(call_kwargs["source_language"], "fr")
                self.assertEqual(call_kwargs["target_language"], "de")
        finally:
            # Clean up
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_with_elevenlabs_params(self, mock_download, mock_elevenlabs, mock_client):
        """Test infer method forwards ElevenLabs params to _impl."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService(default_source_language="en", default_target_language="es")
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            config = SpeechToSpeechConfig(
                source_language="en",
                target_language="es",
                elevenlabs_num_speakers=3,
                elevenlabs_drop_background_audio=True,
                elevenlabs_use_profanity_filter=True,
                elevenlabs_target_accent="castilian",
                elevenlabs_highest_resolution=True,
                elevenlabs_watermark=True,
                elevenlabs_dubbing_studio=True,
            )

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with patch.object(service, "_impl") as mock_impl:
                mock_impl.return_value = iter([SpeechToSpeechResponse(audio_data=b"output")])
                list(service.infer(request_iterator(), context, "test-request"))

                mock_impl.assert_called_once()
                call_kwargs = mock_impl.call_args[1]
                self.assertEqual(call_kwargs["num_speakers"], 3)
                self.assertTrue(call_kwargs["drop_background_audio"])
                self.assertTrue(call_kwargs["use_profanity_filter"])
                self.assertEqual(call_kwargs["target_accent"], "castilian")
                self.assertTrue(call_kwargs["highest_resolution"])
                self.assertTrue(call_kwargs["watermark"])
                self.assertTrue(call_kwargs["dubbing_studio"])
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_invalid_source_language(self, mock_download, mock_elevenlabs, mock_client):
        """Test infer method with invalid source language."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService()
        context = DummyContext()

        # Create test audio file
        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            # Create requests with invalid source language
            config = SpeechToSpeechConfig(source_language="invalid", target_language="es")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            # Call infer (should abort)
            with self.assertRaises(Exception):
                list(service.infer(request_iterator(), context, "test-request"))

            self.assertTrue(context.aborted)
        finally:
            # Clean up
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.el_utils.dubbing.client", new_callable=MagicMock)
    @patch("s2s_service.el_utils.dubbing.ElevenLabs")
    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_invalid_target_language(self, mock_download, mock_elevenlabs, mock_client):
        """Test infer method with invalid target language."""
        from s2s_service.el_utils.dubbing import ELDubbingService

        service = ELDubbingService()
        context = DummyContext()

        # Create test audio file
        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            # Create requests with invalid target language
            config = SpeechToSpeechConfig(source_language="en", target_language="invalid")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            # Call infer (should abort)
            with self.assertRaises(Exception):
                list(service.infer(request_iterator(), context, "test-request"))

            self.assertTrue(context.aborted)
        finally:
            # Clean up
            if Path(test_wav).exists():
                Path(test_wav).unlink()


if __name__ == "__main__":
    unittest.main()
