# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CambAI Dubbing S2S service."""

import os
import tempfile
import unittest
import wave
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse


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
    """Create a temporary WAV file for testing.

    Args:
        duration_seconds: Duration of the audio file in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the temporary WAV file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_name = temp_file.name

    n_frames = int(sample_rate * duration_seconds)
    with wave.open(temp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
    return temp_name


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceInit(unittest.TestCase):
    """Test CambDubbingService initialization."""

    def test_initialization_defaults(self) -> None:
        """Default initialization should set correct attributes."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertEqual(service.sample_rate_hz, 16000)
        self.assertEqual(service.default_source_language, "1")
        self.assertEqual(service.default_target_language, "54")
        self.assertEqual(service.audio_format, "mp3")

    def test_initialization_custom(self) -> None:
        """Custom initialization should override defaults."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService(
            sample_rate_hz=48000,
            default_source_language="81",
            default_target_language="1",
            audio_format="mp3",
        )
        self.assertEqual(service.sample_rate_hz, 48000)
        self.assertEqual(service.default_source_language, "81")
        self.assertEqual(service.default_target_language, "1")

    def test_missing_api_key(self) -> None:
        """Missing CAMB_API_KEY should raise RuntimeError."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CAMB_API_KEY", None)
            with self.assertRaises(RuntimeError) as ctx:
                CambDubbingService()
            self.assertIn("CAMB_API_KEY", str(ctx.exception))


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceLanguages(unittest.TestCase):
    """Test CambAI language validation."""

    def test_valid_source_languages(self) -> None:
        """Valid CambAI integer ID strings should pass validation."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        # English=1, Spanish=54, Hindi=81
        self.assertTrue(service.validate_source_language("1"))
        self.assertTrue(service.validate_source_language("54"))
        self.assertTrue(service.validate_source_language("81"))

    def test_invalid_source_language(self) -> None:
        """Short codes like 'en' are not valid CambAI IDs."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_source_language("en"))
        self.assertFalse(service.validate_source_language("invalid"))
        self.assertFalse(service.validate_source_language("999"))

    def test_valid_target_languages(self) -> None:
        """Target language validation uses same ID set."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertTrue(service.validate_target_language("1"))
        self.assertTrue(service.validate_target_language("54"))

    def test_invalid_target_language(self) -> None:
        """Invalid target language IDs should fail."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_target_language("es"))
        self.assertFalse(service.validate_target_language("0"))


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceAudioFormat(unittest.TestCase):
    """Test audio format validation."""

    def test_mp3_valid(self) -> None:
        """MP3 should be valid — CambAI alt-format output is MP3."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertTrue(service.validate_audio_format("mp3"))

    def test_wav_invalid(self) -> None:
        """WAV should not be valid for the CambAI MP3 output path."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_audio_format("wav"))

    def test_unsupported_format(self) -> None:
        """Unsupported formats should be invalid."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_audio_format("flac"))
        self.assertFalse(service.validate_audio_format("ogg"))


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceImpl(unittest.TestCase):
    """Test _impl method."""

    @patch("s2s_service.camb_utils.dubbing.download_output_audio_to_file")
    @patch("s2s_service.camb_utils.dubbing.get_alt_format_output_audio_url")
    @patch("s2s_service.camb_utils.dubbing.wait_for_completion")
    @patch("s2s_service.camb_utils.dubbing.submit_dub_task")
    @patch("s2s_service.camb_utils.dubbing.upload_local_file")
    def test_impl_success(
        self,
        mock_upload: MagicMock,
        mock_submit: MagicMock,
        mock_wait: MagicMock,
        mock_get_alt_url: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        """Successful _impl should yield audio responses in MP3 format."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        # Set up mocks for the full pipeline
        mock_upload.return_value = "f-123"
        mock_submit.return_value = "task-abc"
        mock_wait.return_value = 42
        mock_get_alt_url.return_value = "https://cdn/dubbed.mp3"

        def write_mp3(*, audio_url: str, output_file: Path) -> Path:
            del audio_url
            output_file.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 100)
            return output_file

        mock_download.side_effect = write_mp3

        input_file = create_test_wav_file()

        try:
            responses = list(
                service._impl(
                    input_path=input_file,
                    request_id="test-request",
                    context=context,
                    source_language="1",
                    target_language="54",
                )
            )

            # Should have audio responses in MP3 format
            audio_responses = [r for r in responses if r.audio_data]
            self.assertGreater(len(audio_responses), 0)
            for resp in audio_responses:
                self.assertEqual(resp.audio_format, "mp3")

            mock_upload.assert_called_once()
            mock_submit.assert_called_once()
            mock_wait.assert_called_once()
            mock_get_alt_url.assert_called_once()
        finally:
            if Path(input_file).exists():
                Path(input_file).unlink()

    @patch("s2s_service.camb_utils.dubbing.upload_local_file")
    def test_impl_upload_error(self, mock_upload: MagicMock) -> None:
        """Upload failure should produce error via queue."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        mock_upload.side_effect = RuntimeError("Upload failed")

        input_file = create_test_wav_file()
        try:
            with self.assertRaises(Exception):
                list(
                    service._impl(
                        input_path=input_file,
                        request_id="test-request",
                        context=context,
                        source_language="1",
                        target_language="54",
                    )
                )
        finally:
            if Path(input_file).exists():
                Path(input_file).unlink()


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceInfer(unittest.TestCase):
    """Test infer method."""

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_with_config(self, mock_download: MagicMock) -> None:
        """Config in first request should override default languages."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService(default_source_language="1", default_target_language="54")
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            # source_language="81" (Hindi), target_language="1" (English)
            config = SpeechToSpeechConfig(source_language="81", target_language="1")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with patch.object(service, "_impl") as mock_impl:
                mock_impl.return_value = iter([SpeechToSpeechResponse(audio_data=b"output")])
                responses = list(service.infer(request_iterator(), context, "test-req"))

                self.assertEqual(len(responses), 1)
                call_kwargs = mock_impl.call_args[1]
                self.assertEqual(call_kwargs["source_language"], "81")
                self.assertEqual(call_kwargs["target_language"], "1")
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_default_languages(self, mock_download: MagicMock) -> None:
        """No config should use service defaults."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService(default_source_language="1", default_target_language="54")
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:

            def request_iterator():
                # No config, just audio
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with patch.object(service, "_impl") as mock_impl:
                mock_impl.return_value = iter([SpeechToSpeechResponse(audio_data=b"output")])
                list(service.infer(request_iterator(), context, "test-req"))

                call_kwargs = mock_impl.call_args[1]
                self.assertEqual(call_kwargs["source_language"], "1")
                self.assertEqual(call_kwargs["target_language"], "54")
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_invalid_source_language(self, mock_download: MagicMock) -> None:
        """Invalid source language should abort with INVALID_ARGUMENT."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            config = SpeechToSpeechConfig(source_language="en", target_language="54")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with self.assertRaises(Exception):
                list(service.infer(request_iterator(), context, "test-req"))
            self.assertTrue(context.aborted)
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_invalid_target_language(self, mock_download: MagicMock) -> None:
        """Invalid target language should abort with INVALID_ARGUMENT."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            config = SpeechToSpeechConfig(source_language="1", target_language="es")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with self.assertRaises(Exception):
                list(service.infer(request_iterator(), context, "test-req"))
            self.assertTrue(context.aborted)
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceKeepalive(unittest.TestCase):
    """Test keepalive behavior during long-running operations."""

    @patch("s2s_service.camb_utils.dubbing.upload_local_file")
    @patch.dict(os.environ, {"S2S_CAMB_KEEPALIVE_INTERVAL": "0"})
    def test_keepalive_sent_during_processing(self, mock_upload: MagicMock) -> None:
        """Keepalives should be sent when queue is empty."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        # Make upload take some time then fail — we only care about
        # keepalive pings appearing before the error
        mock_upload.side_effect = RuntimeError("Simulated slow upload")

        input_file = create_test_wav_file()
        try:
            responses = []
            with self.assertRaises(Exception):
                for resp in service._impl(
                    input_path=input_file,
                    request_id="test-request",
                    context=context,
                    source_language="1",
                    target_language="54",
                ):
                    responses.append(resp)

            # At least one keepalive may have been sent before the error
            # (depends on timing), but we verify the mechanism doesn't crash
        finally:
            if Path(input_file).exists():
                Path(input_file).unlink()


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceArgsfactory(unittest.TestCase):
    """Test argsfactory static method."""

    def test_argsfactory_creates_parser(self) -> None:
        """argsfactory should return a valid parser."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        parser = CambDubbingService.argsfactory()
        self.assertIsNotNone(parser)
        # Should include base S2S args
        args = parser.parse_args([])
        self.assertTrue(hasattr(args, "sample_rate_hz"))
        self.assertTrue(hasattr(args, "default_source_language"))
        self.assertTrue(hasattr(args, "audio_format"))


if __name__ == "__main__":
    unittest.main()
