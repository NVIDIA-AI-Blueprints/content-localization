# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for direct-client configuration behavior."""

import unittest
from argparse import Namespace
from unittest.mock import patch

import pytest

from client.common.audio import AUDIO_CODEC_CONFIGS
from client.direct.config import DirectPipelineConfig

pytestmark = pytest.mark.unit


def _translated_audio_args(lipsync_input_audio_codec: str | None) -> Namespace:
    """Build minimal direct-client args for translated-audio config tests.

    Args:
        lipsync_input_audio_codec (str | None): Explicit LipSync input
            audio codec value, or ``None`` to allow content detection.

    Returns:
        Namespace: Parsed-args stand-in for ``DirectPipelineConfig``.

    Examples:
        >>> args = _translated_audio_args(None)
        >>> args.translated_audio
        'translated.wav'
    """
    return Namespace(
        s2s_server="localhost:50050",
        asd_server="localhost:50055",
        lipsync_server="localhost:50054",
        chunk_size_audio_secs=1.0,
        chunk_size_video_bytes=1048576,
        bypass_asd=True,
        diarization_file=None,
        translated_audio="translated.wav",
        background_audio_input=None,
        lipsync_input_audio_codec=lipsync_input_audio_codec,
        lipsync_extend_audio="unspecified",
        lipsync_extend_video="unspecified",
        lipsync_output_bitrate_mbps=20,
        lipsync_output_idr_interval=8,
        lipsync_head_movement_speed=None,
        lipsync_output_audio_codec=None,
        lipsync_is_speaker_info_provided=False,
        lipsync_lossless=False,
        lipsync_custom_encoding_params=None,
        lipsync_background_audio_codec=None,
        lipsync_background_audio_volume=1.0,
    )


class TestDirectPipelineConfig(unittest.TestCase):
    """Test direct-client config construction."""

    @patch("client.direct.config.is_wav_file")
    def test_explicit_lipsync_codec_skips_translated_audio_detection(self, mock_is_wav_file):
        """Explicit customer codec is preserved for translated-audio bypass."""
        args = _translated_audio_args(lipsync_input_audio_codec="MP3")

        config = DirectPipelineConfig.from_args(args)

        self.assertEqual(config.lipsync_config.input_audio_codec, AUDIO_CODEC_CONFIGS["mp3"])
        mock_is_wav_file.assert_not_called()

    @patch("client.direct.config.is_wav_file", return_value=True)
    def test_omitted_lipsync_codec_detects_translated_audio_content(self, mock_is_wav_file):
        """Omitted codec allows translated-audio content detection."""
        args = _translated_audio_args(lipsync_input_audio_codec=None)

        config = DirectPipelineConfig.from_args(args)

        self.assertEqual(config.lipsync_config.input_audio_codec, AUDIO_CODEC_CONFIGS["wav"])
        mock_is_wav_file.assert_called_once_with("translated.wav")

    @patch("client.direct.config.is_wav_file", return_value=True)
    def test_from_args_populates_io_fields(self, _mock_is_wav_file):
        """I/O paths from the CLI land on the config for validation."""
        args = _translated_audio_args(lipsync_input_audio_codec=None)
        args.input_audio = "in.wav"
        args.output_audio = "out.mp3"
        args.input_mp4 = "in.mp4"
        args.output_mp4 = "out.mp4"

        config = DirectPipelineConfig.from_args(args)

        self.assertEqual(config.input_audio, "in.wav")
        self.assertEqual(config.output_audio, "out.mp3")
        self.assertEqual(config.input_mp4, "in.mp4")
        self.assertEqual(config.output_mp4, "out.mp4")


class TestDirectPipelineConfigValidateIO(unittest.TestCase):
    """Test direct-client I/O validation."""

    def _config(self, **overrides) -> DirectPipelineConfig:
        """Build a config with optional field overrides."""
        fields = {
            "s2s_server": "localhost:50050",
            "asd_server": "localhost:50055",
            "lipsync_server": "localhost:50054",
            "s2s_config": None,
            "asd_config": None,
            "lipsync_config": None,
        }
        fields.update(overrides)
        return DirectPipelineConfig(**fields)

    def test_validate_io_skips_none_paths(self):
        """A config without I/O paths validates successfully."""
        self.assertTrue(self._config().validate_io())

    @patch("client.direct.config.is_file_available", return_value=True)
    def test_validate_io_success(self, _mock_available):
        """Valid inputs and positive chunk sizes pass validation."""
        config = self._config(
            input_audio="in.wav",
            input_mp4="in.mp4",
            diarization_file="diar.json",
        )
        self.assertTrue(config.validate_io())

    @patch("client.direct.config.is_file_available", return_value=False)
    def test_validate_io_invalid_audio(self, _mock_available):
        """An unsupported input audio format raises RuntimeError."""
        config = self._config(input_audio="in.flac")
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @patch("client.direct.config.is_file_available", return_value=False)
    def test_validate_io_invalid_video(self, _mock_available):
        """An unsupported input video format raises RuntimeError."""
        config = self._config(input_mp4="in.avi")
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @patch("client.direct.config.is_file_available", return_value=False)
    def test_validate_io_invalid_diarization(self, _mock_available):
        """An unsupported diarization file format raises RuntimeError."""
        config = self._config(diarization_file="diar.txt")
        with self.assertRaises(RuntimeError):
            config.validate_io()

    def test_validate_io_negative_audio_chunk(self):
        """A non-positive audio chunk size raises RuntimeError."""
        config = self._config(chunk_size_audio_secs=0)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    def test_validate_io_negative_video_chunk(self):
        """A non-positive video chunk size raises RuntimeError."""
        config = self._config(chunk_size_video_bytes=-1)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    def test_validate_io_accepts_bare_output_filenames(self):
        """Bare output filenames must not crash directory creation."""
        config = self._config(output_audio="out.mp3", output_mp4="out.mp4")
        self.assertTrue(config.validate_io())


if __name__ == "__main__":
    unittest.main()
