# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ControllerConfig dataclass."""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from client.common.audio import AUDIO_CODEC_CONFIGS
from client.controller.config import ControllerConfig

_MOCK_S2S = SpeechToSpeechConfig()
_MOCK_ASD = ActiveSpeakerDetectionConfig()
_MOCK_LIPSYNC = LipsyncConfig()


def _patch_nim_builders():
    """Decorator stack that patches the three NIM config builders."""
    return [
        patch(
            "client.controller.config.s2s_config_from_args",
            return_value=_MOCK_S2S,
        ),
        patch(
            "client.controller.config.asd_config_from_args",
            return_value=_MOCK_ASD,
        ),
        patch(
            "client.controller.config.lipsync_config_from_args",
            return_value=_MOCK_LIPSYNC,
        ),
    ]


def _apply_patches(test_func):
    """Apply all NIM builder patches to a test method."""
    for p in reversed(_patch_nim_builders()):
        test_func = p(test_func)
    return test_func


@pytest.mark.unit
class TestControllerConfig(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.controller_server = "localhost:50056"
        self.args.input_audio = "audio.wav"
        self.args.input_mp4 = "video.mp4"
        self.args.output_mp4 = "output.mp4"
        self.args.chunk_size_audio_secs = 1.0
        self.args.chunk_size_video_bytes = 1048576
        self.args.diarization_file = "diar.json"
        self.args.bypass_asd = False
        self.args.translated_audio = None
        self.args.lipsync_input_audio_codec = None

    @_apply_patches
    def test_from_args(self, *_mocks):
        config = ControllerConfig.from_args(self.args)
        self.assertEqual(config.controller_server, "localhost:50056")
        self.assertEqual(config.input_audio, "audio.wav")
        self.assertEqual(config.input_mp4, "video.mp4")
        self.assertEqual(config.output_mp4, "output.mp4")
        self.assertEqual(config.chunk_size_audio_secs, 1.0)
        self.assertEqual(config.chunk_size_video_bytes, 1048576)
        self.assertEqual(config.diarization_file, "diar.json")
        self.assertIs(config.s2s_config, _MOCK_S2S)
        self.assertIs(config.asd_config, _MOCK_ASD)
        self.assertFalse(config.bypass_asd)
        self.assertIs(config.lipsync_config, _MOCK_LIPSYNC)
        self.assertIsNone(config.explicit_lipsync_input_audio_codec)

    @_apply_patches
    def test_from_args_preserves_explicit_lipsync_input_audio_codec(self, *_mocks):
        """User-provided LipSync input codec stays available to callers."""
        self.args.lipsync_input_audio_codec = "MP3"
        config = ControllerConfig.from_args(self.args)
        self.assertEqual(config.explicit_lipsync_input_audio_codec, "MP3")

    @_apply_patches
    def test_from_args_bypass_asd(self, *_mocks):
        self.args.bypass_asd = True
        config = ControllerConfig.from_args(self.args)
        self.assertIsNone(config.asd_config)
        self.assertFalse(config.lipsync_config.is_speaker_info_provided)

    @_apply_patches
    def test_from_args_auto_detect_bypass_asd(self, *_mocks):
        """bypass_asd auto-enabled when no diarization file provided."""
        self.args.bypass_asd = False
        self.args.diarization_file = None
        config = ControllerConfig.from_args(self.args)
        self.assertTrue(config.bypass_asd)
        self.assertIsNone(config.asd_config)

    @_apply_patches
    def test_from_args_diarization_disables_auto_bypass(self, *_mocks):
        """bypass_asd stays False when diarization file is provided."""
        self.args.bypass_asd = False
        self.args.diarization_file = "diar.json"
        config = ControllerConfig.from_args(self.args)
        self.assertFalse(config.bypass_asd)
        self.assertIsNotNone(config.asd_config)

    @_apply_patches
    def test_from_args_auto_bypass_opt_out_keeps_asd(self, *_mocks):
        """auto_bypass_asd=False keeps ASD enabled without a diarization file."""
        self.args.bypass_asd = False
        self.args.diarization_file = None
        config = ControllerConfig.from_args(args=self.args, auto_bypass_asd=False)
        self.assertFalse(config.bypass_asd)
        self.assertIsNotNone(config.asd_config)

    @_apply_patches
    @patch("client.controller.config.is_wav_file", return_value=True)
    def test_from_args_translated_audio_sniffs_wav_codec(self, mock_is_wav, *_mocks):
        """A translated WAV file sets the LipSync input codec to WAV."""
        self.args.translated_audio = "translated.wav"
        self.args.lipsync_input_audio_codec = None
        config = ControllerConfig.from_args(self.args)
        self.assertEqual(
            config.lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["wav"],
        )
        mock_is_wav.assert_called_once_with("translated.wav")

    @_apply_patches
    @patch("client.controller.config.is_wav_file", return_value=False)
    def test_from_args_translated_audio_sniffs_mp3_codec(self, mock_is_wav, *_mocks):
        """MP3 content inside a .wav filename is declared as MP3 to LipSync."""
        self.args.translated_audio = "translated.wav"
        self.args.lipsync_input_audio_codec = None
        config = ControllerConfig.from_args(self.args)
        self.assertEqual(
            config.lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["mp3"],
        )
        mock_is_wav.assert_called_once_with("translated.wav")

    @_apply_patches
    @patch("client.controller.config.is_wav_file")
    def test_from_args_explicit_codec_skips_translated_sniff(self, mock_is_wav, *_mocks):
        """An explicit --lipsync-input-audio-codec disables content sniffing."""
        self.args.translated_audio = "translated.wav"
        self.args.lipsync_input_audio_codec = "MP3"
        ControllerConfig.from_args(self.args)
        mock_is_wav.assert_not_called()

    @_apply_patches
    def test_from_args_without_io(self, *_mocks):
        args = MagicMock(spec=[])
        args.controller_server = "localhost:50056"
        args.chunk_size_audio_secs = 1.0
        args.chunk_size_video_bytes = 1048576
        config = ControllerConfig.from_args(args)
        self.assertIsNone(config.input_audio)
        self.assertIsNone(config.input_mp4)
        self.assertIsNone(config.output_mp4)

    @_apply_patches
    def test_str(self, *_mocks):
        config = ControllerConfig.from_args(self.args)
        output = str(config)
        self.assertIn("Controller Configuration", output)
        self.assertIn("localhost:50056", output)
        self.assertIn("audio.wav", output)
        self.assertIn("video.mp4", output)
        self.assertIn("output.mp4", output)

    @_apply_patches
    @patch("client.controller.config.is_file_available")
    def test_validate_io_success(self, mock_is_file_available, *_mocks):
        mock_is_file_available.return_value = True
        config = ControllerConfig.from_args(self.args)
        self.assertTrue(config.validate_io())

    @_apply_patches
    @patch("client.controller.config.is_file_available")
    def test_validate_io_invalid_audio(self, mock_is_file_available, *_mocks):
        mock_is_file_available.side_effect = lambda path, _exts: path != "audio.wav"
        config = ControllerConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @_apply_patches
    @patch("client.controller.config.is_file_available")
    def test_validate_io_invalid_video(self, mock_is_file_available, *_mocks):
        mock_is_file_available.side_effect = lambda path, _exts: path != "video.mp4"
        config = ControllerConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @_apply_patches
    @patch("client.controller.config.is_file_available")
    def test_validate_io_invalid_diarization(self, mock_is_file_available, *_mocks):
        mock_is_file_available.side_effect = lambda path, _exts: path != "diar.json"
        self.args.diarization_file = "diar.json"
        config = ControllerConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @_apply_patches
    @patch("client.controller.config.is_file_available")
    def test_validate_io_negative_audio_chunk(self, mock_is_file_available, *_mocks):
        mock_is_file_available.return_value = True
        self.args.chunk_size_audio_secs = -1.0
        config = ControllerConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @_apply_patches
    @patch("client.controller.config.is_file_available")
    def test_validate_io_negative_video_chunk(self, mock_is_file_available, *_mocks):
        mock_is_file_available.return_value = True
        self.args.chunk_size_video_bytes = -1
        config = ControllerConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_io()

    @_apply_patches
    def test_validate_io_skips_none_paths(self, *_mocks):
        config = ControllerConfig(
            controller_server="localhost:50056",
            s2s_config=_MOCK_S2S,
            asd_config=_MOCK_ASD,
            lipsync_config=_MOCK_LIPSYNC,
        )
        self.assertTrue(config.validate_io())

    @_apply_patches
    def test_validate_controller_config_alias(self, *_mocks):
        config = ControllerConfig(
            controller_server="localhost:50056",
            s2s_config=_MOCK_S2S,
            asd_config=_MOCK_ASD,
            lipsync_config=_MOCK_LIPSYNC,
        )
        self.assertTrue(config.validate_controller_config())


if __name__ == "__main__":
    unittest.main()
