# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from client.common.audio import AUDIO_CODEC_CONFIGS
from client.lipsync.args import add_lipsync_config_args_to_parser
from client.lipsync.args import argsfactory
from client.lipsync.args import lipsync_config_from_args
from client.lipsync.config import LipSyncConfig
from client.lipsync.encoding import create_output_video_encoding

pytestmark = pytest.mark.unit


class TestLipSyncConfig(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.input_audio = "audio.wav"
        self.args.input_mp4 = "video.mp4"
        self.args.speaker_info_input = "speaker_info.csv"
        self.args.output_mp4 = "output.mp4"
        self.args.lipsync_extend_audio = "silence"
        self.args.lipsync_extend_video = "forward"
        self.args.lipsync_output_bitrate_mbps = 20
        self.args.lipsync_output_idr_interval = 8
        self.args.lipsync_lossless = False
        self.args.lipsync_custom_encoding_params = None
        self.args.lipsync_input_audio_codec = None
        self.args.lipsync_is_speaker_info_provided = False

    def test_from_args(self):
        config = LipSyncConfig.from_args(self.args)
        self.assertEqual(config.audio_filepath, "audio.wav")
        self.assertEqual(config.video_filepath, "video.mp4")
        self.assertEqual(config.speaker_info_filepath, "speaker_info.csv")
        self.assertEqual(config.output_filepath, "output.mp4")
        self.assertEqual(config.extend_audio, "silence")
        self.assertEqual(config.extend_video, "forward")
        self.assertEqual(config.bitrate_mbps, 20)
        self.assertEqual(config.idr_interval, 8)
        self.assertFalse(config.lossless)
        self.assertIsNone(config.audio_codec)
        self.assertIsNone(config.is_speaker_info_provided)
        self.assertIsNone(config.custom_encoding_params)

    def test_from_args_preserves_lipsync_input_audio_codec(self):
        self.args.lipsync_input_audio_codec = "MP3"
        config = LipSyncConfig.from_args(self.args)
        self.assertEqual(config.audio_codec, "MP3")

    def test_parser_default_leaves_input_audio_codec_unset(self):
        parser = add_lipsync_config_args_to_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.lipsync_input_audio_codec)

    def test_str(self):
        config = LipSyncConfig.from_args(self.args)
        output = str(config)
        self.assertIn("LipSync Configuration", output)
        self.assertIn("audio.wav", output)
        self.assertIn("video.mp4", output)
        self.assertIn("output.mp4", output)

    def test_from_args_invalid_custom_encoding_json_raises_value_error(self):
        self.args.lipsync_custom_encoding_params = "{invalid-json}"
        with self.assertRaises(ValueError):
            LipSyncConfig.from_args(self.args)

    def test_lipsync_config_from_args_rejects_non_object_custom_encoding_json(self):
        parsed_args = Namespace(
            lipsync_input_audio_codec="MP3",
            lipsync_extend_audio="unspecified",
            lipsync_extend_video="unspecified",
            lipsync_output_bitrate_mbps=20,
            lipsync_output_idr_interval=8,
            lipsync_head_movement_speed=None,
            lipsync_output_audio_codec=None,
            lipsync_is_speaker_info_provided=False,
            lipsync_lossless=False,
            lipsync_custom_encoding_params="[1]",
            background_audio_input=None,
            lipsync_background_audio_codec=None,
            lipsync_background_audio_volume=1.0,
        )
        with self.assertRaises(ValueError):
            lipsync_config_from_args(parsed_args)

    def test_lipsync_config_from_args_defaults_to_mp3_for_pipeline_configs(self):
        parsed_args = Namespace(
            lipsync_input_audio_codec=None,
            lipsync_extend_audio="unspecified",
            lipsync_extend_video="unspecified",
            lipsync_output_bitrate_mbps=20,
            lipsync_output_idr_interval=8,
            lipsync_head_movement_speed=None,
            lipsync_output_audio_codec=None,
            lipsync_is_speaker_info_provided=False,
            lipsync_lossless=False,
            lipsync_custom_encoding_params=None,
            background_audio_input=None,
            lipsync_background_audio_codec=None,
            lipsync_background_audio_volume=1.0,
        )

        config = lipsync_config_from_args(parsed_args)

        self.assertEqual(config.input_audio_codec, AUDIO_CODEC_CONFIGS["mp3"])

    def test_create_output_video_encoding_rejects_non_object_custom_encoding_json(self):
        self.args.lipsync_custom_encoding_params = "[1]"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(ValueError):
            create_output_video_encoding(config=config)

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config(self, mock_is_file_available):
        # Setup mocks
        mock_is_file_available.side_effect = lambda _path, _exts: True
        config = LipSyncConfig.from_args(self.args)
        # Should not raise
        self.assertTrue(config.validate_lipsync_config())
        self.assertEqual(config.audio_codec, "wav")

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_keeps_explicit_audio_codec(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda _path, _exts: True
        self.args.lipsync_input_audio_codec = "MP3"
        config = LipSyncConfig.from_args(self.args)
        self.assertTrue(config.validate_lipsync_config())
        self.assertEqual(config.audio_codec, "mp3")

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_rejects_unsupported_explicit_audio_codec(
        self, mock_is_file_available
    ):
        mock_is_file_available.side_effect = lambda _path, _exts: True
        self.args.lipsync_input_audio_codec = "FLAC"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_invalid_video(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda path, _exts: path != "video.mp4"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_invalid_audio(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda path, _exts: path != "audio.wav"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_invalid_speaker_info(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda path, _exts: path != "speaker_info.csv"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_explicit_speaker_info_flag_with_file_is_honored(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda _path, _exts: True
        self.args.lipsync_is_speaker_info_provided = True
        config = LipSyncConfig.from_args(self.args)
        self.assertTrue(config.is_speaker_info_provided)
        self.assertTrue(config.validate_lipsync_config())
        self.assertTrue(config.is_speaker_info_provided)

    @patch("client.lipsync.config.is_file_available")
    def test_explicit_speaker_info_flag_without_file_raises(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda _path, _exts: True
        self.args.lipsync_is_speaker_info_provided = True
        self.args.speaker_info_input = None
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_speaker_info_derived_from_file_when_flag_absent(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda _path, _exts: True
        self.args.speaker_info_input = None
        config = LipSyncConfig.from_args(self.args)
        self.assertIsNone(config.is_speaker_info_provided)
        self.assertTrue(config.validate_lipsync_config())
        self.assertFalse(config.is_speaker_info_provided)


class TestLipsyncArgsAliases(unittest.TestCase):
    """The renamed I/O flags and their deprecated aliases share one dest."""

    def test_new_flag_names_parse(self):
        args = argsfactory().parse_args(
            [
                "--lipsync-server",
                "localhost:50054",
                "--input-mp4",
                "v.mp4",
                "--input-audio",
                "a.wav",
                "--output-mp4",
                "o.mp4",
            ]
        )
        self.assertEqual(args.lipsync_server, "localhost:50054")
        self.assertEqual(args.input_mp4, "v.mp4")
        self.assertEqual(args.input_audio, "a.wav")
        self.assertEqual(args.output_mp4, "o.mp4")

    def test_deprecated_aliases_parse_to_same_dest(self):
        args = argsfactory().parse_args(
            [
                "--target",
                "localhost:60000",
                "--video-input",
                "v2.mp4",
                "--audio-input",
                "a2.wav",
                "--output",
                "o2.mp4",
            ]
        )
        self.assertEqual(args.lipsync_server, "localhost:60000")
        self.assertEqual(args.input_mp4, "v2.mp4")
        self.assertEqual(args.input_audio, "a2.wav")
        self.assertEqual(args.output_mp4, "o2.mp4")


if __name__ == "__main__":
    unittest.main()
