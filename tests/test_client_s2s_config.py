# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for S2SConfig dataclass."""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from client.s2s.config import S2SConfig


@pytest.mark.unit
class TestS2SConfig(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.s2s_server = "localhost:50050"
        self.args.input_audio = "audio.wav"
        self.args.output_audio = "output.mp3"
        self.args.chunk_size_audio_secs = 1.0
        self.args.source_language = "en"
        self.args.target_language = "es"
        self.args.voice_name = None
        self.args.elevenlabs_num_speakers = 0
        self.args.elevenlabs_drop_background_audio = False
        self.args.elevenlabs_use_profanity_filter = False
        self.args.elevenlabs_target_accent = None
        self.args.elevenlabs_highest_resolution = False
        self.args.elevenlabs_watermark = False
        self.args.elevenlabs_dubbing_studio = False
        self.args.camb_ai_optimization = True
        self.args.camb_chosen_dictionaries = None

    def test_from_args(self):
        config = S2SConfig.from_args(self.args)
        self.assertEqual(config.s2s_server, "localhost:50050")
        self.assertEqual(config.input_audio, "audio.wav")
        self.assertEqual(config.output_audio, "output.mp3")
        self.assertEqual(config.chunk_size_audio_secs, 1.0)
        self.assertEqual(config.source_language, "en")
        self.assertEqual(config.target_language, "es")
        self.assertIsNone(config.voice_name)
        self.assertEqual(config.elevenlabs_num_speakers, 0)
        self.assertFalse(config.elevenlabs_drop_background_audio)
        self.assertFalse(config.elevenlabs_use_profanity_filter)
        self.assertIsNone(config.elevenlabs_target_accent)
        self.assertFalse(config.elevenlabs_highest_resolution)
        self.assertFalse(config.elevenlabs_watermark)
        self.assertFalse(config.elevenlabs_dubbing_studio)
        self.assertTrue(config.camb_ai_optimization)
        self.assertIsNone(config.camb_chosen_dictionaries)

    def test_from_args_with_camb_params(self):
        self.args.camb_ai_optimization = False
        self.args.camb_chosen_dictionaries = "1,5,12"
        config = S2SConfig.from_args(self.args)
        self.assertFalse(config.camb_ai_optimization)
        self.assertEqual(config.camb_chosen_dictionaries, [1, 5, 12])

    def test_from_args_with_elevenlabs_params(self):
        self.args.elevenlabs_num_speakers = 3
        self.args.elevenlabs_drop_background_audio = True
        self.args.elevenlabs_target_accent = "castilian"
        config = S2SConfig.from_args(self.args)
        self.assertEqual(config.elevenlabs_num_speakers, 3)
        self.assertTrue(config.elevenlabs_drop_background_audio)
        self.assertEqual(config.elevenlabs_target_accent, "castilian")

    def test_str(self):
        config = S2SConfig.from_args(self.args)
        output = str(config)
        self.assertIn("S2S Configuration", output)
        self.assertIn("localhost:50050", output)
        self.assertIn("audio.wav", output)
        self.assertIn("output.mp3", output)
        self.assertIn("en", output)
        self.assertIn("es", output)
        self.assertIn("ElevenLabs Parameters", output)
        self.assertIn("CambAI Parameters", output)

    @patch("client.s2s.config.is_file_available")
    def test_validate_s2s_config_success(self, mock_is_file_available):
        mock_is_file_available.return_value = True
        config = S2SConfig.from_args(self.args)
        self.assertTrue(config.validate_s2s_config())

    @patch("client.s2s.config.is_file_available")
    def test_validate_s2s_config_missing_audio(self, mock_is_file_available):
        mock_is_file_available.return_value = False
        config = S2SConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_s2s_config()

    @patch("client.s2s.config.is_file_available")
    def test_validate_s2s_config_empty_source_language(self, mock_is_file_available):
        mock_is_file_available.return_value = True
        self.args.source_language = ""
        config = S2SConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_s2s_config()

    @patch("client.s2s.config.is_file_available")
    def test_validate_s2s_config_empty_target_language(self, mock_is_file_available):
        mock_is_file_available.return_value = True
        self.args.target_language = ""
        config = S2SConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_s2s_config()


if __name__ == "__main__":
    unittest.main()
