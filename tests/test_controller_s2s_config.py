#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for controller service S2S config handling."""

import os
import sys
import unittest

import pytest

pytestmark = pytest.mark.unit

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

# Add the protos generated directory to the path
sys.path.insert(0, os.path.join(project_root, "protos/generated"))

from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest


class MockRequestHandler:
    """Mock request handler for testing S2S config handling."""

    def generate_s2s_request_from_content_localization_request(self, request):
        """Mock implementation matching the actual controller logic."""
        s2s_request = SpeechToSpeechRequest()

        if request.HasField("audio_data"):
            s2s_request.audio_data = request.audio_data
            s2s_request.audio_sample_rate = 16000
            s2s_request.audio_num_channels = 1
            s2s_request.audio_format = "mp3"

        if request.HasField("s2s_config"):
            s2s_request.config.CopyFrom(request.s2s_config)

        return s2s_request


class TestControllerS2SConfig(unittest.TestCase):
    """Unit tests for controller S2S config handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = MockRequestHandler()

    def test_config_only_request(self):
        """Test request with S2S config only (first message scenario)."""
        # Arrange
        config_request = ContentLocalizationRequest()
        config_request.s2s_config.source_language = "en-US"
        config_request.s2s_config.target_language = "fr-FR"

        # Act
        s2s_request = self.handler.generate_s2s_request_from_content_localization_request(
            config_request
        )

        # Assert
        self.assertTrue(s2s_request.HasField("config"))
        self.assertFalse(s2s_request.HasField("audio_data"))
        self.assertEqual(s2s_request.config.source_language, "en-US")
        self.assertEqual(s2s_request.config.target_language, "fr-FR")

    def test_audio_only_request(self):
        """Test request with audio data only (subsequent message scenario)."""
        # Arrange
        audio_request = ContentLocalizationRequest()
        audio_request.audio_data = b"fake_audio_data"

        # Act
        s2s_request = self.handler.generate_s2s_request_from_content_localization_request(
            audio_request
        )

        # Assert
        self.assertFalse(s2s_request.HasField("config"))
        self.assertTrue(s2s_request.HasField("audio_data"))
        self.assertEqual(len(s2s_request.audio_data), len(b"fake_audio_data"))
        self.assertEqual(s2s_request.audio_sample_rate, 16000)
        self.assertEqual(s2s_request.audio_num_channels, 1)
        self.assertEqual(s2s_request.audio_format, "mp3")

    def test_payload_oneof_keeps_only_the_last_field_set(self):
        """Setting config then audio keeps only the audio payload."""
        # Arrange
        both_request = ContentLocalizationRequest()
        both_request.s2s_config.source_language = "en-US"
        both_request.s2s_config.target_language = "es"
        both_request.audio_data = b"fake_audio_data"

        # Act
        s2s_request = self.handler.generate_s2s_request_from_content_localization_request(
            both_request
        )

        # Assert: the payload oneof holds the audio; the config was displaced.
        self.assertFalse(s2s_request.HasField("config"))
        self.assertTrue(s2s_request.HasField("audio_data"))
        self.assertEqual(len(s2s_request.audio_data), len(b"fake_audio_data"))
        self.assertEqual(s2s_request.audio_sample_rate, 16000)
        self.assertEqual(s2s_request.audio_num_channels, 1)
        self.assertEqual(s2s_request.audio_format, "mp3")

    def test_empty_request(self):
        """Test request with no config or audio data."""
        # Arrange
        empty_request = ContentLocalizationRequest()

        # Act
        s2s_request = self.handler.generate_s2s_request_from_content_localization_request(
            empty_request
        )

        # Assert
        self.assertFalse(s2s_request.HasField("config"))
        self.assertFalse(s2s_request.HasField("audio_data"))

    def test_config_with_voice_name(self):
        """Test S2S config with voice name."""
        # Arrange
        config_request = ContentLocalizationRequest()
        config_request.s2s_config.source_language = "en-US"
        config_request.s2s_config.target_language = "fr-FR"
        config_request.s2s_config.voice_name = "voice_fr_FR_001"

        # Act
        s2s_request = self.handler.generate_s2s_request_from_content_localization_request(
            config_request
        )

        # Assert
        self.assertTrue(s2s_request.HasField("config"))
        self.assertEqual(s2s_request.config.source_language, "en-US")
        self.assertEqual(s2s_request.config.target_language, "fr-FR")
        self.assertEqual(s2s_request.config.voice_name, "voice_fr_FR_001")


if __name__ == "__main__":
    unittest.main()
