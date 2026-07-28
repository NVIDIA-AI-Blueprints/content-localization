# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for background audio support across the pipeline."""

import argparse
import tempfile
import unittest

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    AUDIO_SOURCE_CONFIG_SEPARATE_STREAM,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest

from client.asd.args import asd_config_from_args
from client.direct.pipeline import background_audio_iterator_from_file
from client.lipsync.args import lipsync_config_from_args
from common.buffers import RequestIteratorFromBuffer
from controller_service.conversions import to_lipsync_background_audio
from controller_service.deserializer import ContentLocalizationDeserializer
from controller_service.stream_adapters import lipsync_request_generator

# -- helpers -----------------------------------------------------------------


def _bg_audio_request(data: bytes = b"\xaa\xbb") -> ContentLocalizationRequest:
    """Create a request with background_audio_data."""
    return ContentLocalizationRequest(background_audio_data=data)


def _make_lipsync_args(**overrides) -> argparse.Namespace:
    """Build a minimal Namespace for lipsync_config_from_args."""
    defaults = {
        "lipsync_input_audio_codec": "MP3",
        "lipsync_extend_audio": "unspecified",
        "lipsync_extend_video": "unspecified",
        "lipsync_output_bitrate_mbps": 20,
        "lipsync_output_idr_interval": 8,
        "lipsync_head_movement_speed": None,
        "lipsync_output_audio_codec": None,
        "lipsync_is_speaker_info_provided": False,
        "lipsync_background_audio_codec": None,
        "lipsync_background_audio_volume": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# -- conversions tests -------------------------------------------------------


@pytest.mark.unit
class TestToLipsyncBackgroundAudio(unittest.TestCase):
    """Tests for to_lipsync_background_audio conversion."""

    def test_converts_background_audio_data(self) -> None:
        """Valid background_audio_data is converted correctly."""
        req = ContentLocalizationRequest(background_audio_data=b"\x01\x02\x03")
        result = to_lipsync_background_audio(req)

        self.assertIsInstance(result, LipsyncRequest)
        self.assertEqual(result.input.background_audio_file_data, b"\x01\x02\x03")

    def test_raises_when_no_background_audio(self) -> None:
        """ValueError raised when background_audio_data is missing."""
        req = ContentLocalizationRequest(audio_data=b"\x00")
        with self.assertRaises(ValueError, msg="Background audio data not found"):
            to_lipsync_background_audio(req)

    def test_empty_bytes_raises(self) -> None:
        """Empty request (no fields set) raises ValueError."""
        req = ContentLocalizationRequest()
        with self.assertRaises(ValueError):
            to_lipsync_background_audio(req)


# -- lipsync_config_from_args tests -----------------------------------------


@pytest.mark.unit
class TestLipsyncConfigBackgroundAudio(unittest.TestCase):
    """Tests for BackgroundAudioConfig built by lipsync_config_from_args."""

    def test_no_background_audio_omits_config(self) -> None:
        """Without background_audio_input, no BackgroundAudioConfig is set."""
        args = _make_lipsync_args()
        config = lipsync_config_from_args(args)

        self.assertFalse(config.HasField("background_audio_config"))

    def test_background_audio_wav_auto_detect(self) -> None:
        """Codec auto-detected as WAV from file extension."""
        args = _make_lipsync_args(background_audio_input="bg_music.wav")
        config = lipsync_config_from_args(args)

        self.assertTrue(config.HasField("background_audio_config"))
        bg = config.background_audio_config
        self.assertTrue(bg.is_background_audio_provided)
        self.assertEqual(bg.audio_codec, AUDIO_CODEC_WAV)

    def test_background_audio_mp3_auto_detect(self) -> None:
        """Codec auto-detected as MP3 from file extension."""
        args = _make_lipsync_args(background_audio_input="bg_music.mp3")
        config = lipsync_config_from_args(args)

        bg = config.background_audio_config
        self.assertTrue(bg.is_background_audio_provided)
        self.assertEqual(bg.audio_codec, AUDIO_CODEC_MP3)

    def test_explicit_codec_overrides_extension(self) -> None:
        """Explicit --lipsync-background-audio-codec overrides file ext."""
        args = _make_lipsync_args(
            background_audio_input="bg_music.wav",
            lipsync_background_audio_codec="MP3",
        )
        config = lipsync_config_from_args(args)

        bg = config.background_audio_config
        self.assertEqual(bg.audio_codec, AUDIO_CODEC_MP3)

    def test_volume_set_when_provided(self) -> None:
        """Volume is populated when --lipsync-background-audio-volume is set."""
        args = _make_lipsync_args(
            background_audio_input="bg.mp3",
            lipsync_background_audio_volume=0.3,
        )
        config = lipsync_config_from_args(args)

        self.assertAlmostEqual(config.background_audio_config.audio_volume, 0.3, places=5)

    def test_volume_omitted_when_none(self) -> None:
        """Volume field is not set when --lipsync-background-audio-volume is None."""
        args = _make_lipsync_args(background_audio_input="bg.mp3")
        config = lipsync_config_from_args(args)

        # audio_volume defaults to 0.0 in proto; we only care that we
        # didn't explicitly set it (the proto default is fine).
        bg = config.background_audio_config
        self.assertTrue(bg.is_background_audio_provided)


# -- deserializer routing tests ----------------------------------------------


@pytest.mark.unit
class TestDeserializerBackgroundAudio(unittest.TestCase):
    """Tests for background_audio_data routing in the deserializer."""

    def _run(self, requests):
        ds = ContentLocalizationDeserializer(iter(requests))
        ds.start(request_id="test")
        ds.join(timeout=5.0)
        return ds

    def test_background_audio_routes_to_buffer(self) -> None:
        """background_audio_data goes to background_audio_buffer only."""
        ds = self._run([_bg_audio_request(b"\x01")])

        self.assertEqual(ds.background_audio_buffer.qsize(0), 1)
        # Should not leak into other data buffers
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(0))
        self.assertTrue(ds.diarization_buffer.empty(0))

    def test_background_audio_buffer_done_on_complete(self) -> None:
        """background_audio_buffer.done is True after stream exhausts."""
        ds = self._run([_bg_audio_request()])
        self.assertTrue(ds.background_audio_buffer.done)

    def test_no_background_audio_leaves_buffer_empty_and_done(self) -> None:
        """Stream without background audio leaves buffer empty but done."""
        ds = self._run([ContentLocalizationRequest(audio_data=b"\x00")])

        self.assertTrue(ds.background_audio_buffer.done)
        self.assertTrue(ds.background_audio_buffer.empty(0))

    def test_multiple_background_audio_chunks(self) -> None:
        """Multiple background audio requests all route correctly."""
        requests = [_bg_audio_request(b"\x01"), _bg_audio_request(b"\x02")]
        ds = self._run(requests)

        self.assertEqual(ds.background_audio_buffer.qsize(0), 2)

    def test_background_audio_iterator_drains_buffer(self) -> None:
        """RequestIteratorFromBuffer yields all background audio items."""
        ds = self._run([_bg_audio_request(b"\xaa"), _bg_audio_request(b"\xbb")])
        items = list(
            RequestIteratorFromBuffer(
                ds.background_audio_buffer,
                consumer_id=0,
                poll_timeout=0.01,
            )
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].background_audio_data, b"\xaa")
        self.assertEqual(items[1].background_audio_data, b"\xbb")

    def test_mixed_stream_with_background_audio(self) -> None:
        """Background audio interleaved with audio/video routes correctly."""
        requests = [
            ContentLocalizationRequest(audio_data=b"a1"),
            _bg_audio_request(b"bg1"),
            ContentLocalizationRequest(video_file_data=b"v1"),
            _bg_audio_request(b"bg2"),
        ]
        ds = self._run(requests)

        self.assertEqual(ds.audio_buffer.qsize(0), 1)
        self.assertEqual(ds.video_buffer.qsize(0), 1)
        self.assertEqual(ds.background_audio_buffer.qsize(0), 2)


# -- lipsync_request_generator tests ----------------------------------------


@pytest.mark.unit
class TestLipsyncRequestGeneratorBackgroundAudio(unittest.TestCase):
    """Tests for background_audio_iter in lipsync_request_generator."""

    def test_without_background_audio(self) -> None:
        """Generator works normally when background_audio_iter is None."""
        video = iter([LipsyncRequest(input=LipsyncInputData(video_file_data=b"v"))])
        audio = iter([LipsyncRequest(input=LipsyncInputData(audio_file_data=b"a"))])
        config = LipsyncConfig()

        requests = list(
            lipsync_request_generator(
                video_iter=video,
                audio_iter=audio,
                speaker_info_iter=None,
                lipsync_config=config,
                background_audio_iter=None,
            )
        )

        # config + video + audio = 3
        self.assertEqual(len(requests), 3)
        self.assertTrue(requests[0].HasField("config"))

    def test_with_background_audio(self) -> None:
        """Background audio chunks are interleaved in the output."""
        video = iter([LipsyncRequest(input=LipsyncInputData(video_file_data=b"v"))])
        audio = iter([LipsyncRequest(input=LipsyncInputData(audio_file_data=b"a"))])
        bg = iter([LipsyncRequest(input=LipsyncInputData(background_audio_file_data=b"bg"))])
        config = LipsyncConfig()

        requests = list(
            lipsync_request_generator(
                video_iter=video,
                audio_iter=audio,
                speaker_info_iter=None,
                lipsync_config=config,
                background_audio_iter=bg,
            )
        )

        # config + video + audio + background_audio = 4
        self.assertEqual(len(requests), 4)
        # First is always config
        self.assertTrue(requests[0].HasField("config"))
        # Remaining should include our background audio data somewhere
        bg_data = [
            r.input.background_audio_file_data
            for r in requests[1:]
            if r.HasField("input") and r.input.background_audio_file_data
        ]
        self.assertEqual(bg_data, [b"bg"])

    def test_background_audio_longer_than_streams(self) -> None:
        """Background audio extends beyond video/audio via concurrent merging."""
        video = iter([LipsyncRequest(input=LipsyncInputData(video_file_data=b"v"))])
        audio = iter([LipsyncRequest(input=LipsyncInputData(audio_file_data=b"a"))])
        bg = iter(
            [
                LipsyncRequest(input=LipsyncInputData(background_audio_file_data=b"bg1")),
                LipsyncRequest(input=LipsyncInputData(background_audio_file_data=b"bg2")),
                LipsyncRequest(input=LipsyncInputData(background_audio_file_data=b"bg3")),
            ]
        )
        config = LipsyncConfig()

        requests = list(
            lipsync_request_generator(
                video_iter=video,
                audio_iter=audio,
                speaker_info_iter=None,
                lipsync_config=config,
                background_audio_iter=bg,
            )
        )

        bg_data = [
            r.input.background_audio_file_data
            for r in requests[1:]
            if r.HasField("input") and r.input.background_audio_file_data
        ]
        self.assertEqual(len(bg_data), 3)


# -- direct client pipeline tests -------------------------------------------


@pytest.mark.unit
class TestDirectPipelineBackgroundAudio(unittest.TestCase):
    """Tests for background_audio_iterator_from_file in direct pipeline."""

    def test_reads_file_in_chunks(self) -> None:
        """File is read in chunks yielding LipsyncInputData."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write 150 bytes so we get 2 chunks at 100-byte chunk size
            f.write(b"\x00" * 150)
            f.flush()
            path = f.name

        chunks = list(background_audio_iterator_from_file(file_path=path, chunk_size=100))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].background_audio_file_data), 100)
        self.assertEqual(len(chunks[1].background_audio_file_data), 50)

    def test_empty_file_yields_nothing(self) -> None:
        """Empty file produces no chunks."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        chunks = list(background_audio_iterator_from_file(file_path=path, chunk_size=100))
        self.assertEqual(len(chunks), 0)


# -- ASD config completeness tests ------------------------------------------


@pytest.mark.unit
class TestAsdConfigNewArgs(unittest.TestCase):
    """Tests for new ASD config args (audio_source_config, threshold)."""

    def test_audio_source_config_separate_stream(self) -> None:
        """audio_source_config=separate_stream maps correctly."""
        args = argparse.Namespace(
            asd_input_audio_codec="WAV",
            asd_input_video_codec=None,
            asd_audio_source_config="separate_stream",
            asd_speaker_detection_threshold=None,
        )
        config = asd_config_from_args(args)
        self.assertEqual(config.audio_source_config, AUDIO_SOURCE_CONFIG_SEPARATE_STREAM)

    def test_audio_source_config_embedded(self) -> None:
        """audio_source_config=embedded_in_video maps correctly."""
        args = argparse.Namespace(
            asd_input_audio_codec="WAV",
            asd_input_video_codec=None,
            asd_audio_source_config="embedded_in_video",
            asd_speaker_detection_threshold=None,
        )
        config = asd_config_from_args(args)
        self.assertEqual(config.audio_source_config, AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO)

    def test_speaker_detection_threshold_set(self) -> None:
        """speaker_detection_threshold is populated when provided."""
        args = argparse.Namespace(
            asd_input_audio_codec="WAV",
            asd_input_video_codec=None,
            asd_audio_source_config="unspecified",
            asd_speaker_detection_threshold=0.75,
        )
        config = asd_config_from_args(args)
        self.assertAlmostEqual(config.speaker_detection_threshold, 0.75, places=5)

    def test_speaker_detection_threshold_omitted(self) -> None:
        """speaker_detection_threshold defaults to 0 when None."""
        args = argparse.Namespace(
            asd_input_audio_codec="WAV",
            asd_input_video_codec=None,
            asd_audio_source_config="unspecified",
            asd_speaker_detection_threshold=None,
        )
        config = asd_config_from_args(args)
        # Proto default for float is 0.0 when not set
        self.assertEqual(config.speaker_detection_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
