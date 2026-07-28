# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ContentLocalizationDeserializer."""

import unittest
from unittest.mock import patch

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioSegmentInfo
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from common.buffers import RequestIteratorFromBuffer
from controller_service.deserializer import ContentLocalizationDeserializer

pytestmark = pytest.mark.unit


def _audio_request(audio_bytes: bytes = b"\x00") -> ContentLocalizationRequest:
    """Create a request with audio data."""
    return ContentLocalizationRequest(audio_data=audio_bytes)


def _video_request(video_bytes: bytes = b"\x01") -> ContentLocalizationRequest:
    """Create a request with video data."""
    return ContentLocalizationRequest(video_file_data=video_bytes)


def _request_with_two_payload_fields(
    audio_bytes: bytes = b"\x00", video_bytes: bytes = b"\x01"
) -> ContentLocalizationRequest:
    """Create a request setting two payload fields; the oneof keeps the last."""
    return ContentLocalizationRequest(audio_data=audio_bytes, video_file_data=video_bytes)


def _config_request() -> ContentLocalizationRequest:
    """Create a request with s2s_config only."""
    return ContentLocalizationRequest(s2s_config=SpeechToSpeechConfig())


def _asd_config_request() -> ContentLocalizationRequest:
    """Create a request with asd_config."""
    return ContentLocalizationRequest(
        asd_config=ActiveSpeakerDetectionConfig(
            input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV),
        )
    )


def _lipsync_config_request() -> ContentLocalizationRequest:
    """Create a request with lipsync_config."""
    return ContentLocalizationRequest(
        lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
    )


def _diarization_request(
    start_time: int = 0, end_time: int = 1000, speaker_id: int = 0
) -> ContentLocalizationRequest:
    """Create a request with diarization_info."""
    return ContentLocalizationRequest(
        diarization_info=AudioDiarizationInfo(
            segments=[
                AudioSegmentInfo(start_time=start_time, end_time=end_time, speaker_id=speaker_id)
            ]
        )
    )


class TestContentLocalizationDeserializer(unittest.TestCase):
    """Tests for routing, buffer state, and iterator draining."""

    def _run_deserializer(self, requests):
        """Helper: create, run, and join a deserializer on a list of requests."""
        ds = ContentLocalizationDeserializer(iter(requests))
        ds.start(request_id="test")
        ds.join(timeout=5.0)
        return ds

    def test_audio_only_routes_to_audio_buffer(self) -> None:
        """Audio-only requests go to audio_buffer, not video_buffer."""
        ds = self._run_deserializer([_audio_request(b"a1"), _audio_request(b"a2")])

        self.assertTrue(ds.audio_buffer.done)
        self.assertTrue(ds.video_buffer.done)
        # Audio buffer has 2 consumer queues, both should have 2 items
        self.assertEqual(ds.audio_buffer.qsize(0), 2)
        self.assertEqual(ds.audio_buffer.qsize(1), 2)
        # Video buffer has 2 consumer queues, both should be empty
        self.assertTrue(ds.video_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(1))

    def test_video_only_routes_to_video_buffer(self) -> None:
        """Video-only requests go to video_buffer (both consumers), not audio_buffer."""
        ds = self._run_deserializer([_video_request(b"v1")])

        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.audio_buffer.empty(1))
        # Both consumer queues should have the item
        self.assertEqual(ds.video_buffer.qsize(0), 1)
        self.assertEqual(ds.video_buffer.qsize(1), 1)

    def test_payload_oneof_keeps_only_the_last_field_set(self) -> None:
        """Each request carries exactly one payload; the last field set wins."""
        ds = self._run_deserializer([_request_with_two_payload_fields(b"a", b"v")])

        self.assertEqual(ds.audio_buffer.qsize(0), 0)
        self.assertEqual(ds.audio_buffer.qsize(1), 0)
        self.assertEqual(ds.video_buffer.qsize(0), 1)
        self.assertEqual(ds.video_buffer.qsize(1), 1)

    def test_config_request_routes_to_audio_buffer(self) -> None:
        """s2s_config-only requests go to s2s_config_buffer."""
        ds = self._run_deserializer([_config_request()])

        self.assertEqual(ds.s2s_config_buffer.qsize(0), 1)
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.audio_buffer.empty(1))
        self.assertTrue(ds.video_buffer.empty(0))

    def test_asd_config_routes_to_asd_config_buffer(self) -> None:
        """asd_config requests go to asd_config_buffer only."""
        ds = self._run_deserializer([_asd_config_request()])

        self.assertEqual(ds.asd_config_buffer.qsize(0), 1)
        self.assertTrue(ds.s2s_config_buffer.empty(0))
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(0))

    def test_lipsync_config_routes_to_lipsync_config_buffer(self) -> None:
        """lipsync_config requests go to lipsync_config_buffer only."""
        ds = self._run_deserializer([_lipsync_config_request()])

        self.assertEqual(ds.lipsync_config_buffer.qsize(0), 1)
        self.assertTrue(ds.s2s_config_buffer.empty(0))
        self.assertTrue(ds.asd_config_buffer.empty(0))
        self.assertTrue(ds.audio_buffer.empty(0))

    def test_done_flags_set_on_all_buffers(self) -> None:
        """All buffers are marked done when stream exhausts."""
        ds = self._run_deserializer([])

        self.assertTrue(ds.audio_buffer.done)
        self.assertTrue(ds.s2s_config_buffer.done)
        self.assertTrue(ds.asd_config_buffer.done)
        self.assertTrue(ds.lipsync_config_buffer.done)
        self.assertTrue(ds.video_buffer.done)
        self.assertTrue(ds.diarization_buffer.done)
        self.assertTrue(ds.background_audio_buffer.done)

    def test_s2s_iterator_drains_audio_buffer(self) -> None:
        """s2s_iterator yields all audio requests."""
        ds = self._run_deserializer([_audio_request(b"a1"), _audio_request(b"a2")])

        items = list(RequestIteratorFromBuffer(ds.audio_buffer, consumer_id=0, poll_timeout=0.01))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].audio_data, b"a1")
        self.assertEqual(items[1].audio_data, b"a2")

    def test_asd_and_lipsync_iterators_drain_video_buffer(self) -> None:
        """asd_iterator and lipsync_video_iterator each get all video requests."""
        ds = self._run_deserializer([_video_request(b"v1"), _video_request(b"v2")])

        asd_items = list(
            RequestIteratorFromBuffer(ds.video_buffer, consumer_id=0, poll_timeout=0.01)
        )
        lipsync_items = list(
            RequestIteratorFromBuffer(ds.video_buffer, consumer_id=1, poll_timeout=0.01)
        )

        self.assertEqual(len(asd_items), 2)
        self.assertEqual(len(lipsync_items), 2)
        self.assertEqual(asd_items[0].video_file_data, b"v1")
        self.assertEqual(lipsync_items[0].video_file_data, b"v1")

    def test_video_copies_are_isolated(self) -> None:
        """ASD and LipSync consumer queues hold independent copies."""
        ds = self._run_deserializer([_video_request(b"v")])

        asd_item = ds.video_buffer.get(consumer_id=0)
        ls_item = ds.video_buffer.get(consumer_id=1)

        # They should be equal in value but distinct objects (deepcopy)
        self.assertEqual(asd_item.video_file_data, ls_item.video_file_data)
        self.assertIsNot(asd_item, ls_item)

    def test_audio_copies_are_isolated(self) -> None:
        """Both audio consumer queues hold independent copies."""
        ds = self._run_deserializer([_audio_request(b"a")])

        c0_item = ds.audio_buffer.get(consumer_id=0)
        c1_item = ds.audio_buffer.get(consumer_id=1)

        self.assertEqual(c0_item.audio_data, c1_item.audio_data)
        self.assertIsNot(c0_item, c1_item)

    def test_multiple_payload_requests(self) -> None:
        """Realistic stream with config, audio, and video messages."""
        requests = [
            _config_request(),  # s2s_config_buffer
            _audio_request(b"a1"),  # audio buffer
            _video_request(b"v1"),  # video buffer
            _video_request(b"v2"),  # video buffer
            _audio_request(b"a3"),  # audio buffer
        ]
        ds = self._run_deserializer(requests)

        # audio_buffer: a1, a3 = 2 items per consumer
        self.assertEqual(ds.audio_buffer.qsize(0), 2)
        self.assertEqual(ds.audio_buffer.qsize(1), 2)
        # s2s_config_buffer: config only
        self.assertEqual(ds.s2s_config_buffer.qsize(0), 1)
        # video_buffer: v1, v2 = 2 items per consumer
        self.assertEqual(ds.video_buffer.qsize(0), 2)
        self.assertEqual(ds.video_buffer.qsize(1), 2)

    # -- diarization routing tests ------------------------------------------

    def test_diarization_routes_to_diarization_buffer(self) -> None:
        """Diarization requests go to diarization_buffer only."""
        ds = self._run_deserializer([_diarization_request()])

        self.assertEqual(ds.diarization_buffer.qsize(0), 1)
        # Should not leak into audio or video buffers
        self.assertTrue(ds.s2s_config_buffer.empty(0))
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(0))

    def test_diarization_buffer_done_on_complete(self) -> None:
        """diarization_buffer.done is True after stream exhausts."""
        ds = self._run_deserializer([_diarization_request()])

        self.assertTrue(ds.diarization_buffer.done)

    def test_no_diarization_leaves_buffer_empty(self) -> None:
        """Stream without diarization leaves diarization_buffer empty but done."""
        ds = self._run_deserializer([_audio_request(), _video_request()])

        self.assertTrue(ds.diarization_buffer.done)
        self.assertTrue(ds.diarization_buffer.empty(0))

    def test_multiple_diarization_chunks(self) -> None:
        """Multiple diarization requests all route to diarization_buffer."""
        requests = [
            _diarization_request(0, 1000, 0),
            _diarization_request(1000, 2000, 1),
            _diarization_request(2000, 3000, 0),
        ]
        ds = self._run_deserializer(requests)

        self.assertEqual(ds.diarization_buffer.qsize(0), 3)
        self.assertTrue(ds.s2s_config_buffer.empty(0))
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(0))

    def test_mixed_stream_with_diarization(self) -> None:
        """Realistic stream with config, audio, video, and diarization routes correctly."""
        requests = [
            _config_request(),
            _diarization_request(0, 500, 0),
            _diarization_request(500, 1000, 1),
            _audio_request(b"a1"),
            _video_request(b"v1"),
            _audio_request(b"a2"),
            _video_request(b"v2"),
        ]
        ds = self._run_deserializer(requests)

        # audio_buffer: a1 + a2 = 2 per consumer
        self.assertEqual(ds.audio_buffer.qsize(0), 2)
        self.assertEqual(ds.audio_buffer.qsize(1), 2)
        # s2s_config_buffer: config only
        self.assertEqual(ds.s2s_config_buffer.qsize(0), 1)
        # video_buffer: v1 + v2 = 2 per consumer
        self.assertEqual(ds.video_buffer.qsize(0), 2)
        self.assertEqual(ds.video_buffer.qsize(1), 2)
        # diarization_buffer: 2 chunks
        self.assertEqual(ds.diarization_buffer.qsize(0), 2)

    def test_mixed_stream_with_all_configs(self) -> None:
        """Realistic stream with s2s, asd, lipsync configs, audio, video, diarization."""
        requests = [
            _config_request(),
            _asd_config_request(),
            _lipsync_config_request(),
            _audio_request(b"a1"),
            _video_request(b"v1"),
            _diarization_request(0, 500, 0),
            _audio_request(b"a2"),
            _video_request(b"v2"),
        ]
        ds = self._run_deserializer(requests)

        self.assertEqual(ds.s2s_config_buffer.qsize(0), 1)
        self.assertEqual(ds.asd_config_buffer.qsize(0), 1)
        self.assertEqual(ds.lipsync_config_buffer.qsize(0), 1)
        self.assertEqual(ds.audio_buffer.qsize(0), 2)
        self.assertEqual(ds.audio_buffer.qsize(1), 2)
        self.assertEqual(ds.video_buffer.qsize(0), 2)
        self.assertEqual(ds.video_buffer.qsize(1), 2)
        self.assertEqual(ds.diarization_buffer.qsize(0), 1)

    def test_diarization_iterator_drains_buffer(self) -> None:
        """RequestIteratorFromBuffer yields all diarization requests."""
        ds = self._run_deserializer(
            [_diarization_request(0, 500, 0), _diarization_request(500, 1000, 1)]
        )

        items = list(
            RequestIteratorFromBuffer(ds.diarization_buffer, consumer_id=0, poll_timeout=0.01)
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].diarization_info.segments[0].speaker_id, 0)
        self.assertEqual(items[1].diarization_info.segments[0].speaker_id, 1)

    # -- translated audio routing tests ------------------------------------

    def test_translated_audio_routes_to_translated_audio_buffer(self) -> None:
        """Translated audio requests go to translated_audio_buffer only."""
        req = ContentLocalizationRequest(translated_audio_data=b"ta1")
        ds = self._run_deserializer([req])

        self.assertEqual(ds.translated_audio_buffer.qsize(0), 1)
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(0))
        self.assertTrue(ds.s2s_config_buffer.empty(0))

    def test_multiple_translated_audio_chunks(self) -> None:
        """Multiple translated audio packets all route to translated_audio_buffer."""
        requests = [
            ContentLocalizationRequest(translated_audio_data=b"ta1"),
            ContentLocalizationRequest(translated_audio_data=b"ta2"),
            ContentLocalizationRequest(translated_audio_data=b"ta3"),
        ]
        ds = self._run_deserializer(requests)

        self.assertEqual(ds.translated_audio_buffer.qsize(0), 3)

    def test_translated_audio_iterator_drains_buffer(self) -> None:
        """RequestIteratorFromBuffer yields all translated audio requests."""
        requests = [
            ContentLocalizationRequest(translated_audio_data=b"ta1"),
            ContentLocalizationRequest(translated_audio_data=b"ta2"),
        ]
        ds = self._run_deserializer(requests)

        items = list(
            RequestIteratorFromBuffer(ds.translated_audio_buffer, consumer_id=0, poll_timeout=0.01)
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].translated_audio_data, b"ta1")
        self.assertEqual(items[1].translated_audio_data, b"ta2")

    def test_translated_audio_buffer_done_on_complete(self) -> None:
        """translated_audio_buffer.done is True after stream exhausts."""
        ds = self._run_deserializer([])

        self.assertTrue(ds.translated_audio_buffer.done)

    def test_no_translated_audio_leaves_buffer_empty(self) -> None:
        """Stream without translated audio leaves buffer empty but done."""
        ds = self._run_deserializer([_audio_request()])

        self.assertTrue(ds.translated_audio_buffer.done)
        self.assertTrue(ds.translated_audio_buffer.empty(0))

    # -- controller config routing tests -----------------------------------

    def test_controller_config_routes_to_controller_config_buffer(self) -> None:
        """Controller config requests go to controller_config_buffer only."""
        req = ContentLocalizationRequest(
            controller_config=ContentLocalizationConfig(bypass_s2s=True),
        )
        ds = self._run_deserializer([req])

        self.assertEqual(ds.controller_config_buffer.qsize(0), 1)
        self.assertTrue(ds.audio_buffer.empty(0))
        self.assertTrue(ds.video_buffer.empty(0))
        self.assertTrue(ds.translated_audio_buffer.empty(0))

    def test_controller_config_buffer_done_on_complete(self) -> None:
        """controller_config_buffer.done is True after stream exhausts."""
        ds = self._run_deserializer([])

        self.assertTrue(ds.controller_config_buffer.done)

    def test_controller_config_bypass_s2s_value_preserved(self) -> None:
        """bypass_s2s flag is preserved through the buffer round-trip."""
        req = ContentLocalizationRequest(
            controller_config=ContentLocalizationConfig(bypass_s2s=True),
        )
        ds = self._run_deserializer([req])

        item = ds.controller_config_buffer.get(consumer_id=0)
        self.assertTrue(item.controller_config.bypass_s2s)

    # -- done flags include new buffers ------------------------------------

    def test_done_flags_include_new_buffers(self) -> None:
        """All buffers including translated_audio and controller_config are marked done."""
        ds = self._run_deserializer([])

        self.assertTrue(ds.translated_audio_buffer.done)
        self.assertTrue(ds.controller_config_buffer.done)

    # -- mixed stream with bypass fields -----------------------------------

    def test_mixed_stream_with_bypass_fields(self) -> None:
        """Realistic bypass-S2S stream routes all fields correctly."""
        requests = [
            ContentLocalizationRequest(
                controller_config=ContentLocalizationConfig(bypass_s2s=True),
            ),
            _asd_config_request(),
            _lipsync_config_request(),
            _audio_request(b"a1"),
            _video_request(b"v1"),
            _diarization_request(0, 500, 0),
            ContentLocalizationRequest(translated_audio_data=b"ta1"),
            ContentLocalizationRequest(translated_audio_data=b"ta2"),
        ]
        ds = self._run_deserializer(requests)

        self.assertEqual(ds.controller_config_buffer.qsize(0), 1)
        self.assertEqual(ds.asd_config_buffer.qsize(0), 1)
        self.assertEqual(ds.lipsync_config_buffer.qsize(0), 1)
        self.assertEqual(ds.audio_buffer.qsize(0), 1)
        self.assertEqual(ds.video_buffer.qsize(0), 1)
        self.assertEqual(ds.diarization_buffer.qsize(0), 1)
        self.assertEqual(ds.translated_audio_buffer.qsize(0), 2)
        # S2S config buffer should be empty in bypass mode
        self.assertTrue(ds.s2s_config_buffer.empty(0))

    # -- request id handling ------------------------------------------------

    def test_first_request_id_wins(self) -> None:
        """The first client-supplied request_id is kept for the whole stream."""
        requests = [
            ContentLocalizationRequest(request_id="first", audio_data=b"a1"),
            ContentLocalizationRequest(request_id="first", audio_data=b"a2"),
        ]
        ds = self._run_deserializer(requests)

        self.assertEqual(ds.client_request_id, "first")

    def test_conflicting_late_request_id_warns_once(self) -> None:
        """A differing mid-stream request_id is ignored with a single warning."""
        requests = [
            ContentLocalizationRequest(request_id="first", audio_data=b"a1"),
            ContentLocalizationRequest(request_id="second", audio_data=b"a2"),
            ContentLocalizationRequest(request_id="second", audio_data=b"a3"),
        ]
        with patch("controller_service.deserializer.logger") as mock_logger:
            ds = self._run_deserializer(requests)

        self.assertEqual(ds.client_request_id, "first")
        warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
        id_warnings = [msg for msg in warnings if "'second'" in msg]
        self.assertEqual(len(id_warnings), 1)
        self.assertIn("'first'", id_warnings[0])
