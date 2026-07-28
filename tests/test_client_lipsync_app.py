# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for client LipSync app helpers."""

import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pytest
from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.common.audio import AUDIO_CODEC_CONFIGS
from client.lipsync.app import _build_config_proto
from client.lipsync.config import LipSyncConfig
from client.lipsync.constants import DEFAULT_BITRATE_MBPS
from client.lipsync.constants import DEFAULT_IDR_INTERVAL
from client.lipsync.request_generators import _speaker_info_from_row
from client.lipsync.request_generators import generate_request_for_inference
from client.lipsync.request_generators import group_rows_into_per_frame_infos
from client.lipsync.response_writers import process_response_iter
from client.lipsync.response_writers import write_output_file_from_response

pytestmark = pytest.mark.unit


def test_speaker_info_from_row_basic_bbox_only() -> None:
    """CSV rows with only frame and bbox still parse correctly."""
    row = ["12", "1.0", "2.0", "3.0", "4.0"]
    frame_id, speaker = _speaker_info_from_row(row)

    assert frame_id == 12
    assert speaker.speaker_bbox.x == 1.0
    assert speaker.speaker_bbox.y == 2.0
    assert speaker.speaker_bbox.width == 3.0
    assert speaker.speaker_bbox.height == 4.0
    assert speaker.HasField("speaker_id") is False
    assert speaker.HasField("is_speaking") is False


def test_speaker_info_from_row_preserves_metadata() -> None:
    """CSV rows with ASD metadata populate speaker_id and is_speaking."""
    row = ["42", "10", "20", "30", "40", "0", "7", "True", "0.95"]
    frame_id, speaker = _speaker_info_from_row(row)

    assert frame_id == 42
    assert speaker.speaker_id == 7
    assert speaker.is_speaking is True


def test_speaker_info_from_row_false_speaking_value() -> None:
    """String false-like values are interpreted as not speaking."""
    row = ["99", "1", "2", "3", "4", "0", "11", "false", "0.1"]
    _, speaker = _speaker_info_from_row(row)

    assert speaker.speaker_id == 11
    assert speaker.is_speaking is False


def test_group_rows_into_per_frame_infos_groups_same_frame() -> None:
    """Multiple speakers in the same frame are grouped into one message."""
    rows = [
        ["5", "10", "20", "30", "40", "0", "1", "True", "0.9"],
        ["5", "50", "60", "70", "80", "0", "2", "False", "0.8"],
        ["6", "11", "21", "31", "41", "0", "3", "True", "0.7"],
    ]
    per_frame = group_rows_into_per_frame_infos(rows)

    # Frame 5 has two speakers, frame 6 has one
    assert len(per_frame) == 2
    assert per_frame[0].frame_id == 5
    assert len(per_frame[0].speaker_infos) == 2
    assert per_frame[0].speaker_infos[0].speaker_id == 1
    assert per_frame[0].speaker_infos[1].speaker_id == 2
    assert per_frame[1].frame_id == 6
    assert len(per_frame[1].speaker_infos) == 1


def _make_lipsync_config(output_filepath: str, **overrides) -> LipSyncConfig:
    """Build a LipSyncConfig for response-writer and generator tests."""
    values = {
        "audio_filepath": "a.wav",
        "video_filepath": "v.mp4",
        "speaker_info_filepath": None,
        "output_filepath": output_filepath,
        "extend_audio": "unspecified",
        "extend_video": "unspecified",
        "bitrate_mbps": DEFAULT_BITRATE_MBPS,
        "idr_interval": DEFAULT_IDR_INTERVAL,
        "lossless": False,
        "audio_codec": "wav",
        "is_speaker_info_provided": False,
        "custom_encoding_params": None,
    }
    values.update(overrides)
    return LipSyncConfig(**values)


def _cli_args(**overrides) -> Namespace:
    """Build a parsed-args stand-in for the standalone LipSync client."""
    values = {
        "lipsync_input_audio_codec": None,
        "lipsync_extend_audio": "unspecified",
        "lipsync_extend_video": "unspecified",
        "lipsync_output_bitrate_mbps": DEFAULT_BITRATE_MBPS,
        "lipsync_output_idr_interval": DEFAULT_IDR_INTERVAL,
        "lipsync_head_movement_speed": None,
        "lipsync_output_audio_codec": None,
        "lipsync_is_speaker_info_provided": False,
        "lipsync_lossless": False,
        "lipsync_custom_encoding_params": None,
        "background_audio_input": None,
        "lipsync_background_audio_codec": None,
        "lipsync_background_audio_volume": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


class TestLipsyncResponseWriters(unittest.TestCase):
    """Response writing filters by field content, never by position."""

    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp(prefix="lipsync-writers-"))
        self.addCleanup(shutil.rmtree, self.tmp_path, True)
        self.output = self.tmp_path / "out.mp4"

    def test_write_output_keeps_first_video_chunk(self) -> None:
        """The very first video chunk must be written, not skipped."""
        responses = [
            lipsync_pb2.LipsyncResponse(video_file_data=b"first"),
            lipsync_pb2.LipsyncResponse(video_file_data=b"second"),
        ]

        chunk_count = write_output_file_from_response(
            response_iter=iter(responses),
            output_filepath=str(self.output),
        )

        self.assertEqual(chunk_count, len(responses))
        self.assertEqual(self.output.read_bytes(), b"firstsecond")

    def test_process_response_iter_filters_non_video_responses(self) -> None:
        """Config echo responses are filtered by field, not position."""
        config = _make_lipsync_config(output_filepath=str(self.output))
        responses = iter(
            [
                lipsync_pb2.LipsyncResponse(config=lipsync_pb2.LipsyncConfig()),
                lipsync_pb2.LipsyncResponse(video_file_data=b"video"),
            ]
        )

        process_response_iter(response_iter=responses, lipsync_config=config)

        self.assertEqual(self.output.read_bytes(), b"video")

    def test_process_response_iter_raises_without_video_data(self) -> None:
        """A stream with no video data is an error, not a silent empty file."""
        config = _make_lipsync_config(output_filepath=str(self.output))
        responses = iter([lipsync_pb2.LipsyncResponse(config=lipsync_pb2.LipsyncConfig())])

        with self.assertRaises(RuntimeError):
            process_response_iter(response_iter=responses, lipsync_config=config)


class TestLipsyncRequestGeneration(unittest.TestCase):
    """The standalone client reuses the shared config builder and FeederStream."""

    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp(prefix="lipsync-generator-"))
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def test_build_config_proto_applies_validated_values(self) -> None:
        """The proto reuses the shared builder plus validation results."""
        config = _make_lipsync_config(
            output_filepath="o.mp4",
            speaker_info_filepath="s.csv",
            is_speaker_info_provided=True,
        )

        proto = _build_config_proto(args=_cli_args(), lipsync_config=config)

        self.assertEqual(proto.input_audio_codec, AUDIO_CODEC_CONFIGS["wav"])
        self.assertTrue(proto.is_speaker_info_provided)
        self.assertEqual(proto.output_video_encoding.lossy.bitrate_mbps, DEFAULT_BITRATE_MBPS)
        self.assertEqual(proto.output_video_encoding.lossy.idr_interval, DEFAULT_IDR_INTERVAL)

    def test_generate_request_for_inference_streams_all_inputs(self) -> None:
        """The generator emits config first, then every input stream."""
        video = self.tmp_path / "v.mp4"
        video.write_bytes(b"videodata")
        audio = self.tmp_path / "a.wav"
        audio.write_bytes(b"audiodata")
        speaker_info = self.tmp_path / "s.csv"
        speaker_info.write_text(
            "frame_id,x,y,width,height,diarized_speaker_id,face_id,is_speaking,confidence\n"
            "0,1,2,3,4,0,1,True,0.9\n"
        )
        background = self.tmp_path / "bg.mp3"
        background.write_bytes(b"bgdata")

        config = _make_lipsync_config(
            output_filepath=str(self.tmp_path / "o.mp4"),
            audio_filepath=str(audio),
            video_filepath=str(video),
            speaker_info_filepath=str(speaker_info),
            is_speaker_info_provided=True,
            background_audio_filepath=str(background),
        )
        proto = _build_config_proto(
            args=_cli_args(background_audio_input=str(background)),
            lipsync_config=config,
        )

        requests = list(generate_request_for_inference(lipsync_config=config, config_proto=proto))

        self.assertTrue(requests[0].HasField("config"))

        payload_fields = [self._payload_field(request) for request in requests[1:]]
        self.assertEqual(
            sorted(payload_fields),
            [
                "audio_file_data",
                "background_audio_file_data",
                "per_frame_speaker_infos",
                "video_file_data",
            ],
        )
        by_field = dict(zip(payload_fields, requests[1:], strict=True))
        self.assertEqual(by_field["video_file_data"].input.video_file_data, b"videodata")
        self.assertEqual(by_field["audio_file_data"].input.audio_file_data, b"audiodata")
        self.assertEqual(
            by_field["background_audio_file_data"].input.background_audio_file_data,
            b"bgdata",
        )
        speaker_batch = by_field["per_frame_speaker_infos"].input.per_frame_speaker_infos
        self.assertEqual(speaker_batch[0].frame_id, 0)
        self.assertEqual(speaker_batch[0].speaker_infos[0].speaker_id, 1)

    def test_audio_priming_chunk_is_sent_before_other_streams(self) -> None:
        """The first data message after the config is always audio."""
        video = self.tmp_path / "v.mp4"
        video.write_bytes(b"videodata" * 1024)
        audio = self.tmp_path / "a.wav"
        audio.write_bytes(b"audiodata" * 1024)

        config = _make_lipsync_config(
            output_filepath=str(self.tmp_path / "o.mp4"),
            audio_filepath=str(audio),
            video_filepath=str(video),
        )
        proto = _build_config_proto(args=_cli_args(), lipsync_config=config)

        requests = list(generate_request_for_inference(lipsync_config=config, config_proto=proto))

        self.assertTrue(requests[0].HasField("config"))
        self.assertTrue(requests[1].input.HasField("audio_file_data"))

    def test_primed_audio_stream_is_complete(self) -> None:
        """Priming hands the advanced iterator over without losing chunks."""
        video = self.tmp_path / "v.mp4"
        video.write_bytes(b"videodata")
        audio = self.tmp_path / "a.wav"
        # Spans multiple DATA_CHUNK_SIZE chunks so the priming chunk and the
        # feeder-delivered remainder must join up exactly.
        audio_bytes = bytes(range(256)) * 1024  # 256 KiB
        audio.write_bytes(audio_bytes)

        config = _make_lipsync_config(
            output_filepath=str(self.tmp_path / "o.mp4"),
            audio_filepath=str(audio),
            video_filepath=str(video),
        )
        proto = _build_config_proto(args=_cli_args(), lipsync_config=config)

        requests = list(generate_request_for_inference(lipsync_config=config, config_proto=proto))

        received_audio = b"".join(
            request.input.audio_file_data
            for request in requests[1:]
            if request.input.HasField("audio_file_data")
        )
        self.assertEqual(received_audio, audio_bytes)

    def test_empty_audio_file_skips_priming_chunk(self) -> None:
        """An empty audio file yields no priming chunk and no audio data."""
        video = self.tmp_path / "v.mp4"
        video.write_bytes(b"videodata")
        audio = self.tmp_path / "a.wav"
        audio.write_bytes(b"")

        config = _make_lipsync_config(
            output_filepath=str(self.tmp_path / "o.mp4"),
            audio_filepath=str(audio),
            video_filepath=str(video),
        )
        proto = _build_config_proto(args=_cli_args(), lipsync_config=config)

        requests = list(generate_request_for_inference(lipsync_config=config, config_proto=proto))

        self.assertTrue(requests[0].HasField("config"))
        audio_messages = [
            request for request in requests[1:] if request.input.HasField("audio_file_data")
        ]
        self.assertEqual(audio_messages, [])
        video_messages = [
            request for request in requests[1:] if request.input.HasField("video_file_data")
        ]
        self.assertEqual(len(video_messages), 1)

    def _payload_field(self, request: lipsync_pb2.LipsyncRequest) -> str:
        """Return the name of the populated LipsyncInputData payload field."""
        data = request.input
        if data.HasField("video_file_data"):
            return "video_file_data"
        if data.HasField("audio_file_data"):
            return "audio_file_data"
        if data.HasField("background_audio_file_data"):
            return "background_audio_file_data"
        self.assertGreater(len(data.per_frame_speaker_infos), 0)
        return "per_frame_speaker_infos"
