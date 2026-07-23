# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101,PLR0913

"""Unit tests for ElevenLabs standalone dubbing script."""

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest

from scripts.elevenlabs.s2s_infer import DubbingResult
from scripts.elevenlabs.s2s_infer import download_dubbing_transcript
from scripts.elevenlabs.s2s_infer import extract_elevenlabs_transcript_content
from scripts.elevenlabs.s2s_infer import main
from scripts.elevenlabs.s2s_infer import normalize_elevenlabs_transcript_format
from scripts.elevenlabs.s2s_infer import write_transcript_content

pytestmark = pytest.mark.unit


class FakeJsonPayload:
    """Minimal SDK-like JSON transcript payload."""

    def model_dump(self) -> dict:
        """Return a JSON-serializable transcript."""
        return {"language": "es", "utterances": []}


class FakeTranscriptResponse:
    """Minimal SDK-like transcript response model."""

    def __init__(
        self,
        *,
        json_payload: object | None = None,
        srt: str | None = None,
        webvtt: str | None = None,
    ) -> None:
        self.json_ = json_payload
        self.srt = srt
        self.webvtt = webvtt


def test_normalize_elevenlabs_transcript_format_accepts_vtt_alias() -> None:
    """CLI vtt alias maps to ElevenLabs webvtt API value."""
    assert normalize_elevenlabs_transcript_format("vtt") == "webvtt"


def test_extract_elevenlabs_transcript_content_json() -> None:
    """JSON transcript extraction returns the diarized transcript object."""
    content = extract_elevenlabs_transcript_content(
        FakeTranscriptResponse(json_payload=FakeJsonPayload()),
        "json",
    )

    assert content == {"language": "es", "utterances": []}


def test_extract_elevenlabs_transcript_content_srt() -> None:
    """SRT transcript extraction returns subtitle text."""
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHola\n"

    content = extract_elevenlabs_transcript_content(
        FakeTranscriptResponse(srt=srt),
        "srt",
    )

    assert content == srt


def test_extract_elevenlabs_transcript_content_webvtt() -> None:
    """WebVTT transcript extraction returns subtitle text."""
    webvtt = "WEBVTT\n\n00:00.000 --> 00:01.000\nHola\n"

    content = extract_elevenlabs_transcript_content(
        FakeTranscriptResponse(webvtt=webvtt),
        "vtt",
    )

    assert content == webvtt


def test_write_transcript_content_json(tmp_path: Path) -> None:
    """JSON transcript content is pretty-written to disk."""
    output_path = tmp_path / "transcript.json"

    result = write_transcript_content(
        {"language": "es", "utterances": []},
        output_path,
    )

    assert result == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "language": "es",
        "utterances": [],
    }


def test_write_transcript_content_text(tmp_path: Path) -> None:
    """Text transcript content is written unchanged."""
    output_path = tmp_path / "transcript.vtt"

    result = write_transcript_content("WEBVTT\n", output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "WEBVTT\n"


def test_download_dubbing_transcript_uses_python_sdk(tmp_path: Path) -> None:
    """Transcript download uses the ElevenLabs SDK transcript endpoint."""
    output_path = tmp_path / "target.vtt"
    client = MagicMock()
    client.dubbing.transcripts.get.return_value = FakeTranscriptResponse(webvtt="WEBVTT\n")

    result = download_dubbing_transcript(
        client=client,
        dubbing_id="dub-1",
        language_code="es",
        transcript_format="vtt",
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "WEBVTT\n"
    client.dubbing.transcripts.get.assert_called_once_with(
        dubbing_id="dub-1",
        language_code="es",
        format_type="webvtt",
    )


@patch("scripts.elevenlabs.s2s_infer.download_dubbing_transcript")
@patch("scripts.elevenlabs.s2s_infer.create_dub_from_file")
@patch("scripts.elevenlabs.s2s_infer.convert_video_to_audio_ffmpeg")
@patch("scripts.elevenlabs.s2s_infer.tempfile.mkstemp")
@patch("scripts.elevenlabs.s2s_infer.ElevenLabs")
@patch("scripts.elevenlabs.s2s_infer.os.getenv")
@patch("scripts.elevenlabs.s2s_infer.parse_args")
def test_main_writes_source_and_target_transcripts(
    mock_parse_args: MagicMock,
    mock_getenv: MagicMock,
    mock_elevenlabs: MagicMock,
    mock_mkstemp: MagicMock,
    mock_convert: MagicMock,
    mock_create_dub: MagicMock,
    mock_download_transcript: MagicMock,
) -> None:
    """Main flow fetches both source and target transcripts when requested."""
    client = MagicMock()
    mock_elevenlabs.return_value = client
    mock_getenv.return_value = "test-key"
    mock_mkstemp.return_value = (3, "audio.wav")
    mock_parse_args.return_value = Namespace(
        input_file=Path("inputs/video.mp4"),
        output_file=Path("outputs/audio.wav"),
        source_language_code="en",
        target_language_code="es",
        source_transcript_output_file=Path("outputs/source.vtt"),
        target_transcript_output_file=Path("outputs/target.vtt"),
        transcript_format="vtt",
    )
    mock_create_dub.return_value = DubbingResult(
        output_path=Path("outputs/audio.wav"),
        dubbing_id="dub-1",
    )
    mock_download_transcript.side_effect = [
        Path("outputs/source.vtt"),
        Path("outputs/target.vtt"),
    ]

    main()

    mock_convert.assert_called_once()
    mock_create_dub.assert_called_once()
    mock_download_transcript.assert_has_calls(
        [
            call(
                client=client,
                dubbing_id="dub-1",
                language_code="en",
                transcript_format="vtt",
                output_path=Path("outputs/source.vtt"),
            ),
            call(
                client=client,
                dubbing_id="dub-1",
                language_code="es",
                transcript_format="vtt",
                output_path=Path("outputs/target.vtt"),
            ),
        ]
    )
