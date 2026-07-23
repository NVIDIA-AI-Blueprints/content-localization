# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ElevenLabs diarization segment merging.

Verifies that consecutive words from the same speaker are merged into a
single segment, matching the demo app behavior.
"""

import json
from pathlib import Path

import pytest

from client.common.diarization import load_diarization_info

pytestmark = pytest.mark.unit


def test_consecutive_same_speaker_words_are_merged(tmp_path: Path) -> None:
    """Consecutive words from the same speaker merge into one segment."""
    diarization_path = tmp_path / "el_merge.json"
    diarization_path.write_text(
        json.dumps(
            {
                "language_code": "eng",
                "text": "hello beautiful world",
                "words": [
                    {
                        "text": "hello",
                        "start": 0.5,
                        "end": 0.8,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": " ",
                        "start": 0.8,
                        "end": 0.85,
                        "type": "spacing",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "beautiful",
                        "start": 0.9,
                        "end": 1.2,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": " ",
                        "start": 1.2,
                        "end": 1.25,
                        "type": "spacing",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "world",
                        "start": 1.3,
                        "end": 1.6,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    info = load_diarization_info(
        diarization_file=str(diarization_path),
        diarization_format="elevenlabs-scribe",
    )

    assert info is not None
    # Three words from the same speaker → one merged segment
    assert len(info.segments) == 1
    assert info.segments[0].speaker_id == 0
    assert info.segments[0].word == "hello beautiful world"
    # Start time from first word, end time from last word
    assert info.segments[0].start_time == 500
    assert info.segments[0].end_time == 1600
    assert info.segments[0].language_code == "eng"


def test_speaker_change_creates_new_segment(tmp_path: Path) -> None:
    """Speaker changes split words into separate merged segments."""
    diarization_path = tmp_path / "el_split.json"
    diarization_path.write_text(
        json.dumps(
            {
                "language_code": "eng",
                "text": "hello world goodbye",
                "words": [
                    {
                        "text": "hello",
                        "start": 0.0,
                        "end": 0.3,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "world",
                        "start": 0.4,
                        "end": 0.7,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "goodbye",
                        "start": 1.0,
                        "end": 1.5,
                        "type": "word",
                        "speaker_id": "speaker_1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    info = load_diarization_info(
        diarization_file=str(diarization_path),
        diarization_format="elevenlabs-scribe",
    )

    assert info is not None
    assert len(info.segments) == 2
    # First segment: speaker_0 with two merged words
    assert info.segments[0].speaker_id == 0
    assert info.segments[0].word == "hello world"
    assert info.segments[0].start_time == 0
    assert info.segments[0].end_time == 700
    # Second segment: speaker_1 with one word
    assert info.segments[1].speaker_id == 1
    assert info.segments[1].word == "goodbye"
    assert info.segments[1].start_time == 1000
    assert info.segments[1].end_time == 1500


def test_alternating_speakers_no_merging(tmp_path: Path) -> None:
    """Alternating speakers produce one segment per word (no merging)."""
    diarization_path = tmp_path / "el_alternate.json"
    diarization_path.write_text(
        json.dumps(
            {
                "text": "a b c",
                "words": [
                    {
                        "text": "a",
                        "start": 0.0,
                        "end": 0.1,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "b",
                        "start": 0.2,
                        "end": 0.3,
                        "type": "word",
                        "speaker_id": "speaker_1",
                    },
                    {
                        "text": "c",
                        "start": 0.4,
                        "end": 0.5,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    info = load_diarization_info(
        diarization_file=str(diarization_path),
        diarization_format="elevenlabs-scribe",
    )

    assert info is not None
    assert len(info.segments) == 3
    assert info.segments[0].speaker_id == 0
    assert info.segments[0].word == "a"
    assert info.segments[1].speaker_id == 1
    assert info.segments[1].word == "b"
    assert info.segments[2].speaker_id == 0
    assert info.segments[2].word == "c"


def test_combine_disabled_keeps_one_segment_per_word(tmp_path: Path) -> None:
    """combine_chunks_per_speaker=False keeps consecutive same-speaker words split."""
    diarization_path = tmp_path / "el_no_merge.json"
    diarization_path.write_text(
        json.dumps(
            {
                "language_code": "eng",
                "text": "hello beautiful world",
                "words": [
                    {
                        "text": "hello",
                        "start": 0.5,
                        "end": 0.8,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": " ",
                        "start": 0.8,
                        "end": 0.85,
                        "type": "spacing",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "beautiful",
                        "start": 0.9,
                        "end": 1.2,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                    {
                        "text": "world",
                        "start": 1.3,
                        "end": 1.6,
                        "type": "word",
                        "speaker_id": "speaker_0",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    info = load_diarization_info(
        diarization_file=str(diarization_path),
        diarization_format="elevenlabs-scribe",
        combine_chunks_per_speaker=False,
    )

    assert info is not None
    # Spacing is still filtered, but the three words stay as separate segments.
    assert len(info.segments) == 3
    assert [seg.word for seg in info.segments] == ["hello", "beautiful", "world"]
    assert [seg.speaker_id for seg in info.segments] == [0, 0, 0]
    assert info.segments[0].start_time == 500
    assert info.segments[0].end_time == 800
    # Each word keeps its own language_code.
    assert all(seg.language_code == "eng" for seg in info.segments)


def test_combine_flag_applies_to_non_scribe_format(tmp_path: Path) -> None:
    """combine_chunks_per_speaker merges consecutive same-speaker camb entries too."""
    diarization_path = tmp_path / "camb_same_speaker.json"
    diarization_path.write_text(
        json.dumps(
            [
                {"start": 0.5, "end": 1.2, "text": "hello", "speaker": "Speaker 1"},
                {"start": 1.5, "end": 2.3, "text": "world", "speaker": "Speaker 1"},
            ]
        ),
        encoding="utf-8",
    )

    # Default (combine=True): two same-speaker entries merge into one segment.
    merged = load_diarization_info(
        diarization_file=str(diarization_path),
        diarization_format="camb",
    )
    assert merged is not None
    assert len(merged.segments) == 1
    assert merged.segments[0].speaker_id == 0
    assert merged.segments[0].word == "hello world"
    assert merged.segments[0].start_time == 500
    assert merged.segments[0].end_time == 2300

    # combine=False: each camb entry stays its own segment.
    split = load_diarization_info(
        diarization_file=str(diarization_path),
        diarization_format="camb",
        combine_chunks_per_speaker=False,
    )
    assert split is not None
    assert len(split.segments) == 2
    assert [seg.word for seg in split.segments] == ["hello", "world"]
