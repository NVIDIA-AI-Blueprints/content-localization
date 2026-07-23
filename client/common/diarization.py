# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diarization parsers for ASD provider and service-specific formats."""

import csv
import json

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioSegmentInfo


def _build_segment_from_flat_entry(seg: dict, index: int) -> AudioSegmentInfo:
    """Build ``AudioSegmentInfo`` from a single flat ASD-compatible JSON entry.

    Args:
        seg (dict): Flat JSON object with required keys ``start_time``,
            ``end_time``, ``speaker_id`` and optional ``word``, ``language_code``.
        index (int): Position of this entry in the source list (for error messages).

    Returns:
        AudioSegmentInfo: Populated protobuf message.

    Raises:
        ValueError: If required fields are missing from *seg*.

    Examples:
        >>> seg = {"start_time": 10, "end_time": 50, "speaker_id": 1, "word": "hi"}
        >>> info = _build_segment_from_flat_entry(seg=seg, index=0)
        >>> info.speaker_id
        1
    """
    required_fields = ("start_time", "end_time", "speaker_id")
    missing = [field for field in required_fields if field not in seg]
    if missing:
        raise ValueError(
            f"Invalid flat diarization entry at index {index}: missing fields {missing}."
        )

    segment_kwargs = {
        "start_time": int(seg["start_time"]),
        "end_time": int(seg["end_time"]),
        "speaker_id": int(seg["speaker_id"]),
    }
    if "word" in seg:
        segment_kwargs["word"] = str(seg["word"])
    if "language_code" in seg:
        segment_kwargs["language_code"] = str(seg["language_code"])
    return AudioSegmentInfo(**segment_kwargs)


def _build_segments_from_flat_json(data: list[dict]) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from a flat ASD-compatible JSON list.

    Args:
        data (list[dict]): List of flat segment dictionaries. Each must contain
            ``start_time``, ``end_time``, ``speaker_id``. An optional
            ``transcript`` key on the first entry is used as the overall transcript.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos and
            transcript string (``None`` if absent).

    Raises:
        ValueError: If *data* is not a list.

    Examples:
        >>> data = [{"start_time": 0, "end_time": 100, "speaker_id": 0}]
        >>> segments, transcript = _build_segments_from_flat_json(data=data)
        >>> len(segments)
        1
    """
    if not isinstance(data, list):
        raise ValueError(
            f"Invalid flat diarization JSON: expected a top-level list, got {type(data).__name__}."
        )
    segments = [_build_segment_from_flat_entry(seg=seg, index=idx) for idx, seg in enumerate(data)]
    transcript = None
    if data and isinstance(data[0], dict):
        raw_transcript = data[0].get("transcript")
        if raw_transcript is not None:
            transcript = str(raw_transcript)
    return segments, transcript


def _parse_elevenlabs_speaker_id(speaker_id: str | None) -> int:
    """Convert an ElevenLabs speaker ID string to an integer.

    Handles formats like ``"speaker_0"``, ``"speaker_12"``, or plain
    numeric strings like ``"0"``.  Returns ``0`` when the input is
    ``None`` or cannot be parsed.

    Args:
        speaker_id (str | None): The raw ``speaker_id`` value from the
            ElevenLabs STT response.

    Returns:
        int: Extracted integer speaker ID.

    Examples:
        >>> _parse_elevenlabs_speaker_id(speaker_id="speaker_0")
        0
        >>> _parse_elevenlabs_speaker_id(speaker_id="speaker_12")
        12
        >>> _parse_elevenlabs_speaker_id(speaker_id=None)
        0
    """
    if speaker_id is None:
        return 0
    # "speaker_0" → "0"
    stripped = speaker_id.rsplit("_", 1)[-1]
    try:
        return int(stripped)
    except ValueError:
        return 0


def _build_segments_from_elevenlabs_json(
    data: dict,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from a native ElevenLabs STT JSON response.

    Filters to ``type == "word"`` entries only (skipping spacing and
    punctuation tokens) and emits **one segment per word**.  Converts
    floating-point seconds to integer milliseconds and string speaker
    IDs (``"speaker_0"``) to integers.  Merging consecutive same-speaker
    words into one segment is no longer done here — it is a shared,
    flag-gated post-step applied by :func:`load_diarization_info` (see
    :func:`_merge_consecutive_segments_by_speaker`), so all formats share
    one merge implementation.

    Args:
        data (dict): ElevenLabs ``speech_to_text.convert`` JSON response
            containing a top-level ``words`` list and optional ``text``
            transcript.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of per-word segment
            protos and transcript string (``None`` if absent).

    Raises:
        ValueError: If *data* is not a dict or ``words`` is not a list.

    Examples:
        >>> data = {
        ...     "text": "hello world",
        ...     "language_code": "eng",
        ...     "words": [
        ...         {
        ...             "text": "hello",
        ...             "start": 0.5,
        ...             "end": 0.8,
        ...             "type": "word",
        ...             "speaker_id": "speaker_0",
        ...         },
        ...         {
        ...             "text": " ",
        ...             "start": 0.8,
        ...             "end": 0.9,
        ...             "type": "spacing",
        ...             "speaker_id": "speaker_0",
        ...         },
        ...         {
        ...             "text": "world",
        ...             "start": 0.9,
        ...             "end": 1.2,
        ...             "type": "word",
        ...             "speaker_id": "speaker_1",
        ...         },
        ...     ],
        ... }
        >>> segments, transcript = _build_segments_from_elevenlabs_json(data=data)
        >>> len(segments)
        2
        >>> segments[0].word
        'hello'
    """
    if not isinstance(data, dict):
        raise ValueError(
            "Invalid ElevenLabs diarization JSON: expected a top-level dict, "
            f"got {type(data).__name__}."
        )
    words = data.get("words")
    if not isinstance(words, list):
        raise ValueError("Invalid ElevenLabs diarization JSON: expected 'words' to be a list.")

    language_code = data.get("language_code")
    transcript: str | None = data.get("text")

    # One segment per spoken word; same-speaker merging is handled later
    # by the shared, flag-gated post-step in load_diarization_info.
    segments: list[AudioSegmentInfo] = []
    for word in words:
        if word.get("type") != "word":
            continue

        segment_kwargs: dict[str, int | str] = {
            "start_time": int(word.get("start", 0) * 1000),
            "end_time": int(word.get("end", 0) * 1000),
            "speaker_id": _parse_elevenlabs_speaker_id(word.get("speaker_id")),
        }
        if word.get("text") is not None:
            segment_kwargs["word"] = str(word["text"])
        if language_code:
            segment_kwargs["language_code"] = str(language_code)
        segments.append(AudioSegmentInfo(**segment_kwargs))

    return segments, transcript


def _build_segment_from_elevenlabs_dubbing_api_utterance(
    utterance: dict,
    language_code: str | None,
) -> AudioSegmentInfo:
    """Build an ASD segment from one ElevenLabs Dubbing API utterance.

    Args:
        utterance (dict): One entry from the Dubbing Transcript API
            ``utterances`` array.
        language_code (str | None): Top-level language code from the
            transcript response. Defaults to ``None`` when absent.

    Returns:
        AudioSegmentInfo: Segment with millisecond timestamps and speaker ID.

    Raises:
        ValueError: If *utterance* is not a dictionary.

    Examples:
        >>> segment = _build_segment_from_elevenlabs_dubbing_api_utterance(
        ...     utterance={
        ...         "text": "hola",
        ...         "speaker_id": "speaker_1",
        ...         "start_s": 0.5,
        ...         "end_s": 1.2,
        ...     },
        ...     language_code="es",
        ... )
        >>> segment.start_time
        500
    """
    if not isinstance(utterance, dict):
        raise ValueError(
            "Invalid ElevenLabs Dubbing API utterance: expected a dict, "
            f"got {type(utterance).__name__}."
        )

    segment_kwargs: dict = {
        "start_time": int(float(utterance.get("start_s", 0)) * 1000),
        "end_time": int(float(utterance.get("end_s", 0)) * 1000),
        "speaker_id": _parse_elevenlabs_speaker_id(utterance.get("speaker_id")),
    }
    if utterance.get("text") is not None:
        segment_kwargs["word"] = str(utterance["text"])
    if language_code:
        segment_kwargs["language_code"] = language_code

    return AudioSegmentInfo(**segment_kwargs)


def _build_segments_from_elevenlabs_dubbing_api_json(
    data: dict,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from ElevenLabs Dubbing Transcript API JSON.

    The ElevenLabs Dubbing transcript endpoint returns segment-level
    ``utterances`` instead of the top-level ``words`` list used by the
    ElevenLabs STT/Scribe API.  This parser intentionally keeps that schema
    separate from ``diarization_format="elevenlabs-scribe"`` so callers
    must select the correct provider output format explicitly.

    Args:
        data (dict): Dubbing Transcript API JSON with top-level ``language``
            and ``utterances`` fields.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos
            and the joined utterance transcript.

    Raises:
        ValueError: If *data* is not a dict or ``utterances`` is not a list.

    Examples:
        >>> data = {
        ...     "language": "es",
        ...     "utterances": [
        ...         {
        ...             "text": "hola",
        ...             "speaker_id": "speaker_0",
        ...             "start_s": 0.5,
        ...             "end_s": 1.2,
        ...         }
        ...     ],
        ... }
        >>> segments, transcript = _build_segments_from_elevenlabs_dubbing_api_json(
        ...     data=data,
        ... )
        >>> segments[0].word
        'hola'
    """
    if not isinstance(data, dict):
        raise ValueError(
            "Invalid ElevenLabs Dubbing API diarization JSON: expected a top-level dict, "
            f"got {type(data).__name__}."
        )
    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(
            "Invalid ElevenLabs Dubbing API diarization JSON: expected 'utterances' to be a list."
        )

    language_code = data.get("language")
    if language_code is not None:
        language_code = str(language_code)

    segments: list[AudioSegmentInfo] = []
    transcript_parts: list[str] = []
    for utterance in utterances:
        segments.append(
            _build_segment_from_elevenlabs_dubbing_api_utterance(
                utterance=utterance,
                language_code=language_code,
            )
        )
        if isinstance(utterance, dict) and utterance.get("text"):
            transcript_parts.append(str(utterance["text"]))

    transcript = " ".join(transcript_parts) if transcript_parts else None
    return segments, transcript


def _parse_studio_timestamp(ts: str) -> int:
    """Convert an ElevenLabs Studio ``"HH:MM:SS,ms"`` timestamp to milliseconds.

    Args:
        ts (str): Timestamp string in ``"HH:MM:SS,mmm"`` format
            (e.g. ``"00:01:23,456"``).

    Returns:
        int: Total milliseconds.

    Examples:
        >>> _parse_studio_timestamp(ts="00:01:23,456")
        83456
        >>> _parse_studio_timestamp(ts="00:00:00,000")
        0
    """
    # "HH:MM:SS,ms" → split on comma first for the milliseconds part
    time_part, ms_part = ts.split(",")
    hours, minutes, seconds = time_part.split(":")
    return int(hours) * 3_600_000 + int(minutes) * 60_000 + int(seconds) * 1_000 + int(ms_part)


def _parse_studio_speaker_id(speaker: str) -> int:
    """Convert an ElevenLabs Studio ``"Speaker N"`` label to a zero-based integer.

    Args:
        speaker (str): Speaker label like ``"Speaker 1"`` or ``"Speaker 2"``.

    Returns:
        int: Zero-based speaker ID (``Speaker 1`` → ``0``).

    Examples:
        >>> _parse_studio_speaker_id(speaker="Speaker 1")
        0
        >>> _parse_studio_speaker_id(speaker="Speaker 3")
        2
    """
    # "Speaker 1" → 1 → 0 (zero-based)
    return int(speaker.rsplit(maxsplit=1)[-1]) - 1


def _build_segments_from_elevenlabs_studio_csv(
    file_path: str,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from an ElevenLabs Studio CSV export.

    The CSV has columns: ``speaker``, ``start_time``, ``end_time``,
    ``transcription``, ``translation``.  Each row represents a spoken
    segment with timestamps in ``HH:MM:SS,mmm`` format and speaker
    labels like ``Speaker 1``.

    Args:
        file_path (str): Path to the ElevenLabs Studio CSV file.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos
            and the concatenated transcript.

    Examples:
        >>> segments, transcript = _build_segments_from_elevenlabs_studio_csv(
        ...     file_path="diarization.csv",
        ... )
    """
    segments: list[AudioSegmentInfo] = []
    transcript_parts: list[str] = []

    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment = AudioSegmentInfo(
                start_time=_parse_studio_timestamp(ts=row["start_time"]),
                end_time=_parse_studio_timestamp(ts=row["end_time"]),
                speaker_id=_parse_studio_speaker_id(speaker=row["speaker"]),
                word=row.get("transcription", ""),
            )
            segments.append(segment)
            text = row.get("transcription", "")
            if text:
                transcript_parts.append(text)

    transcript = " ".join(transcript_parts) if transcript_parts else None
    return segments, transcript


def _parse_camb_speaker_id(speaker: str) -> int:
    """Convert a Camb AI speaker label to a zero-based integer.

    Handles two formats returned by different Camb AI API versions:
    - ``"SPEAKER_0"`` (underscore-separated, already zero-based)
    - ``"Speaker 1"`` (space-separated, one-based)

    Args:
        speaker (str): Speaker label like ``"SPEAKER_0"`` or ``"Speaker 1"``.

    Returns:
        int: Zero-based speaker ID.

    Examples:
        >>> _parse_camb_speaker_id(speaker="SPEAKER_0")
        0
        >>> _parse_camb_speaker_id(speaker="SPEAKER_2")
        2
        >>> _parse_camb_speaker_id(speaker="Speaker 1")
        0
        >>> _parse_camb_speaker_id(speaker="Speaker 3")
        2
    """
    # "SPEAKER_0" format: underscore-separated, already zero-based
    if "_" in speaker:
        suffix = speaker.rsplit("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return 0
    # "Speaker 1" format: space-separated, one-based
    return _parse_studio_speaker_id(speaker=speaker)


def _build_segments_from_camb_json(
    data: list[dict] | dict,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from Camb AI transcription JSON.

    Accepts either a raw list of segment objects or the API wrapper
    format ``{"transcript": [...]}``.  Each segment has ``start``,
    ``end`` (seconds as float), ``text``, and ``speaker``
    (``"SPEAKER_0"`` or ``"Speaker N"``).
    Converts seconds to integer milliseconds.

    Args:
        data (list[dict] | dict): Camb AI transcription segments — either
            a raw list or a dict with a ``transcript`` key containing the list.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos
            and the concatenated transcript.

    Raises:
        ValueError: If *data* cannot be resolved to a list of segments.

    Examples:
        >>> data = [
        ...     {"start": 0.5, "end": 1.2, "text": "hello", "speaker": "SPEAKER_0"},
        ...     {"start": 1.5, "end": 2.0, "text": "world", "speaker": "SPEAKER_1"},
        ... ]
        >>> segments, transcript = _build_segments_from_camb_json(data=data)
        >>> len(segments)
        2
        >>> segments[0].start_time
        500
    """
    # Unwrap {"transcript": [...]} wrapper if present
    if isinstance(data, dict):
        if "transcript" in data and isinstance(data["transcript"], list):
            data = data["transcript"]
        else:
            raise ValueError(
                "Invalid Camb AI diarization JSON: expected a list or "
                f"a dict with 'transcript' key, got keys {list(data.keys())}."
            )
    if not isinstance(data, list):
        raise ValueError(
            "Invalid Camb AI diarization JSON: expected a top-level list, "
            f"got {type(data).__name__}."
        )
    segments: list[AudioSegmentInfo] = []
    transcript_parts: list[str] = []

    for seg in data:
        segment_kwargs: dict = {
            "start_time": int(float(seg.get("start", 0)) * 1000),
            "end_time": int(float(seg.get("end", 0)) * 1000),
            "speaker_id": _parse_camb_speaker_id(speaker=str(seg.get("speaker", "Speaker 1"))),
        }
        text = seg.get("text")
        if text is not None:
            segment_kwargs["word"] = str(text)
            if text:
                transcript_parts.append(str(text))

        segments.append(AudioSegmentInfo(**segment_kwargs))

    transcript = " ".join(transcript_parts) if transcript_parts else None
    return segments, transcript


def _merge_consecutive_segments_by_speaker(
    segments: list[AudioSegmentInfo],
) -> list[AudioSegmentInfo]:
    """Merge runs of consecutive same-speaker segments into single segments.

    Walks *segments* in order and coalesces each maximal run of adjacent
    segments sharing a ``speaker_id`` into one segment that spans the run:
    the first segment's ``start_time``, the last segment's ``end_time``,
    the ``word`` values joined with single spaces, and the first non-empty
    ``language_code`` seen in the run.  Input protos are never mutated —
    each kept segment is a fresh copy.

    This is the shared, format-agnostic counterpart to the per-source-unit
    parsers: every ``load_diarization_info`` format produces one segment
    per word/utterance/row, and this step optionally collapses them by
    speaker (see ``combine_chunks_per_speaker``).

    Args:
        segments (list[AudioSegmentInfo]): Per-source-unit segments in
            chronological order.

    Returns:
        list[AudioSegmentInfo]: Segments with consecutive same-speaker runs
            merged. Returns an empty list when *segments* is empty.

    Examples:
        >>> a = AudioSegmentInfo(start_time=0, end_time=80, speaker_id=0, word="hello")
        >>> b = AudioSegmentInfo(start_time=90, end_time=120, speaker_id=0, word="there")
        >>> c = AudioSegmentInfo(start_time=130, end_time=200, speaker_id=1, word="hi")
        >>> merged = _merge_consecutive_segments_by_speaker(segments=[a, b, c])
        >>> len(merged)
        2
        >>> merged[0].word, merged[0].start_time, merged[0].end_time
        ('hello there', 0, 120)
    """
    merged: list[AudioSegmentInfo] = []
    for segment in segments:
        if merged and merged[-1].speaker_id == segment.speaker_id:
            prev = merged[-1]
            prev.end_time = segment.end_time
            if segment.word:
                prev.word = f"{prev.word} {segment.word}".strip() if prev.word else segment.word
            # Backfill language_code only if the run hasn't established one yet.
            if not prev.language_code and segment.language_code:
                prev.language_code = segment.language_code
        else:
            # Copy so the caller's input segments are never mutated.
            kept = AudioSegmentInfo()
            kept.CopyFrom(segment)
            merged.append(kept)
    return merged


VALID_DIARIZATION_FORMATS = (
    "flat",
    "elevenlabs-scribe",
    "elevenlabs-dubbing-api",
    "elevenlabs-studio",
    "camb",
)


def load_diarization_info(
    diarization_file: str,
    diarization_format: str,
    *,
    combine_chunks_per_speaker: bool = True,
) -> AudioDiarizationInfo | None:
    """Load diarization from supported ASD, ASR, and provider transcript formats.

    Each format parser emits one segment per source unit (per word for
    ``elevenlabs-scribe``, per utterance for
    ``elevenlabs-dubbing-api``, per row/entry for the rest).  When
    *combine_chunks_per_speaker* is ``True`` (the default), a single shared
    post-step (:func:`_merge_consecutive_segments_by_speaker`) collapses
    consecutive same-speaker segments into one — this is what keeps the
    historical ``elevenlabs-scribe`` behavior and now applies uniformly to
    every format.  Set it ``False`` to keep the finest granularity (e.g. one
    segment per word), which lets the controller client stream diarization one
    unit at a time via ``--diarization-rows-per-chunk``.

    Supported schemas:
        1) Flat ASD format (``"flat"``):
            ``[{"start_time": 0, "end_time": 320, "speaker_id": 1, ...}]``
        2) Native ElevenLabs STT/Scribe format (``"elevenlabs-scribe"``):
            ``{"text": "...", "words": [{"text": "hello", "start": 0.5,
            "end": 0.8, "type": "word", "speaker_id": "speaker_0"}]}``
        3) ElevenLabs Dubbing Transcript API JSON (``"elevenlabs-dubbing-api"``):
            ``{"language": "es", "utterances": [{"text": "hola",
            "speaker_id": "speaker_0", "start_s": 0.5, "end_s": 1.2}]}``
        4) ElevenLabs Studio CSV (``"elevenlabs-studio"``):
            CSV with ``speaker``, ``start_time``, ``end_time``,
            ``transcription``, ``translation`` columns.
        5) Camb AI transcription format (``"camb"``):
            ``[{"start": 0.5, "end": 1.2, "text": "hello",
            "speaker": "Speaker 1"}]``

    Args:
        diarization_file (str): Path to the diarization file (JSON or CSV).
        diarization_format (str): Explicit format — one of ``"flat"``,
            ``"elevenlabs-scribe"``, ``"elevenlabs-dubbing-api"``,
            ``"elevenlabs-studio"``, ``"camb"``.
        combine_chunks_per_speaker (bool): When ``True`` (default), merge
            consecutive same-speaker segments into one. When ``False``, keep
            one segment per source unit (word/utterance/row).

    Returns:
        AudioDiarizationInfo | None: Parsed diarization message, or ``None``
            if *diarization_file* is falsy.

    Raises:
        ValueError: If *diarization_format* is not a recognised value.

    Examples:
        >>> info = load_diarization_info(
        ...     diarization_file="diarization.json",
        ...     diarization_format="flat",
        ... )
        >>> info = load_diarization_info(
        ...     diarization_file="scribe.json",
        ...     diarization_format="elevenlabs-scribe",
        ...     combine_chunks_per_speaker=False,
        ... )
        >>> info = load_diarization_info(
        ...     diarization_file="camb.json",
        ...     diarization_format="camb",
        ... )
    """
    if not diarization_file:
        return None

    if diarization_format not in VALID_DIARIZATION_FORMATS:
        raise ValueError(
            f"Unknown diarization_format={diarization_format!r}. "
            f"Expected one of {VALID_DIARIZATION_FORMATS}."
        )

    # ElevenLabs Studio uses CSV, not JSON — handle it before json.load
    if diarization_format == "elevenlabs-studio":
        segments, transcript = _build_segments_from_elevenlabs_studio_csv(
            file_path=diarization_file,
        )
    else:
        with open(diarization_file, encoding="utf-8") as f:
            data = json.load(f)

        if diarization_format == "flat":
            segments, transcript = _build_segments_from_flat_json(data=data)
        elif diarization_format == "elevenlabs-scribe":
            segments, transcript = _build_segments_from_elevenlabs_json(data=data)
        elif diarization_format == "elevenlabs-dubbing-api":
            segments, transcript = _build_segments_from_elevenlabs_dubbing_api_json(data=data)
        elif diarization_format == "camb":
            segments, transcript = _build_segments_from_camb_json(data=data)

    if combine_chunks_per_speaker:
        segments = _merge_consecutive_segments_by_speaker(segments=segments)

    kwargs: dict[str, list[AudioSegmentInfo] | str] = {"segments": segments}
    if transcript:
        kwargs["transcript"] = transcript
    return AudioDiarizationInfo(**kwargs)
