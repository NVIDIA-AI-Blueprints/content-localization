# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs Speech-to-Text wrappers for diarization generation.

Wraps the ElevenLabs Python SDK ``speech_to_text.convert`` call and
its response-shape conversion, shared by both the CLI in
``scripts/elevenlabs/diarize.py`` and the batch-processing client in
``client/batch_processing/diarization.py``. Lives under ``src/`` so
that ``scripts/`` and ``client/`` consumers depend on a stable
library rather than on each other.
"""

from typing import Any
from typing import TypedDict

from elevenlabs import ElevenLabs

# ElevenLabs STT accepts between 1 and 32 diarized speakers.
_MIN_SPEAKERS = 1
_MAX_SPEAKERS = 32


class _TranscribeOptions(TypedDict, total=False):
    """STT-tuning overrides accepted by :func:`transcribe`.

    All keys are optional; absent keys fall back to model defaults.
    """

    model_id: str
    language_code: str | None
    max_speakers: int | None
    tag_audio_events: bool


def transcribe(
    file_path: str,
    api_key: str,
    options: _TranscribeOptions | None = None,
) -> dict[str, Any]:
    """Transcribe an audio file with ElevenLabs STT and return the native JSON.

    Always requests diarization with word-level timestamps. Builds a fresh
    SDK client per call, opens the audio file, invokes
    ``speech_to_text.convert``, and converts the SDK response object to a
    JSON-serializable dict via :func:`response_to_native_json`.

    Args:
        file_path (str): Path to the audio file (WAV, MP3, etc.).
        api_key (str): ElevenLabs API key.
        options (dict | None): Optional STT-tuning overrides. Recognized keys:

            - ``model_id`` (str): ElevenLabs STT model ID. Defaults to
              ``"scribe_v2"``.
            - ``language_code`` (str | None): ISO-639-1/3 code (e.g.
              ``"eng"``). When absent or ``None``, the model auto-detects.
            - ``max_speakers`` (int | None): Maximum number of speakers
              (up to 32). When absent or ``None``, the model decides.
              Mapped to the SDK's ``num_speakers`` kwarg.
            - ``tag_audio_events`` (bool): When ``True``, tags non-speech
              events (laughter, footsteps) in the transcript. Defaults to
              ``False``.

    Returns:
        dict: Native ElevenLabs STT JSON response with a ``words`` list.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.

    Examples:
        >>> native = transcribe("audio.wav", api_key="el_key")  # doctest: +SKIP
        >>> "words" in native
        True
    """
    # Validate the boundary now that callers pass an arbitrary mapping
    # instead of argparse-typed values.
    if options is not None and not isinstance(options, dict):
        raise TypeError(f"options must be a dict or None, got {type(options).__name__}")
    opts = options or {}

    stt_kwargs: dict[str, Any] = {
        "model_id": opts.get("model_id", "scribe_v2"),
        "diarize": True,
        "timestamps_granularity": "word",
        "tag_audio_events": opts.get("tag_audio_events", False),
    }
    language_code = opts.get("language_code")
    if language_code is not None:
        stt_kwargs["language_code"] = language_code
    max_speakers = opts.get("max_speakers")
    if max_speakers is not None:
        if not isinstance(max_speakers, int) or isinstance(max_speakers, bool):
            raise TypeError(
                f"options['max_speakers'] must be int, got {type(max_speakers).__name__}"
            )
        # Reject out-of-range values locally so we fail fast instead of on
        # the remote call.
        if not _MIN_SPEAKERS <= max_speakers <= _MAX_SPEAKERS:
            raise ValueError(
                f"options['max_speakers'] must be between {_MIN_SPEAKERS} and "
                f"{_MAX_SPEAKERS}, got {max_speakers}"
            )
        stt_kwargs["num_speakers"] = max_speakers

    client = ElevenLabs(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        response = client.speech_to_text.convert(file=audio_file, **stt_kwargs)

    return response_to_native_json(response)


def response_to_native_json(response: object) -> dict[str, Any]:
    """Convert an ElevenLabs STT response object to a JSON-serializable dict.

    Uses the Pydantic ``model_dump`` method available on the SDK response
    objects.  Falls back to ``dict()`` / ``__dict__`` for older SDK versions.

    Args:
        response (object): ElevenLabs ``SpeechToTextConvertResponse`` object.

    Returns:
        dict: Native ElevenLabs response as a JSON-serializable dictionary.

    Examples:
        >>> native = response_to_native_json(el_response)
        >>> "words" in native
        True
    """
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if hasattr(response, "__dict__"):
        return vars(response)
    raise TypeError(f"Unsupported ElevenLabs response type: {type(response).__name__}")


def extract_diarization_stats(native_response: dict[str, Any]) -> tuple[int, int]:
    """Extract word and speaker counts from native ElevenLabs JSON response.

    Args:
        native_response (dict): Native ElevenLabs STT JSON dictionary
            containing a ``words`` list.

    Returns:
        tuple[int, int]: Tuple of (word_count, unique_speaker_count).

    Examples:
        >>> extract_diarization_stats({"words": []})
        (0, 0)
        >>> extract_diarization_stats(
        ...     {
        ...         "words": [
        ...             {"text": "hello", "speaker_id": "speaker_0"},
        ...             {"text": "world", "speaker_id": "speaker_1"},
        ...         ]
        ...     }
        ... )
        (2, 2)
    """
    words = native_response.get("words", [])
    speaker_ids: set[str] = set()

    for word in words:
        speaker_id = word.get("speaker_id")
        if speaker_id is not None:
            speaker_ids.add(speaker_id)

    return len(words), len(speaker_ids)
