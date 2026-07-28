# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs and Camb AI diarization for batch processing videos.

Generates diarization JSON files using the ElevenLabs Speech-to-Text
(Scribe) API or Camb AI Transcription API, reusing utilities from
``common.diarization.elevenlabs`` and ``common.diarization.camb``.
Skips generation when a diarization file already exists on disk.

Both helpers produce **source-language** diarization aligned with the
input audio — this is what ASD expects. We deliberately do not use the
Camb AI dubbing API (``scripts/camb/s2s_infer.py``) here because its
transcript output is target-language only.
"""

import json
import os

from client.common.paths import ensure_parent_dir
from common.base_utils import logger
from common.diarization.camb import extract_diarization_stats as camb_extract_stats
from common.diarization.camb import get_transcription_result
from common.diarization.camb import submit_transcription
from common.diarization.camb import wait_for_transcription
from common.diarization.elevenlabs import extract_diarization_stats
from common.diarization.elevenlabs import transcribe


def _get_elevenlabs_api_key() -> str:
    """Read the ElevenLabs API key from the environment.

    Returns:
        str: The ElevenLabs API key.

    Raises:
        ValueError: If ``ELEVENLABS_API_KEY`` is not set.

    Examples:
        >>> key = _get_elevenlabs_api_key()  # doctest: +SKIP
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ELEVENLABS_API_KEY environment variable not set. "
            "Export it before running batch processing."
        )
    return api_key


def _get_camb_headers() -> dict[str, str]:
    """Build Camb AI HTTP headers from the environment.

    Returns:
        dict[str, str]: Headers dict with ``x-api-key``.

    Raises:
        ValueError: If ``CAMB_API_KEY`` is not set.

    Examples:
        >>> headers = _get_camb_headers()  # doctest: +SKIP
    """
    api_key = os.getenv("CAMB_API_KEY")
    if not api_key:
        raise ValueError(
            "CAMB_API_KEY environment variable not set. Export it before running batch processing."
        )
    return {"x-api-key": api_key}


def generate_diarization(
    audio_path: str,
    output_json_path: str,
) -> str:
    """Generate diarization JSON for an audio file via ElevenLabs STT.

    Args:
        audio_path (str): Path to the input WAV audio file.
        output_json_path (str): Path to write the diarization JSON.

    Returns:
        str: Path to the diarization JSON file.

    Raises:
        ValueError: If the ElevenLabs API key is missing.
        FileNotFoundError: If the audio file does not exist.

    Examples:
        >>> path = generate_diarization("a.wav", "d.json")  # doctest: +SKIP
    """
    ensure_parent_dir(path=output_json_path)

    api_key = _get_elevenlabs_api_key()

    logger.info(f"  Generating diarization via ElevenLabs STT: {audio_path}")
    native_response = transcribe(file_path=audio_path, api_key=api_key)

    words_count, speaker_count = extract_diarization_stats(native_response)
    logger.info(f"  Diarization: {words_count} words, {speaker_count} speakers")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(native_response, f, indent=2, ensure_ascii=False)

    return output_json_path


def generate_camb_diarization(
    audio_path: str,
    output_json_path: str,
    language_id: int = 1,
) -> str:
    """Generate diarization JSON for an audio file via Camb AI Transcription API.

    Args:
        audio_path (str): Path to the input WAV audio file.
        output_json_path (str): Path to write the diarization JSON.
        language_id (int): Camb AI numeric language ID (default: 1 for English).

    Returns:
        str: Path to the diarization JSON file.

    Raises:
        ValueError: If the Camb AI API key is missing.
        FileNotFoundError: If the audio file does not exist.

    Examples:
        >>> path = generate_camb_diarization("a.wav", "d.json")  # doctest: +SKIP
    """
    ensure_parent_dir(path=output_json_path)

    headers = _get_camb_headers()

    logger.info(f"  Generating diarization via Camb AI: {audio_path}")
    task_id = submit_transcription(
        file_path=audio_path,
        language_id=language_id,
        headers=headers,
    )
    run_id = wait_for_transcription(task_id=task_id, headers=headers)
    result = get_transcription_result(run_id=run_id, headers=headers)

    segment_count, speaker_count = camb_extract_stats(result)
    logger.info(f"  Diarization: {segment_count} segments, {speaker_count} speakers")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_json_path


def ensure_diarization(
    audio_path: str,
    diarization_dir: str,
    video_stem: str,
    s2s_service: str | None = None,
) -> str:
    """Return path to diarization JSON, generating it if absent.

    Checks ``{diarization_dir}/{video_stem}.json`` and skips
    generation when the file already exists. Routes to Camb AI
    when ``s2s_service`` is ``"CAMB_DUBBING"``, otherwise uses
    ElevenLabs.

    Args:
        audio_path (str): Path to the extracted WAV audio file.
        diarization_dir (str): Directory for diarization JSON files.
        video_stem (str): Video filename stem (used as JSON filename).
        s2s_service (str | None): S2S service identifier. When
            ``"CAMB_DUBBING"``, uses Camb AI for diarization.

    Returns:
        str: Path to the diarization JSON file.

    Examples:
        >>> path = ensure_diarization("a.wav", "diar/", "clip1")
        >>> path = ensure_diarization("a.wav", "diar/", "clip1", s2s_service="CAMB_DUBBING")
    """
    diarization_path = os.path.join(diarization_dir, f"{video_stem}.json")

    if os.path.isfile(diarization_path):
        logger.info(f"  Reusing existing diarization: {diarization_path}")
        return diarization_path

    if s2s_service == "CAMB_DUBBING":
        return generate_camb_diarization(
            audio_path=audio_path,
            output_json_path=diarization_path,
        )

    return generate_diarization(
        audio_path=audio_path,
        output_json_path=diarization_path,
    )
