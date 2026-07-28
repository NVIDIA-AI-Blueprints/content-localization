# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol buffer conversion utilities for the Controller Service.

This module provides functions to convert between different protobuf message
formats used in the content localization pipeline:

* ContentLocalizationRequest → SpeechToSpeechRequest (S2S)
* ContentLocalizationRequest → DetectActiveSpeakerRequest (ASD video)
* ContentLocalizationRequest → DetectActiveSpeakerRequest (ASD audio)
* ContentLocalizationRequest → DetectActiveSpeakerRequest (ASD diarization)
* ContentLocalizationRequest → LipsyncRequest (LipSync video)
* ContentLocalizationRequest → LipsyncRequest (LipSync background audio)
* ContentLocalizationRequest → LipsyncRequest (LipSync translated audio,
  bypass S2S mode)

Each conversion produces a ready-to-send request message, writing the
payload directly into the nested data field so every chunk is copied
only once.

These conversions are used in the multi-threaded pipeline where the
ContentLocalizationDeserializer distributes incoming requests to different
service clients (S2S, ASD, LipSync).

Functions:
    to_s2s_request: Convert to S2S service format
    to_asd_video_data: Convert to an ASD request carrying video data
    to_asd_audio_data: Convert to an ASD request carrying audio data
    to_asd_diarization_data: Convert to an ASD request carrying diarization
    to_lipsync_video: Convert to a LipSync request carrying video data
    to_lipsync_background_audio: Convert to a LipSync background audio request
    to_lipsync_translated_audio: Convert to a LipSync translated audio request

Example:
    from controller_service.conversions import to_s2s_request

    s2s_req = to_s2s_request(content_localization_req)
"""

import traceback

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest

from common.audio_utils import create_wav_header  # noqa: F401  # re-exported
from common.base_utils import logger
from controller_service.constants import AUDIO_CONFIG_DEFAULTS


def to_s2s_request(
    request: ContentLocalizationRequest,
    input_audio_format: str = AUDIO_CONFIG_DEFAULTS["audio_format"],
) -> SpeechToSpeechRequest:
    """Convert a ``ContentLocalizationRequest`` to a ``SpeechToSpeechRequest``.

    Uses :data:`AUDIO_CONFIG_DEFAULTS` for sample rate/channel defaults and
    uses ``input_audio_format`` for the request ``audio_format``.

    Args:
        request: Incoming content-localisation request.
        input_audio_format: Source/input audio format (for example, ``"WAV"`` or
            ``"MP3"``) to send to S2S.

    Returns:
        Populated ``SpeechToSpeechRequest``.

    Raises:
        ValueError: If neither ``audio_data`` nor ``s2s_config`` is present.

    Examples:
        >>> req = ContentLocalizationRequest(audio_data=b"\\x00\\x01")
        >>> s2s_req = to_s2s_request(req, input_audio_format="WAV")
        >>> s2s_req.audio_format
        'WAV'
    """
    try:
        s2s_request = SpeechToSpeechRequest()

        if request.HasField("audio_data"):
            s2s_request.audio_data = request.audio_data
            s2s_request.audio_sample_rate = AUDIO_CONFIG_DEFAULTS["audio_sample_rate"]
            s2s_request.audio_num_channels = AUDIO_CONFIG_DEFAULTS["audio_num_channels"]
            s2s_request.audio_format = input_audio_format.upper()

        if request.HasField("s2s_config"):
            s2s_request.config.CopyFrom(request.s2s_config)

        if not request.HasField("s2s_config") and not request.HasField("audio_data"):
            raise ValueError("S2S config or audio data must be provided in the request")
    except Exception as e:
        logger.error(f"Error generating S2S request: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise

    logger.debug(
        f"Generated S2S request: audio_data_present={request.HasField('audio_data')}, "
        f"audio_data_size={len(request.audio_data) if request.HasField('audio_data') else 0}, "
        f"s2s_config_present={request.HasField('s2s_config')}"
    )
    return s2s_request


def to_asd_video_data(request: ContentLocalizationRequest) -> DetectActiveSpeakerRequest:
    """Convert a ``ContentLocalizationRequest`` to an ASD request with video data.

    The video bytes are written directly into the nested ``data``
    field of the request, so each chunk is copied only once.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``DetectActiveSpeakerRequest`` with ``data.video_data`` populated.

    Raises:
        ValueError: If ``video_file_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(video_file_data=b"\\x00")
        >>> to_asd_video_data(req).data.video_data
        b'\\x00'
    """
    if not request.HasField("video_file_data"):
        raise ValueError("Video data not found in request")
    asd_request = DetectActiveSpeakerRequest()
    asd_request.data.video_data = request.video_file_data
    return asd_request


def to_asd_audio_data(request: ContentLocalizationRequest) -> DetectActiveSpeakerRequest:
    """Convert a ``ContentLocalizationRequest`` to an ASD request with audio data.

    The audio bytes are written directly into the nested ``data``
    field of the request, so each chunk is copied only once.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``DetectActiveSpeakerRequest`` with ``data.audio_data`` populated.

    Raises:
        ValueError: If ``audio_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(audio_data=b"\\x00")
        >>> to_asd_audio_data(req).data.audio_data
        b'\\x00'
    """
    if not request.HasField("audio_data"):
        raise ValueError("Audio data not found in request")
    asd_request = DetectActiveSpeakerRequest()
    asd_request.data.audio_data = request.audio_data
    return asd_request


def to_asd_diarization_data(request: ContentLocalizationRequest) -> DetectActiveSpeakerRequest:
    """Convert a ``ContentLocalizationRequest`` to an ASD request with diarization info.

    The diarization message is copied directly into the nested ``data``
    field of the request, so each chunk is copied only once.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``DetectActiveSpeakerRequest`` with ``data.diarization_info``
        populated.

    Raises:
        ValueError: If ``diarization_info`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest()  # doctest: +SKIP
        >>> to_asd_diarization_data(req)  # doctest: +SKIP
    """
    if not request.HasField("diarization_info"):
        raise ValueError("Diarization info not found in request")
    asd_request = DetectActiveSpeakerRequest()
    asd_request.data.diarization_info.CopyFrom(request.diarization_info)
    return asd_request


def to_lipsync_video(request: ContentLocalizationRequest) -> LipsyncRequest:
    """Convert a ``ContentLocalizationRequest`` to a LipSync request with video data.

    The video bytes are written directly into the nested ``input``
    field of the request, so each chunk is copied only once.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``LipsyncRequest`` with ``input.video_file_data`` populated.

    Raises:
        ValueError: If ``video_file_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(video_file_data=b"\\x00")
        >>> to_lipsync_video(req).input.video_file_data
        b'\\x00'
    """
    if not request.HasField("video_file_data"):
        raise ValueError("Video data not found in request")
    lipsync_request = LipsyncRequest()
    lipsync_request.input.video_file_data = request.video_file_data
    return lipsync_request


def to_lipsync_background_audio(
    request: ContentLocalizationRequest,
) -> LipsyncRequest:
    """Convert a ``ContentLocalizationRequest`` to a LipSync request with background audio.

    The audio bytes are written directly into the nested ``input``
    field of the request, so each chunk is copied only once.

    Args:
        request: Incoming content-localisation request with
            ``background_audio_data``.

    Returns:
        ``LipsyncRequest`` with ``input.background_audio_file_data``
        populated.

    Raises:
        ValueError: If ``background_audio_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(background_audio_data=b"\\x00")
        >>> lip = to_lipsync_background_audio(req)
        >>> lip.input.background_audio_file_data
        b'\\x00'
    """
    if not request.HasField("background_audio_data"):
        raise ValueError("Background audio data not found in request")
    lipsync_request = LipsyncRequest()
    lipsync_request.input.background_audio_file_data = request.background_audio_data
    return lipsync_request


def to_lipsync_translated_audio(
    request: ContentLocalizationRequest,
) -> LipsyncRequest:
    """Convert a ``ContentLocalizationRequest`` to a LipSync request with translated audio.

    Used in no-S2S mode: the client provides pre-translated audio that
    bypasses S2S and feeds directly into LipSync. The audio bytes are
    written directly into the nested ``input`` field of the request,
    so each chunk is copied only once.

    Args:
        request: Incoming content-localisation request with
            ``translated_audio_data``.

    Returns:
        ``LipsyncRequest`` with ``input.audio_file_data`` populated.

    Raises:
        ValueError: If ``translated_audio_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(translated_audio_data=b"\\x00")
        >>> lip = to_lipsync_translated_audio(req)
        >>> lip.input.audio_file_data
        b'\\x00'
    """
    if not request.HasField("translated_audio_data"):
        raise ValueError("Translated audio data not found in request")
    lipsync_request = LipsyncRequest()
    lipsync_request.input.audio_file_data = request.translated_audio_data
    return lipsync_request
