# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream adapter utilities for the Controller Service pipeline.

This module provides generator functions that adapt between different service
response streams in the multi-threaded architecture. These adapters:

* Generate ASD requests by merging video and audio request streams
* Transform ASD speaker detection results into LipSync speaker info requests
* Transform S2S audio responses into LipSync audio requests
* Merge video, audio, and speaker info streams into LipSync requests

Every adapter emits ready-to-send request messages, writing payloads
directly into the nested data fields so each chunk is copied only once.

Architecture:
    (video + audio + diarization) → asd_request_generator() → ASD requests
    ASD responses → asd_response_to_lipsync_speaker_info() → LipSync speaker info requests
    S2S responses → s2s_audio_to_lipsync_audio() → LipSync audio requests
    Translated audio → translated_audio_to_lipsync_audio() → LipSync audio requests (bypass S2S)
    (video + audio + speaker_info) → lipsync_request_generator() → LipSync requests
"""

from collections.abc import Iterator

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.base_utils import logger
from common.feeder_stream import FeederSource
from common.feeder_stream import FeederStream
from controller_service.conversions import create_wav_header
from controller_service.conversions import to_lipsync_translated_audio


def asd_request_generator(
    video_iter: Iterator[DetectActiveSpeakerRequest],
    audio_iter: Iterator[DetectActiveSpeakerRequest],
    asd_config: ActiveSpeakerDetectionConfig,
    diarization_iter: Iterator[DetectActiveSpeakerRequest] | None = None,
) -> Iterator[DetectActiveSpeakerRequest]:
    """Merge video, audio, and diarization streams into a ``DetectActiveSpeakerRequest`` stream.

    Emits the client-provided config as the first message, then
    concurrently drains video, audio, and optional diarization
    iterators via :class:`~common.feeder_stream.FeederStream`,
    yielding requests as soon as any source has data ready.

    Args:
        video_iter: ASD video requests (from video_buffer consumer).
        audio_iter: ASD audio requests (from audio_buffer consumer).
        asd_config: Client-provided ``ActiveSpeakerDetectionConfig`` to send
            as the first message.
        diarization_iter: Optional ASD diarization requests (from
            diarization_buffer consumer).

    Yields:
        ``DetectActiveSpeakerRequest`` messages ready for the ASD NIM.

    Examples:
        >>> from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV, AudioConfig
        >>> cfg = ActiveSpeakerDetectionConfig(
        ...     input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV)
        ... )
        >>> reqs = asd_request_generator(
        ...     video_iter=iter([]),
        ...     audio_iter=iter([]),
        ...     asd_config=cfg,
        ... )
        >>> first = next(reqs)
        >>> first.HasField("config")
        True
    """
    # 1. Emit client-provided config as the first message
    yield DetectActiveSpeakerRequest(config=asd_config)
    logger.debug("asd_request_generator: sent config (pass-through from client)")

    # 2. Merge video, audio, and diarization via concurrent feeder threads
    sources: list[FeederSource[DetectActiveSpeakerRequest, DetectActiveSpeakerRequest]] = [
        FeederSource(name="video", iterator=video_iter),
        FeederSource(name="audio", iterator=audio_iter),
    ]
    if diarization_iter is not None:
        sources.append(
            FeederSource(
                name="diarization",
                iterator=diarization_iter,
            )
        )

    stream: FeederStream[DetectActiveSpeakerRequest] = FeederStream(sources=sources)
    stream.start(request_id="asd")
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    logger.info(
        f"asd_request_generator complete: video={counts.get('video', 0)}, "
        f"audio={counts.get('audio', 0)}, "
        f"diarization={counts.get('diarization', 0)}"
    )
    stream.raise_on_error()


def asd_response_to_lipsync_speaker_info(
    response_iter: Iterator[DetectActiveSpeakerResponse],
) -> Iterator[LipsyncRequest]:
    """Yield ``LipsyncRequest`` messages with speaker info from an ASD response stream.

    Converts ASD ``ActiveSpeakerDetectionResult`` to LipSync ``SpeakerInfoPerFrame``,
    mapping bounding boxes and speaker metadata directly into the nested
    ``input`` field of each request so every frame is built in one step.

    Args:
        response_iter: Any iterator of ``DetectActiveSpeakerResponse``.

    Yields:
        ``LipsyncRequest`` with ``input.per_frame_speaker_infos`` populated.
    """
    count = 0
    for response in response_iter:
        result = response.active_speaker_detection_result
        lipsync_request = LipsyncRequest()
        frame = lipsync_request.input.per_frame_speaker_infos.add()
        frame.frame_id = result.frame_id
        for speaker in result.speaker_data:
            speaker_info = frame.speaker_infos.add()
            speaker_info.speaker_bbox.x = speaker.speaker_bbox.x
            speaker_info.speaker_bbox.y = speaker.speaker_bbox.y
            speaker_info.speaker_bbox.width = speaker.speaker_bbox.width
            speaker_info.speaker_bbox.height = speaker.speaker_bbox.height
            speaker_info.speaker_id = speaker.face_id
            speaker_info.is_speaking = speaker.is_speaking
        count += 1
        yield lipsync_request
    logger.info(f"asd_response_to_lipsync_speaker_info: yielded {count} speaker_info frames")


def s2s_audio_to_lipsync_audio(
    response_iter: Iterator[SpeechToSpeechResponse],
    audio_format: str = "mp3",
) -> Iterator[LipsyncRequest]:
    """Yield ``LipsyncRequest`` audio chunks from an S2S response stream.

    On the first audio response the format is validated against
    *audio_format*.  For WAV, a synthetic header is emitted before the
    first audio-data chunk **only when the data is raw PCM** (no
    existing header).  If the first chunk already starts with a RIFF
    header the synthetic header is skipped to avoid a duplicate header
    with a wrong sample-rate.

    Each chunk is written directly into the nested ``input`` field of
    the request so the audio bytes are copied only once.

    Args:
        response_iter: Any iterator of ``SpeechToSpeechResponse``.
        audio_format: Expected audio format (``"mp3"`` or ``"wav"``).

    Yields:
        ``LipsyncRequest`` with ``input.audio_file_data`` populated.

    """
    first_response = True
    for response in response_iter:
        if response.HasField("audio_data"):
            if first_response:
                audio_format_from_s2s = (
                    response.audio_format.lower() if response.audio_format else "mp3"
                )
                if audio_format_from_s2s != audio_format.lower():
                    logger.warning(
                        f"Audio format from S2S service is "
                        f"{audio_format_from_s2s}, but expected "
                        f"{audio_format.lower()}. Continuing with "
                        f"detected format."
                    )
                    audio_format = audio_format_from_s2s
                first_response = False

                # For WAV: only prepend a synthetic header when the
                # data is raw PCM (no header). Some backends stream a
                # complete WAV file whose first bytes are already a
                # RIFF header — adding a second header with a guessed
                # sample-rate makes the audio play at the wrong speed.
                data_already_has_header = response.audio_data[:4] == b"RIFF"
                if audio_format_from_s2s == "wav" and not data_already_has_header:
                    wav_header = create_wav_header(
                        n_channels=response.audio_num_channels or 1,
                        sample_width=2,  # Assuming 16-bit PCM
                        frame_rate=response.audio_sample_rate or 16000,
                        n_frames=0,
                    )
                    logger.debug("s2s_audio_to_lipsync_audio: yielding WAV header")
                    header_request = LipsyncRequest()
                    header_request.input.audio_file_data = wav_header
                    yield header_request
                elif data_already_has_header:
                    logger.debug(
                        "s2s_audio_to_lipsync_audio: first chunk "
                        "already contains a WAV header, skipping "
                        "synthetic header"
                    )
            audio_request = LipsyncRequest()
            audio_request.input.audio_file_data = response.audio_data
            yield audio_request


def lipsync_request_generator(
    video_iter: Iterator[LipsyncRequest],
    audio_iter: Iterator[LipsyncRequest],
    speaker_info_iter: Iterator[LipsyncRequest] | None,
    lipsync_config: LipsyncConfig,
    background_audio_iter: Iterator[LipsyncRequest] | None = None,
) -> Iterator[LipsyncRequest]:
    """Merge video, audio, speaker info, and background audio into a ``LipsyncRequest`` stream.

    Emits the client-provided config as the first message, then
    concurrently drains all input iterators via
    :class:`~common.feeder_stream.FeederStream`, yielding requests as
    soon as any source has data ready.

    Args:
        video_iter (Iterator[LipsyncRequest]): LipSync video requests.
        audio_iter (Iterator[LipsyncRequest]): LipSync audio requests
            (from S2S output).
        speaker_info_iter (Iterator[LipsyncRequest] | None): LipSync
            speaker info requests (from ASD output), or ``None`` when ASD
            is disabled.
        lipsync_config (LipsyncConfig): Client-provided ``LipsyncConfig``
            to send as the first message.
        background_audio_iter (Iterator[LipsyncRequest] | None): Optional
            background audio requests for LipSync mixing. ``None`` when no
            background audio is provided.

    Yields:
        LipsyncRequest: Messages ready for the LipSync service.

    Examples:
        >>> from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
        >>> cfg = LipsyncConfig()
        >>> reqs = lipsync_request_generator(iter([]), iter([]), None, cfg)
        >>> first = next(reqs)
        >>> first.HasField("config")
        True
    """
    # 1. Emit client-provided config as the first message
    yield LipsyncRequest(config=lipsync_config)
    logger.debug("lipsync_request_generator: sent config (pass-through from client)")

    # 2. Prime audio early so the LipSync NIM initializes its audio reader
    # (sample rate, resampler) before the first video frame triggers
    # _initialize_video_writer.  Without this, the video writer may see
    # sample_rate=None when the video arrives before any audio data.
    try:
        yield next(audio_iter)
        logger.debug("lipsync_request_generator: audio priming chunk sent")
    except StopIteration:
        logger.warning("lipsync_request_generator: audio stream empty, no priming chunk")

    # 3. Merge remaining input streams via concurrent feeder threads
    sources: list[FeederSource[LipsyncRequest, LipsyncRequest]] = [
        FeederSource(name="video", iterator=video_iter),
        FeederSource(name="audio", iterator=audio_iter),
    ]
    if speaker_info_iter is not None:
        sources.append(
            FeederSource(
                name="speaker_info",
                iterator=speaker_info_iter,
            )
        )
    if background_audio_iter is not None:
        sources.append(
            FeederSource(
                name="background_audio",
                iterator=background_audio_iter,
            )
        )

    stream: FeederStream[LipsyncRequest] = FeederStream(sources=sources)
    stream.start(request_id="lipsync")
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    logger.info(
        f"lipsync_request_generator complete: "
        f"video={counts.get('video', 0)}, "
        f"audio={counts.get('audio', 0)}, "
        f"speaker_info={counts.get('speaker_info', 0)}, "
        f"background_audio={counts.get('background_audio', 0)}"
    )
    stream.raise_on_error()


def translated_audio_to_lipsync_audio(
    request_iter: Iterator[ContentLocalizationRequest],
) -> Iterator[LipsyncRequest]:
    """Yield ``LipsyncRequest`` audio chunks from pre-translated audio requests.

    Used in no-S2S mode: the client sends already-translated audio in
    ``translated_audio_data``, which is passed directly to LipSync
    without any S2S processing.

    Args:
        request_iter: Iterator of ``ContentLocalizationRequest`` with
            ``translated_audio_data`` populated.

    Yields:
        ``LipsyncRequest`` with ``input.audio_file_data`` from the
            translated audio bytes.

    Examples:
        >>> from nvidia.ai4m.controller.v1.controller_pb2 import (
        ...     ContentLocalizationRequest,
        ... )
        >>> reqs = [ContentLocalizationRequest(translated_audio_data=b"\\x00")]
        >>> items = list(translated_audio_to_lipsync_audio(iter(reqs)))
        >>> len(items)
        1
    """
    count = 0
    for request in request_iter:
        count += 1
        yield to_lipsync_translated_audio(request)
    logger.info(f"translated_audio_to_lipsync_audio: yielded {count} audio chunks")
