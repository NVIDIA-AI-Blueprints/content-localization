# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream adapters for the direct client (ASD, LipSync request/response conversion).

NOTE: These adapters are client-side counterparts to the server-side adapters in
``controller_service.stream_adapters``.  They have diverged (print vs logger,
different backpressure logic) and are maintained separately.
"""

from collections.abc import Iterator

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.common.v1.common_pb2 import BoundingBox
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import SpeakerInfo as LipsyncSpeakerInfo
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import SpeakerInfoPerFrame

from common.base_utils import logger
from common.feeder_stream import FeederSource
from common.feeder_stream import FeederStream

# Delay between chunks to avoid overwhelming gRPC servers with data
# faster than they can consume it.
BACKPRESSURE_DELAY_SECS = 0.1


def to_asd_request(
    item: ActiveSpeakerDetectionData,
) -> DetectActiveSpeakerRequest:
    """Wrap an ``ActiveSpeakerDetectionData`` item in a ``DetectActiveSpeakerRequest``.

    Args:
        item: ASD data payload (video or audio).

    Returns:
        A ``DetectActiveSpeakerRequest`` with the ``data`` field set.

    Examples:
        >>> data = ActiveSpeakerDetectionData(video_data=b"\\x00")
        >>> req = to_asd_request(data)
        >>> req.HasField("data")
        True
    """
    return DetectActiveSpeakerRequest(data=item)


def to_lipsync_request(
    item: LipsyncInputData,
) -> LipsyncRequest:
    """Wrap a ``LipsyncInputData`` item in a ``LipsyncRequest``.

    Args:
        item: LipSync input payload (video, audio, speaker info, or
            background audio).

    Returns:
        A ``LipsyncRequest`` with the ``input`` field set.

    Examples:
        >>> data = LipsyncInputData(video_file_data=b"\\x00")
        >>> req = to_lipsync_request(data)
        >>> req.HasField("input")
        True
    """
    return LipsyncRequest(input=item)


def asd_request_generator_with_audio(
    video_iter: Iterator[ActiveSpeakerDetectionData],
    audio_iter: Iterator[ActiveSpeakerDetectionData],
    asd_config: ActiveSpeakerDetectionConfig,
    diarization_info: AudioDiarizationInfo | None = None,
) -> Iterator[DetectActiveSpeakerRequest]:
    """Merge video and audio streams into a DetectActiveSpeakerRequest stream.

    Emits a config message first, then optional diarization info as a
    standalone message, then concurrently drains video and audio
    iterators via :class:`~common.feeder_stream.FeederStream`.

    Args:
        video_iter (Iterator[ActiveSpeakerDetectionData]): ASD video data chunks.
        audio_iter (Iterator[ActiveSpeakerDetectionData]): ASD audio data chunks.
        asd_config (ActiveSpeakerDetectionConfig): Pre-built
            ``ActiveSpeakerDetectionConfig`` protobuf message.
        diarization_info (AudioDiarizationInfo | None): Optional diarization
            metadata sent as a standalone message before data streaming.

    Yields:
        DetectActiveSpeakerRequest: Messages ready for the ASD NIM.

    Examples:
        >>> gen = asd_request_generator_with_audio(
        ...     video_iter=video_data,
        ...     audio_iter=audio_data,
        ...     asd_config=config,
        ... )  # doctest: +SKIP
    """
    # 1. Emit config as the first message
    yield DetectActiveSpeakerRequest(config=asd_config)
    logger.debug(f"ASD: sent config: {asd_config}")

    # 2. Send diarization info as a standalone message if provided
    if diarization_info is not None:
        yield DetectActiveSpeakerRequest(
            data=ActiveSpeakerDetectionData(diarization_info=diarization_info),
        )
        logger.debug(f"ASD: sent diarization info ({len(diarization_info.segments)} segments)")

    # 3. Merge video and audio via concurrent feeder threads
    sources: list[FeederSource[ActiveSpeakerDetectionData, DetectActiveSpeakerRequest]] = [
        FeederSource(name="video", iterator=video_iter, transform=to_asd_request),
        FeederSource(name="audio", iterator=audio_iter, transform=to_asd_request),
    ]

    stream: FeederStream[DetectActiveSpeakerRequest] = FeederStream(
        sources=sources,
        backpressure_delay=BACKPRESSURE_DELAY_SECS,
    )
    stream.start(request_id="asd-direct")
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    logger.info(f"ASD complete: video={counts.get('video', 0)}, audio={counts.get('audio', 0)}")
    stream.raise_on_error()


def speaker_info_from_asd_response(
    response_iter: Iterator[DetectActiveSpeakerResponse],
) -> Iterator[LipsyncInputData]:
    """Create a LipsyncInputData iterator from the ASD response stream.

    Converts ASD ActiveSpeakerDetectionResult to LipSync SpeakerInfoPerFrame,
    mapping bounding boxes and speaker metadata.

    Args:
        response_iter (Iterator[DetectActiveSpeakerResponse]): Iterator
            of DetectActiveSpeakerResponse objects.

    Yields:
        LipsyncInputData: Objects with per_frame_speaker_infos populated.

    Examples:
        >>> gen = speaker_info_from_asd_response(
        ...     response_iter=asd_responses,
        ... )  # doctest: +SKIP
    """
    for response in response_iter:
        result = response.active_speaker_detection_result
        speaker_infos = []
        for speaker in result.speaker_data:
            speaker_infos.append(
                LipsyncSpeakerInfo(
                    speaker_bbox=BoundingBox(
                        x=speaker.speaker_bbox.x,
                        y=speaker.speaker_bbox.y,
                        width=speaker.speaker_bbox.width,
                        height=speaker.speaker_bbox.height,
                    ),
                    speaker_id=speaker.face_id,
                    is_speaking=speaker.is_speaking,
                )
            )

        yield LipsyncInputData(
            per_frame_speaker_infos=[
                SpeakerInfoPerFrame(
                    frame_id=result.frame_id,
                    speaker_infos=speaker_infos,
                )
            ]
        )


def _filter_keepalive(
    it: Iterator[LipsyncInputData],
) -> Iterator[LipsyncInputData]:
    """Strip keepalive messages from a ``LipsyncInputData`` stream.

    Args:
        it: Upstream iterator that may contain keepalive messages.

    Yields:
        Only non-keepalive ``LipsyncInputData`` items.
    """
    for item in it:
        if hasattr(item, "keepalive") and item.keepalive is not None:
            logger.debug("lipsync | skipping keep-alive chunk")
            continue
        yield item


def lipsync_input_request_generator(
    video_iterator: Iterator[LipsyncInputData],
    audio_iterator: Iterator[LipsyncInputData],
    speaker_info_iterator: Iterator[LipsyncInputData] | None,
    lipsync_config: LipsyncConfig,
    background_audio_iterator: Iterator[LipsyncInputData] | None = None,
) -> Iterator[LipsyncRequest]:
    """Generate a stream of LipsyncRequest messages for the LipSync service.

    Sends audio priming chunk first so the server initializes its
    sample rate/resampler, then concurrently drains all input
    iterators via :class:`~common.feeder_stream.FeederStream` with
    backpressure delay.

    Args:
        video_iterator (Iterator[LipsyncInputData]): Iterator of
            LipsyncInputData for video.
        audio_iterator (Iterator[LipsyncInputData]): Iterator of
            LipsyncInputData for audio.
        speaker_info_iterator (Iterator[LipsyncInputData] | None): Iterator
            of LipsyncInputData for speaker info. If None, no speaker info
            input will be sent.
        lipsync_config (LipsyncConfig): Pre-built ``LipsyncConfig`` protobuf
            message.
        background_audio_iterator (Iterator[LipsyncInputData] | None):
            Optional iterator of LipsyncInputData for background audio.
            ``None`` when no background audio is provided.

    Yields:
        LipsyncRequest: Messages containing either configuration or chunks
            of input data.

    Examples:
        >>> gen = lipsync_input_request_generator(
        ...     video_iterator=video_iter,
        ...     audio_iterator=audio_iter,
        ...     speaker_info_iterator=None,
        ...     lipsync_config=config,
        ... )  # doctest: +SKIP
    """
    logger.debug(f"lipsync_input_request_generator called with config: {lipsync_config}")

    # 1. Send configuration
    yield LipsyncRequest(config=lipsync_config)

    # 2. Prime audio early so the server initializes sample rate/resampler
    # before video/speaker-info arrives. Keepalive chunks are skipped.
    filtered_audio = _filter_keepalive(audio_iterator)
    try:
        primed_audio_chunk = next(filtered_audio)
        yield LipsyncRequest(input=primed_audio_chunk)
        logger.info("lipsync | audio priming chunk sent")
    except StopIteration:
        logger.info("lipsync | audio stream empty, no priming chunk")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Audio priming failed ({type(e).__name__}), continuing without prime: {e}")

    # 3. Merge remaining streams via concurrent feeder threads
    # Filter keepalive from speaker_info too
    filtered_speaker_info = (
        _filter_keepalive(speaker_info_iterator) if speaker_info_iterator is not None else None
    )

    sources: list[FeederSource[LipsyncInputData, LipsyncRequest]] = [
        FeederSource(
            name="video",
            iterator=video_iterator,
            transform=to_lipsync_request,
        ),
        FeederSource(
            name="audio",
            iterator=filtered_audio,
            transform=to_lipsync_request,
        ),
    ]
    if filtered_speaker_info is not None:
        sources.append(
            FeederSource(
                name="speaker_info",
                iterator=filtered_speaker_info,
                transform=to_lipsync_request,
            )
        )
    if background_audio_iterator is not None:
        sources.append(
            FeederSource(
                name="background_audio",
                iterator=background_audio_iterator,
                transform=to_lipsync_request,
            )
        )

    stream: FeederStream[LipsyncRequest] = FeederStream(
        sources=sources,
        backpressure_delay=BACKPRESSURE_DELAY_SECS,
    )
    stream.start(request_id="lipsync-direct")
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    logger.info(
        f"Transmission complete: video: {counts.get('video', 0)}, "
        f"audio: {counts.get('audio', 0)}, "
        f"speaker_info: {counts.get('speaker_info', 0)}, "
        f"background_audio: {counts.get('background_audio', 0)}"
    )
    stream.raise_on_error()
