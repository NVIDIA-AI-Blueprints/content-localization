# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASD request stream generators for the standalone ASD client."""

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

from common.base_utils import logger
from common.feeder_stream import FeederSource
from common.feeder_stream import FeederStream
from common.source_sink.grpc.audio import AudioSourceSimulator
from common.source_sink.grpc.video import VideoSourceSimulator


def asd_request_generator(
    video_source: VideoSourceSimulator,
    audio_source: AudioSourceSimulator,
    chunk_size_video_bytes: int,
    chunk_size_audio_secs: float,
    asd_config: ActiveSpeakerDetectionConfig,
    diarization_info: AudioDiarizationInfo | None = None,
) -> Iterator[DetectActiveSpeakerRequest]:
    """Generate a stream of DetectActiveSpeakerRequest messages for the ASD service.

    Sends config first, then concurrently merges video, audio, and
    optional diarization into a single request stream via
    :class:`~common.feeder_stream.FeederStream` (diarization is supplied as
    an additional :class:`~common.feeder_stream.FeederSource`, so it is
    interleaved with media chunks rather than pre-sent).

    Args:
        video_source (VideoSourceSimulator): Video source simulator for
            reading video chunks.
        audio_source (AudioSourceSimulator): Audio source simulator for
            reading audio chunks.
        chunk_size_video_bytes (int): Size of each video chunk in bytes.
        chunk_size_audio_secs (float): Duration of each audio chunk in
            seconds.
        asd_config (ActiveSpeakerDetectionConfig): Pre-built
            ``ActiveSpeakerDetectionConfig`` protobuf message.
        diarization_info (AudioDiarizationInfo | None): Optional
            diarization metadata.

    Yields:
        DetectActiveSpeakerRequest: Messages containing config or data.

    Examples:
        >>> gen = asd_request_generator(
        ...     video_source=video_src,
        ...     audio_source=audio_src,
        ...     chunk_size_video_bytes=65536,
        ...     chunk_size_audio_secs=1.0,
        ...     asd_config=config,
        ... )  # doctest: +SKIP
    """
    # 1. Send config as the first message
    yield DetectActiveSpeakerRequest(config=asd_config)
    logger.debug(f"ASD: sent config: {asd_config}")

    # 2. Create iterators and merge via concurrent feeder threads
    video_iter = video_source.read(chunk_size=chunk_size_video_bytes)
    audio_iter = audio_source.read(chunk_duration_secs=chunk_size_audio_secs)

    sources: list[FeederSource] = [
        FeederSource(
            name="video",
            iterator=video_iter,
            transform=lambda c: DetectActiveSpeakerRequest(
                data=ActiveSpeakerDetectionData(video_data=c),
            ),
        ),
        FeederSource(
            name="audio",
            iterator=audio_iter,
            transform=lambda c: DetectActiveSpeakerRequest(
                data=ActiveSpeakerDetectionData(audio_data=c),
            ),
        ),
    ]

    if diarization_info is not None:
        sources.append(
            FeederSource(
                name="diarization",
                iterator=iter([diarization_info]),
                transform=lambda info: DetectActiveSpeakerRequest(
                    data=ActiveSpeakerDetectionData(diarization_info=info),
                ),
            )
        )

    stream: FeederStream[DetectActiveSpeakerRequest] = FeederStream(sources=sources)
    stream.start(request_id="asd-standalone")
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    logger.debug(
        f"ASD data sending complete: "
        f"video={counts.get('video', 0)}, "
        f"audio={counts.get('audio', 0)}, "
        f"diarization={counts.get('diarization', 0)}"
    )
    stream.raise_on_error()
