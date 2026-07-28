# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller client request stream generators and related constants."""

import uuid
from collections.abc import Iterator

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from common.base_utils import logger
from common.feeder_stream import FeederSource
from common.feeder_stream import FeederStream
from common.source_sink.base import BaseFileSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator
from common.source_sink.grpc.audio import simulated_audio_chunk_generator_raw
from common.source_sink.grpc.video import VideoSourceSimulator
from common.source_sink.grpc.video import simulated_video_chunk_generator_raw

BACKPRESSURE_DELAY_SECS = 0
FORWARD_PRESSURE_DELAY_SECS = 0
LOG_FOR_EVERY_N_CHUNKS = 1


def chunk_diarization_info(
    diarization_info: AudioDiarizationInfo,
    rows_per_chunk: int | None = None,
) -> list[AudioDiarizationInfo]:
    """Split diarization info into chunks of N segments each.

    Args:
        diarization_info (AudioDiarizationInfo): The full diarization info to chunk.
        rows_per_chunk (int | None): Number of segment rows per chunk.
            ``None`` sends all segments in a single message.

    Returns:
        list[AudioDiarizationInfo]: List of chunked diarization info messages.
            Empty list if no segments.

    Examples:
        >>> info = AudioDiarizationInfo(segments=[seg1, seg2, seg3])
        >>> chunks = chunk_diarization_info(info, rows_per_chunk=2)
        >>> len(chunks)
        2
        >>> chunks_all = chunk_diarization_info(info)
        >>> len(chunks_all)
        1
    """
    segments = list(diarization_info.segments)
    if not segments:
        return []

    # Send all segments in one message when rows_per_chunk is None
    if rows_per_chunk is None:
        return [diarization_info]

    chunks = []
    for i in range(0, len(segments), rows_per_chunk):
        chunk_segments = segments[i : i + rows_per_chunk]
        chunks.append(AudioDiarizationInfo(segments=chunk_segments))
    return chunks


def _log_diarization_chunks(
    chunks: list[AudioDiarizationInfo],
) -> Iterator[AudioDiarizationInfo]:
    """Yield diarization chunks while debug-logging exactly what is shipped.

    Wraps the chunk list so each chunk is logged at the moment it is
    pulled into the request stream, giving a precise on-the-wire record
    of segment counts and time spans without changing what is sent.

    Args:
        chunks (list[AudioDiarizationInfo]): Diarization chunks to ship,
            as produced by :func:`chunk_diarization_info`.

    Yields:
        AudioDiarizationInfo: Each chunk unchanged, in order.

    Examples:
        >>> info = AudioDiarizationInfo(segments=[seg1, seg2])
        >>> shipped = list(_log_diarization_chunks([info]))
        >>> len(shipped)
        1
    """
    for index, chunk in enumerate(chunks):
        segments = chunk.segments
        # start_time/end_time are uint32 millisecond offsets; report the
        # span so a sparse-looking stream is obviously few-but-wide
        # segments, not dropped data.
        span = f"{segments[0].start_time}-{segments[-1].end_time}ms" if segments else "empty"
        speaker_ids = sorted({seg.speaker_id for seg in segments})
        logger.debug(
            f"Controller client: shipping diarization chunk {index + 1}/"
            f"{len(chunks)} — {len(segments)} segment(s), span={span}, "
            f"speaker_ids={speaker_ids}"
        )
        yield chunk


def create_controller_request_generator(  # noqa: PLR0913
    audio_source: AudioSourceSimulator,
    video_source: VideoSourceSimulator,
    chunk_size_audio_secs: float,
    chunk_size_video_bytes: int,
    s2s_config: SpeechToSpeechConfig | None,
    asd_config: ActiveSpeakerDetectionConfig | None,
    lipsync_config: LipsyncConfig,
    diarization_info: AudioDiarizationInfo | None = None,
    background_audio_source: BaseFileSimulator | None = None,
    translated_audio_source: BaseFileSimulator | None = None,
    bypass_asd: bool = False,
    diarization_rows_per_chunk: int | None = 10,
    input_audio_codec: int = AUDIO_CODEC_WAV,
    request_id: str | None = None,
) -> Iterator[ContentLocalizationRequest]:
    """Create a generator that yields ContentLocalizationRequest objects.

    This generator sends NIM config messages first (controller config,
    optionally S2S, optionally ASD, LipSync), then interleaves audio,
    video, and optional diarization/translated audio chunks into a single
    request stream. Every message carries the request's correlation id
    (``request_id``), which the controller echoes in its responses.

    When ``translated_audio_source`` is provided, S2S is bypassed: the
    controller config signals ``bypass_s2s=True`` and translated audio
    chunks are streamed alongside original audio (needed for ASD).

    Args:
        audio_source (AudioSourceSimulator): Audio source simulator.
        video_source (VideoSourceSimulator): Video source simulator.
        chunk_size_audio_secs (float): Audio chunk size in seconds.
        chunk_size_video_bytes (int): Video chunk size in bytes.
        s2s_config (SpeechToSpeechConfig | None): S2S config protobuf
            message, or ``None`` when S2S is bypassed.
        asd_config (ActiveSpeakerDetectionConfig | None): ASD config
            protobuf message, or ``None`` when ASD is disabled.
        lipsync_config (LipsyncConfig): LipSync config protobuf message.
        diarization_info (AudioDiarizationInfo | None): Optional diarization
            metadata.
        background_audio_source (AudioSourceSimulator | None): Optional
            background audio source for LipSync mixing. ``None`` when no
            background audio is provided.
        translated_audio_source (AudioSourceSimulator | None): Optional
            pre-translated audio source. When provided, S2S is bypassed
            and this audio feeds directly into LipSync.
        bypass_asd (bool): When True, tells the controller to skip ASD
            and use LipSync's internal face detection. Defaults to False.
        diarization_rows_per_chunk (int | None): Number of diarization
            segment rows per chunk. ``None`` sends all segments in a
            single message. Defaults to ``10``.
        input_audio_codec (int): ``audio.v1.AudioCodec`` value describing
            the original input audio stream; sent to the controller as
            ``input_audio_config`` so downstream services receive the
            correct codec. Defaults to ``AUDIO_CODEC_WAV``.
        request_id (str | None): Correlation id stamped on every request
            message and echoed by the controller in its responses.
            ``None`` (the default) generates a fresh UUID4 for this
            request stream.

    Yields:
        ContentLocalizationRequest: Requests containing configs, diarization,
            audio and video data.

    Examples:
        >>> gen = create_controller_request_generator(
        ...     audio_source=audio_src,
        ...     video_source=video_src,
        ...     chunk_size_audio_secs=1.0,
        ...     chunk_size_video_bytes=1048576,
        ...     s2s_config=s2s_cfg,
        ...     asd_config=asd_cfg,
        ...     lipsync_config=ls_cfg,
        ... )  # doctest: +SKIP
    """
    # None sentinel: a UUID default in the signature would be evaluated
    # once at import time and shared by every request stream. Blank ids
    # fall back to a generated one like omitted ids.
    if request_id is None or not request_id.strip():
        request_id = str(uuid.uuid4())
    logger.info(f"Controller | request_id={request_id}")

    bypass_s2s = translated_audio_source is not None

    # --- 1. Send all configs first ---
    # Controller config so the server knows the mode and the input codec
    # (the controller assumes WAV when input_audio_config is omitted).
    yield ContentLocalizationRequest(
        controller_config=ContentLocalizationConfig(
            bypass_s2s=bypass_s2s,
            bypass_asd=bypass_asd,
            input_audio_config=AudioConfig(encoding=input_audio_codec),
        ),
        request_id=request_id,
    )

    if asd_config is not None:
        yield ContentLocalizationRequest(asd_config=asd_config, request_id=request_id)
    yield ContentLocalizationRequest(lipsync_config=lipsync_config, request_id=request_id)

    if s2s_config is not None:
        logger.debug(
            f"Controller | sending S2S config: "
            f"source_language={s2s_config.source_language}, "
            f"target_language={s2s_config.target_language}, "
            f"voice_name={s2s_config.voice_name or 'None'}"
        )
        yield ContentLocalizationRequest(s2s_config=s2s_config, request_id=request_id)
    else:
        logger.info("Controller | S2S bypassed — using translated audio")

    # --- 2. Create generators for audio, video, and conditional streams ---
    audio_generator = simulated_audio_chunk_generator_raw(
        simulator=audio_source, chunk_size_secs=chunk_size_audio_secs
    )
    video_generator = simulated_video_chunk_generator_raw(
        simulator=video_source, chunk_size=chunk_size_video_bytes
    )

    # --- 3. Build feeder sources — only include streams relevant to the mode ---
    sources: list[FeederSource] = [
        FeederSource(
            name="audio",
            iterator=audio_generator,
            transform=lambda chunk: ContentLocalizationRequest(
                audio_data=chunk,
                request_id=request_id,
            ),
        ),
        FeederSource(
            name="video",
            iterator=video_generator,
            transform=lambda chunk: ContentLocalizationRequest(
                video_file_data=chunk,
                request_id=request_id,
            ),
        ),
    ]

    # Diarization is only needed when ASD is active
    if not bypass_asd and diarization_info is not None:
        # None means "send all in one message"; any explicit count must be
        # positive — 0 would make chunk_diarization_info hit range(_, _, 0).
        if diarization_rows_per_chunk is not None and diarization_rows_per_chunk <= 0:
            raise ValueError(
                "diarization_rows_per_chunk must be a positive integer or None, "
                f"got {diarization_rows_per_chunk}"
            )
        diarization_chunks = chunk_diarization_info(
            diarization_info=diarization_info,
            rows_per_chunk=diarization_rows_per_chunk,
        )
        # Surface chunking math up front: chunk count is
        # ceil(segments / rows_per_chunk), so a single chunk from a
        # small file is expected, not a dropped stream.
        logger.debug(
            f"Controller client: prepared {len(diarization_chunks)} "
            f"diarization chunk(s) from {len(diarization_info.segments)} "
            f"segment(s) (rows_per_chunk={diarization_rows_per_chunk})"
        )
        sources.append(
            FeederSource(
                name="diarization",
                iterator=_log_diarization_chunks(chunks=diarization_chunks),
                transform=lambda chunk: ContentLocalizationRequest(
                    diarization_info=chunk,
                    request_id=request_id,
                ),
            )
        )

    # Background audio is optional
    if background_audio_source is not None:
        bg_audio_generator = simulated_audio_chunk_generator_raw(
            simulator=background_audio_source,
            chunk_size_secs=chunk_size_audio_secs,
        )
        sources.append(
            FeederSource(
                name="background_audio",
                iterator=bg_audio_generator,
                transform=lambda chunk: ContentLocalizationRequest(
                    background_audio_data=chunk,
                    request_id=request_id,
                ),
            )
        )

    # Translated audio is only needed in bypass-S2S mode
    if bypass_s2s and translated_audio_source is not None:
        translated_audio_generator = simulated_audio_chunk_generator_raw(
            simulator=translated_audio_source,
            chunk_size_secs=chunk_size_audio_secs,
        )
        sources.append(
            FeederSource(
                name="translated_audio",
                iterator=translated_audio_generator,
                transform=lambda chunk: ContentLocalizationRequest(
                    translated_audio_data=chunk,
                    request_id=request_id,
                ),
            )
        )

    stream: FeederStream[ContentLocalizationRequest] = FeederStream(
        sources=sources,
        backpressure_delay=BACKPRESSURE_DELAY_SECS,
    )
    # The correlation id doubles as the feeder-thread log label so client
    # logs line up with the controller's per-request logs.
    stream.start(request_id=request_id)
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    total = sum(counts.values())
    logger.info(f"Controller | finished transmitting {total} requests to controller service")
    logger.info(f"Controller | audio chunks: {counts.get('audio', 0)}")
    logger.info(f"Controller | video chunks: {counts.get('video', 0)}")
    logger.info(f"Controller | diarization chunks: {counts.get('diarization', 0)}")
    logger.info(f"Controller | bg audio chunks: {counts.get('background_audio', 0)}")
    logger.info(f"Controller | translated audio chunks: {counts.get('translated_audio', 0)}")
    stream.raise_on_error()
