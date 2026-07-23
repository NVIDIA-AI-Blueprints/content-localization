# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct client implementation.

Drives the three services individually — Speech-to-Speech, Active
Speaker Detection, and LipSync — without the controller: S2S output
audio and ASD speaker info are streamed straight into LipSync from the
client process.

Supports pre-translated audio (``--translated-audio``), which skips the
S2S leg entirely, and ``--bypass-asd``, which skips the ASD leg so
LipSync uses internal face detection.
"""

import argparse
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from client.common.diarization import load_diarization_info
from client.common.timing import StageTimer
from client.common.worker import ClientWorker
from client.direct.args import argsfactory
from client.direct.config import DirectPipelineConfig
from client.direct.pipeline import audio_iterator_from_file
from client.direct.pipeline import audio_iterator_from_s2s_response_with_format
from client.direct.pipeline import background_audio_iterator_from_file
from client.direct.pipeline import video_iterator_from_source
from client.direct.stream_adapters import asd_request_generator_with_audio
from client.direct.stream_adapters import lipsync_input_request_generator
from client.direct.stream_adapters import speaker_info_from_asd_response
from common.base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.clients import Client
from common.context import LocalContext
from common.health import check_service_health
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionHandle
from common.nims import LipsyncClient
from common.nims import LipsyncHandle
from common.nims import SpeechToSpeechClient
from common.nims import SpeechToSpeechHandle
from common.source_sink.grpc.audio import AudioSinkSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator
from common.source_sink.grpc.audio import simulated_audio_chunk_generator
from common.source_sink.grpc.video import VideoSinkSimulator
from common.source_sink.grpc.video import VideoSourceSimulator
from common.source_sink.grpc.video import simulated_video_chunk_generator_raw

BUFFER_POLL_TIMEOUT_SECS = 0.1


@dataclass
class _S2SLeg:
    """Resources owned by the Speech-to-Speech leg of the pipeline.

    Attributes:
        audio_source: Source simulator streaming the input audio.
        audio_sink: Sink simulator writing the translated audio output.
        handle: gRPC service handle for the S2S NIM.
        client: Streaming client bound to the handle.
        output_buffer: Buffer receiving S2S responses.
        worker: Background worker thread, set once inference starts.
    """

    audio_source: AudioSourceSimulator
    audio_sink: AudioSinkSimulator
    handle: SpeechToSpeechHandle
    client: SpeechToSpeechClient
    output_buffer: Buffer[SpeechToSpeechResponse] = field(default_factory=Buffer)
    worker: ClientWorker | None = None


@dataclass
class _ASDLeg:
    """Resources owned by the Active Speaker Detection leg of the pipeline.

    Attributes:
        video_source: Source simulator streaming the input video.
        audio_source: Source simulator streaming the input audio.
        request_generator: Merged config/video/audio/diarization request
            stream for the ASD NIM.
        handle: gRPC service handle for the ASD NIM.
        client: Streaming client bound to the handle.
        output_buffer: Buffer receiving ASD responses.
        worker: Background worker thread, set once inference starts.
    """

    video_source: VideoSourceSimulator
    audio_source: AudioSourceSimulator
    request_generator: Iterator[DetectActiveSpeakerRequest]
    handle: ActiveSpeakerDetectionHandle
    client: ActiveSpeakerDetectionClient
    output_buffer: Buffer[DetectActiveSpeakerResponse] = field(default_factory=Buffer)
    worker: ClientWorker | None = None


@dataclass
class _LipsyncLeg:
    """Resources owned by the LipSync leg of the pipeline.

    Attributes:
        video_source: Source simulator streaming the input video.
        video_sink: Sink simulator writing the lip-synced output video.
        handle: gRPC service handle for the LipSync NIM.
        client: Streaming client bound to the handle.
        output_buffer: Buffer receiving LipSync responses.
        worker: Background worker thread, set once inference starts.
    """

    video_source: VideoSourceSimulator
    video_sink: VideoSinkSimulator
    handle: LipsyncHandle
    client: LipsyncClient
    output_buffer: Buffer[LipsyncResponse] = field(default_factory=Buffer)
    worker: ClientWorker | None = None


def _check_services_health(cfg: DirectPipelineConfig) -> None:
    """Probe the health endpoint of every service used in this run.

    S2S is skipped when pre-translated audio is provided; ASD is
    skipped when bypassed.

    Args:
        cfg (DirectPipelineConfig): Pipeline configuration with server
            addresses and bypass flags.

    Returns:
        None

    Raises:
        ConnectionError: If a required service is unreachable.

    Examples:
        >>> _check_services_health(cfg=cfg)  # doctest: +SKIP
    """
    if cfg.translated_audio is None:
        check_service_health(server=cfg.s2s_server)
        logger.info("S2S service is healthy")
    else:
        logger.info(f"pre-translated audio: {cfg.translated_audio} (S2S skipped)")
    if not cfg.bypass_asd:
        check_service_health(server=cfg.asd_server)
        logger.info("ASD service is healthy")
    check_service_health(server=cfg.lipsync_server)
    logger.info("LipSync service is healthy")


def _setup_s2s_leg(cfg: DirectPipelineConfig) -> _S2SLeg:
    """Create the S2S input source, output sink, and service client.

    Args:
        cfg (DirectPipelineConfig): Pipeline configuration with the S2S
            server address and audio I/O paths.

    Returns:
        _S2SLeg: Initialized S2S leg resources.

    Examples:
        >>> leg = _setup_s2s_leg(cfg=cfg)  # doctest: +SKIP
    """
    audio_source = AudioSourceSimulator(file_path=cfg.input_audio)
    audio_sink = AudioSinkSimulator(
        frame_rate=audio_source.frame_rate,
        sample_width=audio_source.sample_width,
        n_channels=audio_source.n_channels,
        n_frames=audio_source.n_frames,
        file_path=cfg.output_audio,
        chunk_duration_secs=cfg.chunk_size_audio_secs,
        audio_format=cfg.output_audio.split(".")[-1],
    )
    host, port = cfg.s2s_server.split(":", 1)
    handle = SpeechToSpeechHandle(host=host, port=int(port))
    return _S2SLeg(
        audio_source=audio_source,
        audio_sink=audio_sink,
        handle=handle,
        client=SpeechToSpeechClient(handle=handle),
    )


def _asd_video_data_iterator(
    source: VideoSourceSimulator,
    chunk_size_bytes: int,
) -> Iterator[ActiveSpeakerDetectionData]:
    """Yield ASD video-data messages from a video source.

    Args:
        source (VideoSourceSimulator): Video source simulator to read.
        chunk_size_bytes (int): Bytes per video chunk.

    Yields:
        ActiveSpeakerDetectionData: Messages with ``video_data`` set.

    Examples:
        >>> it = _asd_video_data_iterator(
        ...     source=video_source,
        ...     chunk_size_bytes=65536,
        ... )  # doctest: +SKIP
    """
    for chunk in source.read(chunk_size=chunk_size_bytes):
        yield ActiveSpeakerDetectionData(video_data=chunk)


def _asd_audio_data_iterator(
    source: AudioSourceSimulator,
    chunk_size_secs: float,
) -> Iterator[ActiveSpeakerDetectionData]:
    """Yield ASD audio-data messages from an audio source.

    Args:
        source (AudioSourceSimulator): Audio source simulator to read.
        chunk_size_secs (float): Seconds of audio per chunk.

    Yields:
        ActiveSpeakerDetectionData: Messages with ``audio_data`` set.

    Examples:
        >>> it = _asd_audio_data_iterator(
        ...     source=audio_source,
        ...     chunk_size_secs=1.0,
        ... )  # doctest: +SKIP
    """
    for chunk in source.read(chunk_duration_secs=chunk_size_secs):
        yield ActiveSpeakerDetectionData(audio_data=chunk)


def _setup_asd_leg(cfg: DirectPipelineConfig, args: argparse.Namespace) -> _ASDLeg:
    """Create the ASD sources, request generator, and service client.

    Args:
        cfg (DirectPipelineConfig): Pipeline configuration with the ASD
            server address, media I/O paths, and chunk sizes.
        args (argparse.Namespace): Parsed CLI arguments providing the
            diarization file, format, and chunking mode.

    Returns:
        _ASDLeg: Initialized ASD leg resources.

    Examples:
        >>> leg = _setup_asd_leg(cfg=cfg, args=args)  # doctest: +SKIP
    """
    diarization_info = load_diarization_info(
        diarization_file=args.diarization_file,
        diarization_format=args.diarization_format,
        combine_chunks_per_speaker=not args.diarization_chunked_per_segment,
    )
    if diarization_info:
        logger.info(f"diarization: {len(diarization_info.segments)} segments")

    video_source = VideoSourceSimulator(file_path=cfg.input_mp4)
    audio_source = AudioSourceSimulator(file_path=cfg.input_audio)
    request_generator = asd_request_generator_with_audio(
        video_iter=_asd_video_data_iterator(
            source=video_source,
            chunk_size_bytes=cfg.chunk_size_video_bytes,
        ),
        audio_iter=_asd_audio_data_iterator(
            source=audio_source,
            chunk_size_secs=cfg.chunk_size_audio_secs,
        ),
        asd_config=cfg.asd_config,
        diarization_info=diarization_info,
    )
    host, port = cfg.asd_server.split(":", 1)
    handle = ActiveSpeakerDetectionHandle(host=host, port=int(port))
    return _ASDLeg(
        video_source=video_source,
        audio_source=audio_source,
        request_generator=request_generator,
        handle=handle,
        client=ActiveSpeakerDetectionClient(handle=handle),
    )


def _setup_lipsync_leg(cfg: DirectPipelineConfig) -> _LipsyncLeg:
    """Create the LipSync video source, output sink, and service client.

    Args:
        cfg (DirectPipelineConfig): Pipeline configuration with the
            LipSync server address and video I/O paths.

    Returns:
        _LipsyncLeg: Initialized LipSync leg resources.

    Examples:
        >>> leg = _setup_lipsync_leg(cfg=cfg)  # doctest: +SKIP
    """
    video_source = VideoSourceSimulator(file_path=cfg.input_mp4)
    video_sink = VideoSinkSimulator(
        file_path=cfg.output_mp4,
        chunk_size=cfg.chunk_size_video_bytes,
    )
    host, port = cfg.lipsync_server.split(":", 1)
    handle = LipsyncHandle(host=host, port=int(port))
    if cfg.background_audio_input:
        logger.info(f"background audio: {cfg.background_audio_input}")
    return _LipsyncLeg(
        video_source=video_source,
        video_sink=video_sink,
        handle=handle,
        client=LipsyncClient(handle=handle),
    )


def _s2s_request_generator(
    cfg: DirectPipelineConfig,
    audio_source: AudioSourceSimulator,
) -> Iterator[SpeechToSpeechRequest]:
    """Yield the S2S config message followed by audio chunks.

    Args:
        cfg (DirectPipelineConfig): Pipeline configuration providing the
            S2S protobuf config and audio chunk size.
        audio_source (AudioSourceSimulator): Input audio simulator.

    Yields:
        SpeechToSpeechRequest: Config message, then audio data messages.

    Examples:
        >>> gen = _s2s_request_generator(
        ...     cfg=cfg,
        ...     audio_source=audio_source,
        ... )  # doctest: +SKIP
    """
    yield SpeechToSpeechRequest(config=cfg.s2s_config)
    yield from simulated_audio_chunk_generator(
        simulator=audio_source,
        chunk_size_secs=cfg.chunk_size_audio_secs,
    )


def _start_client_worker(
    name: str,
    client: Client,
    request_iterator: Iterator,
    output_buffer: Buffer,
) -> ClientWorker:
    """Start a background worker that streams requests through a client.

    Args:
        name (str): Worker/request identifier used in logs and thread
            names (e.g. ``"direct-s2s"``).
        client (Client): Streaming NIM client to invoke.
        request_iterator (Iterator): Request stream passed to the client.
        output_buffer (Buffer): Buffer receiving the client's responses.

    Returns:
        ClientWorker: The started worker thread.

    Examples:
        >>> worker = _start_client_worker(
        ...     name="direct-s2s",
        ...     client=s2s_client,
        ...     request_iterator=requests,
        ...     output_buffer=buffer,
        ... )  # doctest: +SKIP
    """

    def run() -> None:
        logger.debug(f"{name} client running on thread: {threading.current_thread().name}")
        client(
            request_iterator=request_iterator,
            output_buffer=output_buffer,
            context=LocalContext(),
            request_id=name,
        )

    worker = ClientWorker(target=run, name=name)
    worker.start()
    return worker


def _build_lipsync_request_generator(
    cfg: DirectPipelineConfig,
    s2s_leg: _S2SLeg | None,
    asd_leg: _ASDLeg | None,
    lipsync_leg: _LipsyncLeg,
) -> Iterator[LipsyncRequest]:
    """Assemble the LipSync request stream from the upstream legs.

    Video always comes from the LipSync leg's own source. Audio comes
    from the pre-translated file when S2S is bypassed, otherwise from
    the live S2S response stream. Speaker info comes from the live ASD
    response stream unless ASD is bypassed.

    Args:
        cfg (DirectPipelineConfig): Pipeline configuration with chunk
            sizes, the LipSync protobuf config, and optional
            translated/background audio paths.
        s2s_leg (_S2SLeg | None): S2S leg, or ``None`` when using
            pre-translated audio.
        asd_leg (_ASDLeg | None): ASD leg, or ``None`` when ASD is
            bypassed.
        lipsync_leg (_LipsyncLeg): LipSync leg providing the video
            source.

    Returns:
        Iterator[LipsyncRequest]: Merged request stream for LipSync.

    Examples:
        >>> gen = _build_lipsync_request_generator(
        ...     cfg=cfg,
        ...     s2s_leg=s2s_leg,
        ...     asd_leg=None,
        ...     lipsync_leg=lipsync_leg,
        ... )  # doctest: +SKIP
    """
    video_iterator = video_iterator_from_source(
        source_iterator=simulated_video_chunk_generator_raw(
            simulator=lipsync_leg.video_source,
            chunk_size=cfg.chunk_size_video_bytes,
        )
    )
    if s2s_leg is None:
        audio_iterator = audio_iterator_from_file(
            file_path=cfg.translated_audio,
            chunk_size_secs=cfg.chunk_size_audio_secs,
        )
    else:
        audio_iterator = audio_iterator_from_s2s_response_with_format(
            response_iter=RequestIteratorFromBuffer(
                s2s_leg.output_buffer,
                poll_timeout=BUFFER_POLL_TIMEOUT_SECS,
            ),
            audio_format=cfg.output_audio.split(".")[-1],
            output_sink=s2s_leg.audio_sink,
        )
    speaker_info_iterator = (
        speaker_info_from_asd_response(
            response_iter=RequestIteratorFromBuffer(
                asd_leg.output_buffer,
                poll_timeout=BUFFER_POLL_TIMEOUT_SECS,
            )
        )
        if asd_leg is not None
        else None
    )
    background_audio_iterator = (
        background_audio_iterator_from_file(file_path=cfg.background_audio_input)
        if cfg.background_audio_input
        else None
    )
    return lipsync_input_request_generator(
        video_iterator=video_iterator,
        audio_iterator=audio_iterator,
        speaker_info_iterator=speaker_info_iterator,
        lipsync_config=cfg.lipsync_config,
        background_audio_iterator=background_audio_iterator,
    )


def _write_lipsync_output(
    response_iter: Iterator[LipsyncResponse],
    video_sink: VideoSinkSimulator,
) -> int:
    """Drain LipSync responses and write video chunks to the sink.

    Args:
        response_iter (Iterator[LipsyncResponse]): LipSync response
            stream to drain.
        video_sink (VideoSinkSimulator): Output video sink.

    Returns:
        int: Number of video chunks written.

    Raises:
        grpc.RpcError: If the LipSync stream fails.

    Examples:
        >>> count = _write_lipsync_output(
        ...     response_iter=responses,
        ...     video_sink=video_sink,
        ... )  # doctest: +SKIP
    """
    chunk_count = 0
    try:
        for lipsync_response in response_iter:
            if lipsync_response.video_file_data:
                chunk_count += 1
                video_sink.write(video_bytes=lipsync_response.video_file_data)
                if chunk_count % 1000 == 0:
                    logger.debug(f"lipsync | received chunk: {chunk_count}")
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()}: {e.details()}")
        logger.error(
            "This might be due to a timeout or connection issue. Try running the client again."
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    return chunk_count


def _close_legs(
    s2s_leg: _S2SLeg | None,
    asd_leg: _ASDLeg | None,
    lipsync_leg: _LipsyncLeg,
) -> None:
    """Close all open sources, sinks, and gRPC channels.

    Args:
        s2s_leg (_S2SLeg | None): S2S leg, or ``None`` when skipped.
        asd_leg (_ASDLeg | None): ASD leg, or ``None`` when bypassed.
        lipsync_leg (_LipsyncLeg): LipSync leg (always present).

    Returns:
        None

    Examples:
        >>> _close_legs(
        ...     s2s_leg=s2s_leg,
        ...     asd_leg=None,
        ...     lipsync_leg=lipsync_leg,
        ... )  # doctest: +SKIP
    """
    if s2s_leg is not None:
        if s2s_leg.audio_source.is_open():
            s2s_leg.audio_source.close()
        if s2s_leg.audio_sink.is_open():
            s2s_leg.audio_sink.close()
    if asd_leg is not None:
        if asd_leg.video_source.is_open():
            asd_leg.video_source.close()
        if asd_leg.audio_source.is_open():
            asd_leg.audio_source.close()
    if lipsync_leg.video_source.is_open():
        lipsync_leg.video_source.close()
    if lipsync_leg.video_sink.is_open():
        lipsync_leg.video_sink.flush()
        lipsync_leg.video_sink.close()
    # Explicitly close gRPC channels to the NIMs so connections terminate
    # immediately rather than at process exit / GC.
    lipsync_leg.handle.close()
    if s2s_leg is not None:
        s2s_leg.handle.close()
    if asd_leg is not None:
        asd_leg.handle.close()


def main() -> None:
    """Run the direct three-service pipeline (S2S, ASD, and LipSync).

    Validates the configuration, checks service health, sets up the
    per-service legs, streams the media through all three services
    concurrently, writes the lip-synced output video, and cleans up.
    """
    args = argsfactory().parse_args()

    # Build and validate the pipeline config
    # (server addresses + NIM configs + streaming params + I/O paths)
    cfg = DirectPipelineConfig.from_args(args)
    cfg.validate_io()

    timer = StageTimer()

    with timer.stage("health_check"):
        _check_services_health(cfg=cfg)

    with timer.stage("setup"):
        s2s_leg = None if cfg.translated_audio else _setup_s2s_leg(cfg=cfg)
        asd_leg = None if cfg.bypass_asd else _setup_asd_leg(cfg=cfg, args=args)
        lipsync_leg = _setup_lipsync_leg(cfg=cfg)

    with timer.stage("inference"):
        # Start the upstream workers first: their responses feed LipSync.
        if s2s_leg is not None:
            s2s_leg.worker = _start_client_worker(
                name="direct-s2s",
                client=s2s_leg.client,
                request_iterator=_s2s_request_generator(
                    cfg=cfg,
                    audio_source=s2s_leg.audio_source,
                ),
                output_buffer=s2s_leg.output_buffer,
            )
        if asd_leg is not None:
            asd_leg.worker = _start_client_worker(
                name="direct-asd",
                client=asd_leg.client,
                request_iterator=asd_leg.request_generator,
                output_buffer=asd_leg.output_buffer,
            )
        lipsync_leg.worker = _start_client_worker(
            name="direct-lipsync",
            client=lipsync_leg.client,
            request_iterator=_build_lipsync_request_generator(
                cfg=cfg,
                s2s_leg=s2s_leg,
                asd_leg=asd_leg,
                lipsync_leg=lipsync_leg,
            ),
            output_buffer=lipsync_leg.output_buffer,
        )

        chunk_count = _write_lipsync_output(
            response_iter=RequestIteratorFromBuffer(
                lipsync_leg.output_buffer,
                poll_timeout=BUFFER_POLL_TIMEOUT_SECS,
            ),
            video_sink=lipsync_leg.video_sink,
        )
        logger.info(f"inference complete: {chunk_count} LipSync response chunks")

        # Surface any worker failure on the main thread so the CLI exits non-zero.
        lipsync_leg.worker.join_and_raise()
        if s2s_leg is not None and s2s_leg.worker is not None:
            s2s_leg.worker.join_and_raise()
        if asd_leg is not None and asd_leg.worker is not None:
            asd_leg.worker.join_and_raise()

    with timer.stage("cleanup"):
        _close_legs(s2s_leg=s2s_leg, asd_leg=asd_leg, lipsync_leg=lipsync_leg)

    timer.log_summary()


if __name__ == "__main__":
    main()
