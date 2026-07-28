# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-orchestration helpers for the content localization pipeline.

This module contains the per-request pipeline building blocks used by the
Controller: functions that launch the S2S, ASD, and LipSync client threads,
drain unused buffer queues, stream LipSync responses back to the client,
validate the translated-audio contract, and join every pipeline thread
during cleanup.

Request Flow
============

.. code-block:: text

    Client Request Stream
      |
    ContentLocalizationDeserializer (background thread)
      |-> controller_config_buffer -> determines bypass_s2s mode
      |-> audio_buffer (queue 0) -> S2S Client Thread -> s2s_output_buffer  [skip if bypass]
      |-> audio_buffer (queue 1) -> ASD Client Thread (audio input)
      |-> video_buffer (queue 0) -> ASD Client Thread (video input) -> asd_output_buffer
      |-> video_buffer (queue 1) -> LipSync Client Thread -> lipsync_output_buffer
      |-> diarization_buffer     -> ASD Client Thread (diarization input)
      |-> background_audio_buffer -> LipSync Client Thread (optional)
      |-> translated_audio_buffer -> LipSync Client Thread (bypass S2S only)
      |
    Main Thread yields ContentLocalizationResponse

Both ``bypass_s2s`` and ``bypass_asd`` are per-request flags read from
``ContentLocalizationConfig``. When a stage is bypassed, its input queues
are drained so the deserializer's fan-out never accumulates data for the
lifetime of the request.
"""

import threading
from collections.abc import Iterator

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.errors import PipelineInputError
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionHandle
from common.nims import LipsyncClient
from common.nims import LipsyncHandle
from common.nims import SpeechToSpeechClient
from common.nims import SpeechToSpeechHandle
from controller_service.config import _PipelineConfig
from controller_service.conversions import to_asd_audio_data
from controller_service.conversions import to_asd_diarization_data
from controller_service.conversions import to_asd_video_data
from controller_service.conversions import to_lipsync_background_audio
from controller_service.conversions import to_lipsync_video
from controller_service.conversions import to_s2s_request
from controller_service.deserializer import AudioQueueConsumer
from controller_service.deserializer import ContentLocalizationDeserializer
from controller_service.deserializer import DiarizationQueueConsumer
from controller_service.deserializer import VideoQueueConsumer
from controller_service.helpers import CONTROLLER_CLEANUP_TIMEOUT
from controller_service.helpers import _extract_config
from controller_service.stream_adapters import asd_request_generator
from controller_service.stream_adapters import asd_response_to_lipsync_speaker_info
from controller_service.stream_adapters import lipsync_request_generator
from controller_service.stream_adapters import s2s_audio_to_lipsync_audio
from controller_service.stream_adapters import translated_audio_to_lipsync_audio


def _start_s2s_thread(
    deserializer: ContentLocalizationDeserializer,
    s2s_server: SpeechToSpeechHandle | None,
    s2s_output_buffer: "Buffer[SpeechToSpeechResponse]",
    input_audio_format: str,
    bypass_s2s: bool,
    context: grpc.ServicerContext,
    request_id: str,
) -> threading.Thread | None:
    """Start S2S client thread or drain thread when bypassed.

    The request generator reads the single ``s2s_config`` message with a
    bounded wait and then streams audio as soon as it arrives, so S2S
    processing starts without waiting for the full inbound stream.

    Args:
        deserializer (ContentLocalizationDeserializer): Active
            deserializer.
        s2s_server (SpeechToSpeechHandle | None): Per-request S2S
            handle. Required when ``bypass_s2s`` is False.
        s2s_output_buffer (Buffer): Output buffer for S2S
            responses.
        input_audio_format (str): Audio format string.
        bypass_s2s (bool): Whether to skip S2S.
        context (grpc.ServicerContext): gRPC context.
        request_id (str): Request identifier.

    Returns:
        threading.Thread | None: Started thread, or ``None``
            if precondition check aborted.

    Examples:
        >>> thr = _start_s2s_thread(  # doctest: +SKIP
        ...     deserializer=des,
        ...     s2s_server=services.s2s_server,
        ...     s2s_output_buffer=buf,
        ...     input_audio_format="wav",
        ...     bypass_s2s=False,
        ...     context=ctx,
        ...     request_id="r1",
        ... )
    """
    if not bypass_s2s:
        # Precondition validated in _check_services_health
        def _s2s_request_generator() -> Iterator:
            # The config is a single message; a bounded wait keeps audio
            # streaming decoupled from the end of the inbound stream.
            s2s_config = _extract_config(deserializer.s2s_config_buffer, "s2s_config")
            if s2s_config is not None:
                config_request = SpeechToSpeechRequest()
                config_request.config.CopyFrom(s2s_config)
                yield config_request
            for req in RequestIteratorFromBuffer(
                deserializer.audio_buffer,
                consumer_id=AudioQueueConsumer.S2S,
            ):
                yield to_s2s_request(req, input_audio_format=input_audio_format)
            # Exhaust any late config packets so the queue never accumulates.
            for _ in RequestIteratorFromBuffer(deserializer.s2s_config_buffer, consumer_id=0):
                pass

        def _run_s2s() -> None:
            logger.debug(f"S2S client thread started: {threading.current_thread().name}")
            s2s_client = SpeechToSpeechClient(handle=s2s_server)
            s2s_client(
                request_iterator=_s2s_request_generator(),
                output_buffer=s2s_output_buffer,
                context=context,
                request_id=request_id,
            )

        thread = threading.Thread(
            target=_run_s2s,
            daemon=True,
            name=f"S2S-{request_id}",
        )
        thread.start()
        logger.debug("S2S client thread launched")
        return thread

    # Drain audio_buffer queue 0 in bypass mode
    thread = _start_drain_thread(
        drains=[
            (deserializer.audio_buffer, AudioQueueConsumer.S2S),
            (deserializer.s2s_config_buffer, 0),
        ],
        name=f"S2S-drain-{request_id}",
    )
    logger.debug("S2S thread skipped (bypass_s2s mode), drain launched")
    return thread


def _start_drain_thread(
    drains: "list[tuple[Buffer, int]]",
    name: str,
) -> threading.Thread:
    """Start a daemon thread that exhausts unused buffer queues.

    The deserializer fans every request field out to fixed consumer queues
    even when the consuming stage is bypassed or the input is unused in the
    current mode; draining keeps those queues from accumulating data for the
    lifetime of the request.

    Args:
        drains (list[tuple[Buffer, int]]): Pairs of (buffer, consumer_id)
            queues to exhaust sequentially. Each queue terminates once its
            buffer is done and empty.
        name (str): Thread name, used in cleanup logs.

    Returns:
        threading.Thread: The started daemon drain thread.

    Examples:
        >>> thr = _start_drain_thread(  # doctest: +SKIP
        ...     drains=[(des.audio_buffer, 0)],
        ...     name="S2S-drain-r1",
        ... )
    """

    def _drain() -> None:
        for buffer, consumer_id in drains:
            for _ in RequestIteratorFromBuffer(buffer, consumer_id=consumer_id):
                pass

    thread = threading.Thread(
        target=_drain,
        daemon=True,
        name=name,
    )
    thread.start()
    return thread


def _has_background_audio(lipsync_config: "LipsyncConfig") -> bool:
    """Return True when the request supplies a separate background-audio stream.

    Args:
        lipsync_config (LipsyncConfig): Effective LipSync configuration.

    Returns:
        bool: True when ``background_audio_config`` is present and declares
            a provided background-audio stream.

    Examples:
        >>> _has_background_audio(  # doctest: +SKIP
        ...     lipsync_config=LipsyncConfig(),
        ... )
        False
    """
    return (
        lipsync_config.HasField("background_audio_config")
        and lipsync_config.background_audio_config.is_background_audio_provided
    )


def _start_unused_input_drains(
    deserializer: ContentLocalizationDeserializer,
    pipeline_config: "_PipelineConfig",
    request_id: str,
) -> list[threading.Thread]:
    """Drain optional input buffers that the current mode never consumes.

    ``translated_audio_buffer`` is only read in bypass-S2S mode and
    ``background_audio_buffer`` only when background audio is declared;
    draining them keeps payloads sent in a non-consuming mode from
    accumulating.

    Args:
        deserializer (ContentLocalizationDeserializer): Active deserializer.
        pipeline_config (_PipelineConfig): Effective pipeline configuration.
        request_id (str): Request identifier.

    Returns:
        list[threading.Thread]: Zero or one started drain thread.

    Examples:
        >>> threads = _start_unused_input_drains(  # doctest: +SKIP
        ...     deserializer=des,
        ...     pipeline_config=cfg,
        ...     request_id="r1",
        ... )
    """
    drains: list[tuple[Buffer, int]] = []
    if not pipeline_config.bypass_s2s:
        drains.append((deserializer.translated_audio_buffer, 0))
    if not _has_background_audio(lipsync_config=pipeline_config.lipsync_config):
        drains.append((deserializer.background_audio_buffer, 0))
    if not drains:
        return []
    drain_thread = _start_drain_thread(
        drains=drains,
        name=f"UnusedInput-drain-{request_id}",
    )
    logger.debug(f"Unused-input drain launched for {len(drains)} buffer(s)")
    return [drain_thread]


def _start_asd_thread(
    deserializer: ContentLocalizationDeserializer,
    asd_server: "ActiveSpeakerDetectionHandle | None",
    asd_config: "ActiveSpeakerDetectionConfig | None",
    asd_active: bool,
    context: grpc.ServicerContext,
    request_id: str,
) -> tuple[threading.Thread | None, "Buffer | None"]:
    """Start ASD client thread or drain threads when ASD is inactive.

    When ``asd_active`` is False, drains the four ASD-related buffer
    queues (video queue 0, audio queue 1, diarization queue 0, ASD
    config) to prevent accumulation.

    Args:
        deserializer (ContentLocalizationDeserializer): Active
            deserializer.
        asd_server (ActiveSpeakerDetectionHandle | None): Per-request
            ASD handle. Required when ``asd_active`` is True.
        asd_config (ActiveSpeakerDetectionConfig | None): ASD
            config protobuf or ``None``.
        asd_active (bool): Whether ASD runs in this request.
        context (grpc.ServicerContext): gRPC context.
        request_id (str): Request identifier.

    Returns:
        tuple[threading.Thread | None, Buffer | None]: The ASD
            thread (or drain thread) and the ASD output buffer
            (``None`` when ASD is inactive).

    Examples:
        >>> thr, buf = _start_asd_thread(  # doctest: +SKIP
        ...     deserializer=des,
        ...     asd_server=services.asd_server,
        ...     asd_config=cfg,
        ...     asd_active=True,
        ...     context=ctx,
        ...     request_id="r1",
        ... )
    """
    if asd_active:
        asd_output_buffer: Buffer[DetectActiveSpeakerResponse] = Buffer(num_queues=1)

        def _asd_video_iter() -> Iterator:
            for req in RequestIteratorFromBuffer(
                deserializer.video_buffer,
                consumer_id=VideoQueueConsumer.ASD,
            ):
                yield to_asd_video_data(req)

        def _asd_audio_iter() -> Iterator:
            for req in RequestIteratorFromBuffer(
                deserializer.audio_buffer,
                consumer_id=AudioQueueConsumer.ASD,
            ):
                yield to_asd_audio_data(req)

        def _asd_diarization_iter() -> Iterator:
            for req in RequestIteratorFromBuffer(
                deserializer.diarization_buffer,
                consumer_id=DiarizationQueueConsumer.ASD,
            ):
                yield to_asd_diarization_data(req)

        def _run_asd() -> None:
            logger.debug(f"ASD client thread started: {threading.current_thread().name}")
            asd_client = ActiveSpeakerDetectionClient(handle=asd_server)
            asd_request_iter = asd_request_generator(
                video_iter=_asd_video_iter(),
                audio_iter=_asd_audio_iter(),
                asd_config=asd_config,
                diarization_iter=_asd_diarization_iter(),
            )
            asd_client(
                request_iterator=asd_request_iter,
                output_buffer=asd_output_buffer,
                context=context,
                request_id=request_id,
            )

        asd_thread = threading.Thread(
            target=_run_asd,
            daemon=True,
            name=f"ASD-{request_id}",
        )
        asd_thread.start()
        logger.debug("ASD client thread launched")
        return asd_thread, asd_output_buffer

    # Drain ASD-related buffers so items don't accumulate
    drain_thread = _start_drain_thread(
        drains=[
            (deserializer.video_buffer, VideoQueueConsumer.ASD),
            (deserializer.audio_buffer, AudioQueueConsumer.ASD),
            (deserializer.diarization_buffer, DiarizationQueueConsumer.ASD),
            (deserializer.asd_config_buffer, 0),
        ],
        name=f"ASD-drain-{request_id}",
    )
    logger.debug("ASD bypassed; drain thread launched for ASD buffers")
    return drain_thread, None


def _start_lipsync_thread(
    deserializer: ContentLocalizationDeserializer,
    lipsync_server: LipsyncHandle,
    s2s_output_buffer: "Buffer[SpeechToSpeechResponse]",
    asd_output_buffer: "Buffer[DetectActiveSpeakerResponse] | None",
    pipeline_config: "_PipelineConfig",
    context: grpc.ServicerContext,
    request_id: str,
) -> tuple[threading.Thread, "Buffer[LipsyncResponse]"]:
    """Start LipSync client thread with appropriate inputs.

    Wires video, audio, speaker info, and background audio
    iterators based on the pipeline configuration.

    Args:
        deserializer (ContentLocalizationDeserializer): Active
            deserializer.
        lipsync_server (LipsyncHandle): Per-request LipSync handle.
        s2s_output_buffer (Buffer): S2S output buffer.
        asd_output_buffer (Buffer | None): ASD output buffer,
            or ``None`` when ASD is inactive.
        pipeline_config (_PipelineConfig): Pipeline config.
        context (grpc.ServicerContext): gRPC context.
        request_id (str): Request identifier.

    Returns:
        tuple[threading.Thread, Buffer]: LipSync thread and
            output buffer.

    Examples:
        >>> thr, buf = _start_lipsync_thread(  # doctest: +SKIP
        ...     deserializer=des,
        ...     lipsync_server=services.lipsync_server,
        ...     s2s_output_buffer=s2s_buf,
        ...     asd_output_buffer=asd_buf,
        ...     pipeline_config=cfg,
        ...     context=ctx,
        ...     request_id="r1",
        ... )
    """
    lipsync_output_buffer: Buffer[LipsyncResponse] = Buffer(num_queues=1)
    lipsync_config = pipeline_config.lipsync_config

    # Video input
    lipsync_video_iter = (
        to_lipsync_video(req)
        for req in RequestIteratorFromBuffer(
            deserializer.video_buffer,
            consumer_id=VideoQueueConsumer.LIPSYNC,
        )
    )

    # Audio input: translated audio or S2S output
    if pipeline_config.bypass_s2s:
        lipsync_audio_iter = translated_audio_to_lipsync_audio(
            request_iter=RequestIteratorFromBuffer(
                deserializer.translated_audio_buffer,
                consumer_id=0,
            ),
        )
    else:
        lipsync_audio_iter = s2s_audio_to_lipsync_audio(
            response_iter=RequestIteratorFromBuffer(s2s_output_buffer),
            audio_format=pipeline_config.s2s_output_format,
        )

    # Speaker info: from ASD output when ASD runs in this request
    if pipeline_config.asd_active:
        if asd_output_buffer is None:
            raise ValueError("asd_output_buffer is required when ASD is active")
        lipsync_speaker_info_iter = asd_response_to_lipsync_speaker_info(
            response_iter=RequestIteratorFromBuffer(asd_output_buffer),
        )
    else:
        lipsync_speaker_info_iter = None

    # Background audio
    has_background_audio = _has_background_audio(lipsync_config=lipsync_config)
    if has_background_audio:
        lipsync_bg_audio_iter = (
            to_lipsync_background_audio(req)
            for req in RequestIteratorFromBuffer(
                deserializer.background_audio_buffer,
                consumer_id=0,
            )
        )
    else:
        lipsync_bg_audio_iter = None

    lipsync_request_iter = lipsync_request_generator(
        video_iter=lipsync_video_iter,
        audio_iter=lipsync_audio_iter,
        speaker_info_iter=lipsync_speaker_info_iter,
        lipsync_config=lipsync_config,
        background_audio_iter=lipsync_bg_audio_iter,
    )

    def _run_lipsync() -> None:
        logger.debug(f"LipSync client thread started: {threading.current_thread().name}")
        lipsync_client = LipsyncClient(handle=lipsync_server)
        lipsync_client(
            request_iterator=lipsync_request_iter,
            output_buffer=lipsync_output_buffer,
            context=context,
            request_id=request_id,
        )

    lipsync_thread = threading.Thread(
        target=_run_lipsync,
        daemon=True,
        name=f"LipSync-{request_id}",
    )
    lipsync_thread.start()
    logger.debug("LipSync client thread launched")
    return lipsync_thread, lipsync_output_buffer


def _yield_responses(
    lipsync_output_buffer: "Buffer[LipsyncResponse]",
    request_id: str,
    deserializer: ContentLocalizationDeserializer,
    bypass_s2s: bool,
) -> Iterator[ContentLocalizationResponse]:
    """Stream LipSync responses to the gRPC client.

    LipSync keepalives are forwarded as controller keepalives so the end
    client observes app-level liveness while the pipeline waits on
    long-running work (for example a dubbing job). Exceptions propagate to
    the servicer, the pipeline's single abort point.

    Before every yield, the deserializer's routing counters are checked so
    a request that illegally streams ``translated_audio_data`` without
    ``bypass_s2s`` is rejected before any output reaches the client (an
    invalid request must produce no output).

    Args:
        lipsync_output_buffer (Buffer): Buffer of LipSync
            responses.
        request_id (str): Request identifier.
        deserializer (ContentLocalizationDeserializer): Deserializer whose
            routing counters record translated-audio arrivals.
        bypass_s2s (bool): Effective bypass flag for this request.

    Yields:
        ContentLocalizationResponse: Responses with video data or a
            keepalive payload.

    Raises:
        PipelineInputError: When forbidden translated audio has arrived;
            the servicer reports it as ``INVALID_ARGUMENT``.

    Examples:
        >>> for resp in _yield_responses(  # doctest: +SKIP
        ...     lipsync_output_buffer=buf,
        ...     request_id="r1",
        ...     deserializer=des,
        ...     bypass_s2s=False,
        ... ):
        ...     pass
    """
    lipsync_response_iter = RequestIteratorFromBuffer(lipsync_output_buffer)
    logger.info("Starting to read LipSync responses.")
    _raise_if_forbidden_translated_audio(deserializer=deserializer, bypass_s2s=bypass_s2s)
    response_count = 0
    _keepalive_count = 0
    for lipsync_response in lipsync_response_iter:
        response_count += 1
        logger.debug(f"Processing LipSync response #{response_count}")
        # Re-check before each yield: the deserializer routes on a
        # background thread, so a forbidden chunk can arrive mid-stream.
        _raise_if_forbidden_translated_audio(deserializer=deserializer, bypass_s2s=bypass_s2s)
        if lipsync_response.HasField("video_file_data"):
            yield ContentLocalizationResponse(
                video_file_data=lipsync_response.video_file_data,
                request_id=request_id,
            )
        elif lipsync_response.HasField("keepalive"):
            logger.debug(f"Forwarding keep-alive response {_keepalive_count}.")
            _keepalive_count += 1
            yield ContentLocalizationResponse(
                keepalive=lipsync_response.keepalive,
                request_id=request_id,
            )
        else:
            logger.debug("Ignoring non-video LipSync responses.")
            logger.debug(f"Lipsync response: {lipsync_response}")
    logger.debug(
        f"LipSync response iterator finished after "
        f"{response_count} responses and "
        f"{_keepalive_count} keep-alive responses"
    )


def _forbidden_translated_audio_message(translated_chunks: int) -> str:
    """Build the error message for translated audio sent without bypass_s2s.

    Shared by the eager per-response check and the end-of-stream
    validation so both paths report the violation identically.

    Args:
        translated_chunks (int): Number of translated-audio chunks routed
            so far.

    Returns:
        str: Human-readable INVALID_ARGUMENT message.

    Examples:
        >>> _forbidden_translated_audio_message(translated_chunks=2)[:8]
        'Received'
    """
    return (
        f"Received {translated_chunks} translated_audio_data "
        "chunk(s) but bypass_s2s is False. Set bypass_s2s=True in "
        "ContentLocalizationConfig to use pre-translated audio."
    )


def _raise_if_forbidden_translated_audio(
    deserializer: ContentLocalizationDeserializer,
    bypass_s2s: bool,
) -> None:
    """Raise when translated audio has arrived although S2S is active.

    Reads the translated-audio buffer's ``put_count`` (lock-guarded and
    incremented at distribute time), so the check is safe against the
    deserializer's background thread and needs no extra error plumbing.
    A chunk arriving between two checks is caught before the next
    response is yielded, and ultimately by the end-of-stream validation.

    Args:
        deserializer (ContentLocalizationDeserializer): Deserializer whose
            routing counters record translated-audio arrivals.
        bypass_s2s (bool): Effective bypass flag for this request.

    Returns:
        None.

    Raises:
        PipelineInputError: When ``bypass_s2s`` is False and translated
            audio has been received.

    Examples:
        >>> _raise_if_forbidden_translated_audio(  # doctest: +SKIP
        ...     deserializer=des,
        ...     bypass_s2s=False,
        ... )
    """
    if bypass_s2s:
        return
    translated_chunks = deserializer.translated_audio_buffer.put_count
    if translated_chunks > 0:
        raise PipelineInputError(
            _forbidden_translated_audio_message(translated_chunks=translated_chunks)
        )


def _validate_translated_audio_usage(
    deserializer: ContentLocalizationDeserializer,
    bypass_s2s: bool,
) -> None:
    """Validate the pairing of ``bypass_s2s`` and ``translated_audio_data``.

    ``translated_audio_data`` is consumed only in bypass-S2S mode, and
    bypass-S2S mode requires at least one translated-audio chunk. The
    deserializer is joined first so the routing counters reflect the fully
    consumed inbound stream.

    Args:
        deserializer (ContentLocalizationDeserializer): Deserializer whose
            routing counters reflect the completed inbound stream.
        bypass_s2s (bool): Effective bypass flag for this request.

    Returns:
        None.

    Raises:
        PipelineInputError: When the pairing is violated; the servicer
            reports it as ``INVALID_ARGUMENT``.

    Examples:
        >>> _validate_translated_audio_usage(  # doctest: +SKIP
        ...     deserializer=des,
        ...     bypass_s2s=True,
        ... )
    """
    deserializer.join(timeout=CONTROLLER_CLEANUP_TIMEOUT)
    # The buffer's put-count is used rather than its occupancy: consumers (and
    # the unused-input drain) empty the queues during streaming, so qsize()
    # is 0 at this point whether or not translated audio ever arrived. The
    # put-count records arrivals independently of consumption.
    translated_chunks = deserializer.translated_audio_buffer.put_count
    if bypass_s2s and translated_chunks == 0:
        raise PipelineInputError(
            "bypass_s2s=True requires translated_audio_data in the "
            "request stream, but none was received."
        )
    if not bypass_s2s and translated_chunks > 0:
        raise PipelineInputError(
            _forbidden_translated_audio_message(translated_chunks=translated_chunks)
        )


def _warn_on_unconsumed_items(deserializer: ContentLocalizationDeserializer) -> None:
    """Log a warning for every buffer with routed but unconsumed items.

    A fully consumed buffer sees one ``get()`` per queue per item, so
    ``get_count == put_count * num_queues`` once the pipeline threads have
    finished. A mismatch means items were routed into the buffer and then
    discarded with the request — a witness in the logs for consumers that
    ended early or inputs no stage read.

    Args:
        deserializer (ContentLocalizationDeserializer): Deserializer whose
            threads have been joined, so the counters are final.

    Returns:
        None.

    Examples:
        >>> _warn_on_unconsumed_items(deserializer=des)  # doctest: +SKIP
    """
    for name, buffer in deserializer.named_buffers().items():
        expected_gets = buffer.put_count * buffer.num_queues
        if buffer.get_count != expected_gets:
            logger.warning(
                f"Buffer {name} ends the request partially consumed: "
                f"{buffer.put_count} item(s) put, {buffer.get_count} of "
                f"{expected_gets} expected get(s) observed"
            )


def _cleanup_threads(
    deserializer: ContentLocalizationDeserializer,
    threads: list[threading.Thread],
) -> None:
    """Stop deserializer and join all pipeline threads.

    After the joins, when the buffer counters are final, buffers that end
    the request partially consumed are reported via
    :func:`_warn_on_unconsumed_items`.

    Args:
        deserializer (ContentLocalizationDeserializer): Active
            deserializer to stop.
        threads (list[threading.Thread]): Threads to join.

    Examples:
        >>> _cleanup_threads(  # doctest: +SKIP
        ...     deserializer=des,
        ...     threads=[t1, t2],
        ... )
    """
    logger.debug("Cleanup: stopping deserializer and joining threads")
    try:
        deserializer.stop(timeout=CONTROLLER_CLEANUP_TIMEOUT)
    except Exception as e:
        logger.error(f"Error stopping deserializer: {e}")

    for thr in threads:
        try:
            thr.join(timeout=CONTROLLER_CLEANUP_TIMEOUT)
            if thr.is_alive():
                logger.warning(f"Thread {thr.name} did not stop within timeout")
        except Exception as e:
            logger.error(f"Error joining thread {thr.name}: {e}")

    _warn_on_unconsumed_items(deserializer=deserializer)
    logger.debug("Cleanup completed")
