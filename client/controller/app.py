# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller client implementation.

Streams audio, video, and optional diarization data to the Controller
gRPC service and writes the lip-synced output video.

Supports a **bypass-S2S mode** via ``--translated-audio``: when a
pre-translated audio file is provided, S2S is skipped and the
translated audio is sent directly to LipSync alongside the original
audio (still needed for ASD) and video.
"""

import grpc
from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerStub

from client.common.audio import create_audio_source
from client.common.audio import detect_audio_codec
from client.common.diarization import load_diarization_info
from client.common.timing import StageTimer
from client.controller.args import argsfactory
from client.controller.config import ControllerConfig
from client.controller.request_generators import create_controller_request_generator
from client.controller.response_writers import write_output_from_response
from common.base_utils import logger
from common.health import check_service_health
from common.source_sink.base import BaseFileSimulator
from common.source_sink.grpc.video import VideoSourceSimulator
from common.tls import create_channel_credentials


def _close_sources(sources: list[BaseFileSimulator | None]) -> None:
    """Close every media source that is still open.

    Args:
        sources (list[BaseFileSimulator | None]): Source simulators to
            close. ``None`` entries (optional sources that were never
            created) are skipped.

    Returns:
        None

    Examples:
        >>> _close_sources(sources=[audio_source, None])  # doctest: +SKIP
    """
    for source in sources:
        if source is not None and source.is_open():
            source.close()


def main() -> None:
    """Main function for the controller client."""
    args = argsfactory().parse_args()

    # Build and validate configuration
    cfg = ControllerConfig.from_args(args)
    cfg.validate_io()
    logger.info(f"Controller config: {cfg}")

    timer = StageTimer()

    # Optional TLS/mTLS for the controller channel, mirroring the
    # standalone NIM clients.
    channel_credentials = None
    if args.ssl_mode != "DISABLED":
        channel_credentials = create_channel_credentials(args)

    with timer.stage("health_check"):
        check_service_health(
            server=cfg.controller_server,
            channel_credentials=channel_credentials,
        )
        logger.info("Controller service is healthy")

    with timer.stage("setup"):
        # Audio sources are selected by file content (RIFF sniffing), so MP3
        # data inside a .wav filename still streams as raw bytes.
        input_audio_source = create_audio_source(file_path=cfg.input_audio)
        input_video_source = VideoSourceSimulator(file_path=cfg.input_mp4)

        # Create background audio source only when provided
        bg_audio_source = None
        if cfg.background_audio_input:
            bg_audio_source = create_audio_source(file_path=cfg.background_audio_input)
            logger.info(f"background audio: {cfg.background_audio_input}")

        # Create translated audio source for no-S2S bypass mode
        translated_audio_source = None
        if cfg.translated_audio:
            translated_audio_source = create_audio_source(file_path=cfg.translated_audio)
            logger.info(f"translated audio: {cfg.translated_audio} (S2S bypassed)")

        # Load optional diarization info
        diarization_info = load_diarization_info(
            diarization_file=cfg.diarization_file,
            diarization_format=args.diarization_format,
            combine_chunks_per_speaker=cfg.combine_chunks_per_speaker,
        )
        if diarization_info:
            logger.info(f"diarization: {len(diarization_info.segments)} segments")
        if cfg.bypass_asd:
            logger.info("ASD bypassed — LipSync will use internal face detection")

        # Connect to the controller service
        if channel_credentials is not None:
            channel = grpc.secure_channel(cfg.controller_server, channel_credentials)
        else:
            channel = grpc.insecure_channel(cfg.controller_server)
        stub = ContentLocalizationControllerStub(channel)

        controller_request_generator = create_controller_request_generator(
            audio_source=input_audio_source,
            video_source=input_video_source,
            chunk_size_audio_secs=cfg.chunk_size_audio_secs,
            chunk_size_video_bytes=cfg.chunk_size_video_bytes,
            s2s_config=cfg.s2s_config,
            asd_config=cfg.asd_config,
            lipsync_config=cfg.lipsync_config,
            diarization_info=diarization_info,
            background_audio_source=bg_audio_source,
            translated_audio_source=translated_audio_source,
            bypass_asd=cfg.bypass_asd,
            diarization_rows_per_chunk=cfg.diarization_rows_per_chunk,
            input_audio_codec=detect_audio_codec(file_path=cfg.input_audio),
            request_id=cfg.request_id,
        )

    with timer.stage("inference"):
        controller_response_iter = stub.StreamContentLocalization(controller_request_generator)
        try:
            write_output_from_response(
                response_iter=controller_response_iter,
                output_mp4_path=cfg.output_mp4,
                chunk_size_video_bytes=cfg.chunk_size_video_bytes,
            )
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            logger.error(
                "This might be due to a timeout or connection issue. Try running the client again."
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    with timer.stage("cleanup"):
        _close_sources(
            sources=[
                input_audio_source,
                input_video_source,
                bg_audio_source,
                translated_audio_source,
            ]
        )
        channel.close()

    timer.log_summary()


if __name__ == "__main__":
    main()
