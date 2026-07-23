# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple ASD (Active Speaker Detection) client implementation."""

import threading

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)

from client.asd.args import argsfactory
from client.asd.args import asd_config_from_args
from client.asd.config import ASDConfig
from client.asd.request_generators import asd_request_generator
from client.asd.response_writers import write_asd_outputs_from_response
from client.common.diarization import load_diarization_info
from client.common.timing import StageTimer
from client.common.worker import ClientWorker
from common.base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.context import LocalContext
from common.health import check_service_health
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionHandle
from common.source_sink.file import FileSourceSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator
from common.source_sink.grpc.video import VideoSourceSimulator


def main() -> None:
    """Main function for the ASD client.

    This function:
    1. Checks service health
    2. Streams both video and audio to the ASD service
    3. Writes the speaker detection data to a CSV file
    4. Logs processing statistics
    """
    args = argsfactory().parse_args()

    # Build and validate configuration
    asd_cfg = ASDConfig.from_args(args)
    asd_cfg.validate_asd_config()
    logger.info(f"ASD config: {asd_cfg}")

    timer = StageTimer()

    with timer.stage("health_check"):
        check_service_health(server=args.asd_server)
        logger.info("ASD service is healthy")

    with timer.stage("setup"):
        input_video_source = VideoSourceSimulator(file_path=args.input_mp4)
        if args.asd_input_audio_codec == "MP3":
            input_audio_source = FileSourceSimulator(file_path=args.input_audio)
        else:
            input_audio_source = AudioSourceSimulator(file_path=args.input_audio)

        diarization_info = load_diarization_info(
            diarization_file=args.diarization_file,
            diarization_format=args.diarization_format,
            combine_chunks_per_speaker=not args.diarization_chunked_per_segment,
        )
        if diarization_info:
            logger.info(f"diarization: {len(diarization_info.segments)} segments")

        asd_config = asd_config_from_args(args)
        host, port = args.asd_server.split(":", 1)
        handle = ActiveSpeakerDetectionHandle(host=host, port=int(port))
        client = ActiveSpeakerDetectionClient(handle=handle)
        request_generator = asd_request_generator(
            video_source=input_video_source,
            audio_source=input_audio_source,
            chunk_size_video_bytes=args.chunk_size_video_bytes,
            chunk_size_audio_secs=args.chunk_size_audio_secs,
            asd_config=asd_config,
            diarization_info=diarization_info,
        )

    output_buffer: Buffer[DetectActiveSpeakerResponse] = Buffer()
    context = LocalContext()

    def run_client() -> None:
        logger.debug(f"ASD client running on thread: {threading.current_thread().name}")
        client(
            request_iterator=request_generator,
            output_buffer=output_buffer,
            context=context,
            request_id="asd-client",
        )

    with timer.stage("inference"):
        client_worker = ClientWorker(target=run_client, name="asd-client")
        client_worker.start()
        asd_response_iter = RequestIteratorFromBuffer(output_buffer, poll_timeout=0.1)
        write_asd_outputs_from_response(
            response_iter=asd_response_iter,
            output_csv_path=args.output_speaker_info,
        )
        # Surface any worker failure on the main thread so the CLI exits non-zero.
        client_worker.join_and_raise()

    with timer.stage("cleanup"):
        if input_video_source.is_open():
            input_video_source.close()
        if input_audio_source.is_open():
            input_audio_source.close()
        # Explicitly close the gRPC channel to the ASD NIM so the connection
        # terminates immediately rather than at process exit / GC.
        handle.close()

    timer.log_summary()


if __name__ == "__main__":
    main()
