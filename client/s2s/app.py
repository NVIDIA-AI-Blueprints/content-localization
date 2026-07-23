# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main S2S client implementation."""

import threading
from collections.abc import Iterator

from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from client.common.audio import create_audio_source
from client.common.timing import StageTimer
from client.common.worker import ClientWorker
from client.s2s.args import argsfactory
from client.s2s.args import s2s_config_from_args
from client.s2s.config import S2SConfig
from client.s2s.latency_analysis import calculate_output_stream_latencies
from client.s2s.latency_analysis import calculate_per_chunk_latencies
from client.s2s.latency_analysis import plot_latency
from client.s2s.latency_analysis import write_latency_json
from client.s2s.response_writers import write_outputs_from_response
from common.base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.context import LocalContext
from common.health import check_service_health
from common.nims import SpeechToSpeechClient
from common.nims import SpeechToSpeechHandle
from common.source_sink.grpc.audio import AudioSinkSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator
from common.source_sink.grpc.audio import simulated_audio_chunk_generator


def main() -> None:
    """Main function for the S2S client.

    Checks service health, streams audio to S2S, writes the translated
    audio file, reports latency, and optionally writes a machine-readable
    latency summary for the performance aggregator.
    """
    args = argsfactory().parse_args()

    # Build and validate configuration
    s2s_cfg = S2SConfig.from_args(args)
    s2s_cfg.validate_s2s_config()
    logger.info(f"S2S config: {s2s_cfg}")

    timer = StageTimer()

    with timer.stage("health_check"):
        check_service_health(server=args.s2s_server)
        logger.info("S2S service is healthy")

    with timer.stage("setup"):
        # The source simulator is selected by file content (RIFF sniffing),
        # so MP3 data inside a .wav filename still streams as raw bytes.
        input_file_generator = create_audio_source(file_path=args.input_audio)
        is_wav_input = isinstance(input_file_generator, AudioSourceSimulator)
        if is_wav_input:
            sink_kwargs = {
                "sample_width": input_file_generator.sample_width,
                "n_channels": input_file_generator.n_channels,
                "n_frames": input_file_generator.n_frames,
            }
        else:
            # WAV-specific params are not meaningful for compressed input; defaults suffice
            # since AudioSinkSimulator either writes raw bytes (mp3 output) or detects
            # RIFF and switches to passthrough (wav output from a backend that returns WAV).
            sink_kwargs = {"sample_width": 2, "n_channels": 1, "n_frames": 0}

        output_file_generator = AudioSinkSimulator(
            frame_rate=16000,
            file_path=args.output_audio,
            chunk_duration_secs=args.chunk_size_audio_secs,
            audio_format=args.output_audio.split(".")[-1],
            **sink_kwargs,
        )
        host, port = args.s2s_server.split(":", 1)
        handle = SpeechToSpeechHandle(host=host, port=int(port))
        client = SpeechToSpeechClient(handle=handle)
        config = s2s_config_from_args(args)

    def s2s_request_generator() -> Iterator[SpeechToSpeechRequest]:
        yield SpeechToSpeechRequest(config=config)
        if is_wav_input:
            yield from simulated_audio_chunk_generator(
                simulator=input_file_generator, chunk_size_secs=args.chunk_size_audio_secs
            )
        else:
            for chunk in input_file_generator.read(chunk_duration_secs=args.chunk_size_audio_secs):
                yield SpeechToSpeechRequest(audio_data=chunk, audio_format="mp3")

    output_buffer: Buffer[SpeechToSpeechResponse] = Buffer()
    context = LocalContext()

    def run_client() -> None:
        logger.debug(f"S2S client running on thread: {threading.current_thread().name}")
        client(
            request_iterator=s2s_request_generator(),
            output_buffer=output_buffer,
            context=context,
            request_id="s2s-client",
        )

    with timer.stage("inference"):
        client_worker = ClientWorker(target=run_client, name="s2s-client")
        client_worker.start()
        response_iter = RequestIteratorFromBuffer(output_buffer, poll_timeout=0.1)
        write_outputs_from_response(
            response_iter=response_iter,
            output_file_generator=output_file_generator,
        )
        # Surface any worker failure on the main thread so the CLI exits non-zero.
        client_worker.join_and_raise()

    # Explicitly close the gRPC channel to the S2S service so the connection
    # terminates immediately rather than at process exit / GC.
    handle.close()

    with timer.stage("analysis"):
        latencies = calculate_per_chunk_latencies(
            input_ledger=input_file_generator.ledger, output_ledger=output_file_generator.ledger
        )
        output_stream_latencies = calculate_output_stream_latencies(
            input_ledger=input_file_generator.ledger, output_ledger=output_file_generator.ledger
        )
        if args.latency_plot:
            plot_latency(
                output_stream_latencies=output_stream_latencies,
                per_chunk_latencies=latencies,
                chunk_size_secs=args.chunk_size_audio_secs,
                output_path=args.latency_plot,
            )
        # An empty latency series is not "real-time" — guard against all() over an
        # empty iterable returning True (which would mislabel a run that produced
        # no output chunks).
        is_realtime = bool(output_stream_latencies) and all(
            latency < args.chunk_size_audio_secs for latency in output_stream_latencies
        )
        logger.info(f"realtime output stream: {is_realtime}")

    timer.log_summary()

    # Persist a machine-readable summary so the perf aggregator can ingest
    # S2S numbers without scraping logs. This is a side artifact: a write
    # failure must not mark an otherwise-successful run as failed.
    if args.latency_json:
        if is_wav_input:
            duration_secs = input_file_generator.n_frames / input_file_generator.frame_rate
        else:
            duration_secs = 0.0  # not derivable without decoding compressed input
        wall_time_secs = timer.as_dict().get("inference", 0.0)
        try:
            write_latency_json(
                per_chunk_latencies=latencies,
                output_stream_latencies=output_stream_latencies,
                chunk_size_secs=args.chunk_size_audio_secs,
                is_realtime=is_realtime,
                output_path=args.latency_json,
                asset=args.input_audio,
                duration_secs=duration_secs,
                wall_time_secs=wall_time_secs,
            )
            logger.info(f"S2S latency summary written to: {args.latency_json}")
        except OSError as exc:
            logger.error(f"Failed to write latency summary to {args.latency_json}: {exc}")


if __name__ == "__main__":
    main()
