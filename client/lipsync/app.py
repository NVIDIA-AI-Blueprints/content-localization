# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the NVIDIA AI4M Lipsync service.

Thin entry-point that orchestrates configuration, channel setup, and the
request/response pipeline.  Business logic lives in the sibling modules
``encoding``, ``request_generators``, and ``response_writers``.
"""

import argparse
import sys
import threading

from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.common.audio import AUDIO_CODEC_CONFIGS
from client.common.timing import StageTimer
from client.common.worker import ClientWorker
from client.lipsync.args import argsfactory
from client.lipsync.args import lipsync_config_from_args
from client.lipsync.config import LipSyncConfig
from client.lipsync.request_generators import generate_request_for_inference
from client.lipsync.response_writers import process_response_iter
from common.base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.context import LocalContext
from common.health import check_service_health
from common.nims import LipsyncClient
from common.nims import LipsyncHandle
from common.tls import create_channel_credentials


def _build_config_proto(
    args: argparse.Namespace,
    lipsync_config: LipSyncConfig,
) -> lipsync_pb2.LipsyncConfig:
    """Build the LipSync protobuf config from CLI args and validated config.

    Uses the shared ``lipsync_config_from_args`` builder (the same one
    used by the direct and controller clients), then applies the values
    resolved by validation: the input audio codec derived from the file
    and the speaker-info flag derived from the file presence.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        lipsync_config (LipSyncConfig): Validated client configuration.

    Returns:
        lipsync_pb2.LipsyncConfig: Protobuf configuration message.

    Examples:
        >>> proto = _build_config_proto(
        ...     args=args,
        ...     lipsync_config=cfg,
        ... )  # doctest: +SKIP
    """
    config_proto = lipsync_config_from_args(args)
    config_proto.input_audio_codec = AUDIO_CODEC_CONFIGS[lipsync_config.audio_codec]
    config_proto.is_speaker_info_provided = bool(lipsync_config.is_speaker_info_provided)
    return config_proto


def main() -> int:
    """Main entry point for the LipSync client.

    Handles:
    1. Argument parsing
    2. Configuration validation
    3. Service health check
    4. Channel setup (secure/insecure)
    5. Request processing

    Returns:
        int: ``0`` on success, ``1`` on failure.
    """
    parser = argsfactory()
    args = parser.parse_args()
    lipsync_config = LipSyncConfig.from_args(args)

    try:
        lipsync_config.validate_lipsync_config()
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    logger.info(f"LipSync config: {lipsync_config}")
    config_proto = _build_config_proto(args=args, lipsync_config=lipsync_config)

    timer = StageTimer()

    try:
        channel_credentials = None
        if args.ssl_mode != "DISABLED":
            channel_credentials = create_channel_credentials(args)

        with timer.stage("health_check"):
            check_service_health(
                server=args.lipsync_server,
                channel_credentials=channel_credentials,
            )
            logger.info("LipSync service is healthy")

        with timer.stage("setup"):
            host, port = args.lipsync_server.split(":", 1)
            handle = LipsyncHandle(
                host=host,
                port=int(port),
                channel_credentials=channel_credentials,
            )
            client = LipsyncClient(handle=handle)
            output_buffer: Buffer[lipsync_pb2.LipsyncResponse] = Buffer()

        def run_client() -> None:
            logger.debug(f"LipSync client running on thread: {threading.current_thread().name}")
            context = LocalContext()
            client(
                request_iterator=generate_request_for_inference(
                    lipsync_config=lipsync_config,
                    config_proto=config_proto,
                ),
                output_buffer=output_buffer,
                context=context,
                request_id="lipsync-client",
            )

        with timer.stage("inference"):
            client_worker = ClientWorker(target=run_client, name="lipsync-client")
            client_worker.start()
            response_iter = RequestIteratorFromBuffer(output_buffer, poll_timeout=0.1)
            process_response_iter(response_iter=response_iter, lipsync_config=lipsync_config)
            # Surface any worker failure into the surrounding handler so the CLI
            # returns a non-zero exit code.
            client_worker.join_and_raise()

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
    finally:
        with timer.stage("cleanup"):
            # Explicitly close the gRPC channel to the LipSync NIM so the
            # connection terminates immediately rather than at process exit / GC.
            if "handle" in locals():
                handle.close()

    timer.log_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
