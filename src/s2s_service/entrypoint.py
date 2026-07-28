#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech-to-Speech (S2S) gRPC entrypoint."""

import argparse

from common.base_utils import logger
from s2s_service.camb_utils.dubbing import CambDubbingService
from s2s_service.el_utils.dubbing import ELDubbingService


def main() -> None:
    """Main function to run the S2S service.

    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Creates and initializes the S2S service
    4. Starts the gRPC server
    5. Handles the service lifecycle
    """
    parser = argparse.ArgumentParser(
        description=(
            "Speech-to-Speech (S2S) gRPC entrypoint supporting ElevenLabs and CambAI backends."
        )
    )
    subparsers = parser.add_subparsers(dest="service", required=True, help="S2S backend to use")

    # EL Dubbing subcommand
    el_dubbing_parser = subparsers.add_parser("el_dubbing", help="Run with ElevenLabs backend")
    ELDubbingService.argsfactory(parser=el_dubbing_parser)

    # CambAI Dubbing subcommand
    camb_dubbing_parser = subparsers.add_parser(
        "camb_dubbing", help="Run with CambAI dubbing backend"
    )
    CambDubbingService.argsfactory(parser=camb_dubbing_parser)

    args = parser.parse_args()

    logger.debug(f"args: {args}")

    if args.service == "el_dubbing":
        service = ELDubbingService(
            sample_rate_hz=args.sample_rate_hz,
            default_source_language=args.default_source_language,
            default_target_language=args.default_target_language,
            message_size=args.message_size,
            audio_format=args.audio_format,
        )
    elif args.service == "camb_dubbing":
        service = CambDubbingService(
            sample_rate_hz=args.sample_rate_hz,
            default_source_language=args.default_source_language,
            default_target_language=args.default_target_language,
            message_size=args.message_size,
            audio_format=args.audio_format,
        )
    else:
        parser.error(f"Unknown service: {args.service}")

    # Forward the server-side SSL surface the vendored
    # GRPCServiceBase.argsfactory already exposes (--use-ssl,
    # --ssl_server_key_path, --ssl_server_cert_path,
    # --ssl_root_cert_path), matching the controller entrypoint.
    # Plaintext remains the default.
    service.serve(
        service_uri=args.service_uri,
        max_concurrency=args.max_concurrency,
        use_ssl=args.use_ssl,
        ssl_server_key_path=args.ssl_server_key_path,
        ssl_server_cert_path=args.ssl_server_cert_path,
        ssl_root_cert_path=args.ssl_root_cert_path,
        concurrency_mode=args.concurrency_mode,
        threads_per_process=args.threads_per_process,
    )


if __name__ == "__main__":
    main()
