# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller service gRPC entrypoint."""

import argparse
import sys

import grpc

from common.base_utils import logger
from common.handles import GRPCServiceHandle
from common.nims import ActiveSpeakerDetectionHandle
from common.nims import LipsyncHandle
from common.nims import SpeechToSpeechHandle
from common.tls import create_channel_credentials
from controller_service.service import ControllerService


def _nim_channel_credentials(
    args: argparse.Namespace,
    ssl_mode: str,
    service_label: str,
) -> grpc.ChannelCredentials | None:
    """Build channel credentials for one downstream NIM connection.

    Mirrors the client-side ``--ssl-mode`` pattern: ``DISABLED`` keeps the
    hop plaintext, while ``TLS``/``MTLS`` build
    :class:`grpc.ChannelCredentials` from the shared PEM files supplied
    via ``--ssl-key``, ``--ssl-cert``, and ``--ssl-root-cert``. The
    effective mode is logged per hop so mixed topologies (e.g. TLS NIMs
    next to a plaintext S2S) are auditable at startup.

    Args:
        args (argparse.Namespace): Parsed controller arguments containing
            ``ssl_key``, ``ssl_cert``, and ``ssl_root_cert``.
        ssl_mode (str): Effective SSL mode for this hop (``DISABLED``,
            ``TLS``, or ``MTLS``), already resolved from the per-service
            override and the global ``--ssl-mode``.
        service_label (str): Human-readable hop name for the startup log
            (e.g. ``"S2S"``).

    Returns:
        grpc.ChannelCredentials | None: Credentials for a secure channel,
            or ``None`` when the effective mode is ``DISABLED``.

    Examples:
        >>> credentials = _nim_channel_credentials(
        ...     args=args,
        ...     ssl_mode="TLS",
        ...     service_label="LipSync",
        ... )  # doctest: +SKIP
    """
    logger.info(f"{service_label} downstream channel security: {ssl_mode}")
    if ssl_mode == "DISABLED":
        return None
    return create_channel_credentials(args=args, ssl_mode=ssl_mode)


def main() -> None:
    """Main function to run the controller service.

    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Creates the controller service and optional NIM channel credentials
    4. Starts the gRPC server (TLS-terminated when ``--use-ssl`` is set)
    5. Handles the service lifecycle
    """
    parser = ControllerService.argsfactory()
    args = parser.parse_args()

    logger.debug(f"args: {args}")

    try:
        # Parse service URIs
        service_uri = GRPCServiceHandle.from_string(url=args.service_uri)
        lipsync_server = GRPCServiceHandle.from_string(url=args.lipsync_server)

        # S2S is optional — not needed when running bypass-S2S-only
        s2s_server = None
        if args.s2s_server:
            s2s_server = GRPCServiceHandle.from_string(url=args.s2s_server)

        # ASD is optional — when omitted, only bypass_asd requests work
        asd_server = None
        if args.asd_server:
            asd_server = GRPCServiceHandle.from_string(url=args.asd_server)

        logger.info(f"Starting Controller Service on {service_uri.host}:{service_uri.port}")
        if s2s_server is not None:
            logger.info(f"S2S Server: {s2s_server.host}:{s2s_server.port}")
        else:
            logger.info("S2S Server: NOT CONFIGURED (bypass-S2S only)")
        if asd_server is not None:
            logger.info(f"ASD NIM Server: {asd_server.host}:{asd_server.port}")
        else:
            logger.info("ASD NIM Server: NOT CONFIGURED (only bypass_asd=True requests supported)")
        logger.info(f"LipSync NIM Server: {lipsync_server.host}:{lipsync_server.port}")

        # Optional client-side TLS for the downstream NIM channels. Plaintext
        # by default — inside the compose network the hops stay on the
        # trusted bridge; enable for deployments crossing untrusted networks.
        # Each hop resolves its own mode (per-service override falling back
        # to the global --ssl-mode) because the services can differ in TLS
        # capability: hops still serving plaintext (e.g. NIMs without a TLS
        # surface) pair --ssl-mode TLS with --<service>-ssl-mode DISABLED.
        lipsync_credentials = _nim_channel_credentials(
            args=args,
            ssl_mode=args.lipsync_ssl_mode or args.ssl_mode,
            service_label="LipSync",
        )

        # Create service instances
        lipsync_nim = LipsyncHandle(
            host=lipsync_server.host,
            port=lipsync_server.port,
            channel_credentials=lipsync_credentials,
        )
        s2s_nim = None
        if s2s_server is not None:
            s2s_nim = SpeechToSpeechHandle(
                host=s2s_server.host,
                port=s2s_server.port,
                channel_credentials=_nim_channel_credentials(
                    args=args,
                    ssl_mode=args.s2s_ssl_mode or args.ssl_mode,
                    service_label="S2S",
                ),
            )

        # Only create ASD service if not disabled
        asd_nim = None
        if asd_server is not None:
            asd_nim = ActiveSpeakerDetectionHandle(
                host=asd_server.host,
                port=asd_server.port,
                channel_credentials=_nim_channel_credentials(
                    args=args,
                    ssl_mode=args.asd_ssl_mode or args.ssl_mode,
                    service_label="ASD",
                ),
            )

        # Create and start the controller service
        controller_service = ControllerService(
            lipsync_server=lipsync_nim,
            s2s_server=s2s_nim,
            asd_server=asd_nim,
            message_size=args.message_size,
        )

        # Start the service, forwarding the server-side SSL surface that the
        # vendored GRPCServiceBase.argsfactory already exposes
        # (--use-ssl, --ssl_server_key_path, --ssl_server_cert_path,
        # --ssl_root_cert_path).
        controller_service.serve(
            service_uri=args.service_uri,
            max_concurrency=args.max_concurrency,
            use_ssl=args.use_ssl,
            ssl_server_key_path=args.ssl_server_key_path,
            ssl_server_cert_path=args.ssl_server_cert_path,
            ssl_root_cert_path=args.ssl_root_cert_path,
            concurrency_mode=args.concurrency_mode,
            threads_per_process=args.threads_per_process,
        )

    except Exception as e:
        logger.error(f"Failed to start controller service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
