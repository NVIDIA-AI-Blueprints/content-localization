# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TLS/mTLS channel credential helpers.

Builds :class:`grpc.ChannelCredentials` from PEM-encoded key and
certificate files supplied on the command line, for clients that connect
to TLS-secured service endpoints.
"""

import argparse
import os

import grpc


def read_file_content(file_path: os.PathLike) -> bytes:
    """Read file content as bytes.

    Used to load PEM-encoded keys and certificates for channel
    credentials.

    Args:
        file_path (os.PathLike): Path to the input file.

    Returns:
        bytes: The file contents.

    Examples:
        >>> root_cert = read_file_content("root.pem")  # doctest: +SKIP
    """
    with open(file_path, "rb") as file:
        return file.read()


def create_channel_credentials(
    args: argparse.Namespace,
    ssl_mode: str | None = None,
) -> grpc.ChannelCredentials:
    """Create channel credentials based on SSL mode.

    Args:
        args (argparse.Namespace): Command line arguments containing SSL
            configuration (``ssl_mode``, ``ssl_key``, ``ssl_cert``, and
            ``ssl_root_cert``).
        ssl_mode (str | None): Optional mode override (``TLS`` or
            ``MTLS``). When ``None``, ``args.ssl_mode`` is used. Lets
            callers build per-connection credentials from the shared PEM
            options.

    Returns:
        grpc.ChannelCredentials: Configured channel credentials.

    Raises:
        RuntimeError: If required SSL files are missing.

    Examples:
        >>> credentials = create_channel_credentials(args)  # doctest: +SKIP
        >>> tls_only = create_channel_credentials(
        ...     args=args,
        ...     ssl_mode="TLS",
        ... )  # doctest: +SKIP
    """
    if ssl_mode is None:
        ssl_mode = args.ssl_mode
    channel_credentials = None
    if ssl_mode == "MTLS":
        if not (args.ssl_key and args.ssl_cert and args.ssl_root_cert):
            raise RuntimeError(
                "If --ssl-mode is MTLS, --ssl-key, --ssl-cert and --ssl-root-cert are required."
            )
        private_key = read_file_content(args.ssl_key)
        certificate_chain = read_file_content(args.ssl_cert)
        root_certificates = read_file_content(args.ssl_root_cert)
        channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )
    else:
        if not (args.ssl_root_cert):
            raise RuntimeError("If --ssl-mode is TLS, --ssl-root-cert is required.")
        root_certificates = read_file_content(args.ssl_root_cert)
        channel_credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
    return channel_credentials
