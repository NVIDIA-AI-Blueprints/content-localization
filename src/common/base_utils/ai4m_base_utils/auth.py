#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication utilities for gRPC services.

This module provides utilities for configuring SSL/TLS authentication for gRPC
servers and clients. It includes functions for setting up SSL credentials,
managing certificates, and handling command-line arguments for SSL configuration.
"""

import argparse
import os
import grpc


from ai4m_base_utils.file_utils import FileUtils
from ai4m_base_utils.error_utils import SSLConfigurationError
from ai4m_base_utils.logger import logger


class Auth:
    """Authentication utilities for gRPC services.

    This class provides methods for configuring SSL/TLS authentication
    for both gRPC servers and clients. It handles certificate management,
    credential creation, and command-line argument parsing for SSL settings.
    """

    @staticmethod
    def validate_server_config(
        ssl_key_path: os.PathLike | None = None,
        ssl_cert_path: os.PathLike | None = None,
        ssl_root_cert_path: os.PathLike | None = None,
        use_ssl: bool = False,
    ) -> None:
        """Validate server configuration parameters.

        Args:
            ssl_key_path (os.PathLike | None): Path to SSL key file
            ssl_cert_path (os.PathLike | None): Path to SSL certificate file
            ssl_root_cert_path (os.PathLike | None): Path to SSL root certificate file
            use_ssl (bool): Whether SSL is enabled

        Raises:
            ServiceConfigurationError: If configuration is invalid
            SSLConfigurationError: If SSL configuration is invalid
        """

        if use_ssl:
            if not ssl_key_path:
                raise SSLConfigurationError("SSL key file must be specified when SSL is enabled")
            if not ssl_cert_path:
                raise SSLConfigurationError(
                    "SSL certificate file must be specified when SSL is enabled"
                )

            # Validate SSL file existence
            for file_path, file_type in [
                (ssl_key_path, "SSL key"),
                (ssl_cert_path, "SSL certificate"),
            ]:
                if file_path and not Auth._is_file_accessible(file_path):
                    raise SSLConfigurationError(
                        f"{file_type} file is not accessible: {file_path}"
                    )

            # Only check root cert if mTLS is enabled
            if ssl_root_cert_path:
                if not Auth._is_file_accessible(ssl_root_cert_path):
                    raise SSLConfigurationError(
                        f"SSL root certificate file is not accessible: {ssl_root_cert_path}"
                    )

    @staticmethod
    def _is_file_accessible(file_path: os.PathLike) -> bool:
        """Check if a file exists, is a regular file, and is readable.

        Args:
            file_path (os.PathLike): Path to the file to check

        Returns:
            bool: True if file exists, is a regular file, and is readable; False otherwise
        """
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

    @staticmethod
    def configure_ssl_credentials(
        ssl_server_key_path: os.PathLike,
        ssl_server_cert_path: os.PathLike,
        ssl_root_cert_path: os.PathLike | None,
        use_ssl: bool,
    ) -> grpc.ServerCredentials:
        """Configure SSL credentials for a gRPC server.

        This method validates the SSL configuration, reads the certificate files,
        and creates the appropriate server credentials for SSL/TLS encryption.

        Args:
            ssl_server_key_path: Path to the server's private key file.
            ssl_server_cert_path: Path to the server's certificate file.
            ssl_root_cert_path: Path to the root certificate file for mTLS, if used.
            use_ssl: Whether to enable SSL encryption.

        Returns:
            Configured SSL server credentials.

        Raises:
            SSLConfigurationError: If SSL configuration is invalid or files are inaccessible.
        """
        try:
            Auth.validate_server_config(
                ssl_key_path=ssl_server_key_path,
                ssl_cert_path=ssl_server_cert_path,
                ssl_root_cert_path=ssl_root_cert_path,
                use_ssl=use_ssl,
            )

            server_key = FileUtils.read_file_bytes(ssl_server_key_path)
            server_cert = FileUtils.read_file_bytes(ssl_server_cert_path)
            root_cert = (
                FileUtils.read_file_bytes(ssl_root_cert_path) if ssl_root_cert_path else None
            )

            key_cert_pair = (server_key, server_cert)
            creds = grpc.ssl_server_credentials(
                [key_cert_pair],
                root_certificates=root_cert,
                require_client_auth=bool(root_cert),
            )
            logger.info("Using SSL Credentials")
            return creds
        except (IOError, FileNotFoundError) as e:
            raise SSLConfigurationError(f"Failed to read SSL files: {str(e)}") from e
        except Exception as e:
            raise SSLConfigurationError(f"Failed to configure SSL: {str(e)}") from e

    @staticmethod
    def get_client_ssl_credentials(
        server_key_path: os.PathLike,
        server_cert_path: os.PathLike,
        root_cert_path: os.PathLike,
    ) -> grpc.ChannelCredentials:
        """Create SSL credentials for a gRPC client.

        This method reads the necessary certificate files and creates SSL
        credentials for client-side authentication.

        Args:
            server_key_path: Path to the client's private key file.
            server_cert_path: Path to the client's certificate file.
            root_cert_path: Path to the root certificate file.

        Returns:
            SSL credentials configured for the client.

        Raises:
            SSLConfigurationError: If SSL files are inaccessible or invalid.
        """
        try:
            private_key = FileUtils.read_file_bytes(server_key_path)
            certificate_chain = FileUtils.read_file_bytes(server_cert_path)
            root_certificates = FileUtils.read_file_bytes(root_cert_path)

            return grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain,
            )
        except (IOError, FileNotFoundError) as e:
            raise SSLConfigurationError(f"Failed to read SSL files: {str(e)}") from e
        except Exception as e:
            raise SSLConfigurationError(f"Failed to configure SSL: {str(e)}") from e

    @staticmethod
    def argsfactory(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add SSL configuration arguments to an argument parser.

        This method adds command-line arguments for SSL configuration to the
        provided argument parser. These arguments control SSL/TLS settings
        for both server and client authentication.

        Args:
            parser: The argument parser to add SSL arguments to.

        Returns:
            The updated argument parser with SSL configuration options.
        """
        parser.add_argument(
            "--use-ssl",
            action="store_true",
            help="Enable SSL encrypted channel",
            default=False,
        )
        parser.add_argument(
            "--ssl_server_key_path",
            type=str,
            help="Path to server private key for SSL encryption",
            default=None,
        )
        parser.add_argument(
            "--ssl_server_cert_path",
            type=str,
            help="Path to server certificate for SSL encryption",
            default=None,
        )
        parser.add_argument(
            "--ssl_root_cert_path",
            type=str,
            help="Path to root certificate for mTLS authentication",
            default=None,
        )

        return parser
