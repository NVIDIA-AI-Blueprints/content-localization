#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for file handling operations."""

import os
from typing import Iterator

from ai4m_base_utils.config import (
    AI4M_DEFAULT_MESSAGE_SIZE,
    AI4M_MAX_INPUT_FILE_SIZE_MB,
)
from ai4m_base_utils.error_utils import FileSizeError


class FileUtils:
    """Class providing file utility methods for handling file operations in gRPC services."""

    @staticmethod
    def write_bytes_to_file(filepath: os.PathLike, data_generator: Iterator[bytes]) -> None:
        """Write binary data from a generator to a file.

        Args:
            filepath (os.PathLike): Path to the file to write.
            data_generator (Iterator): A generator that produces byte stream.

        Raises:
            FileSizeError: If the file size exceeds the maximum allowed size.
            IOError: If there are issues writing to the file.
        """
        total_bytes_received = 0

        try:
            with open(filepath, "wb") as f:
                for data in data_generator:
                    if AI4M_MAX_INPUT_FILE_SIZE_MB is not None:
                        max_size_bytes = AI4M_MAX_INPUT_FILE_SIZE_MB * 1024 * 1024
                        total_bytes_received += len(data)
                        if total_bytes_received > max_size_bytes:
                            raise FileSizeError(
                                f"The input file size exceeds the "
                                f"{AI4M_MAX_INPUT_FILE_SIZE_MB} MB limit. "
                                f"Please try again with a file of smaller size."
                            )
                    f.write(data)
        except IOError as e:
            raise IOError(f"Failed to write to file {filepath}: {str(e)}") from e

    @staticmethod
    def read_bytes_in_chunks(
        filepath: os.PathLike, chunk_size: int = AI4M_DEFAULT_MESSAGE_SIZE
    ) -> Iterator[bytes]:
        """Read a file in chunks and return a byte stream generator.

        Args:
            filepath (os.PathLike): Path to the file to read.
            chunk_size (int): Size of the data chunks to read, defaults to
                AI4M_DEFAULT_MESSAGE_SIZE.

        Returns:
            Iterator[bytes]: A generator that produces a byte stream of the file content.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there are issues reading from the file.
        """
        try:
            with open(filepath, "rb") as fd:
                while True:
                    data = fd.read(chunk_size)
                    if data == b"":
                        break
                    yield data
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {filepath}") from exc
        except IOError as e:
            raise IOError(f"Failed to read from file {filepath}: {str(e)}") from e

    @staticmethod
    def read_file_bytes(file_path: os.PathLike) -> bytes:
        """Read the entire contents of a file as bytes.

        Args:
            file_path (os.PathLike): The path to the file to be read.

        Returns:
            bytes: The complete content of the file in binary format.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            IOError: If an error occurs while opening or reading the file.
        """
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {file_path}") from exc
        except IOError as e:
            raise IOError(f"Failed to read file {file_path}: {str(e)}") from e
