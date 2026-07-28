# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-agnostic file source simulator for streaming raw bytes."""

import os
from collections.abc import Generator

from common.source_sink.base import BaseFileSimulator

# 64 KB — reasonable trade-off between request count and per-message
# overhead for gRPC streaming
_DEFAULT_CHUNK_SIZE_BYTES = 64 * 1024


class FileSourceSimulator(BaseFileSimulator):
    """Concrete ``BaseFileSimulator`` that streams any file as raw bytes.

    Unlike ``AudioSourceSimulator`` (WAV-only), this class treats the
    file as an opaque byte stream and chunks it by a fixed byte size.
    Useful for MP3 files, or any file that should be sent without
    format-specific parsing.

    The ``read`` interface is compatible with ``AudioSourceSimulator``
    so both can be used interchangeably with
    ``simulated_audio_chunk_generator_raw``.

    Examples:
        >>> src = FileSourceSimulator(file_path="translated.mp3")  # doctest: +SKIP
        >>> for chunk in src.read():
        ...     process(chunk)
    """

    def __init__(
        self,
        file_path: str | os.PathLike[str],
        chunk_size_bytes: int = _DEFAULT_CHUNK_SIZE_BYTES,
    ) -> None:
        """Initialise the file source.

        Args:
            file_path: Path to the file (any format).
            chunk_size_bytes: Number of bytes per read chunk.
                Defaults to 64 KB. Must be positive.

        Raises:
            ValueError: If ``chunk_size_bytes`` is not positive.
            FileNotFoundError: If ``file_path`` is not an existing
                regular file.
        """
        super().__init__(file_path=file_path)
        if chunk_size_bytes <= 0:
            raise ValueError(f"chunk_size_bytes must be > 0, got {chunk_size_bytes}")
        self._file_opened = open(file_path, "rb")  # noqa: SIM115
        self._chunk_size = chunk_size_bytes

    def validate_file_path(self, value: str | os.PathLike[str]) -> None:
        """Validate the path points to an existing regular file.

        Args:
            value: Path to the file.

        Raises:
            FileNotFoundError: If the path does not exist or is not a
                regular file (e.g. a directory).
        """
        if not os.path.isfile(value):
            raise FileNotFoundError(f"File not found or not a regular file: {value}")

    def read(
        self,
        chunk_duration_secs: float = 1.0,  # noqa: ARG002
    ) -> Generator[bytes, None, None]:
        """Yield fixed-size byte chunks from the file.

        *chunk_duration_secs* is accepted for API compatibility with
        ``AudioSourceSimulator`` but ignored — byte-level chunking
        via *chunk_size_bytes* is used instead.

        Args:
            chunk_duration_secs: Ignored (kept for interface compat).

        Yields:
            bytes: Raw data chunks from the file.
        """
        while True:
            data = self._file_opened.read(self._chunk_size)
            if not data:
                break
            yield data
