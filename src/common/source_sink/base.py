# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract classes for S2S source simulators."""

import os
from abc import ABC
from abc import abstractmethod
from types import TracebackType
from typing import Self


class BaseFileSimulator(ABC):
    """Base class for an audio source or sink simulator."""

    @property
    def file_path(self) -> os.PathLike:
        """Return the file path backing this simulator.

        Returns:
            os.PathLike: The currently configured file path.
        """
        return self._file_path

    @file_path.setter
    def file_path(self, value: os.PathLike) -> None:
        """Validate and set the file path backing this simulator.

        Args:
            value (os.PathLike): The new file path. Passed through
                :meth:`validate_file_path` before being stored.

        Returns:
            None
        """
        self.validate_file_path(value)
        self._file_path = value

    @abstractmethod
    def validate_file_path(self, value: os.PathLike) -> None:
        """Validate a candidate file path for this simulator.

        Args:
            value (os.PathLike): The file path to validate.

        Returns:
            None

        Raises:
            FileNotFoundError: If the path is not valid for the
                concrete simulator (subclass-defined).
        """

    def __init__(self, file_path: os.PathLike) -> None:
        """Initialize the simulator with a file path.

        Args:
            file_path (os.PathLike): Path to the file this simulator
                reads from or writes to.

        Returns:
            None
        """
        self.file_path = file_path

        # Create a ledger to track the timestamps of the audio samples going, in/out.
        # Maps chunk index -> wall-clock timestamp.
        self.ledger: dict[int, float] = {}

    def __enter__(self) -> Self:
        """Enter the runtime context for this simulator.

        The underlying file handle is opened by the concrete
        simulator's constructor, so entering the context requires no
        additional work.

        Returns:
            Self: This simulator instance.

        Examples:
            >>> with VideoSinkSimulator("out.mp4") as sink:  # doctest: +SKIP
            ...     sink.write(b"data")
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the runtime context and close the file handle.

        Args:
            exc_type (type[BaseException] | None): Exception class raised in
                the context, if any.
            exc_value (BaseException | None): Exception instance raised in
                the context, if any.
            traceback (TracebackType | None): Traceback of the exception, if
                any.

        Returns:
            None

        Examples:
            >>> with VideoSinkSimulator("out.mp4") as sink:  # doctest: +SKIP
            ...     sink.write(b"data")
        """
        self.close()

    def __del__(self) -> None:
        """Close the underlying file handle on garbage collection.

        Returns:
            None
        """
        self.close()

    # Intentionally non-abstract (noqa B027): flushing is an optional hook
    # that only sinks with write-side buffering need to override.
    def flush(self) -> None:  # noqa: B027
        """Write internally buffered data to the underlying file.

        The default implementation is a no-op; sink simulators that
        accumulate write-side buffers override this so partial data is
        persisted before the file handle is closed.

        Returns:
            None

        Examples:
            >>> sink = VideoSinkSimulator("out.mp4")  # doctest: +SKIP
            ... sink.write(b"partial")
            ... sink.flush()
        """

    def close(self) -> None:
        """Flush buffered data and close the underlying file handle if it is open.

        Returns:
            None
        """
        if hasattr(self, "_file_opened") and self._file_opened is not None:
            try:
                self.flush()
            finally:
                # A flush failure must still release the file handle; the
                # flush error propagates to the caller afterwards.
                self._file_opened.close()
                self._file_opened = None

    def is_open(self) -> bool:
        """Report whether the underlying file handle is open.

        Returns:
            bool: ``True`` if a file handle is currently open,
            ``False`` otherwise.
        """
        return hasattr(self, "_file_opened") and self._file_opened is not None
