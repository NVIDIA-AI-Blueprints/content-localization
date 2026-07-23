# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ai4m_base_utils.file_utils module."""

import os
from unittest.mock import patch

import pytest

from common.base_utils import FileSizeError
from common.base_utils import FileUtils

pytestmark = pytest.mark.unit


class TestFileUtils:
    """Test file utility functions."""

    def test_write_bytes_to_file(self, temp_dir):
        """Test writing bytes to file."""
        filepath = os.path.join(temp_dir, "test_file.bin")
        test_data = [b"chunk1", b"chunk2", b"chunk3"]
        FileUtils.write_bytes_to_file(filepath, iter(test_data))
        assert os.path.exists(filepath)
        with open(filepath, "rb") as f:
            content = f.read()
        assert content == b"chunk1chunk2chunk3"

    @patch("ai4m_base_utils.file_utils.AI4M_MAX_INPUT_FILE_SIZE_MB", 1)
    def test_write_bytes_size_limit_exceeded(self, temp_dir):
        """Test that FileSizeError is raised when size limit is exceeded."""
        filepath = os.path.join(temp_dir, "large_file.bin")
        large_data = [b"x" * (512 * 1024), b"y" * (512 * 1024), b"z" * 1024]
        with pytest.raises(FileSizeError) as exc_info:
            FileUtils.write_bytes_to_file(filepath, iter(large_data))
        error_msg = str(exc_info.value)
        assert "size limit exceeded" in error_msg.lower() or "exceeds" in error_msg.lower()

    @patch("ai4m_base_utils.file_utils.AI4M_MAX_INPUT_FILE_SIZE_MB", None)
    def test_write_bytes_no_size_limit(self, temp_dir):
        """Test no size limit is enforced when AI4M_MAX_INPUT_FILE_SIZE_MB is None."""
        filepath = os.path.join(temp_dir, "large_file_no_limit.bin")
        large_data = [b"x" * (1024 * 1024), b"y" * (1024 * 1024)]
        FileUtils.write_bytes_to_file(filepath, iter(large_data))
        with open(filepath, "rb") as f:
            content = f.read()
        expected_content = b"x" * (1024 * 1024) + b"y" * (1024 * 1024)
        assert content == expected_content

    def test_read_bytes_in_chunks(self, temp_dir):
        """Test reading file in chunks."""
        filepath = os.path.join(temp_dir, "test_read.bin")
        test_content = b"0123456789" * 100
        with open(filepath, "wb") as f:
            f.write(test_content)
        chunks = list(FileUtils.read_bytes_in_chunks(filepath, chunk_size=100))
        assert len(chunks) == 10
        reconstructed = b"".join(chunks)
        assert reconstructed == test_content

    def test_read_bytes_in_chunks_nonexistent_file(self):
        """Test reading nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            list(FileUtils.read_bytes_in_chunks("/nonexistent/path.bin"))

    def test_read_file_bytes(self, temp_dir):
        """Test reading entire file as bytes."""
        filepath = os.path.join(temp_dir, "test_read_all.bin")
        test_content = b"Hello, World! This is test content."
        with open(filepath, "wb") as f:
            f.write(test_content)
        result = FileUtils.read_file_bytes(filepath)
        assert result == test_content

    def test_read_file_bytes_nonexistent_file(self):
        """Test reading nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            FileUtils.read_file_bytes("/nonexistent/path.bin")

    def test_read_empty_file(self, temp_dir):
        """Test reading empty file."""
        filepath = os.path.join(temp_dir, "empty.bin")
        with open(filepath, "wb") as f:
            pass
        result = FileUtils.read_file_bytes(filepath)
        assert result == b""
        chunks = list(FileUtils.read_bytes_in_chunks(filepath))
        assert chunks == []

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_write_bytes_io_error(self, mock_open):
        """Test that IOError during write is handled properly."""
        with pytest.raises(IOError, match="Failed to write"):
            FileUtils.write_bytes_to_file("/some/path.bin", iter([b"data"]))

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_read_file_bytes_io_error(self, mock_open):
        """Test that IOError during read is handled properly."""
        with pytest.raises(IOError, match="Failed to read"):
            FileUtils.read_file_bytes("/some/path.bin")
