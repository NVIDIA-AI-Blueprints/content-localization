# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for video simulator classes."""

import os
import tempfile

import pytest

from common.source_sink.grpc.video import VideoSinkSimulator
from common.source_sink.grpc.video import VideoSourceSimulator

pytestmark = pytest.mark.unit


class TestVideoSourceSimulator:
    """Test cases for VideoSourceSimulator class."""

    def test_init_with_valid_file(self):
        """Test VideoSourceSimulator initialization with valid file."""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"fake video data")
            temp_path = temp_file.name

        try:
            simulator = VideoSourceSimulator(temp_path)
            assert simulator.file_path == temp_path
            assert simulator.is_open()
            assert simulator.file_size > 0
            assert simulator._chunk_count == 0
        finally:
            os.unlink(temp_path)

    def test_init_with_invalid_file(self):
        """Test VideoSourceSimulator initialization with invalid file."""
        with pytest.raises(FileNotFoundError):
            VideoSourceSimulator("nonexistent_file.mp4")

    def test_frames_generator(self):
        """Test frames generator yields video data."""
        # Create a temporary video file with some data
        test_data = b"fake video frame data" * 100
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(test_data)
            temp_path = temp_file.name

        try:
            simulator = VideoSourceSimulator(temp_path)
            frames = list(simulator.frames(chunk_size=64))

            # Should have yielded some frames
            assert len(frames) > 0

            # All frames should be bytes
            for frame in frames:
                assert isinstance(frame, bytes)
                assert len(frame) > 0

            # Chunk count should be incremented
            assert simulator._chunk_count > 0

        finally:
            os.unlink(temp_path)

    def test_read_generator(self):
        """Test read generator yields video data."""
        # Create a temporary video file with some data
        test_data = b"fake video frame data" * 100
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(test_data)
            temp_path = temp_file.name

        try:
            simulator = VideoSourceSimulator(temp_path)
            frames = list(simulator.read(chunk_size=64))

            # Should have yielded some frames
            assert len(frames) > 0

            # All frames should be bytes
            for frame in frames:
                assert isinstance(frame, bytes)
                assert len(frame) > 0

            # Chunk count should be incremented
            assert simulator._chunk_count > 0

        finally:
            os.unlink(temp_path)

    def test_ledger_tracking(self):
        """Test that ledger tracks frame timestamps."""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"fake video data")
            temp_path = temp_file.name

        try:
            simulator = VideoSourceSimulator(temp_path)
            frames = list(simulator.frames(chunk_size=64))

            # Ledger should have entries for each frame
            assert len(simulator.ledger) == len(frames)

            # Each ledger entry should be a timestamp
            for timestamp in simulator.ledger.values():
                assert isinstance(timestamp, float)

        finally:
            os.unlink(temp_path)

    def test_chunk_count_increment(self):
        """Test that chunk count increments correctly."""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"fake video data" * 10)
            temp_path = temp_file.name

        try:
            simulator = VideoSourceSimulator(temp_path)
            initial_count = simulator._chunk_count

            # Read some frames
            frames = list(simulator.frames(chunk_size=64))

            # Chunk count should have increased
            assert simulator._chunk_count > initial_count
            assert simulator._chunk_count == len(frames)

        finally:
            os.unlink(temp_path)


class TestVideoSinkSimulator:
    """Test cases for VideoSinkSimulator class."""

    def test_init_with_valid_directory(self):
        """Test VideoSinkSimulator initialization with valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            simulator = VideoSinkSimulator(output_path)

            assert simulator.file_path == output_path
            assert simulator.is_open()
            assert simulator._chunk_count == 0
            assert simulator.chunk_size == 64 * 1024

    def test_init_with_invalid_directory(self):
        """Test VideoSinkSimulator initialization with invalid directory."""
        with pytest.raises(FileNotFoundError):
            VideoSinkSimulator("/nonexistent/directory/output.mp4")

    def test_write_video_data(self):
        """Test writing video data to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            # Use small chunk size so test data fills the buffer
            simulator = VideoSinkSimulator(output_path, chunk_size=21)

            # Write test data that exactly fills the chunk size
            test_data = b"fake video frame data"  # 21 bytes
            assert len(test_data) == 21  # Verify the actual length
            simulator.write(test_data)

            # Check that data was written
            assert simulator._chunk_count > 0
            assert os.path.exists(output_path)

            # Close the simulator to ensure data is flushed to disk
            simulator.close()

            # Check file content - the exact 21 bytes should be written
            with open(output_path, "rb") as f:
                written_data = f.read()
                assert len(written_data) == 21
                assert written_data == test_data

    def test_write_with_callback(self):
        """Test writing video data with callback function."""
        callback_called = False
        callback_data = None

        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            # Use small chunk size so test data fills the buffer
            simulator = VideoSinkSimulator(output_path, chunk_size=20)

            # Write test data that exactly fills the chunk size
            test_data = b"fake video frame data"  # 20 bytes
            simulator.write(test_data, process_video_callback=test_callback)

            # Check that callback was called
            assert callback_called
            assert callback_data is not None
            assert isinstance(callback_data, bytes)

    def test_flush_remaining_data(self):
        """Test flushing remaining buffered data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            simulator = VideoSinkSimulator(output_path, chunk_size=100)

            # Write data smaller than chunk size
            test_data = b"small data"
            simulator.write(test_data)

            # Should be buffered but not written yet
            assert simulator._chunk_count == 0
            assert len(simulator._buffer) > 0

            # Flush the data
            simulator.flush()

            # Should be written now
            assert simulator._chunk_count > 0
            assert len(simulator._buffer) == 0

            # Close the simulator to ensure data is flushed to disk
            simulator.close()

            # Check file content
            with open(output_path, "rb") as f:
                written_data = f.read()
                assert test_data in written_data

    def test_ledger_tracking(self):
        """Test that ledger tracks write timestamps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            # Use small chunk size so test data fills the buffer
            simulator = VideoSinkSimulator(output_path, chunk_size=20)

            # Write test data that exactly fills the chunk size
            test_data = b"fake video frame data"  # 20 bytes
            simulator.write(test_data)

            # Ledger should have entries for each write
            assert len(simulator.ledger) > 0

            # Each ledger entry should be a timestamp
            for timestamp in simulator.ledger.values():
                assert isinstance(timestamp, float)

    def test_chunk_count_increment(self):
        """Test that chunk count increments correctly during writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            # Use small chunk size so test data fills the buffer
            simulator = VideoSinkSimulator(output_path, chunk_size=20)

            initial_count = simulator._chunk_count

            # Write test data that exactly fills the chunk size
            test_data = b"fake video frame data"  # 20 bytes
            simulator.write(test_data)

            # Chunk count should have increased
            assert simulator._chunk_count > initial_count

    def test_buffer_handling(self):
        """Test buffer handling with different chunk sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            simulator = VideoSinkSimulator(output_path, chunk_size=100)

            # Write data smaller than chunk size
            small_data = b"small"
            simulator.write(small_data)

            # Should be buffered
            assert len(simulator._buffer) > 0
            assert simulator._chunk_count == 0

            # Write more data to fill buffer
            more_data = b"more data to fill the buffer" * 10  # Make it large enough
            simulator.write(more_data)

            # Should have written some chunks
            assert simulator._chunk_count > 0


class TestBaseFileSimulatorContextManager:
    """Test cases for the context-manager protocol on file simulators."""

    def test_source_context_manager_closes_file(self):
        """Exiting the context closes the underlying file handle."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"fake video data")
            temp_path = temp_file.name

        try:
            with VideoSourceSimulator(temp_path) as simulator:
                assert simulator.is_open()
                frames = list(simulator.frames(chunk_size=64))
                assert len(frames) > 0
            assert not simulator.is_open()
        finally:
            os.unlink(temp_path)

    def test_sink_context_manager_closes_file(self):
        """Exiting the context closes the sink and its written data persists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            test_data = b"fake video frame data"  # 21 bytes

            with VideoSinkSimulator(output_path, chunk_size=21) as simulator:
                assert simulator.is_open()
                simulator.write(test_data)
            assert not simulator.is_open()

            with open(output_path, "rb") as f:
                assert f.read() == test_data

    def test_context_manager_closes_file_on_exception(self):
        """The file handle is closed even when the context body raises."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")

            with (
                pytest.raises(RuntimeError, match="expected failure"),
                VideoSinkSimulator(output_path) as simulator,
            ):
                raise RuntimeError("expected failure")
            assert not simulator.is_open()

    def test_close_is_idempotent(self):
        """Calling close() after the context has exited is a no-op."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")

            with VideoSinkSimulator(output_path) as simulator:
                pass
            simulator.close()
            assert not simulator.is_open()

    def test_sink_context_manager_flushes_partial_chunk(self):
        """A trailing partial chunk is persisted when the context exits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")

            with VideoSinkSimulator(output_path, chunk_size=1024) as simulator:
                simulator.write(b"partial")  # 7 bytes < chunk_size, stays buffered

            with open(output_path, "rb") as f:
                assert f.read() == b"partial"

    def test_close_without_context_flushes_partial_chunk(self):
        """A plain close() persists a buffered partial chunk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")

            simulator = VideoSinkSimulator(output_path, chunk_size=1024)
            simulator.write(b"partial")
            simulator.close()

            with open(output_path, "rb") as f:
                assert f.read() == b"partial"

    def test_explicit_flush_then_close_writes_once(self):
        """flush() followed by close() does not duplicate the buffered data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")

            simulator = VideoSinkSimulator(output_path, chunk_size=1024)
            simulator.write(b"partial")
            simulator.flush()
            simulator.close()

            with open(output_path, "rb") as f:
                assert f.read() == b"partial"
