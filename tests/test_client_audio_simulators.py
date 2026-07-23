# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio simulators module."""

import gc
import tempfile
import wave
from pathlib import Path

import pytest

from common.source_sink.base import BaseFileSimulator
from common.source_sink.grpc.audio import AudioSinkSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator

pytestmark = pytest.mark.unit


def _assert_no_unraisable_warning(recwarn) -> None:
    gc.collect()
    unraisable = [
        warning
        for warning in recwarn
        if issubclass(warning.category, pytest.PytestUnraisableExceptionWarning)
    ]
    if unraisable:
        pytest.fail(f"Unexpected unraisable warning(s): {unraisable}")


class TestBaseFileSimulator:
    """Test cases for BaseFileSimulator."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseFileSimulator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFileSimulator("test.wav")

    def test_validate_file_path_abstract_method(self):
        """Test that validate_file_path is abstract."""

        # Create a concrete subclass for testing
        class ConcreteSimulator(BaseFileSimulator):
            def validate_file_path(self, value):
                pass

        simulator = ConcreteSimulator("test.wav")
        assert simulator.file_path == "test.wav"

    def test_ledger_initialization(self):
        """Test that ledger is initialized as empty dict."""

        class ConcreteSimulator(BaseFileSimulator):
            def validate_file_path(self, value):
                pass

        simulator = ConcreteSimulator("test.wav")
        assert simulator.ledger == {}

    def test_close_method(self):
        """Test close method handles missing _file_opened gracefully."""

        class ConcreteSimulator(BaseFileSimulator):
            def validate_file_path(self, value):
                pass

        simulator = ConcreteSimulator("test.wav")
        # Should not raise an exception
        simulator.close()

    def test_is_open_method(self):
        """Test is_open method returns False when _file_opened is not set."""

        class ConcreteSimulator(BaseFileSimulator):
            def validate_file_path(self, value):
                pass

        simulator = ConcreteSimulator("test.wav")
        assert not simulator.is_open()


class TestAudioSourceSimulator:
    """Test cases for AudioSourceSimulator."""

    def test_initialization_with_valid_file(self):
        """Test AudioSourceSimulator initialization with valid WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple WAV file
            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 1600)  # 0.1 seconds of silence

            try:
                simulator = AudioSourceSimulator(tmp_file.name)
                assert simulator.n_channels == 1
                assert simulator.sample_width == 2
                assert simulator.frame_rate == 16000
                assert simulator.n_frames > 0
                assert len(simulator.header) > 0
                assert simulator.ledger == {}
            finally:
                Path(tmp_file.name).unlink()

    def test_initialization_with_invalid_file(self):
        """Test AudioSourceSimulator initialization with invalid file."""
        with pytest.raises(FileNotFoundError):
            AudioSourceSimulator("nonexistent.wav")

    def test_validate_file_path(self):
        """Test validate_file_path method."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple WAV file
            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 1600)  # 0.1 seconds of silence

            try:
                simulator = AudioSourceSimulator(tmp_file.name)
                # Should not raise an exception for existing file
                simulator.validate_file_path(tmp_file.name)

                # Should raise for non-existent file
                with pytest.raises(FileNotFoundError):
                    simulator.validate_file_path("nonexistent.wav")
            finally:
                Path(tmp_file.name).unlink()

    def test_samples_generator(self):
        """Test samples generator yields audio data."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple WAV file
            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 3200)  # 0.2 seconds of silence

            try:
                simulator = AudioSourceSimulator(tmp_file.name)
                samples = list(simulator.samples(chunk_duration_secs=0.1))

                assert len(samples) == 2  # Should yield 2 chunks
                assert len(samples[0]) > 0  # Each chunk should have data
                assert len(simulator.ledger) == 2  # Should record timestamps
            finally:
                Path(tmp_file.name).unlink()

    def test_read_generator(self):
        """Test read generator yields header and samples."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple WAV file
            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 1600)  # 0.1 seconds of silence

            try:
                simulator = AudioSourceSimulator(tmp_file.name)
                chunks = list(simulator.read(chunk_duration_secs=0.1))

                assert len(chunks) == 2  # Header + 1 sample chunk
                assert chunks[0] == simulator.header  # First chunk should be header
                assert len(chunks[1]) > 0  # Second chunk should have audio data
            finally:
                Path(tmp_file.name).unlink()

    def test_close_method(self):
        """Test close method closes the wave file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Create a simple WAV file
            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 1600)

            try:
                simulator = AudioSourceSimulator(tmp_file.name)
                assert simulator.is_open()
                simulator.close()
                assert not simulator.is_open()
            finally:
                Path(tmp_file.name).unlink()


class TestAudioSinkSimulator:
    """Test cases for AudioSinkSimulator."""

    def test_initialization(self):
        """Test AudioSinkSimulator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            simulator = AudioSinkSimulator(
                frame_rate=16000,
                sample_width=2,
                n_channels=1,
                file_path=str(output_path),
                chunk_duration_secs=0.128,
            )

            assert simulator.frame_rate == 16000
            assert simulator.sample_width == 2
            assert simulator.chunk_duration_secs == 0.128
            assert simulator.ledger == {}
            assert simulator.is_open()

    def test_validate_file_path_with_valid_directory(self):
        """Test validate_file_path with valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            simulator = AudioSinkSimulator(file_path=str(output_path))
            # Should not raise an exception
            simulator.validate_file_path(str(output_path))

    def test_validate_file_path_with_invalid_directory(self, recwarn):
        """Test validate_file_path with invalid directory."""
        with pytest.raises(FileNotFoundError):
            AudioSinkSimulator(file_path="nonexistent/output.wav")
        _assert_no_unraisable_warning(recwarn)

    def test_write_method(self):
        """Test write method processes audio data correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            simulator = AudioSinkSimulator(
                frame_rate=16000,
                sample_width=2,
                n_channels=1,
                file_path=str(output_path),
                chunk_duration_secs=0.128,
            )

            # Write audio data
            audio_data = b"\x00\x00" * 2048  # 4096 bytes
            simulator.write(audio_data)

            # Check that ledger was updated
            assert len(simulator.ledger) > 0

            simulator.close()

    def test_write_method_with_callback(self):
        """Test write method with callback function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            simulator = AudioSinkSimulator(
                frame_rate=16000,
                sample_width=2,
                n_channels=1,
                file_path=str(output_path),
                chunk_duration_secs=0.128,
            )

            callback_called = False

            def callback(data):
                nonlocal callback_called
                callback_called = True
                assert len(data) > 0

            # Write audio data
            audio_data = b"\x00\x00" * 2048  # 4096 bytes
            simulator.write(audio_data, process_audio_callback=callback)

            assert callback_called
            simulator.close()

    def test_write_method_with_closed_file(self):
        """Test write method raises error when file is closed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            simulator = AudioSinkSimulator(file_path=str(output_path))
            simulator.close()

            # Write data to trigger the write operation
            with pytest.raises(RuntimeError, match="Output audio file is closed"):
                simulator.write(b"\x00\x00" * 2048)  # 4096 bytes

    def test_close_method(self):
        """Test close method closes the wave file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            simulator = AudioSinkSimulator(file_path=str(output_path))

            assert simulator.is_open()
            simulator.close()
            assert not simulator.is_open()

    def test_wav_format_support(self):
        """Test AudioSinkSimulator with WAV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"

            # Test WAV sink
            wav_sink = AudioSinkSimulator(
                frame_rate=16000,
                sample_width=2,
                n_channels=1,
                n_frames=0,
                file_path=str(output_path),
                chunk_duration_secs=0.128,
                audio_format="wav",
            )

            # Write some test data
            test_data = b"\x00\x00" * 1600  # 0.1 seconds of silence
            wav_sink.write(test_data)
            wav_sink.close()

            # Verify the file was created and has WAV header
            with open(output_path, "rb") as f:
                data = f.read()
                assert data.startswith(b"RIFF"), "WAV file should start with RIFF"
                assert b"WAVE" in data, "WAV file should contain WAVE"

    def test_mp3_format_support(self):
        """Test AudioSinkSimulator with MP3 format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp3"

            # Test MP3 sink
            mp3_sink = AudioSinkSimulator(
                frame_rate=16000,
                sample_width=2,
                n_channels=1,
                n_frames=0,
                file_path=str(output_path),
                chunk_duration_secs=0.128,
                audio_format="mp3",
            )

            # Write some test data (simulated MP3 data)
            test_data = b"ID3" + b"\x00" * 100  # Simulated MP3 header
            mp3_sink.write(test_data)
            mp3_sink.close()

            # Verify the file was created
            with open(output_path, "rb") as f:
                data = f.read()
                assert len(data) > 0, "MP3 file should not be empty"

    def test_invalid_audio_format(self, recwarn):
        """Test AudioSinkSimulator with invalid audio format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.ogg"

            # Test invalid format
            with pytest.raises(ValueError, match="Unsupported audio format"):
                AudioSinkSimulator(
                    frame_rate=16000,
                    sample_width=2,
                    n_channels=1,
                    n_frames=0,
                    file_path=str(output_path),
                    chunk_duration_secs=0.128,
                    audio_format="ogg",  # Unsupported format
                )
            _assert_no_unraisable_warning(recwarn)

    def test_audio_format_case_insensitive(self):
        """Test that audio format is case insensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.wav"

            # Test with uppercase format
            wav_sink = AudioSinkSimulator(
                frame_rate=16000,
                sample_width=2,
                n_channels=1,
                n_frames=0,
                file_path=str(output_path),
                chunk_duration_secs=0.128,
                audio_format="WAV",  # Uppercase
            )

            assert wav_sink.audio_format == "wav"  # Should be converted to lowercase
            wav_sink.close()


if __name__ == "__main__":
    pytest.main([__file__])
