# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio simulator classes for S2S client."""

import os
import time
import wave
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator

from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest

from common.audio_utils import create_wav_header
from common.base_utils import logger
from common.source_sink.base import BaseFileSimulator


class AudioSourceSimulator(BaseFileSimulator):
    """Simulate an audio source as an iterator."""

    def __init__(self, file_path: os.PathLike) -> None:
        """Initialize AudioSourceSimulator with a WAV file path."""
        super().__init__(file_path=file_path)

        # Get WAV file parameters
        self._file_opened = wave.open(file_path, "rb")
        self.n_channels = self._file_opened.getnchannels()
        self.sample_width = self._file_opened.getsampwidth()
        self.frame_rate = self._file_opened.getframerate()
        self.n_frames = self._file_opened.getnframes()

        # Create a dummy header for the WAV file
        self.header = create_wav_header(
            n_channels=self.n_channels,
            sample_width=self.sample_width,
            frame_rate=self.frame_rate,
            n_frames=self.n_frames,
        )

    def validate_file_path(self, value: os.PathLike) -> None:
        """Validate the file path of the WAV file."""
        if not os.path.exists(value):
            raise FileNotFoundError(f"File not found: {value}")

    def samples(
        self, chunk_duration_secs: float, process_audio_callback: Callable | None = None
    ) -> Generator[bytes, None, None]:
        """Generator that yields audio samples from the WAV file.

        Args:
            chunk_duration_secs (float): The duration of the chunk in seconds.
                Once set, will not be changed until entire file is read, but can be
                changed for a second read.
            process_audio_callback (Callable): A callback function to process the audio samples
                once read.

        Returns:
            Generator[bytes, None, None]: A generator that yields audio samples in bytes.
        """
        # Calculate samples per chunk
        samples_per_chunk = int(chunk_duration_secs * self.frame_rate)
        count = 0
        while True:
            samples = self._file_opened.readframes(samples_per_chunk)
            if not samples:  # End of file
                break
            # Note down the timestamp of the sample taken out.
            self.ledger[count] = time.time()
            if process_audio_callback:
                process_audio_callback(samples)
            count += 1
            yield samples

    def read(
        self, chunk_duration_secs: float, process_audio_callback: Callable | None = None
    ) -> Generator[bytes, None, None]:
        """Yield a WAV file as a generator.

        First will yield a header, then will start yielding samples.

        Args:
            chunk_duration_secs (float): The duration of the chunk in seconds.
            process_audio_callback (Callable | None): A callback function to process
                the audio samples once read.

        Yields:
            Generator[bytes, None, None]: A generator that yields audio samples in bytes.
        """
        yield self.header
        # generate samples
        yield from self.samples(
            chunk_duration_secs=chunk_duration_secs, process_audio_callback=process_audio_callback
        )


class AudioSinkSimulator(BaseFileSimulator):
    """Simulate an audio sink to consume audio samples from a generator."""

    def validate_file_path(self, value: os.PathLike) -> None:
        """Validate the file path of the audio file."""
        if not os.path.exists(os.path.dirname(value)):
            raise FileNotFoundError(f"Directory not found: {os.path.dirname(value)}")

    def __init__(
        self,
        frame_rate: int = 16000,
        sample_width: int = 2,
        n_channels: int = 1,
        n_frames: int = 0,
        file_path: str = "output.wav",
        chunk_duration_secs: float = 0.128,
        audio_format: str = "wav",
    ) -> None:
        """Initialize AudioSinkSimulator.

        Args:
            frame_rate (int): The frame rate in Hz. (sample rate).
            sample_width (int): The sample width in bytes, usually 2 for 16-bit PCM.
            n_channels (int): The number of channels in the audio file.
            n_frames (int): The number of frames in the audio file.
            file_path (str): The path to the output audio file.
            chunk_duration_secs (float): The duration of the chunk in seconds.
            audio_format (str): The audio format ("wav" or "mp3").
        """
        self._file_opened = None
        super().__init__(file_path=file_path)

        self.audio_format = audio_format.lower()
        self.frame_rate = frame_rate
        self.sample_width = sample_width
        self.n_channels = n_channels

        # When True, incoming WAV data already contains a RIFF header. We
        # write raw bytes instead of using wave.writeframes() which would
        # corrupt the output by treating the header as PCM samples.
        self._wav_passthrough = False

        # Open file in write mode based on format
        if self.audio_format == "wav":
            self._file_opened = wave.open(file_path, mode="wb")
            self._file_opened.setnchannels(nchannels=n_channels)
            self._file_opened.setsampwidth(sampwidth=sample_width)
            self._file_opened.setframerate(framerate=frame_rate)
            self._file_opened.setnframes(nframes=n_frames)
            # Keep path so we can reopen in passthrough mode
            self._file_path = file_path
        elif self.audio_format == "mp3":
            # For MP3, we'll write raw bytes to the file
            self._file_opened = open(file_path, "wb")
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

        # Open file in write mode.
        self._chunk_count = 0
        self._buffer = b""
        self.chunk_duration_secs = chunk_duration_secs

    def write(
        self,
        wave_bytes: bytes,
        process_audio_callback: Callable | None = None,
    ) -> None:
        """Write audio data to the output file.

        On the first WAV write, checks whether the data already contains a
        RIFF header. If so, switches to raw-byte passthrough to avoid
        corrupting the output.

        Args:
            wave_bytes (bytes): Audio data from the S2S service.
            process_audio_callback (Callable | None): A callback function to
                process the audio samples once read.
        """
        # Write audio data to the file
        if self._file_opened is None:
            raise RuntimeError("Output audio file is closed")

        if process_audio_callback:
            process_audio_callback(wave_bytes)

        # On first WAV chunk, detect if data already has a RIFF header
        if (
            self.audio_format == "wav"
            and self._chunk_count == 0
            and wave_bytes[:4] == b"RIFF"
            and not self._wav_passthrough
        ):
            # Data is a complete WAV file — close the wave writer
            # and reopen as raw binary so we don't double-wrap.
            self._file_opened.close()
            self._file_opened = open(self._file_path, "wb")  # noqa: SIM115
            self._wav_passthrough = True

        # Write the audio data based on format
        if self.audio_format == "wav" and not self._wav_passthrough:
            # Write raw PCM data directly to the WAV file
            self._file_opened.writeframes(wave_bytes)
        else:
            # MP3, or WAV passthrough — write bytes directly
            self._file_opened.write(wave_bytes)

        self.ledger[self._chunk_count] = time.time()
        self._chunk_count += 1
        if self._chunk_count % 10 == 0:
            logger.debug(f"Audio sink | received chunk: {self._chunk_count}")


def simulated_audio_chunk_generator(
    simulator: AudioSourceSimulator, chunk_size_secs: float = 0.128
) -> Iterator[SpeechToSpeechRequest]:
    """Generate SpeechToSpeechRequest messages from a simulated audio source
        and return a request chunk.

    Args:
        simulator (AudioSourceSimulator): The simulated audio source.
        chunk_size_secs (float): The chunk size in seconds for streaming audio.

    Returns:
        Iterator[SpeechToSpeechRequest]: An iterator of SpeechToSpeechRequest messages.
    """
    for chunk in simulator.read(chunk_duration_secs=chunk_size_secs):
        yield SpeechToSpeechRequest(
            audio_data=chunk,
            audio_sample_rate=simulator.frame_rate,
            audio_num_channels=simulator.n_channels,
            audio_format="LINEAR_PCM",
        )


def simulated_audio_chunk_generator_raw(
    simulator: "BaseFileSimulator", chunk_size_secs: float = 0.128
) -> Iterator[bytes]:
    """Generate raw audio chunks from a simulated audio source and return a request chunk.

    Args:
        simulator (BaseFileSimulator): The simulated audio source.
        chunk_size_secs (float): The chunk size for streaming audio. Default is 0.128 seconds.

    Yields:
        bytes: Raw audio chunks read from the simulator.
    """
    yield from simulator.read(chunk_duration_secs=chunk_size_secs)
