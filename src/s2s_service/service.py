# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for S2S services."""

import argparse
import os
import tempfile
import threading
import traceback
import uuid
import wave
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator

import grpc
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse
from nvidia.ai4m.s2s.v1.s2s_pb2_grpc import SpeechToSpeechServicer
from nvidia.ai4m.s2s.v1.s2s_pb2_grpc import add_SpeechToSpeechServicer_to_server

from common.audio_utils import download_audio_file_from_iterator
from common.audio_utils import is_wav_file
from common.base_utils import GRPCServiceBase
from common.base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer


def download_input_audio_file(
    request_iterator: Iterator[SpeechToSpeechRequest],
    context: grpc.ServicerContext,
    request_id: str,
    poll_timeout: float = 0.1,
) -> str:
    """Download streamed audio into a temp WAV file with buffering and header fix.
    Use only in transactional use-cases.

    Args:
        request_iterator (Iterator[SpeechToSpeechRequest]): The request iterator.
        context (grpc.ServicerContext): The gRPC context.
        request_id (str): The request ID.
        poll_timeout (float): The poll timeout in seconds.

    Returns:
        str: The path to the downloaded audio file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as temp_in:
        input_path = temp_in.name

    _input_buffer: Buffer[SpeechToSpeechRequest] = Buffer(num_queues=1)

    def collect_requests() -> None:
        try:
            for chunk in request_iterator:
                _input_buffer.put(chunk)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Error collecting audio data in request id {request_id}: {exc}")
        finally:
            _input_buffer.done = True

    collector_thread = threading.Thread(target=collect_requests, daemon=True)
    collector_thread.start()

    request_iterator_from_input_buffer = RequestIteratorFromBuffer(
        _input_buffer, poll_timeout=poll_timeout
    )
    try:
        input_path = download_audio_file_from_iterator(
            chunks=request_iterator_from_input_buffer,
            file_path=input_path,
        )
        # All input packets have arrived
        collector_thread.join()

    except Exception as e:
        os.remove(input_path)
        tb = traceback.format_exc()
        logger.error(f"Error collecting audio data in request id {request_id}: {e}\n{tb}")
        context.abort(grpc.StatusCode.INTERNAL, f"Collecting audio data failed: {e}\n{tb}")

    # Non-WAV inputs (e.g. MP3) must not go through the WAV header-fix path.
    if not is_wav_file(input_path):
        non_wav_path = input_path.removesuffix(".wav") + ".mp3"
        os.rename(input_path, non_wav_path)
        logger.info(f"Non-WAV input detected for {request_id}, saved as {non_wav_path}")
        logger.debug(
            f"Input file streamed in for request id {request_id}: {non_wav_path} of "
            f"size {os.path.getsize(non_wav_path)} bytes"
        )
        return non_wav_path

    # Fix WAV header if nframes is 0 (common when streaming).
    # Capture everything inside the with block — wave.Wave_read is closed on exit
    # and calling methods on it afterwards is undefined behaviour.
    with wave.open(input_path, "rb") as wav_check:
        nframes = wav_check.getnframes()
        if nframes == 0:
            sample_rate = wav_check.getframerate()
            channels = wav_check.getnchannels()
            sample_width = wav_check.getsampwidth()

    if nframes == 0:
        with open(input_path, "rb") as raw_file:
            raw_file.seek(44)
            pcm_data = raw_file.read()

        with wave.open(input_path, "wb") as fixed_wav:
            fixed_wav.setnchannels(channels)
            fixed_wav.setsampwidth(sample_width)
            fixed_wav.setframerate(sample_rate)
            fixed_wav.writeframes(pcm_data)

        logger.info(f"Fixed WAV header with {len(pcm_data) // (channels * sample_width)} frames")

    logger.debug(
        f"Input file streamed in for request id {request_id}: {input_path} of "
        f" size Input file size: {os.path.getsize(input_path)} bytes"
    )
    return input_path


class S2SServiceServicer(SpeechToSpeechServicer):
    """Speech-to-Speech servicer implementation.

    This class handles the gRPC streaming service for speech-to-speech conversion.
    It delegates processing to the configured S2S service implementation, which
    uses queue-based background work to produce streaming responses.
    """

    def __init__(self, service: "S2SService") -> None:
        """Initialize the S2S servicer.

        Args:
            service (S2SService): The parent S2SService instance that provides for an S2S
            inference.
        """
        self.service = service

    def StreamSpeechToSpeech(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Process an audio stream and return dubbed speech.

        This is the RPC method that the client will call into.

        This method implements the bidirectional streaming RPC. It:

        1. Receives audio chunks from the client
        2. Submits the collected audio to the dubbing API (ElevenLabs or CambAI)
        3. Sends keepalive responses while the dubbing job runs
        4. Streams the dubbed audio chunks back to the client

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): Async iterator of incoming
                SpeechToSpeechRequest messages.
            context (grpc.ServicerContext): The gRPC servicer context.

        Yields:
            SpeechToSpeechResponse: Processed audio chunks with synthesized speech.

        Raises:
            grpc.RpcError: If there's an error in processing the stream.
        """
        # Get the first request to extract the request_id
        logger.debug("Creating request id.")
        request_id = str(uuid.uuid4())
        peer = context.peer() if hasattr(context, "peer") else "unknown"
        logger.debug(f"Request received | id={request_id} | peer={peer}")

        # It is the responsibility of the infer method to handle S2S and yield chunks of audio
        # in the SpeechToSpeechResponse format.
        try:
            logger.debug("Running S2S call.")
            s2s_response = self.service.infer(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in S2S Service: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
            return

        logger.debug("S2S Service response passed.")

        # Yield S2S responses directly (ASD is now handled by separate RPC)
        yield from s2s_response


class S2SService(GRPCServiceBase, ABC):
    """Abstract base class for S2S services."""

    # Validation for service parameters
    @property
    def nchannels(self) -> int:
        """Number of audio channels.

        Args:
            value (int): The number of audio channels.

        Returns:
            int: The number of audio channels.
        """
        return self._nchannels

    @nchannels.setter
    def nchannels(self, value: int) -> None:
        """Set the number of audio channels.

        Args:
            value (int): The number of audio channels.

        Raises:
            ValueError: If the number of channels is less than 1.
        """
        if value < 1:
            raise ValueError("Number of channels must be >= 1")
        self._nchannels = value

    @property
    def sample_rate_hz(self) -> int:
        """Sample rate in Hz.

        Args:
            value (int): The sample rate in Hz.

        Returns:
            int: The sample rate in Hz.
        """
        return self._sample_rate_hz

    @sample_rate_hz.setter
    def sample_rate_hz(self, value: int) -> None:
        """Set the sample rate in Hz.

        Args:
            value (int): The sample rate in Hz.

        Raises:
            ValueError: If the sample rate is not 8000, 16000, 24000, or 48000.
        """
        if value not in [8000, 16000, 24000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 24000, or 48000")
        self._sample_rate_hz = value

    def validate_source_language(self, value: str) -> bool:
        """Validate the source language.

        Supported languages: en-US, es-US, fr-FR.

        Args:
            value (str): The source language.

        Returns:
            bool: True if the source language is supported, False otherwise.
        """
        return value in self.supported_source_languages

    def validate_target_language(self, value: str) -> bool:
        """Validate the target language.

        Supported languages: en-US, es-US, fr-FR.

        Args:
            value (str): The target language.

        Returns:
            bool: True if the target language is supported, False otherwise.
        """
        return value in self.supported_target_languages

    @abstractmethod
    def validate_audio_format(self, value: str) -> bool:
        """Validate the audio format.

        Args:
            value (str): The audio format.
        """

    def __init__(
        self,
        message_size: int = 1024 * 1024 * 4,
        sample_rate_hz: int = 16000,
        nchannels: int = 1,
        audio_format: str = "mp3",
        default_source_language: str = "en",
        default_target_language: str = "es",
        supported_source_languages: list[str] = [],
        supported_target_languages: list[str] = [],
        # TODO: Add voice name generalization.
    ) -> None:
        """Initialize the S2S service.

        Args:
            message_size (int): The maximum message size in bytes. Defaults to 1024 * 1024 * 4.
            sample_rate_hz (int): The sample rate in Hz. Defaults to 16000.
            nchannels (int): The number of audio channels. Defaults to 1.
            audio_format (str): The audio format. Defaults to "mp3".
            default_source_language (str): The default source language. Defaults to "en".
            default_target_language (str): The default target language. Defaults to "es".
            supported_source_languages (list[str]): The supported source languages. Defaults to [].
            supported_target_languages (list[str]): The supported target languages. Defaults to [].
        """
        super().__init__(message_size=message_size)
        self.sample_rate_hz = sample_rate_hz
        self.nchannels = nchannels
        self.audio_format = audio_format.lower()
        self.default_source_language = default_source_language
        self.default_target_language = default_target_language
        self.supported_source_languages = supported_source_languages
        self.supported_target_languages = supported_target_languages

    def add_servicer_to_server(self, server: grpc.Server) -> None:
        """Add the S2S servicer to the gRPC server.

        This method is called by the base class to register the servicer
        with the gRPC server. It creates a new S2SServiceServicer instance
        and adds it to the server.

        Args:
            server (grpc.Server): The gRPC server instance to add the servicer to.
        """
        add_SpeechToSpeechServicer_to_server(S2SServiceServicer(service=self), server)
        logger.debug("Added S2S servicer to gRPC server")

    @abstractmethod
    def infer(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Write your inference logic here.

        Make sure to yield chunks of audio in the SpeechToSpeechResponse format.

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): The audio to translate.
            context (grpc.ServicerContext): The context of the request.
            request_id (str): The id of the request.

        Returns:
            Iterator[SpeechToSpeechResponse]: The translated audio.

        Raises:
            grpc.RpcError: If there's an error in processing the stream.
        """

    def download_input_audio(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> str:
        """Download streamed audio to a temp file using the shared helper."""
        return download_input_audio_file(
            request_iterator=request_iterator,
            context=context,
            request_id=request_id,
        )

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Parser for command line arguments.

        Args:
            parser (argparse.ArgumentParser | None): Optional existing parser to extend.

        Returns:
            argparse.ArgumentParser: Unparsed command line arguments
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Speech-to-Speech Service")

        parser = GRPCServiceBase.argsfactory(parser)

        parser.add_argument(
            "--sample-rate-hz",
            type=int,
            default=16000,
            help="The sample rate in Hz.",
        )
        parser.add_argument(
            "--default-source-language",
            type=str,
            default="en",
            help="The source language.",
        )
        parser.add_argument(
            "--default-target-language",
            type=str,
            default="es",
            help="The target language.",
        )
        # TODO: Move this to input audio config.
        parser.add_argument(
            "--audio-format",
            type=str,
            default="mp3",
            help="The audio format.",
        )
        return parser
