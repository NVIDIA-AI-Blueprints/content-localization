# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eleven Labs-based Speech-to-Speech (S2S) gRPC service."""

import argparse
import os
import queue
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from queue import Empty as QueueEmpty

import grpc

# Importing the ElevenLabs client
from elevenlabs.client import ElevenLabs
from google.protobuf.empty_pb2 import Empty

# RequestId is not available in the generated protos, removing import
# Importing the auto-generated proto
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.audio_utils import audio_mime_type
from common.audio_utils import download_audio_file_from_iterator
from common.base_utils import logger

# Importing the base S2S service
from s2s_service.service import S2SService

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY environment variable not set.")
    client = None
else:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Supported languages for ElevenLabs Dubbing API
# (see: https://elevenlabs.io/docs/capabilities/dubbing)
SUPPORTED_SOURCE_LANGUAGES = [
    "en",  # English
    "hi",  # Hindi
    "pt",  # Portuguese
    "zh",  # Chinese
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "ja",  # Japanese
    "ar",  # Arabic
    "ru",  # Russian
    "ko",  # Korean
    "id",  # Indonesian
    "it",  # Italian
    "nl",  # Dutch
    "tr",  # Turkish
    "pl",  # Polish
    "sv",  # Swedish
    "fil",  # Filipino
    "ms",  # Malay
    "ro",  # Romanian
    "uk",  # Ukrainian
    "el",  # Greek
    "cs",  # Czech
    "da",  # Danish
    "fi",  # Finnish
    "bg",  # Bulgarian
    "hr",  # Croatian
    "sk",  # Slovak
    "ta",  # Tamil
    "hu",  # Hungarian
    "no",  # Norwegian
    "vi",  # Vietnamese
    "auto",  # Auto-detect language
]
SUPPORTED_TARGET_LANGUAGES = SUPPORTED_SOURCE_LANGUAGES.copy()


def download_dubbed_file(dubbing_id: str, language_code: str) -> Iterator[bytes]:
    """Downloads the dubbed file for a given dubbing ID and language code.

    Args:
        dubbing_id: The ID of the dubbing project.
        language_code: The language code for the dubbing.

    Returns:
        Iterator[bytes]: The generator for the dubbed file.
    """
    # Use the current ElevenLabs API
    chunk_count = 0
    total_bytes = 0
    try:
        for chunk in client.dubbing.audio.get(dubbing_id, language_code):
            chunk_count += 1
            total_bytes += len(chunk)
            logger.debug(
                "Downloaded from 11 labs chunk %s: %s bytes (total: %s)",
                chunk_count,
                len(chunk),
                total_bytes,
            )
            yield chunk
        logger.info(f"Downloaded complete file: {chunk_count} chunks, {total_bytes} total bytes")
    except Exception as e:
        logger.error(f"Error downloading dubbed file: {e}")
        raise e


def wait_for_dubbing_completion(dubbing_id: str) -> bool:
    """Waits for the dubbing process to complete by checking the status often.

    Args:
        dubbing_id (str): The dubbing project id.

    Returns:
        bool: True if the dubbing is successful, False otherwise.
    """
    MAX_ATTEMPTS = int(os.environ.get("S2S_EL_DUBBING_MAX_ATTEMPTS", "120"))
    CHECK_INTERVAL = int(os.environ.get("S2S_EL_DUBBING_POLL_INTERVAL", "10"))

    in_progress_statuses = {"dubbing", "queued", "processing", "in_progress", "preparing"}
    failure_statuses = {"failed", "error", "canceled", "cancelled"}

    for _ in range(MAX_ATTEMPTS):
        metadata = client.dubbing.get(dubbing_id)
        status = getattr(metadata, "status", None)
        error = getattr(metadata, "error", None)
        if status == "dubbed":
            return True
        elif status in in_progress_statuses:
            logger.debug(
                f"Dubbing in progress... Will check status again in {CHECK_INTERVAL} seconds."
            )
            time.sleep(CHECK_INTERVAL)
        elif status in failure_statuses:
            logger.error(f"Dubbing failed: status={status}, error={error}")
            return False
        else:
            logger.warning(f"Unexpected dubbing status: {status}, error={error}")
            time.sleep(CHECK_INTERVAL)

    logger.error(f"Dubbing timed out after {MAX_ATTEMPTS} attempts")
    return False


def create_dub_from_file(
    input_file_path: Path,
    source_language: str,
    target_language: str,
    *,
    num_speakers: int = 0,
    drop_background_audio: bool = False,
    use_profanity_filter: bool = False,
    target_accent: str | None = None,
    highest_resolution: bool = False,
    watermark: bool = False,
    dubbing_studio: bool = False,
) -> Iterator[bytes]:
    """Dub an audio or video file from one language to another via ElevenLabs.

    Args:
        input_file_path (Path): The file path of the audio or video to dub.
        source_language (str): The language of the input file.
        target_language (str): The target language to dub into.
        num_speakers (int): Number of speakers. 0 = auto-detect. Defaults to ``0``.
        drop_background_audio (bool): Drop background audio from the final dub.
            Defaults to ``False``.
        use_profanity_filter (bool): Censor profanities in transcripts.
            Defaults to ``False``.
        target_accent (str | None): Experimental accent to apply. Defaults to ``None``.
        highest_resolution (bool): Use highest resolution output.
            Defaults to ``False``.
        watermark (bool): Apply watermark to output. Defaults to ``False``.
        dubbing_studio (bool): Prepare dub for dubbing studio edits.
            Defaults to ``False``.

    Returns:
        Iterator[bytes]: The generator for the dubbed file.

    Examples:
        >>> it = create_dub_from_file(Path("in.wav"), "en", "es", num_speakers=2)
    """
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    logger.debug(f"Creating dubbing in EL Cloud for request id {input_file_path}")

    create_kwargs: dict = {
        "file": (os.path.basename(input_file_path), None, "audio/wav"),
        "target_lang": target_language,
        "source_lang": source_language,
        "mode": "automatic",
        "num_speakers": num_speakers,
        "drop_background_audio": drop_background_audio,
        "use_profanity_filter": use_profanity_filter,
        "highest_resolution": highest_resolution,
        "watermark": watermark,
        "dubbing_studio": dubbing_studio,
        "disable_voice_cloning": False,
    }
    if target_accent:
        create_kwargs["target_accent"] = target_accent

    mime_type = audio_mime_type(input_file_path)
    with open(input_file_path, "rb") as audio_file:
        create_kwargs["file"] = (os.path.basename(input_file_path), audio_file, mime_type)
        response = client.dubbing.create(**create_kwargs)

    dubbing_id = response.dubbing_id
    logger.debug(f"Dubbing ID: {dubbing_id}")
    if wait_for_dubbing_completion(dubbing_id=dubbing_id):
        file_downloader = download_dubbed_file(
            dubbing_id=dubbing_id,
            language_code=target_language,
        )
        yield from file_downloader
    else:
        raise Exception("Dubbing failed")


class ELDubbingService(S2SService):
    """Speech-to-Speech service using ElevenLabs streaming dubbing API.

    Note: This service is transactional only, even though it streams in audio.
    This is because the ElevenLabs API only supports non-streaming dubbing.

    .. code-block:: text

        S2SServiceServicer (from service.py)
          |
          | 1. Extract request_id, wrap iterator
          | 2. Call: service.infer(request_iterator, context, request_id)
          v
        S2SService (abstract, implemented by ELDubbingService or CambDubbingService)
          |
          | (ElevenLabs Path)
          |
          | 3. ELDubbingService.infer()
          |    |
          |    4. Extract config from first request
          |       (source_language, target_language, elevenlabs_* params)
          |       v
          |    5. self.download_input_audio()
          |       - Collects all audio data from the stream
          |       - Writes to a temp WAV file
          |       v
          |    6. self._impl(**el_kwargs)
          |       |
          |       |-- Background thread: process_audio() ----------------------.
          |       |   create_dub_from_file(**el_kwargs)                       |
          |       |     -> client.dubbing.create(num_speakers, ...)           |
          |       |     -> download_dubbed_file()                             |
          |       |   read MP3 -> enqueue SpeechToSpeechResponse + completed  |
          |       |                                                        |
          |       `-- Main thread: read queue (timeout) ---------------------'
          |            - yield audio chunks
          |            - send keep-alive when queue is empty
          |
          v
        Return response stream to client

    """

    def validate_audio_format(self, value: str) -> bool:
        """Validate the audio format.
        Supported formats: mp3, wav.
        """
        return value in ["mp3"]

    def __init__(
        self,
        message_size: int = 1024 * 1024 * 4,
        sample_rate_hz: int = 16000,
        default_source_language: str = "en",
        default_target_language: str = "es",
        audio_format: str = "mp3",
    ) -> None:
        """Initialize the EL S2S service.

        Args:
            message_size (int, optional): The maximum message size in bytes.
                Defaults to 1024*1024*4.
            sample_rate_hz (int, optional): The sample rate in Hz. Defaults to 16000.
            default_source_language (str, optional): The source language. Defaults to "en".
            default_target_language (str, optional): The target language. Defaults to "es".
            audio_format (str, optional): The audio format. Defaults to "mp3".

        Raises:
            ValueError: If the ElevenLabs API key is not set.
            Exception: If there is an error getting the ElevenLabs API key.
        """
        if not client:
            raise RuntimeError("ElevenLabs client not initialized.")

        super().__init__(
            message_size=message_size,
            sample_rate_hz=sample_rate_hz,
            default_source_language=default_source_language,
            default_target_language=default_target_language,
            audio_format=audio_format,
            supported_source_languages=SUPPORTED_SOURCE_LANGUAGES,
            supported_target_languages=SUPPORTED_TARGET_LANGUAGES,
        )
        logger.debug(
            "Initialized EL S2S service for dubbing from %s to %s",
            default_source_language,
            default_target_language,
        )

    def _impl(
        self,
        input_path: str,
        request_id: str,
        context: grpc.ServicerContext,
        source_language: str = "en",
        target_language: str = "es",
        *,
        num_speakers: int = 0,
        drop_background_audio: bool = False,
        use_profanity_filter: bool = False,
        target_accent: str | None = None,
        highest_resolution: bool = False,
        watermark: bool = False,
        dubbing_studio: bool = False,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Call the ElevenLabs Dubbing API and stream the dubbed audio back.

        Args:
            input_path (str): The path to the input audio file.
            request_id (str): The request ID.
            context (grpc.ServicerContext): The gRPC context.
            source_language (str): The source language. Defaults to ``"en"``.
            target_language (str): The target language. Defaults to ``"es"``.
            num_speakers (int): Number of speakers. 0 = auto-detect. Defaults to ``0``.
            drop_background_audio (bool): Drop background audio. Defaults to ``False``.
            use_profanity_filter (bool): Censor profanities. Defaults to ``False``.
            target_accent (str | None): Accent to apply. Defaults to ``None``.
            highest_resolution (bool): Highest resolution output. Defaults to ``False``.
            watermark (bool): Apply watermark. Defaults to ``False``.
            dubbing_studio (bool): Prepare for dubbing studio. Defaults to ``False``.

        Returns:
            Iterator[SpeechToSpeechResponse]: Audio response stream.
        """
        # Queue to store audio chunks for streaming
        audio_queue = queue.Queue()
        processing_done = threading.Event()

        el_kwargs = {
            "num_speakers": num_speakers,
            "drop_background_audio": drop_background_audio,
            "use_profanity_filter": use_profanity_filter,
            "target_accent": target_accent,
            "highest_resolution": highest_resolution,
            "watermark": watermark,
            "dubbing_studio": dubbing_studio,
        }

        def process_audio():
            """Process audio in background thread."""
            dubbed_audio_path: Path | None = None
            try:
                logger.debug(f"Calling ElevenLabs API for request id {request_id}")
                logger.debug(f"Target language: {target_language}")
                try:
                    file_downloader = create_dub_from_file(
                        input_file_path=Path(input_path),
                        source_language=source_language,
                        target_language=target_language,
                        **el_kwargs,
                    )
                except Exception as e:
                    logger.error(f"Error calling ElevenLabs API for request id {request_id}: {e}")
                    logger.error(f"Exception traceback: {traceback.format_exc()}")
                    raise e

                # Download the dubbed audio file to a temp file
                logger.debug(f"Downloading dubbed audio file for request id {request_id}")
                dubbed_audio_path_tmp = tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False, dir="/tmp"
                )
                dubbed_audio_path = download_audio_file_from_iterator(
                    chunks=file_downloader, file_path=Path(dubbed_audio_path_tmp.name)
                )

                # Log the file size to debug
                file_size = os.path.getsize(dubbed_audio_path)
                logger.info(f"Downloaded MP3 file size: {file_size} bytes")

                # Read audio file and put chunks in queue
                logger.info(f"Starting to read MP3 file: {dubbed_audio_path}")
                chunk_count = 0
                with open(dubbed_audio_path, "rb") as f:
                    while True:
                        chunk = f.read(8192)  # Read in 8KB chunks
                        if not chunk:
                            break
                        chunk_count += 1
                        logger.debug(f"Read chunk {chunk_count}: {len(chunk)} bytes")
                        audio_queue.put(
                            SpeechToSpeechResponse(audio_data=chunk, audio_format=self.audio_format)
                        )
                logger.info(f"Finished reading MP3 file: {chunk_count} chunks total")

                # Clean up
                os.remove(dubbed_audio_path)
                # Signal completion AFTER all chunks have been put in queue
                logger.info("Putting completion signal in queue")
                audio_queue.put("completed")  # Signal completion

            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                logger.error(f"Exception traceback: {traceback.format_exc()}")
                audio_queue.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                if dubbed_audio_path is not None and Path(dubbed_audio_path).exists():
                    try:
                        os.remove(dubbed_audio_path)
                    except Exception:
                        logger.warning(
                            f"Failed to clean up temporary dubbed file: {dubbed_audio_path}"
                        )
                # Set processing done AFTER sending completion signal
                logger.info("Setting processing done flag")
                processing_done.set()

        # Start audio processing thread
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        audio_thread.start()

        # Stream responses with keep-alive
        keepalive_ping_interval_secs = int(os.environ.get("S2S_EL_KEEPALIVE_INTERVAL", "1"))
        response_count = 0
        keepalive_count = 0
        while True:
            try:
                response = audio_queue.get(timeout=keepalive_ping_interval_secs)
                if isinstance(response, tuple) and response[0] == "error":
                    context.abort(
                        grpc.StatusCode.INTERNAL, f"ElevenLabs dubbing failed: {response[1]}"
                    )
                if response == "completed":
                    logger.info(
                        f"Received completed from queue, processing complete. "
                        f"Total responses: {response_count}, keepalives: {keepalive_count}"
                    )
                    break  # Processing complete
                response_count += 1
                logger.debug(
                    f"Yielding audio response {response_count}: {len(response.audio_data)} bytes"
                )
                yield response
            except QueueEmpty:
                # No response available, send keep-alive
                keepalive_count += 1
                logger.debug(f"Sending keepalive response {keepalive_count}")
                yield SpeechToSpeechResponse(keepalive=Empty())

            # Note: We don't need to check processing_done.is_set() here because
            # the "completed" signal from the queue is sufficient to indicate
            # when all audio chunks have been processed and sent.

        # Clean up the input file
        if os.path.exists(input_path):
            os.remove(input_path)

    def infer(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
        source_language: str = "en",
        target_language: str = "es",
    ) -> Iterator[SpeechToSpeechResponse]:
        """Run ElevenLabs streaming dubbing pipeline.

        This method:
        1. Extracts config (languages + ElevenLabs params) from the first request.
        2. Collects all audio data from the stream.
        3. Calls ElevenLabs Dubbing API (file-based).
        4. Streams the dubbed audio file back to the client in chunks.

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): The request iterator.
            context (grpc.ServicerContext): The gRPC context.
            request_id (str): The request ID.
            source_language (str): The source language. Defaults to ``"en"``.
            target_language (str): The target language. Defaults to ``"es"``.

        Returns:
            Iterator[SpeechToSpeechResponse]: The response iterator.
        """
        logger.info(f"Received request id: {request_id}")

        config_request = next(request_iterator)
        if config_request.audio_data:
            original_request_iterator = request_iterator

            def replay_request_iterator() -> Iterator[SpeechToSpeechRequest]:
                yield config_request
                yield from original_request_iterator

            request_iterator = replay_request_iterator()

        # -- Extract ElevenLabs-specific parameters (defaults for all) --
        el_kwargs: dict = {
            "num_speakers": 0,
            "drop_background_audio": False,
            "use_profanity_filter": False,
            "target_accent": None,
            "highest_resolution": False,
            "watermark": False,
            "dubbing_studio": False,
        }

        if config_request.HasField("config"):
            config = config_request.config
            if config.HasField("source_language"):
                source_language = config.source_language
            else:
                source_language = self.default_source_language
            if config.HasField("target_language"):
                target_language = config.target_language
            else:
                target_language = self.default_target_language

            if config.HasField("elevenlabs_num_speakers"):
                el_kwargs["num_speakers"] = config.elevenlabs_num_speakers
            if config.HasField("elevenlabs_drop_background_audio"):
                el_kwargs["drop_background_audio"] = config.elevenlabs_drop_background_audio
            if config.HasField("elevenlabs_use_profanity_filter"):
                el_kwargs["use_profanity_filter"] = config.elevenlabs_use_profanity_filter
            if config.HasField("elevenlabs_target_accent"):
                el_kwargs["target_accent"] = config.elevenlabs_target_accent
            if config.HasField("elevenlabs_highest_resolution"):
                el_kwargs["highest_resolution"] = config.elevenlabs_highest_resolution
            if config.HasField("elevenlabs_watermark"):
                el_kwargs["watermark"] = config.elevenlabs_watermark
            if config.HasField("elevenlabs_dubbing_studio"):
                el_kwargs["dubbing_studio"] = config.elevenlabs_dubbing_studio
        else:
            source_language = self.default_source_language
            target_language = self.default_target_language

        if not self.validate_source_language(source_language):
            logger.error(f"Invalid source language: {source_language}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid source language: {source_language}"
            )
        if not self.validate_target_language(target_language):
            logger.error(f"Invalid target language: {target_language}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid target language: {target_language}"
            )

        logger.info(f"Using source language: {source_language}")
        logger.info(f"Using target language: {target_language}")
        logger.info(f"ElevenLabs params: {el_kwargs}")

        try:
            input_path = self.download_input_audio(
                request_iterator=request_iterator, context=context, request_id=request_id
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in streaming inputs: {request_id}: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"Error in streaming inputs: {e}\n{tb}")

        if not os.path.exists(input_path):
            logger.error(f"Error in streaming inputs: {request_id}: {input_path}.")
            context.abort(grpc.StatusCode.INTERNAL, f"Error in streaming inputs: {input_path}.")

        try:
            yield from self._impl(
                input_path=input_path,
                request_id=request_id,
                context=context,
                source_language=source_language,
                target_language=target_language,
                **el_kwargs,
            )
        except Exception as e:
            os.remove(input_path)
            tb = traceback.format_exc()
            logger.error(f"Stream back from client failed in request id {request_id}: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"Streamback failed: {e}\n{tb}")

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Factory method for creating an argument parser for the EL Dubbing service.

        Args:
            parser (argparse.ArgumentParser, optional): The argument parser to use.
                Defaults to None.

        Returns:
            argparse.ArgumentParser: The argument parser.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="ElevenLabs Speech-to-Speech Service")
        parser = S2SService.argsfactory(parser=parser)
        return parser
