# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CambAI-based Speech-to-Speech (S2S) gRPC dubbing service.

Mirrors the ElevenLabs ``ELDubbingService`` pattern: transactional
file-based dubbing with queue+thread streaming and keepalive pings.
"""

import argparse
import os
import queue
import tempfile
import threading
import traceback
from collections.abc import Iterator
from pathlib import Path
from queue import Empty as QueueEmpty

import grpc
from google.protobuf.empty_pb2 import Empty
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.base_utils import logger
from s2s_service.camb_utils.api import download_output_audio_to_file
from s2s_service.camb_utils.api import get_alt_format_output_audio_url
from s2s_service.camb_utils.api import submit_dub_task
from s2s_service.camb_utils.api import upload_local_file
from s2s_service.camb_utils.api import wait_for_completion
from s2s_service.service import S2SService

# CambAI uses integer language IDs passed as strings (e.g. "1" for English).
# Users must provide the resolved numeric ID — no short-code lookup in the
# service. To get the full language-to-ID mapping, query the CambAI API:
#   GET https://client.camb.ai/apis/source-languages
#   GET https://client.camb.ai/apis/target-languages
# Docs: https://docs.camb.ai/api-reference/endpoint/get-source-languages
#       https://docs.camb.ai/api-reference/endpoint/get-target-languages
SUPPORTED_SOURCE_LANGUAGES = [
    "1",  # English (US)
    "2",  # Afrikaans (ZA)
    "3",  # Amharic (ET)
    "16",  # Arabic (SA)
    "20",  # Azerbaijani (AZ)
    "21",  # Bulgarian (BG)
    "22",  # Bengali (BD)
    "24",  # Bosnian (BA)
    "25",  # Catalan (ES)
    "26",  # Czech (CZ)
    "27",  # Welsh (GB)
    "28",  # Danish (DK)
    "31",  # German (DE)
    "32",  # Greek (GR)
    "54",  # Spanish (ES)
    "68",  # Estonian (EE)
    "69",  # Basque (ES)
    "70",  # Persian (IR)
    "71",  # Finnish (FI)
    "72",  # Filipino (PH)
    "76",  # French (FR)
    "77",  # Irish (IE)
    "78",  # Galician (ES)
    "79",  # Gujarati (IN)
    "80",  # Hebrew (IL)
    "81",  # Hindi (IN)
    "82",  # Croatian (HR)
    "83",  # Hungarian (HU)
    "84",  # Armenian (AM)
    "85",  # Indonesian (ID)
    "86",  # Icelandic (IS)
    "87",  # Italian (IT)
    "88",  # Japanese (JP)
    "89",  # Javanese (ID)
    "90",  # Georgian (GE)
    "91",  # Kazakh (KZ)
    "92",  # Khmer (KH)
    "93",  # Kannada (IN)
    "94",  # Korean (KR)
    "95",  # Lao (LA)
    "96",  # Lithuanian (LT)
    "97",  # Latvian (LV)
    "98",  # Macedonian (MK)
    "99",  # Malayalam (IN)
    "100",  # Mongolian (MN)
    "101",  # Marathi (IN)
    "102",  # Malay (MY)
    "103",  # Maltese (MT)
    "104",  # Burmese (MM)
    "105",  # Norwegian Bokmål (NO)
    "106",  # Nepali (NP)
    "108",  # Dutch (NL)
    "109",  # Polish (PL)
    "110",  # Pashto (AF)
    "111",  # Portuguese (BR)
    "113",  # Romanian (RO)
    "114",  # Russian (RU)
    "115",  # Sinhala (LK)
    "116",  # Slovak (SK)
    "117",  # Slovenian (SI)
    "118",  # Somali (SO)
    "119",  # Albanian (AL)
    "120",  # Serbian (RS)
    "121",  # Sundanese (ID)
    "122",  # Swedish (SE)
    "123",  # Swahili (KE)
    "125",  # Tamil (IN)
    "129",  # Telugu (IN)
    "130",  # Thai (TH)
    "131",  # Turkish (TR)
    "132",  # Ukrainian (UA)
    "134",  # Urdu (PK)
    "135",  # Uzbek (UZ)
    "136",  # Vietnamese (VN)
    "139",  # Chinese Mandarin (CN)
    "147",  # Zulu (ZA)
    "148",  # Punjabi (IN)
    "149",  # Sanskrit (IN)
    "151",  # Tagalog (PH)
]
SUPPORTED_TARGET_LANGUAGES = SUPPORTED_SOURCE_LANGUAGES.copy()


def _run_camb_pipeline(
    input_path: str,
    request_id: str,
    source_language: str,
    target_language: str,
    headers: dict[str, str],
    audio_queue: queue.Queue,
    *,
    chosen_dictionaries: list[int] | None = None,
    ai_optimization: bool = True,
) -> None:
    """Execute the full CambAI upload → dub → MP3 alt-format download pipeline.

    Runs on a background thread. Requests CambAI MP3 alt-format output,
    downloads it, then enqueues ``SpeechToSpeechResponse`` chunks followed by
    a ``"completed"`` sentinel, or an ``("error", detail)`` tuple on failure.

    Args:
        input_path (str): Path to the input audio file.
        request_id (str): The request ID for logging.
        source_language (str): CambAI source language ID string.
        target_language (str): CambAI target language ID string.
        headers (dict[str, str]): HTTP headers for CambAI API calls.
        audio_queue (queue.Queue): Queue for streaming results back.
        chosen_dictionaries (list[int] | None): CambAI dictionary IDs.
            Defaults to ``None``.
        ai_optimization (bool): Enable CambAI AI optimization.
            Defaults to ``True``.

    Examples:
        >>> q = queue.Queue()
        >>> _run_camb_pipeline("/tmp/in.wav", "r1", "1", "54", h, q)
    """
    dubbed_audio_path: Path | None = None
    try:
        dubbed_audio_path = _upload_dub_and_download(
            input_path=input_path,
            request_id=request_id,
            source_language=source_language,
            target_language=target_language,
            headers=headers,
            chosen_dictionaries=chosen_dictionaries,
            ai_optimization=ai_optimization,
        )
        _enqueue_audio_chunks(
            dubbed_audio_path=dubbed_audio_path,
            audio_queue=audio_queue,
        )
        os.remove(dubbed_audio_path)
        dubbed_audio_path = None
        audio_queue.put("completed")

    except Exception as e:
        logger.error(f"Error in CambAI audio processing thread: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        audio_queue.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        if dubbed_audio_path is not None and dubbed_audio_path.exists():
            try:
                os.remove(dubbed_audio_path)
            except Exception:
                logger.warning(f"Failed to clean up temporary dubbed file: {dubbed_audio_path}")


def _upload_dub_and_download(
    input_path: str,
    request_id: str,
    source_language: str,
    target_language: str,
    headers: dict[str, str],
    *,
    chosen_dictionaries: list[int] | None = None,
    ai_optimization: bool = True,
) -> Path:
    """Upload file, submit dub, poll, and download MP3 alt-format output.

    CambAI's default dubbing result is not used directly. This function
    requests the alt-format API with ``output_format="mp3"`` so downstream
    consumers receive the same container as the ElevenLabs S2S path.

    Args:
        input_path (str): Local audio file path to upload.
        request_id (str): Request ID for logging.
        source_language (str): CambAI source language ID.
        target_language (str): CambAI target language ID.
        headers (dict[str, str]): HTTP headers for CambAI API calls.
        chosen_dictionaries (list[int] | None): CambAI dictionary IDs.
            Defaults to ``None``.
        ai_optimization (bool): Enable CambAI AI optimization.
            Defaults to ``True``.

    Returns:
        Path: Path to the downloaded MP3 audio file.

    Examples:
        >>> _upload_dub_and_download("/tmp/in.wav", "r1", "1", "54", h)
        PosixPath('/tmp/tmpXXXX.mp3')
    """
    logger.debug(f"Uploading to CambAI for request id {request_id}")
    file_id = upload_local_file(file_path=Path(input_path), headers=headers)

    logger.debug(f"Submitting dub task for request id {request_id}")
    task_id = submit_dub_task(
        source_language_id=int(source_language),
        target_language_id=int(target_language),
        headers=headers,
        file_id=file_id,
        chosen_dictionaries=chosen_dictionaries,
        ai_optimization=ai_optimization,
    )
    logger.info(f"CambAI task_id={task_id} for request {request_id}")

    run_id = wait_for_completion(task_id=task_id, headers=headers)
    logger.info(f"CambAI dubbing completed run_id={run_id}")

    audio_url = get_alt_format_output_audio_url(
        run_id=run_id,
        language=target_language,
        headers=headers,
        output_format="mp3",
    )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="/tmp") as mp3_tmp:
        mp3_path = Path(mp3_tmp.name)

    download_output_audio_to_file(audio_url=audio_url, output_file=mp3_path)
    mp3_size = os.path.getsize(mp3_path)
    logger.info(f"Downloaded dubbed MP3: {mp3_size} bytes")
    return mp3_path


def _enqueue_audio_chunks(
    dubbed_audio_path: Path,
    audio_queue: queue.Queue,
) -> None:
    """Read an MP3 file in 8 KB chunks and enqueue as responses.

    The MP3 container carries its own audio metadata, so only ``audio_data`` and
    ``audio_format`` are populated on each ``SpeechToSpeechResponse``.

    Args:
        dubbed_audio_path (Path): Path to the MP3 file.
        audio_queue (queue.Queue): Queue to put chunks into.

    Examples:
        >>> _enqueue_audio_chunks(Path("/tmp/out.mp3"), q)
    """
    chunk_count = 0
    with dubbed_audio_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            chunk_count += 1
            logger.debug(f"Read chunk {chunk_count}: {len(chunk)} bytes")
            response = SpeechToSpeechResponse(
                audio_data=chunk,
                audio_format="mp3",
            )
            audio_queue.put(response)
    logger.info(f"Finished reading MP3 file: {chunk_count} chunks total")


def _extract_languages(
    config_request: SpeechToSpeechRequest,
    default_source: str,
    default_target: str,
) -> tuple[str, str]:
    """Extract source and target language from the config request.

    Args:
        config_request (SpeechToSpeechRequest): The first request message.
        default_source (str): Fallback source language ID.
        default_target (str): Fallback target language ID.

    Returns:
        tuple[str, str]: ``(source_language, target_language)``.

    Examples:
        >>> _extract_languages(req, "1", "54")
        ('1', '54')
    """
    if not config_request.HasField("config"):
        return default_source, default_target

    config = config_request.config
    source = config.source_language if config.HasField("source_language") else default_source
    target = config.target_language if config.HasField("target_language") else default_target
    return source, target


def _extract_camb_params(
    config_request: SpeechToSpeechRequest,
) -> dict:
    """Extract CambAI-specific params from the config request.

    Args:
        config_request (SpeechToSpeechRequest): The first request message.

    Returns:
        dict: Keyword arguments for ``_impl``
            (``chosen_dictionaries``, ``ai_optimization``).

    Examples:
        >>> _extract_camb_params(req)
        {'chosen_dictionaries': None, 'ai_optimization': True}
    """
    result: dict = {
        "chosen_dictionaries": None,
        "ai_optimization": True,
    }
    if not config_request.HasField("config"):
        return result

    config = config_request.config

    if config.HasField("camb_ai_optimization"):
        result["ai_optimization"] = config.camb_ai_optimization

    # repeated field — empty list means not set
    if len(config.camb_chosen_dictionaries) > 0:
        result["chosen_dictionaries"] = list(config.camb_chosen_dictionaries)

    return result


class CambDubbingService(S2SService):
    """Speech-to-Speech service using CambAI direct dubbing API.

    Transactional only — collects all input audio, uploads to CambAI,
    polls for completion, requests MP3 alt-format output, then streams
    the dubbed MP3 back.

    .. code-block:: text

        S2SServiceServicer (from service.py)
          |
          | 1. Extract request_id, wrap iterator
          | 2. Call: service.infer(request_iterator, context, request_id)
          v
        CambDubbingService
          |
          | 3. infer()
          |    |
          |    4. Extract config from first request
          |       (source_language, target_language as CambAI int ID strings)
          |       v
          |    5. self.download_input_audio()
          |       - Collects all audio data from the stream
          |       - Writes to a temp WAV file
          |       v
          |    6. self._impl(...)
          |       |
          |       |-- Background thread: _run_camb_pipeline() -------.
          |       |   upload_local_file() → submit_dub_task()        |
          |       |   → wait_for_completion()                        |
          |       |   → get_alt_format_output_audio_url()            |
          |       |   → download MP3                                 |
          |       |   → enqueue SpeechToSpeechResponse chunks        |
          |       |   + "completed" sentinel                         |
          |       |                                                  |
          |       `-- Main thread: read queue (timeout) -------------'
          |            - yield audio chunks
          |            - send keep-alive when queue is empty
          |
          v
        Return response stream to client

    """

    def validate_audio_format(self, value: str) -> bool:
        """Validate the audio format. CambAI outputs MP3 via alt-format API.

        Args:
            value (str): The audio format string.

        Returns:
            bool: ``True`` if the format is ``"mp3"``.

        Examples:
            >>> service.validate_audio_format("mp3")
            True
        """
        return value == "mp3"

    def __init__(
        self,
        message_size: int = 1024 * 1024 * 4,
        sample_rate_hz: int = 16000,
        default_source_language: str = "1",
        default_target_language: str = "54",
        audio_format: str = "mp3",
    ) -> None:
        """Initialize the CambAI dubbing S2S service.

        Args:
            message_size (int): Maximum gRPC message size in bytes.
                Defaults to ``4194304``.
            sample_rate_hz (int): Sample rate in Hz. Defaults to ``16000``.
            default_source_language (str): Default CambAI source language ID.
                Defaults to ``"1"`` (English).
            default_target_language (str): Default CambAI target language ID.
                Defaults to ``"54"`` (Spanish).
            audio_format (str): Output audio format. Defaults to ``"mp3"``.

        Raises:
            RuntimeError: If ``CAMB_API_KEY`` environment variable is not set.

        Examples:
            >>> service = CambDubbingService()
        """
        self._camb_api_key = os.getenv("CAMB_API_KEY")
        if not self._camb_api_key:
            raise RuntimeError("CAMB_API_KEY environment variable not set.")

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
            "Initialized CambAI S2S service for dubbing from %s to %s",
            default_source_language,
            default_target_language,
        )

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for CambAI API calls.

        Returns:
            dict[str, str]: Headers dict with ``x-api-key``.

        Examples:
            >>> service._build_headers()
            {'x-api-key': '...'}
        """
        return {"x-api-key": self._camb_api_key}

    def _impl(
        self,
        input_path: str,
        request_id: str,
        context: grpc.ServicerContext,
        source_language: str = "1",
        target_language: str = "54",
        *,
        chosen_dictionaries: list[int] | None = None,
        ai_optimization: bool = True,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Upload to CambAI, dub, and stream the dubbed audio back.

        Args:
            input_path (str): Path to the input audio file.
            request_id (str): The request ID.
            context (grpc.ServicerContext): The gRPC context.
            source_language (str): CambAI source language ID string.
                Defaults to ``"1"``.
            target_language (str): CambAI target language ID string.
                Defaults to ``"54"``.
            chosen_dictionaries (list[int] | None): CambAI dictionary
                IDs for custom terminology. Defaults to ``None``.
            ai_optimization (bool): Enable CambAI AI optimization.
                Defaults to ``True``.

        Yields:
            SpeechToSpeechResponse: Audio data chunks or keepalive pings.

        Examples:
            >>> list(service._impl("/tmp/in.wav", "req-1", ctx))
        """
        audio_queue: queue.Queue = queue.Queue()
        headers = self._build_headers()

        audio_thread = threading.Thread(
            target=_run_camb_pipeline,
            kwargs={
                "input_path": input_path,
                "request_id": request_id,
                "source_language": source_language,
                "target_language": target_language,
                "headers": headers,
                "audio_queue": audio_queue,
                "chosen_dictionaries": chosen_dictionaries,
                "ai_optimization": ai_optimization,
            },
            daemon=True,
        )
        audio_thread.start()

        yield from self._drain_queue(audio_queue=audio_queue, context=context)

        # Clean up the input file
        if os.path.exists(input_path):
            os.remove(input_path)

    def _drain_queue(
        self,
        audio_queue: queue.Queue,
        context: grpc.ServicerContext,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Yield responses from queue, sending keepalives on timeout.

        Args:
            audio_queue (queue.Queue): Queue of responses or sentinels.
            context (grpc.ServicerContext): The gRPC context.

        Yields:
            SpeechToSpeechResponse: Audio chunks or keepalive pings.

        Examples:
            >>> list(service._drain_queue(q, ctx))
        """
        keepalive_interval = int(os.environ.get("S2S_CAMB_KEEPALIVE_INTERVAL", "1"))
        response_count = 0
        keepalive_count = 0
        while True:
            try:
                response = audio_queue.get(timeout=keepalive_interval)
                if isinstance(response, tuple) and response[0] == "error":
                    context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"CambAI dubbing failed: {response[1]}",
                    )
                if response == "completed":
                    logger.info(
                        f"CambAI processing complete. Total responses: "
                        f"{response_count}, keepalives: {keepalive_count}"
                    )
                    break
                response_count += 1
                logger.debug(
                    f"Yielding audio response {response_count}: {len(response.audio_data)} bytes"
                )
                yield response
            except QueueEmpty:
                keepalive_count += 1
                logger.debug(f"Sending keepalive response {keepalive_count}")
                yield SpeechToSpeechResponse(keepalive=Empty())

    def infer(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
        source_language: str = "1",
        target_language: str = "54",
    ) -> Iterator[SpeechToSpeechResponse]:
        """Run CambAI dubbing pipeline.

        Extracts config (language IDs) from the first request, collects
        all audio, then delegates to ``_impl`` for upload+dub+stream.

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): Incoming
                audio stream.
            context (grpc.ServicerContext): The gRPC context.
            request_id (str): The request ID.
            source_language (str): Default source language ID.
                Defaults to ``"1"``.
            target_language (str): Default target language ID.
                Defaults to ``"54"``.

        Yields:
            SpeechToSpeechResponse: Dubbed audio chunks or keepalives.

        Examples:
            >>> list(service.infer(req_iter, ctx, "req-1"))
        """
        logger.info(f"Received request id: {request_id}")

        config_request = next(request_iterator)
        if config_request.audio_data:
            # First message has audio — replay it into the iterator
            original_request_iterator = request_iterator

            def replay_request_iterator() -> Iterator[SpeechToSpeechRequest]:
                yield config_request
                yield from original_request_iterator

            request_iterator = replay_request_iterator()

        source_language, target_language = _extract_languages(
            config_request=config_request,
            default_source=self.default_source_language,
            default_target=self.default_target_language,
        )
        camb_kwargs = _extract_camb_params(config_request)

        self._validate_languages(
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

        logger.info(f"Using CambAI source language ID: {source_language}")
        logger.info(f"Using CambAI target language ID: {target_language}")

        input_path = self._collect_input_audio(
            request_iterator=request_iterator,
            context=context,
            request_id=request_id,
        )

        try:
            yield from self._impl(
                input_path=input_path,
                request_id=request_id,
                context=context,
                source_language=source_language,
                target_language=target_language,
                **camb_kwargs,
            )
        except Exception as e:
            if os.path.exists(input_path):
                os.remove(input_path)
            tb = traceback.format_exc()
            logger.error(f"Stream back from client failed in request id {request_id}: {e}\n{tb}")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Streamback failed: {e}\n{tb}",
            )

    def _validate_languages(
        self,
        source_language: str,
        target_language: str,
        context: grpc.ServicerContext,
    ) -> None:
        """Validate source and target languages, aborting on invalid.

        Args:
            source_language (str): CambAI source language ID string.
            target_language (str): CambAI target language ID string.
            context (grpc.ServicerContext): gRPC context for aborting.

        Examples:
            >>> service._validate_languages("1", "54", ctx)
        """
        if not self.validate_source_language(source_language):
            logger.error(f"Invalid CambAI source language ID: {source_language}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid CambAI source language ID: {source_language}. "
                f"Must be a valid CambAI integer ID string.",
            )
        if not self.validate_target_language(target_language):
            logger.error(f"Invalid CambAI target language ID: {target_language}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid CambAI target language ID: {target_language}. "
                f"Must be a valid CambAI integer ID string.",
            )

    def _collect_input_audio(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> str:
        """Download input audio from the request stream.

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): Incoming
                audio stream.
            context (grpc.ServicerContext): The gRPC context.
            request_id (str): The request ID.

        Returns:
            str: Path to the downloaded audio file.

        Examples:
            >>> service._collect_input_audio(iter_, ctx, "req-1")
            '/tmp/tmpXXXX.mp3'
        """
        try:
            input_path = self.download_input_audio(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in streaming inputs: {request_id}: {e}\n{tb}")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Error in streaming inputs: {e}\n{tb}",
            )

        if not os.path.exists(input_path):
            logger.error(f"Error in streaming inputs: {request_id}: {input_path}.")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Error in streaming inputs: {input_path}.",
            )

        return input_path

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Factory method for creating an argument parser.

        Args:
            parser (argparse.ArgumentParser | None): Existing parser to extend.
                Defaults to ``None``.

        Returns:
            argparse.ArgumentParser: The argument parser with CambAI args.

        Examples:
            >>> parser = CambDubbingService.argsfactory()
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="CambAI Speech-to-Speech Service")
        parser = S2SService.argsfactory(parser=parser)
        return parser
