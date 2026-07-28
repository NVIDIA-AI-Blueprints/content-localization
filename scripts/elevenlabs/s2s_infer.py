#!/bin/python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke ElevenLabs end-to-end dubbing for local media and save translated audio output."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elevenlabs import ElevenLabs

ELEVENLABS_TRANSCRIPT_FORMATS = ("json", "srt", "webvtt", "vtt")


@dataclass(frozen=True)
class DubbingResult:
    """Metadata produced by a completed ElevenLabs dubbing run."""

    output_path: Path
    dubbing_id: str


def convert_video_to_audio_ffmpeg(
    video_file: Path, audio_file: Path, sample_rate_hz: int = 16000
) -> None:
    """Convert a video file to WAV audio using ``ffmpeg``.

    Extracts mono PCM audio at the given sample rate via the ``ffmpeg``
    command-line tool.

    Args:
        video_file (Path): Input video file path.
        audio_file (Path): Output WAV audio file path.
        sample_rate_hz (int): Sampling rate for output audio. Defaults to ``16000``.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``video_file`` does not exist.
        subprocess.CalledProcessError: If ``ffmpeg`` exits with a non-zero status.

    Examples:
        >>> convert_video_to_audio_ffmpeg(Path("input.mp4"), Path("output.wav"))
    """
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"Input video file {video_file} not found.")
    print(f"Converting {video_file} to {audio_file}.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        f"{video_file}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        f"{sample_rate_hz}",
        "-acodec",
        "pcm_s16le",
        f"{audio_file}",
    ]
    subprocess.run(cmd, check=True)  # noqa: S603


def download_dubbed_file(
    client: ElevenLabs, dubbing_id: str, language_code: str, output_path: Path
) -> None:
    """Download the dubbed file for a given dubbing ID and language code.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs client instance.
        dubbing_id (str): The ID of the dubbing project.
        language_code (str): The language code for the dubbing.
        output_path (Path): Destination file path for the downloaded audio.

    Returns:
        None.

    Examples:
        >>> download_dubbed_file(client, "dub-123", "es", Path("output.wav"))
    """
    with open(output_path, "wb") as file:
        for chunk in client.dubbing.audio.get(dubbing_id, language_code):
            file.write(chunk)


def wait_for_dubbing_completion(client: ElevenLabs, dubbing_id: str) -> bool:
    """Wait for the dubbing process to complete by polling status.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs client instance.
        dubbing_id (str): The dubbing project ID.

    Returns:
        bool: ``True`` if the dubbing succeeded, ``False`` if it failed or timed out.

    Examples:
        >>> wait_for_dubbing_completion(client, "dub-123")
        True
    """
    max_attempts = 120
    check_interval = 10  # In seconds

    for _ in range(max_attempts):
        metadata = client.dubbing.get(dubbing_id)
        status = (metadata.status or "").lower()

        if status == "dubbed":
            return True
        elif status in {"dubbing", "queued", "in_progress", "processing", "pending", "created"}:
            print(
                "Dubbing status:",
                metadata.status,
                "- will check again in",
                check_interval,
                "seconds.",
            )
            time.sleep(check_interval)
        elif status in {"failed", "error", "cancelled"}:
            print(f"Dubbing failed with status={metadata.status}: {metadata.error}")
            return False
        else:
            print(f"Unknown dubbing status={metadata.status}; continuing to poll for completion.")
            time.sleep(check_interval)

    print("Dubbing timed out")
    return False


def create_dub_from_file(  # noqa: PLR0913
    client: ElevenLabs,
    input_file_path: Path,
    file_format: str,
    source_language: str,
    target_language: str,
    output_file_path: Path,
) -> DubbingResult | None:
    """Dub an audio or video file from one language to another and save the output.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs client instance.
        input_file_path (Path): The file path of the audio or video to dub.
        file_format (str): The MIME type of the input file (e.g. ``"audio/wav"``).
        source_language (str): The language of the input file.
        target_language (str): The target language to dub into.
        output_file_path (Path): The file path of the output file.

    Returns:
        DubbingResult | None: The dubbed file path and dubbing ID, or ``None``
        if the operation failed.

    Examples:
        >>> create_dub_from_file(
        ...     client, Path("input.wav"), "audio/wav", "en", "es", Path("output.wav")
        ... )
        DubbingResult(output_path=PosixPath('output.wav'), dubbing_id='dub-123')
    """
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    with open(input_file_path, "rb") as audio_file:
        response = client.dubbing.create(
            file=(os.path.basename(input_file_path), audio_file, file_format),
            target_lang=target_language,
            mode="automatic",
            source_lang=source_language,
        )

    dubbing_id = response.dubbing_id
    if wait_for_dubbing_completion(client, dubbing_id):
        download_dubbed_file(
            client=client,
            dubbing_id=dubbing_id,
            language_code=target_language,
            output_path=output_file_path,
        )
        return DubbingResult(output_path=output_file_path, dubbing_id=dubbing_id)
    else:
        return None


def normalize_elevenlabs_transcript_format(transcript_format: str) -> str:
    """Normalize CLI transcript format aliases to ElevenLabs API values.

    Args:
        transcript_format (str): CLI transcript format value.

    Returns:
        str: ElevenLabs API format value.

    Raises:
        ValueError: If ``transcript_format`` is unsupported.
    """
    normalized = transcript_format.lower()
    if normalized == "vtt":
        return "webvtt"
    if normalized in {"json", "srt", "webvtt"}:
        return normalized
    raise ValueError(f"Unsupported ElevenLabs transcript format: {transcript_format}")


def to_json_serializable(value: Any) -> Any:
    """Convert SDK response objects to JSON-serializable Python values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_serializable(item) for item in value]
    return value


def extract_elevenlabs_transcript_content(response: object, transcript_format: str) -> Any:
    """Extract the requested transcript payload from an ElevenLabs SDK response.

    Args:
        response (object): ElevenLabs transcript response model.
        transcript_format (str): Normalized transcript format (``json``, ``srt``,
            or ``webvtt``).

    Returns:
        Any: A JSON-serializable object for JSON transcripts or text for subtitles.

    Raises:
        RuntimeError: If the response does not contain the expected transcript field.
    """
    normalized_format = normalize_elevenlabs_transcript_format(transcript_format)

    if normalized_format == "json":
        json_payload = getattr(response, "json_", None)
        if json_payload is not None:
            return to_json_serializable(json_payload)

        dumped = to_json_serializable(response)
        if isinstance(dumped, dict):
            for key in ("json", "json_"):
                value = dumped.get(key)
                if value is not None:
                    return to_json_serializable(value)
            if "utterances" in dumped or "language" in dumped:
                return dumped

        raise RuntimeError("ElevenLabs transcript response missing JSON payload")

    transcript_text = getattr(response, normalized_format, None)
    if isinstance(transcript_text, str):
        return transcript_text
    if isinstance(response, str):
        return response

    dumped = to_json_serializable(response)
    if isinstance(dumped, dict) and isinstance(dumped.get(normalized_format), str):
        return dumped[normalized_format]

    raise RuntimeError(f"ElevenLabs transcript response missing {normalized_format} payload")


def write_transcript_content(content: Any, output_path: Path) -> Path:
    """Write JSON or text transcript content to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        output_path.write_text(content, encoding="utf-8")
    else:
        output_path.write_text(
            json.dumps(content, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return output_path


def download_dubbing_transcript(
    client: ElevenLabs,
    dubbing_id: str,
    language_code: str,
    transcript_format: str,
    output_path: Path,
) -> Path:
    """Fetch a transcript for an ElevenLabs dubbing run and write it to disk.

    The user-supplied ``transcript_format`` is normalized via
    :func:`normalize_elevenlabs_transcript_format` to the format token the
    ElevenLabs SDK expects, the response payload is unwrapped via
    :func:`extract_elevenlabs_transcript_content`, and the resulting JSON or
    text is persisted with :func:`write_transcript_content`.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs SDK client.
        dubbing_id (str): Dubbing job ID returned by ``create_dub_from_file``.
        language_code (str): Language code of the transcript to fetch — pass
            the source language code for the source transcript and the target
            language code for the target transcript.
        transcript_format (str): Requested transcript format. Accepts the
            user-facing aliases ``"json"``, ``"srt"``, ``"vtt"``, and
            ``"webvtt"`` (case-insensitive); normalized internally before the
            API call.
        output_path (Path): Destination path. Parent directories are created
            by ``write_transcript_content`` if they do not yet exist.

    Returns:
        Path: ``output_path`` after the transcript has been written.

    Raises:
        ValueError: If ``transcript_format`` is not a supported alias (raised
            from :func:`normalize_elevenlabs_transcript_format`).
        RuntimeError: If the SDK response does not contain the expected
            transcript payload (raised from
            :func:`extract_elevenlabs_transcript_content`).
        Exception: Any HTTP/SDK error from
            ``client.dubbing.transcripts.get`` or filesystem error from
            :func:`write_transcript_content` propagates to the caller.

    Examples:
        >>> from pathlib import Path
        >>> path = download_dubbing_transcript(
        ...     client=client,
        ...     dubbing_id=result.dubbing_id,
        ...     language_code="es",
        ...     transcript_format="srt",
        ...     output_path=Path("outputs/transcript.srt"),
        ... )
        >>> print(path)
        outputs/transcript.srt
    """
    normalized_format = normalize_elevenlabs_transcript_format(transcript_format)
    response = client.dubbing.transcripts.get(
        dubbing_id=dubbing_id,
        language_code=language_code,
        format_type=normalized_format,
    )
    content = extract_elevenlabs_transcript_content(response, normalized_format)
    return write_transcript_content(content=content, output_path=output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ElevenLabs dubbing pipeline.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``source_language_code``: ``"en"``
            - ``target_language_code``: ``"es"``
            - ``output_file``: ``"output.wav"``

    Examples:
        >>> args = parse_args()  # with appropriate sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Speech translation service using ElevenLabs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Path to a local audio or video file to dub.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default="output.wav",
        required=True,
        help="Output file .wav file to write audio.",
    )
    parser.add_argument(
        "--source-language-code",
        default="en",
        help="Language code of the source input.",
    )
    parser.add_argument(
        "--target-language-code",
        default="es",
        help="Language code of the target language.",
    )
    parser.add_argument(
        "--source-transcript-output-file",
        type=Path,
        default=None,
        help="Optional output file for the source-language transcript.",
    )
    parser.add_argument(
        "--target-transcript-output-file",
        type=Path,
        default=None,
        help="Optional output file for the target-language transcript.",
    )
    parser.add_argument(
        "--transcript-format",
        choices=ELEVENLABS_TRANSCRIPT_FORMATS,
        default="json",
        help="Transcript format to fetch when transcript output files are provided.",
    )
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    if args.output_file is not None:
        args.output_file = args.output_file.expanduser()
    if args.source_transcript_output_file is not None:
        args.source_transcript_output_file = args.source_transcript_output_file.expanduser()
    if args.target_transcript_output_file is not None:
        args.target_transcript_output_file = args.target_transcript_output_file.expanduser()
    return args


def main() -> None:
    """Run ElevenLabs end-to-end dubbing flow and save translated audio.

    Reads ``ELEVENLABS_API_KEY`` from the environment, extracts audio from
    the input video, submits a dubbing request, and writes the result.

    Returns:
        None.

    Raises:
        ValueError: If ``ELEVENLABS_API_KEY`` is not set.

    Examples:
        >>> # CLI usage:
        >>> # python scripts/elevenlabs/s2s_infer.py --input-file assets/sample_audio.wav -o output.wav
    """
    start_time = time.time()
    args = parse_args()

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")
    client = ElevenLabs(api_key=elevenlabs_api_key)

    _, extracted_audio = tempfile.mkstemp(prefix="audio_", suffix=".wav")
    extracted_audio_path = Path(extracted_audio)
    convert_video_to_audio_ffmpeg(args.input_file, extracted_audio_path)

    result = create_dub_from_file(
        client=client,
        input_file_path=extracted_audio_path,
        file_format="audio/wav",
        source_language=args.source_language_code,
        target_language=args.target_language_code,
        output_file_path=args.output_file,
    )
    if result:
        print(f"Dubbing was successful! File saved at: {result.output_path}.")
        if args.source_transcript_output_file:
            source_transcript_path = download_dubbing_transcript(
                client=client,
                dubbing_id=result.dubbing_id,
                language_code=args.source_language_code,
                transcript_format=args.transcript_format,
                output_path=args.source_transcript_output_file,
            )
            print(f"Source transcript saved at: {source_transcript_path}.")
        if args.target_transcript_output_file:
            target_transcript_path = download_dubbing_transcript(
                client=client,
                dubbing_id=result.dubbing_id,
                language_code=args.target_language_code,
                transcript_format=args.transcript_format,
                output_path=args.target_transcript_output_file,
            )
            print(f"Target transcript saved at: {target_transcript_path}.")
        print(f"Time taken for invocation: {time.time() - start_time}")
    else:
        print("Dubbing failed or timed out.")


if __name__ == "__main__":
    main()
