# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the ASD client."""

from dataclasses import dataclass

from client.common.paths import ensure_parent_dir
from common.media import is_file_available


@dataclass
class ASDConfig:
    """Configuration class for Active Speaker Detection client parameters.

    Attributes:
        asd_server: Address and port of the ASD gRPC service.
        input_mp4: Path to input video file (MP4; streamable MP4 recommended).
        input_audio: Path to input audio file.
        output_speaker_info: Path to output CSV file for speaker info data.
        chunk_size_video_bytes: Video chunk size in bytes.
        chunk_size_audio_secs: Audio chunk size in seconds.
        input_audio_codec: Audio codec for ASD input (WAV or MP3).
        input_video_codec: Video codec for ASD input (optional, e.g. H264).
        diarization_file: Path to optional JSON diarization file.
    """

    asd_server: str
    input_mp4: str
    input_audio: str
    output_speaker_info: str
    chunk_size_video_bytes: int
    chunk_size_audio_secs: float
    input_audio_codec: str
    input_video_codec: str | None
    diarization_file: str | None

    @classmethod
    def from_args(cls, args: object) -> "ASDConfig":
        """Create config from parsed command-line arguments.

        Args:
            args: Parsed argument namespace from ``argparse``.

        Returns:
            ASDConfig: Populated configuration instance.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     asd_server="localhost:50055",
            ...     input_mp4="video.mp4",
            ...     input_audio="audio.wav",
            ...     output_speaker_info="output.csv",
            ...     chunk_size_video_bytes=1048576,
            ...     chunk_size_audio_secs=1.0,
            ...     asd_input_audio_codec="WAV",
            ...     asd_input_video_codec=None,
            ...     diarization_file=None,
            ... )
            >>> config = ASDConfig.from_args(args)
            >>> config.input_audio_codec
            'WAV'
        """
        return cls(
            asd_server=args.asd_server,
            input_mp4=args.input_mp4,
            input_audio=args.input_audio,
            output_speaker_info=args.output_speaker_info,
            chunk_size_video_bytes=args.chunk_size_video_bytes,
            chunk_size_audio_secs=args.chunk_size_audio_secs,
            input_audio_codec=args.asd_input_audio_codec,
            input_video_codec=getattr(args, "asd_input_video_codec", None),
            diarization_file=getattr(args, "diarization_file", None),
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the ASD configuration."""
        sep = "=" * 60
        lines = [
            sep,
            "ASD Configuration",
            sep,
            f"Server               : {self.asd_server}",
            f"Input video          : {self.input_mp4}",
            f"Input audio          : {self.input_audio}",
            f"Output speaker info  : {self.output_speaker_info}",
            f"Chunk size video (B) : {self.chunk_size_video_bytes}",
            f"Chunk size audio (s) : {self.chunk_size_audio_secs}",
            f"Input audio codec    : {self.input_audio_codec}",
            f"Input video codec    : {self.input_video_codec or 'None'}",
            f"Diarization file     : {self.diarization_file or 'None'}",
            sep,
        ]
        return "\n".join(lines)

    def validate_asd_config(self) -> bool:
        """Validate the ASD configuration.

        Checks that input files exist, video is MP4, audio codec is supported,
        and diarization file is valid if provided.

        Returns:
            bool: ``True`` if validation passes.

        Raises:
            RuntimeError: If input files are missing or formats are unsupported.

        Examples:
            >>> config = ASDConfig.from_args(args)
            >>> config.validate_asd_config()
            True
        """
        if not is_file_available(self.input_mp4, ["mp4"]):
            raise RuntimeError(
                f"Input video file not found or unsupported format: {self.input_mp4}. "
                "Only MP4 format is supported."
            )

        if not is_file_available(self.input_audio, ["wav", "mp3"]):
            raise RuntimeError(
                f"Input audio file not found or unsupported format: {self.input_audio}. "
                "Only WAV and MP3 formats are supported."
            )

        if self.input_audio_codec.upper() not in ("WAV", "MP3"):
            raise RuntimeError(
                f"Unsupported audio codec: {self.input_audio_codec}. "
                "Only WAV and MP3 are supported."
            )

        if self.diarization_file and not is_file_available(self.diarization_file, ["json"]):
            raise RuntimeError(
                f"Diarization file not found or unsupported format: {self.diarization_file}. "
                "Only JSON format is supported."
            )

        ensure_parent_dir(path=self.output_speaker_info)

        return True
