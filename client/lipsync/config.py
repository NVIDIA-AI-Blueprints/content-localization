# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from dataclasses import dataclass

from client.common.audio import AUDIO_CODEC_CONFIGS
from common.media import is_file_available


@dataclass
class LipSyncConfig:
    """Configuration class for LipSync parameters.

    Attributes:
        audio_filepath: Path to input audio file
        video_filepath: Path to input video file
        speaker_info_filepath: Path to speaker info CSV file
        output_filepath: Path for output video
        extend_audio: How to handle audio extension
        extend_video: How to handle video extension
        bitrate_mbps: Output video bitrate in Mbps
        idr_interval: IDR frame interval
        lossless: Enable lossless video encoding
        audio_codec: Audio codec (mp3/wav) using AudioCodec from audio.v1
        is_speaker_info_provided: Whether speaker info is user-provided
        custom_encoding_params: Custom encoding parameters in JSON format
        background_audio_filepath: Path to background audio file (optional)
    """

    audio_filepath: os.PathLike
    video_filepath: os.PathLike
    speaker_info_filepath: os.PathLike
    output_filepath: os.PathLike
    extend_audio: str
    extend_video: str
    bitrate_mbps: int
    idr_interval: int
    lossless: bool
    audio_codec: str | None
    is_speaker_info_provided: bool | None
    custom_encoding_params: dict | None
    background_audio_filepath: os.PathLike | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LipSyncConfig":
        """Create config from command line arguments.

        ``is_speaker_info_provided`` honors an explicit
        ``--lipsync-is-speaker-info-provided`` flag; when the flag is
        absent it stays ``None`` so validation can derive it from the
        presence of a speaker-info file.

        Args:
            args (argparse.Namespace): Parsed CLI arguments from
                ``argsfactory``.

        Returns:
            LipSyncConfig: Populated configuration instance.

        Raises:
            ValueError: If ``--lipsync-custom-encoding-params`` is not
                valid JSON.

        Examples:
            >>> config = LipSyncConfig.from_args(args)  # doctest: +SKIP
        """
        # Parse custom encoding parameters if provided
        custom_params = None
        custom_json = getattr(args, "lipsync_custom_encoding_params", None)
        if custom_json:
            try:
                custom_params = json.loads(custom_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format for custom encoding parameters: {e}") from e

        # A store_true flag can only signal an explicit True; leave None
        # otherwise so validation derives it from the speaker-info file.
        explicit_speaker_info = getattr(args, "lipsync_is_speaker_info_provided", False)

        return cls(
            audio_filepath=args.input_audio,
            video_filepath=args.input_mp4,
            speaker_info_filepath=args.speaker_info_input,
            output_filepath=args.output_mp4,
            extend_audio=args.lipsync_extend_audio,
            extend_video=args.lipsync_extend_video,
            bitrate_mbps=args.lipsync_output_bitrate_mbps,
            idr_interval=args.lipsync_output_idr_interval,
            lossless=getattr(args, "lipsync_lossless", False),
            audio_codec=getattr(args, "lipsync_input_audio_codec", None),
            is_speaker_info_provided=True if explicit_speaker_info else None,
            custom_encoding_params=custom_params,
            background_audio_filepath=getattr(args, "background_audio_input", None),
        )

    def __str__(self) -> str:
        """Return string representation of config."""
        output = (
            "=" * 60
            + "\n"
            + "LipSync Configuration\n"
            + "=" * 60
            + "\n"
            + f"Video input      : {self.video_filepath}\n"
            + f"Audio input      : {self.audio_filepath}\n"
            + f"Speaker info file: {self.speaker_info_filepath}\n"
            + f"Audio codec      : {self.audio_codec}\n"
            + f"Extend audio     : {self.extend_audio}\n"
            + f"Extend video     : {self.extend_video}\n"
        )
        if self.lossless:
            output += "Encoding         : Lossless\n"
        elif self.custom_encoding_params:
            output += f"Encoding         : Custom parameters: {self.custom_encoding_params}\n"
        else:
            output += (
                f"Bitrate          : {self.bitrate_mbps} Mbps\n"
                f"IDR interval     : {self.idr_interval}\n"
            )
        output += (
            f"Output file      : {self.output_filepath}\n"
            f"Lossless         : {self.lossless}\n"
            f"Speaker info     : {self.is_speaker_info_provided}\n"
            f"Background audio : {self.background_audio_filepath or 'None'}\n" + "=" * 60
        )
        return output

    def validate_lipsync_config(self) -> bool:
        """Validate the lipsync configuration.

        Checks that input files exist with correct formats, the audio
        codec is supported, the speaker-info file format is valid when
        provided, and an explicit ``--lipsync-is-speaker-info-provided``
        flag is accompanied by a ``--speaker-info-input`` file.

        Returns:
            bool: ``True`` if validation passes.

        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If file formats are invalid or the speaker-info
                flag is set without a speaker-info file

        Examples:
            >>> config.validate_lipsync_config()  # doctest: +SKIP
            True
        """
        validation_success = False
        # Validate video file
        is_video_available = is_file_available(self.video_filepath, ["mp4"])
        if not is_video_available:
            raise RuntimeError("Only MP4 video format is supported")

        # Validate audio file
        is_audio_available = is_file_available(
            self.audio_filepath, list(AUDIO_CODEC_CONFIGS.keys())
        )
        if not is_audio_available:
            raise RuntimeError("Only WAV and MP3 audio formats are supported")
        file_audio_codec = os.path.splitext(self.audio_filepath)[1].lower().lstrip(".")
        if self.audio_codec is None:
            self.audio_codec = file_audio_codec
        else:
            self.audio_codec = self.audio_codec.lower()
            if self.audio_codec not in AUDIO_CODEC_CONFIGS:
                raise RuntimeError(
                    f"Unsupported audio codec: {self.audio_codec}. Supported codecs are: WAV, MP3"
                )

        # Validate speaker info file if provided
        if self.speaker_info_filepath:
            is_speaker_info_available = is_file_available(self.speaker_info_filepath, ["csv"])
            if not is_speaker_info_available:
                raise RuntimeError("Only CSV format is supported for speaker info file")
            self.is_speaker_info_provided = True
        elif self.is_speaker_info_provided:
            # The flag was passed explicitly but there is no data to send.
            raise RuntimeError(
                "--lipsync-is-speaker-info-provided requires --speaker-info-input "
                "so the speaker bounding boxes can be streamed to the service."
            )
        else:
            self.is_speaker_info_provided = False

        # Validate background audio file if provided
        if self.background_audio_filepath:
            is_bg_audio_available = is_file_available(
                self.background_audio_filepath, ["wav", "mp3"]
            )
            if not is_bg_audio_available:
                raise RuntimeError("Only WAV and MP3 formats are supported for background audio")

        validation_success = True
        return validation_success
