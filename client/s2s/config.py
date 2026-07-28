# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the S2S client."""

from dataclasses import dataclass

from client.common.paths import ensure_parent_dir
from common.media import is_file_available


def _parse_camb_dictionaries(raw: str | None) -> list[int] | None:
    """Parse a comma-separated string of dictionary IDs into a list.

    Args:
        raw: Comma-separated IDs (e.g. ``"1,5,12"``) or ``None``.

    Returns:
        list[int] | None: Parsed list of IDs, or ``None`` if *raw*
            is ``None`` or empty.

    Examples:
        >>> _parse_camb_dictionaries("1,5,12")
        [1, 5, 12]
        >>> _parse_camb_dictionaries(None) is None
        True
    """
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass
class S2SConfig:
    """Configuration class for Speech-to-Speech client parameters.

    Attributes:
        s2s_server: Address and port of the S2S gRPC service.
        input_audio: Path to input audio file.
        output_audio: Path to output audio file.
        chunk_size_audio_secs: Audio chunk size in seconds for streaming.
        source_language: Source language code (e.g. "en", "auto").
        target_language: Target language code (e.g. "de").
        voice_name: Optional voice name for TTS.
        elevenlabs_num_speakers: Number of speakers (0 = auto-detect).
        elevenlabs_drop_background_audio: Drop background audio from final dub.
        elevenlabs_use_profanity_filter: Censor profanities in transcripts.
        elevenlabs_target_accent: Experimental accent to apply.
        elevenlabs_highest_resolution: Use highest resolution output.
        elevenlabs_watermark: Apply watermark to output.
        elevenlabs_dubbing_studio: Prepare dub for editing in dubbing studio.
        camb_ai_optimization: Enable CambAI AI optimization.
        camb_chosen_dictionaries: CambAI dictionary IDs for custom
            terminology, or ``None`` if unused.
    """

    s2s_server: str
    input_audio: str
    output_audio: str
    chunk_size_audio_secs: float
    source_language: str
    target_language: str
    voice_name: str | None
    elevenlabs_num_speakers: int
    elevenlabs_drop_background_audio: bool
    elevenlabs_use_profanity_filter: bool
    elevenlabs_target_accent: str | None
    elevenlabs_highest_resolution: bool
    elevenlabs_watermark: bool
    elevenlabs_dubbing_studio: bool
    camb_ai_optimization: bool
    camb_chosen_dictionaries: list[int] | None

    @classmethod
    def from_args(cls, args: object) -> "S2SConfig":
        """Create config from parsed command-line arguments.

        Args:
            args: Parsed argument namespace from ``argparse``.

        Returns:
            S2SConfig: Populated configuration instance.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     s2s_server="localhost:50050",
            ...     input_audio="in.wav",
            ...     output_audio="out.mp3",
            ...     chunk_size_audio_secs=1.0,
            ...     source_language="en",
            ...     target_language="de",
            ...     voice_name=None,
            ...     elevenlabs_num_speakers=0,
            ...     elevenlabs_drop_background_audio=False,
            ...     elevenlabs_use_profanity_filter=False,
            ...     elevenlabs_target_accent=None,
            ...     elevenlabs_highest_resolution=False,
            ...     elevenlabs_watermark=False,
            ...     elevenlabs_dubbing_studio=False,
            ...     camb_ai_optimization=True,
            ...     camb_chosen_dictionaries=None,
            ... )
            >>> config = S2SConfig.from_args(args)
            >>> config.target_language
            'de'
        """
        return cls(
            s2s_server=args.s2s_server,
            input_audio=args.input_audio,
            output_audio=args.output_audio,
            chunk_size_audio_secs=args.chunk_size_audio_secs,
            source_language=args.source_language,
            target_language=args.target_language,
            voice_name=args.voice_name,
            elevenlabs_num_speakers=getattr(args, "elevenlabs_num_speakers", 0),
            elevenlabs_drop_background_audio=getattr(
                args, "elevenlabs_drop_background_audio", False
            ),
            elevenlabs_use_profanity_filter=getattr(args, "elevenlabs_use_profanity_filter", False),
            elevenlabs_target_accent=getattr(args, "elevenlabs_target_accent", None),
            elevenlabs_highest_resolution=getattr(args, "elevenlabs_highest_resolution", False),
            elevenlabs_watermark=getattr(args, "elevenlabs_watermark", False),
            elevenlabs_dubbing_studio=getattr(args, "elevenlabs_dubbing_studio", False),
            camb_ai_optimization=getattr(args, "camb_ai_optimization", True),
            camb_chosen_dictionaries=_parse_camb_dictionaries(
                getattr(args, "camb_chosen_dictionaries", None)
            ),
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the S2S configuration."""
        sep = "=" * 60
        lines = [
            sep,
            "S2S Configuration",
            sep,
            f"Server           : {self.s2s_server}",
            f"Input audio      : {self.input_audio}",
            f"Output audio     : {self.output_audio}",
            f"Chunk size (s)   : {self.chunk_size_audio_secs}",
            f"Source language   : {self.source_language}",
            f"Target language   : {self.target_language}",
            f"Voice name       : {self.voice_name or 'None'}",
            "",
            "ElevenLabs Parameters",
            "-" * 40,
            f"Num speakers             : {self.elevenlabs_num_speakers}",
            f"Drop background audio    : {self.elevenlabs_drop_background_audio}",
            f"Use profanity filter     : {self.elevenlabs_use_profanity_filter}",
            f"Target accent            : {self.elevenlabs_target_accent or 'None'}",
            f"Highest resolution       : {self.elevenlabs_highest_resolution}",
            f"Watermark                : {self.elevenlabs_watermark}",
            f"Dubbing studio           : {self.elevenlabs_dubbing_studio}",
            "",
            "CambAI Parameters",
            "-" * 40,
            f"AI optimization          : {self.camb_ai_optimization}",
            f"Chosen dictionaries      : {self.camb_chosen_dictionaries}",
            sep,
        ]
        return "\n".join(lines)

    def validate_s2s_config(self) -> bool:
        """Validate the S2S configuration.

        Checks that input audio exists, output directory is writable,
        and language fields are non-empty.

        Returns:
            bool: ``True`` if validation passes.

        Raises:
            FileNotFoundError: If the input audio file does not exist.
            RuntimeError: If the audio format is unsupported or languages are empty.

        Examples:
            >>> config = S2SConfig.from_args(args)
            >>> config.validate_s2s_config()
            True
        """
        if not is_file_available(self.input_audio, ["wav", "mp3"]):
            raise RuntimeError(
                f"Input audio file not found or unsupported format: {self.input_audio}. "
                "Only WAV and MP3 formats are supported."
            )

        ensure_parent_dir(path=self.output_audio)

        if not self.source_language:
            raise RuntimeError("Source language must not be empty.")
        if not self.target_language:
            raise RuntimeError("Target language must not be empty.")

        return True
