# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the Direct client pipeline."""

import argparse
from dataclasses import dataclass

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from client.asd.args import asd_config_from_args
from client.common.audio import AUDIO_CODEC_CONFIGS
from client.common.bypass import resolve_bypass_asd
from client.common.paths import ensure_parent_dir
from client.lipsync.args import lipsync_config_from_args
from client.s2s.args import s2s_config_from_args
from common.audio_utils import is_wav_file
from common.media import is_file_available

KB = 1024
MB = 1024 * KB


@dataclass
class DirectPipelineConfig:
    """Grouped configuration for a direct-client pipeline run.

    Bundles per-service server addresses, protobuf NIM configs, and
    streaming chunk sizes into a single object so they can be
    constructed once and threaded through the direct client pipeline.

    Attributes:
        s2s_server: S2S gRPC address (``host:port``).
        asd_server: ASD gRPC address (``host:port``).
        lipsync_server: LipSync gRPC address (``host:port``).
        s2s_config: Speech-to-Speech protobuf configuration,
            or ``None`` when using pre-translated audio.
        asd_config: Active Speaker Detection protobuf configuration,
            or ``None`` when ASD is bypassed.
        lipsync_config: LipSync protobuf configuration.
        chunk_size_audio_secs: Audio chunk duration in seconds.
        chunk_size_video_bytes: Video chunk size in bytes.
        bypass_asd: If ``True``, skip ASD and use LipSync internal
            face detection instead.
        background_audio_input: Path to background audio file for
            LipSync mixing (optional, WAV or MP3).
        translated_audio: Path to pre-translated audio file (WAV or
            MP3). When provided, S2S is bypassed and this audio is
            sent directly to LipSync (optional).
        input_audio: Path to input audio file (optional).
        output_audio: Path to output audio file (optional).
        input_mp4: Path to input video file (optional).
        output_mp4: Path to output video file (optional).
        diarization_file: Path to the diarization file (optional).

    Examples:
        >>> from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
        >>> from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
        ...     ActiveSpeakerDetectionConfig,
        ... )
        >>> from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
        >>> cfg = DirectPipelineConfig(
        ...     s2s_server="localhost:50050",
        ...     asd_server="localhost:50055",
        ...     lipsync_server="localhost:50054",
        ...     s2s_config=SpeechToSpeechConfig(),
        ...     asd_config=ActiveSpeakerDetectionConfig(),
        ...     lipsync_config=LipsyncConfig(),
        ... )
        >>> cfg.s2s_server
        'localhost:50050'
    """

    s2s_server: str
    asd_server: str
    lipsync_server: str
    s2s_config: SpeechToSpeechConfig | None
    asd_config: ActiveSpeakerDetectionConfig | None
    lipsync_config: LipsyncConfig
    chunk_size_audio_secs: float = 1.0
    chunk_size_video_bytes: int = 1 * MB
    bypass_asd: bool = False
    background_audio_input: str | None = None
    translated_audio: str | None = None
    input_audio: str | None = None
    output_audio: str | None = None
    input_mp4: str | None = None
    output_mp4: str | None = None
    diarization_file: str | None = None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        auto_bypass_asd: bool = True,
    ) -> "DirectPipelineConfig":
        """Build a ``DirectPipelineConfig`` from parsed CLI arguments.

        Delegates to ``s2s_config_from_args``, ``asd_config_from_args``,
        and ``lipsync_config_from_args`` to construct the protobuf
        config messages.

        Args:
            args (argparse.Namespace): Parsed argument namespace with
                direct-client, S2S, ASD, and LipSync attributes.
            auto_bypass_asd (bool): When ``True`` (default), bypass ASD
                automatically if no diarization file was provided.

        Returns:
            DirectPipelineConfig: Populated configuration instance.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     s2s_server="localhost:50050",
            ...     asd_server="localhost:50055",
            ...     lipsync_server="localhost:50054",
            ...     chunk_size_audio_secs=1.0,
            ...     chunk_size_video_bytes=1048576,
            ...     bypass_asd=False,
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
            ...     asd_input_audio_codec="WAV",
            ...     asd_input_video_codec=None,
            ...     lipsync_input_audio_codec="MP3",
            ...     lipsync_extend_audio="unspecified",
            ...     lipsync_extend_video="unspecified",
            ...     lipsync_output_bitrate_mbps=20,
            ...     lipsync_output_idr_interval=8,
            ...     lipsync_head_movement_speed=None,
            ...     lipsync_output_audio_codec=None,
            ...     lipsync_is_speaker_info_provided=False,
            ... )
            >>> cfg = DirectPipelineConfig.from_args(args)
            >>> cfg.s2s_server
            'localhost:50050'
        """
        lipsync_config = lipsync_config_from_args(args)

        bypass_asd = resolve_bypass_asd(args=args, auto_bypass_asd=auto_bypass_asd)

        # When ASD is enabled, LipSync must know to expect speaker info
        if not bypass_asd:
            lipsync_config.is_speaker_info_provided = True

        translated_audio = getattr(args, "translated_audio", None)

        # Detect the actual audio codec from file content only when the
        # customer did not provide --lipsync-input-audio-codec. ElevenLabs
        # sometimes returns MP3 data inside a .wav filename.
        if translated_audio and getattr(args, "lipsync_input_audio_codec", None) is None:
            actual_codec = "wav" if is_wav_file(translated_audio) else "mp3"
            lipsync_config.input_audio_codec = AUDIO_CODEC_CONFIGS[actual_codec]

        s2s_config = None if translated_audio else s2s_config_from_args(args)

        asd_config = None if bypass_asd else asd_config_from_args(args)

        return cls(
            s2s_server=args.s2s_server,
            asd_server=args.asd_server,
            lipsync_server=args.lipsync_server,
            s2s_config=s2s_config,
            asd_config=asd_config,
            lipsync_config=lipsync_config,
            chunk_size_audio_secs=args.chunk_size_audio_secs,
            chunk_size_video_bytes=args.chunk_size_video_bytes,
            bypass_asd=bypass_asd,
            background_audio_input=getattr(args, "background_audio_input", None),
            translated_audio=translated_audio,
            input_audio=getattr(args, "input_audio", None),
            output_audio=getattr(args, "output_audio", None),
            input_mp4=getattr(args, "input_mp4", None),
            output_mp4=getattr(args, "output_mp4", None),
            diarization_file=getattr(args, "diarization_file", None),
        )

    def validate_io(self) -> bool:
        """Validate I/O paths and chunk sizes.

        Checks that input audio and video files exist with supported
        formats, output parent directories exist (creating them when
        needed), the diarization file is valid (when provided), and
        chunk sizes are positive.

        I/O path checks are skipped when the corresponding field is
        ``None``, so callers that manage their own files can safely
        skip validation.

        Returns:
            bool: ``True`` if validation passes.

        Raises:
            RuntimeError: If input files are missing, formats are
                unsupported, or chunk sizes are non-positive.

        Examples:
            >>> cfg = DirectPipelineConfig.from_args(args)
            >>> cfg.validate_io()
            True
        """
        if self.input_audio is not None and not is_file_available(
            file_path=self.input_audio, file_types=["wav", "mp3"]
        ):
            raise RuntimeError(
                f"Input audio file not found or unsupported"
                f" format: {self.input_audio}. "
                "Only WAV and MP3 formats are supported."
            )

        if self.input_mp4 is not None and not is_file_available(
            file_path=self.input_mp4, file_types=["mp4"]
        ):
            raise RuntimeError(
                f"Input video file not found or unsupported"
                f" format: {self.input_mp4}. "
                "Only MP4 format is supported."
            )

        if self.output_audio is not None:
            ensure_parent_dir(path=self.output_audio)

        if self.output_mp4 is not None:
            ensure_parent_dir(path=self.output_mp4)

        if self.diarization_file and not is_file_available(
            file_path=self.diarization_file, file_types=["json", "csv"]
        ):
            raise RuntimeError(
                f"Diarization file not found or unsupported"
                f" format: {self.diarization_file}. "
                "Only JSON and CSV formats are supported."
            )

        if self.background_audio_input is not None and not is_file_available(
            file_path=self.background_audio_input, file_types=["wav", "mp3"]
        ):
            raise RuntimeError(
                f"Background audio file not found or unsupported"
                f" format: {self.background_audio_input}. "
                "Only WAV and MP3 formats are supported."
            )

        if self.translated_audio is not None and not is_file_available(
            file_path=self.translated_audio, file_types=["wav", "mp3"]
        ):
            raise RuntimeError(
                f"Translated audio file not found or unsupported"
                f" format: {self.translated_audio}. "
                "Only WAV and MP3 formats are supported."
            )

        if self.chunk_size_audio_secs <= 0:
            raise RuntimeError(
                f"Audio chunk size must be positive, got {self.chunk_size_audio_secs}."
            )

        if self.chunk_size_video_bytes <= 0:
            raise RuntimeError(
                f"Video chunk size must be positive, got {self.chunk_size_video_bytes}."
            )

        return True
