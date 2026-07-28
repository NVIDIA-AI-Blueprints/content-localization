# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the Controller pipeline.

``ControllerConfig`` groups every parameter needed to run a content
localization pipeline through the controller service: server address,
NIM protobuf configs, streaming chunk sizes, and (optionally) I/O paths.

I/O fields (``input_audio``, ``input_mp4``, ``output_mp4``,
``diarization_file``) default to ``None`` so the config can be used by
callers that manage their own I/O (e.g. the batch processing tool).
Call ``validate_io()`` to verify the I/O paths when they are set.
"""

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

MB = 1024 * 1024


@dataclass
class ControllerConfig:
    """Full configuration for a controller pipeline run.

    Required fields capture the pipeline execution parameters (server
    address, NIM configs, chunk sizes).  Optional I/O fields are only
    needed by the controller CLI client and can be left as ``None``
    when the caller manages its own file paths.

    Attributes:
        controller_server: Controller gRPC address (``host:port``).
        s2s_config: Speech-to-Speech protobuf configuration, or ``None``
            when S2S is bypassed (``--translated-audio`` provided).
        asd_config: Active Speaker Detection protobuf configuration,
            or ``None`` when ASD is bypassed (``--bypass-asd``).
        lipsync_config: LipSync protobuf configuration.
        chunk_size_audio_secs: Audio chunk duration in seconds.
        chunk_size_video_bytes: Video chunk size in bytes.
        input_audio: Path to input audio file (optional).
        input_mp4: Path to input video file (optional).
        output_mp4: Path to output video file (optional).
        diarization_file: Path to JSON diarization file (optional).
        background_audio_input: Path to background audio file for
            LipSync mixing (optional, WAV or MP3).
        translated_audio: Path to pre-translated audio file (WAV or
            MP3). When provided, S2S is bypassed and this audio is
            sent directly to LipSync (optional).
        explicit_lipsync_input_audio_codec: User-provided
            ``--lipsync-input-audio-codec`` value, if any.
        combine_chunks_per_speaker: When ``True`` (default), consecutive
            same-speaker diarization segments are merged into one before
            streaming. When ``False``, one segment per source unit (e.g.
            per word) is kept, enabling fine-grained diarization streaming.
        request_id: Correlation id stamped on every request message and
            echoed by the controller in responses. ``None`` lets the
            request generator create a UUID4 per request stream.

    Examples:
        >>> from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
        >>> from nvidia.ai4m.activespeakerdetection.v1 import (
        ...     activespeakerdetection_pb2,
        ... )
        >>> from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
        >>> cfg = ControllerConfig(
        ...     controller_server="localhost:50056",
        ...     s2s_config=SpeechToSpeechConfig(),
        ...     asd_config=(activespeakerdetection_pb2.ActiveSpeakerDetectionConfig()),
        ...     lipsync_config=LipsyncConfig(),
        ... )
        >>> cfg.controller_server
        'localhost:50056'
    """

    controller_server: str
    s2s_config: SpeechToSpeechConfig | None
    asd_config: ActiveSpeakerDetectionConfig | None
    lipsync_config: LipsyncConfig
    chunk_size_audio_secs: float = 1.0
    chunk_size_video_bytes: int = 1 * MB
    input_audio: str | None = None
    input_mp4: str | None = None
    output_mp4: str | None = None
    diarization_file: str | None = None
    background_audio_input: str | None = None
    translated_audio: str | None = None
    explicit_lipsync_input_audio_codec: str | None = None
    bypass_asd: bool = False
    diarization_rows_per_chunk: int | None = 10
    combine_chunks_per_speaker: bool = True
    request_id: str | None = None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        auto_bypass_asd: bool = True,
    ) -> "ControllerConfig":
        """Build a ``ControllerConfig`` from parsed CLI arguments.

        Builds NIM protobuf configs via ``s2s_config_from_args``,
        ``asd_config_from_args``, and ``lipsync_config_from_args``.
        When ``bypass_asd`` is True (either via ``args.bypass_asd``
        or auto-detected from missing diarization file),
        ``asd_config`` is set to ``None`` and
        ``lipsync_config.is_speaker_info_provided`` is forced to
        ``False``.  When translated audio is provided and
        ``--lipsync-input-audio-codec`` is omitted, the LipSync input
        codec is detected from the file content.  I/O fields are
        populated only when the corresponding attributes exist on
        *args*.

        Args:
            args (argparse.Namespace): Parsed argument namespace with
                controller, S2S, ASD, and LipSync attributes.  I/O
                attributes (``input_audio``, ``input_mp4``,
                ``output_mp4``, ``diarization_file``) are optional.
                ``diarization_chunked_per_segment`` (optional, default
                ``False``) is the inverse of
                ``combine_chunks_per_speaker``: when set, one diarization
                chunk per source segment is kept instead of merging
                consecutive same-speaker segments.
            auto_bypass_asd (bool): When ``True`` (default), bypass ASD
                automatically if no diarization file was provided.
                Callers that supply diarization later (e.g. the batch
                processing client) should pass ``False`` to keep ASD
                enabled unless ``--bypass-asd`` was set explicitly.

        Returns:
            ControllerConfig: Populated configuration instance.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     controller_server="localhost:50056",
            ...     chunk_size_audio_secs=1.0,
            ...     chunk_size_video_bytes=1048576,
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
            >>> cfg = ControllerConfig.from_args(args)
            >>> cfg.controller_server
            'localhost:50056'
        """
        bypass_asd = resolve_bypass_asd(args=args, auto_bypass_asd=auto_bypass_asd)

        asd_config = None if bypass_asd else asd_config_from_args(args)

        lipsync_config = lipsync_config_from_args(args)
        if bypass_asd:
            lipsync_config.is_speaker_info_provided = False

        translated_audio = getattr(args, "translated_audio", None)

        # Detect the actual audio codec from file content only when the
        # customer did not provide --lipsync-input-audio-codec. ElevenLabs
        # sometimes returns MP3 data inside a .wav filename.
        if translated_audio and getattr(args, "lipsync_input_audio_codec", None) is None:
            actual_codec = "wav" if is_wav_file(translated_audio) else "mp3"
            lipsync_config.input_audio_codec = AUDIO_CODEC_CONFIGS[actual_codec]

        # When translated audio is provided, S2S is bypassed — skip s2s_config
        s2s_config = None if translated_audio else s2s_config_from_args(args)

        # Normalize -1 to None (both mean "send all at once")
        raw_rows = getattr(args, "diarization_rows_per_chunk", None)
        if not isinstance(raw_rows, int) or raw_rows < 0:
            diarization_rows_per_chunk = None
        else:
            diarization_rows_per_chunk = raw_rows

        return cls(
            controller_server=args.controller_server,
            s2s_config=s2s_config,
            asd_config=asd_config,
            lipsync_config=lipsync_config,
            chunk_size_audio_secs=args.chunk_size_audio_secs,
            chunk_size_video_bytes=args.chunk_size_video_bytes,
            input_audio=getattr(args, "input_audio", None),
            input_mp4=getattr(args, "input_mp4", None),
            output_mp4=getattr(args, "output_mp4", None),
            diarization_file=getattr(args, "diarization_file", None),
            background_audio_input=getattr(args, "background_audio_input", None),
            translated_audio=translated_audio,
            explicit_lipsync_input_audio_codec=getattr(
                args,
                "lipsync_input_audio_codec",
                None,
            ),
            bypass_asd=bypass_asd,
            diarization_rows_per_chunk=diarization_rows_per_chunk,
            # CLI exposes the inverse (--diarization-chunked-per-segment); merging
            # by speaker stays the default when the flag is absent.
            combine_chunks_per_speaker=not getattr(args, "diarization_chunked_per_segment", False),
            request_id=getattr(args, "request_id", None),
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            str: Multi-line formatted string.

        Examples:
            >>> print(cfg)  # doctest: +SKIP
        """
        sep = "=" * 60
        lines = [
            sep,
            "Controller Configuration",
            sep,
            f"Server               : {self.controller_server}",
            f"Chunk size audio (s) : {self.chunk_size_audio_secs}",
            f"Chunk size video (B) : {self.chunk_size_video_bytes}",
        ]
        if self.input_audio is not None:
            lines.append(f"Input audio          : {self.input_audio}")
        if self.input_mp4 is not None:
            lines.append(f"Input video          : {self.input_mp4}")
        if self.output_mp4 is not None:
            lines.append(f"Output video         : {self.output_mp4}")
        lines.append(f"Diarization file     : {self.diarization_file or 'None'}")
        lines.append(f"Background audio     : {self.background_audio_input or 'None'}")
        lines.append(f"Translated audio     : {self.translated_audio or 'None'}")
        lines.append(f"Bypass S2S           : {self.translated_audio is not None}")
        lines.append(f"Bypass ASD           : {self.bypass_asd}")
        lines.append(sep)
        return "\n".join(lines)

    def validate_io(self) -> bool:
        """Validate I/O paths and chunk sizes.

        Checks that input audio and video files exist with supported
        formats, the output directory is writable, the diarization file
        is valid (when provided), and chunk sizes are positive.

        I/O path checks are skipped when the corresponding field is
        ``None``, so callers that manage their own files (e.g. the
        batch processing tool) can safely skip validation.

        Returns:
            bool: ``True`` if validation passes.

        Raises:
            RuntimeError: If input files are missing, formats are
                unsupported, or chunk sizes are non-positive.

        Examples:
            >>> cfg = ControllerConfig.from_args(args)
            >>> cfg.validate_io()
            True
        """
        if self.input_audio is not None and not is_file_available(self.input_audio, ["wav", "mp3"]):
            raise RuntimeError(
                f"Input audio file not found or unsupported"
                f" format: {self.input_audio}. "
                "Only WAV and MP3 formats are supported."
            )

        if self.input_mp4 is not None and not is_file_available(self.input_mp4, ["mp4"]):
            raise RuntimeError(
                f"Input video file not found or unsupported"
                f" format: {self.input_mp4}. "
                "Only MP4 format is supported."
            )

        if self.output_mp4 is not None:
            ensure_parent_dir(path=self.output_mp4)

        if self.diarization_file and not is_file_available(self.diarization_file, ["json", "csv"]):
            raise RuntimeError(
                f"Diarization file not found or unsupported"
                f" format: {self.diarization_file}. "
                "Only JSON and CSV formats are supported."
            )

        if self.background_audio_input is not None and not is_file_available(
            self.background_audio_input, ["wav", "mp3"]
        ):
            raise RuntimeError(
                f"Background audio file not found or unsupported"
                f" format: {self.background_audio_input}. "
                "Only WAV and MP3 formats are supported."
            )

        if self.translated_audio is not None and not is_file_available(
            self.translated_audio, ["wav", "mp3"]
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

    def validate_controller_config(self) -> bool:
        """Validate the configuration (alias for ``validate_io``).

        Retained for backward compatibility with existing callers.

        Returns:
            bool: ``True`` if validation passes.

        Raises:
            RuntimeError: If validation fails (see ``validate_io``).

        Examples:
            >>> cfg.validate_controller_config()  # doctest: +SKIP
            True
        """
        return self.validate_io()
