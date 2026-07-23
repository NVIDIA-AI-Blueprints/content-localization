# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch processing: run the full pipeline on every video in a directory."""

import os
import time
from pathlib import Path

from client.batch_processing.args import argsfactory
from client.batch_processing.diarization import ensure_diarization
from client.batch_processing.preprocessing import preprocess_video
from client.batch_processing.report import BatchResult
from client.batch_processing.report import print_report
from client.batch_processing.report import save_report
from client.batch_processing.runner import run_single_video
from client.common.audio import AUDIO_CODEC_CONFIGS
from client.common.diarization import load_diarization_info
from client.common.timing import StageTimer
from client.controller.config import ControllerConfig
from common.audio_utils import is_wav_file
from common.base_utils import logger
from common.health import check_service_health

VIDEO_EXTENSIONS = {".mp4"}


def discover_videos(input_dir: str) -> list[str]:
    """Find all video files in a directory, sorted by name.

    Args:
        input_dir (str): Directory to scan for video files.

    Returns:
        list[str]: Sorted list of absolute video file paths.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.

    Examples:
        >>> videos = discover_videos("videos/")
        >>> len(videos) > 0
        True
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]


def _diarization_format_for_service(s2s_service: str) -> str:
    """Return the diarization format for the given S2S service.

    Args:
        s2s_service (str): S2S service identifier
            (e.g. ``"CAMB_DUBBING"``).

    Returns:
        str: Diarization format name compatible with
            ``load_diarization_info``.

    Examples:
        >>> _diarization_format_for_service("CAMB_DUBBING")
        'camb'
        >>> _diarization_format_for_service("EL_DUBBING")
        'elevenlabs-scribe'
    """
    if s2s_service == "CAMB_DUBBING":
        return "camb"
    return "elevenlabs-scribe"


def _resolve_translated_audio(
    translated_audio_dir: str | None,
    stem: str,
    fallback_wav_path: str,
) -> str:
    """Resolve the pre-translated audio path for a bypass-S2S run.

    Args:
        translated_audio_dir (str | None): Directory holding
            ``{stem}.wav`` or ``{stem}.mp3`` pre-translated audio files.
            When ``None``, the source audio is used as a stand-in.
        stem (str): Video filename stem used to locate the file.
        fallback_wav_path (str): Extracted source-audio WAV to fall
            back to when no translated-audio directory is provided.

    Returns:
        str: Path to the audio file to stream in place of S2S output.

    Raises:
        FileNotFoundError: If a directory is given but no supported
            ``{stem}`` audio file is present.

    Examples:
        >>> _resolve_translated_audio(None, "clip", "clip.wav")
        'clip.wav'
    """
    if translated_audio_dir is None:
        # Perf stand-in: bypass timing measures ASD + LipSync, not
        # translation quality, so the source audio (matching duration)
        # is sufficient when no real translations are supplied.
        return fallback_wav_path
    for extension in ("wav", "mp3"):
        candidate = os.path.join(translated_audio_dir, f"{stem}.{extension}")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Translated audio not found for '{stem}' in {translated_audio_dir}. "
        f"Provide '{stem}.wav' or '{stem}.mp3', or omit --translated-audio-dir "
        "to use the source audio as a perf stand-in."
    )


def _set_lipsync_codec_for_translated_audio(
    config: ControllerConfig,
    translated_audio_path: str,
    explicit_input_audio_codec: str | None = None,
) -> None:
    """Set LipSync input codec for translated audio.

    An explicit customer-provided codec takes precedence. File-content
    detection is only used when the CLI flag was omitted.

    Args:
        config (ControllerConfig): Pipeline configuration whose
            ``lipsync_config`` will be updated.
        translated_audio_path (str): Resolved translated or fallback
            audio file streamed directly to LipSync.
        explicit_input_audio_codec (str | None): User-provided
            ``--lipsync-input-audio-codec`` value, if any.

    Examples:
        >>> _set_lipsync_codec_for_translated_audio(cfg, "clip.wav")  # doctest: +SKIP
    """
    if explicit_input_audio_codec is not None:
        codec = explicit_input_audio_codec.lower()
        config.lipsync_config.input_audio_codec = AUDIO_CODEC_CONFIGS[codec]
        logger.info(
            f"Using explicit {explicit_input_audio_codec.upper()} LipSync input codec "
            f"for translated audio: {translated_audio_path}"
        )
        return

    actual_codec = "wav" if is_wav_file(translated_audio_path) else "mp3"
    config.lipsync_config.input_audio_codec = AUDIO_CODEC_CONFIGS[actual_codec]
    logger.info(
        f"Using {actual_codec.upper()} LipSync input codec for translated audio: "
        f"{translated_audio_path}"
    )


def _process_single_video(
    video_path: str,
    output_dir: str,
    target_language: str,
    config: ControllerConfig,
    s2s_service: str = "EL_DUBBING",
    bypass_s2s: bool = False,
    translated_audio_dir: str | None = None,
    combine_chunks_per_speaker: bool = True,
) -> BatchResult:
    """Preprocess and run a single video, returning its result.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Base output directory.
        target_language (str): Target language code for naming.
        config (ControllerConfig): Pipeline configuration bundle.
        s2s_service (str): S2S backend identifier. Routes
            diarization to Camb AI when ``"CAMB_DUBBING"``.
        bypass_s2s (bool): When True, skip the S2S service and stream
            pre-translated audio directly to LipSync. ASD still runs.
        translated_audio_dir (str | None): Directory of ``{stem}.wav``
            pre-translated audio used when ``bypass_s2s`` is True. When
            None, the extracted source audio is used as a perf stand-in.
        combine_chunks_per_speaker (bool): When True (default), merge
            consecutive same-speaker diarization segments before
            streaming; when False, keep one chunk per source segment.

    Returns:
        BatchResult: Timing and status for this video.

    Examples:
        >>> result = _process_single_video(
        ...     video_path="v.mp4",
        ...     output_dir="out/",
        ...     target_language="de",
        ...     config=cfg,
        ... )  # doctest: +SKIP
    """
    video_name = os.path.basename(video_path)
    stem = Path(video_path).stem
    total_start = time.time()
    timer = StageTimer()

    try:
        # -- Preprocess --
        with timer.stage("preprocess"):
            wav_path, duration, video_width, video_height, video_frame_count = preprocess_video(
                video_path=video_path,
                output_dir=output_dir,
            )
        logger.info(
            f"[{video_name}] preprocessed: "
            f"duration={duration:.1f}s, resolution={video_width}x{video_height}, "
            f"frames={video_frame_count}"
        )

        # -- Translated audio (S2S bypass) --
        # Resolve up front so a missing pre-translated file fails fast, before
        # the expensive diarization call.
        translated_audio_path = (
            _resolve_translated_audio(
                translated_audio_dir=translated_audio_dir,
                stem=stem,
                fallback_wav_path=wav_path,
            )
            if bypass_s2s
            else None
        )
        if translated_audio_path is not None:
            _set_lipsync_codec_for_translated_audio(
                config=config,
                translated_audio_path=translated_audio_path,
                explicit_input_audio_codec=config.explicit_lipsync_input_audio_codec,
            )

        # -- Diarization --
        diarization_dir = os.path.join(output_dir, "diarization")
        diarization_format = _diarization_format_for_service(s2s_service)
        with timer.stage("diarization"):
            diarization_path = ensure_diarization(
                audio_path=wav_path,
                diarization_dir=diarization_dir,
                video_stem=stem,
                s2s_service=s2s_service,
            )
            diarization_info = load_diarization_info(
                diarization_file=diarization_path,
                diarization_format=diarization_format,
                combine_chunks_per_speaker=combine_chunks_per_speaker,
            )
        if diarization_info:
            logger.info(f"[{video_name}] diarization: {len(diarization_info.segments)} segments")

        # -- Pipeline --
        output_mp4 = os.path.join(output_dir, f"{stem}_{target_language}.mp4")
        with timer.stage("pipeline"):
            run_single_video(
                audio_path=wav_path,
                video_path=video_path,
                output_path=output_mp4,
                config=config,
                diarization_info=diarization_info,
                translated_audio_path=translated_audio_path,
            )

        output_size = os.path.getsize(output_mp4)
        total_time = time.time() - total_start
        timer.log_summary(label=f"[{video_name}]")

        timings = timer.as_dict()
        return BatchResult(
            video_name=video_name,
            video_duration_secs=duration,
            video_width=video_width,
            video_height=video_height,
            video_frame_count=video_frame_count,
            preprocess_time_secs=timings.get("preprocess", 0.0),
            diarization_time_secs=timings.get("diarization", 0.0),
            pipeline_time_secs=timings.get("pipeline", 0.0),
            total_time_secs=total_time,
            stage_timings=timings,
            output_path=output_mp4,
            output_size_bytes=output_size,
            success=True,
        )

    except Exception as exc:
        total_time = time.time() - total_start
        logger.exception(f"[{video_name}] FAILED: {exc}")
        timings = timer.as_dict()
        return BatchResult(
            video_name=video_name,
            video_duration_secs=0.0,
            video_width=0,
            video_height=0,
            video_frame_count=0,
            preprocess_time_secs=timings.get("preprocess", 0.0),
            diarization_time_secs=timings.get("diarization", 0.0),
            pipeline_time_secs=timings.get("pipeline", 0.0),
            total_time_secs=total_time,
            stage_timings=timings,
            output_path="",
            output_size_bytes=0,
            success=False,
            error_message=str(exc),
        )


def main() -> None:
    """Run the batch processing pipeline.

    Discovers videos in ``--input-dir``, preprocesses each one
    (audio extraction), runs it through the controller pipeline,
    and produces a batch processing report.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    args = argsfactory().parse_args()

    # Discover videos
    videos = discover_videos(args.input_dir)
    if not videos:
        logger.info(f"No video files found in {args.input_dir}")
        return
    logger.info(f"Found {len(videos)} video(s) in {args.input_dir}")

    # Check controller health
    check_service_health(server=args.controller_server)
    logger.info("Controller service is healthy")

    # Build pipeline config. Batch discovers a diarization file per video
    # later, so ASD must stay enabled instead of being auto-bypassed for the
    # missing CLI --diarization-file.
    pipeline_config = ControllerConfig.from_args(args=args, auto_bypass_asd=False)

    # In bypass-S2S mode the S2S config is unused; drop it so the controller
    # request stream signals bypass cleanly (ASD still runs on the original audio).
    if args.bypass_s2s:
        pipeline_config.s2s_config = None
        logger.info("S2S bypass enabled — streaming pre-translated audio to LipSync")
    elif args.translated_audio_dir is not None:
        # Guard against a silent no-op: the directory is only consulted in
        # bypass mode, so a user who forgot --bypass-s2s should be told.
        logger.warning(
            "--translated-audio-dir is ignored without --bypass-s2s; running the full S2S pipeline."
        )

    combine_chunks_per_speaker = not args.diarization_chunked_per_segment

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each video
    results: list[BatchResult] = []
    for idx, video_path in enumerate(videos, start=1):
        logger.info(
            f"\n{'=' * 72}\n[{idx}/{len(videos)}] {os.path.basename(video_path)}\n{'=' * 72}"
        )
        result = _process_single_video(
            video_path=video_path,
            output_dir=args.output_dir,
            target_language=args.target_language,
            config=pipeline_config,
            s2s_service=args.s2s_service,
            bypass_s2s=args.bypass_s2s,
            translated_audio_dir=args.translated_audio_dir,
            combine_chunks_per_speaker=combine_chunks_per_speaker,
        )
        results.append(result)

    # Report
    print_report(results)
    report_path = os.path.join(args.output_dir, "batch_processing_report.json")
    save_report(results=results, output_path=report_path)


if __name__ == "__main__":
    main()
