# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch processing reporting: result storage, console output, JSON export."""

import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

from client.common.paths import ensure_parent_dir
from common.base_utils import logger

_SECONDS_PER_MINUTE = 60
KB = 1024
MB = KB * KB


@dataclass
class BatchResult:
    """Result of processing a single video through the pipeline.

    Attributes:
        video_name: Input video filename.
        video_duration_secs: Duration of the input video.
        video_width: Width of the input video in pixels.
        video_height: Height of the input video in pixels.
        video_frame_count: Total frames in the source video (from ffprobe packet count).
        preprocess_time_secs: Audio extraction time (mirrors ``stage_timings["preprocess"]``).
        diarization_time_secs: Diarization API call + parse time
            (mirrors ``stage_timings["diarization"]``).
        pipeline_time_secs: Controller gRPC call time
            (mirrors ``stage_timings["pipeline"]``).
        total_time_secs: Wall-clock time from first byte to last (includes all overhead).
        stage_timings: Per-stage timing dict produced by :class:`~client.common.timing.StageTimer`.
            Keys are stage names (``"preprocess"``, ``"diarization"``, ``"pipeline"``);
            values are elapsed seconds. Only stages that completed are present, so a failure
            mid-run shows partial timings. Serialised directly into the JSON report for
            downstream consumption by ``aggregate_perf.py``.
        output_path: Path to the output video file.
        output_size_bytes: Size of the output file in bytes.
        success: Whether the pipeline completed successfully.
        error_message: Error details if success is False.
    """

    video_name: str
    video_duration_secs: float
    video_width: int
    video_height: int
    video_frame_count: int
    preprocess_time_secs: float
    diarization_time_secs: float
    pipeline_time_secs: float
    total_time_secs: float
    stage_timings: dict[str, float] = field(default_factory=dict)
    output_path: str = ""
    output_size_bytes: int = 0
    success: bool = False
    error_message: str | None = None

    @property
    def realtime_factor(self) -> float:
        """Pipeline time divided by video duration.

        Returns:
            float: Real-time factor (< 1.0 = faster than real-time).

        Examples:
            >>> r = BatchResult(
            ...     video_name="v.mp4",
            ...     video_duration_secs=10.0,
            ...     video_width=1920,
            ...     video_height=1080,
            ...     video_frame_count=300,
            ...     preprocess_time_secs=1.0,
            ...     diarization_time_secs=2.0,
            ...     pipeline_time_secs=5.0,
            ...     total_time_secs=8.0,
            ...     output_path="o.mp4",
            ...     output_size_bytes=100,
            ...     success=True,
            ... )
            >>> r.realtime_factor
            0.5
        """
        if self.video_duration_secs <= 0:
            return 0.0
        return self.pipeline_time_secs / self.video_duration_secs


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted string (e.g. ``'1m 23.4s'``).

    Examples:
        >>> _fmt_duration(83.4)
        '1m 23.4s'
    """
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    minutes = int(seconds // _SECONDS_PER_MINUTE)
    remaining = seconds % _SECONDS_PER_MINUTE
    return f"{minutes}m {remaining:.1f}s"


def _fmt_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string.

    Args:
        size_bytes (int): Size in bytes.

    Returns:
        str: Formatted string (e.g. ``'12.3 MB'``).

    Examples:
        >>> _fmt_size(12_345_678)
        '11.8 MB'
    """
    if size_bytes < KB:
        return f"{size_bytes} B"
    if size_bytes < MB:
        return f"{size_bytes / KB:.1f} KB"
    return f"{size_bytes / MB:.1f} MB"


def print_report(results: list[BatchResult]) -> None:
    """Print a formatted batch processing summary to the console.

    Args:
        results (list[BatchResult]): Batch processing results.

    Examples:
        >>> print_report([result1, result2])  # doctest: +SKIP
    """
    sep = "=" * 72
    logger.info(f"\n{sep}")
    logger.info("BATCH PROCESSING REPORT")
    logger.info(sep)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    for result in results:
        status = "OK" if result.success else "FAIL"
        logger.info(f"\n  [{status}] {result.video_name}")
        logger.info(f"    Duration:    {_fmt_duration(result.video_duration_secs)}")
        logger.info(f"    Resolution:  {result.video_width}x{result.video_height}")
        logger.info(f"    Preprocess:  {_fmt_duration(result.preprocess_time_secs)}")
        logger.info(f"    Diarization: {_fmt_duration(result.diarization_time_secs)}")
        logger.info(f"    Pipeline:    {_fmt_duration(result.pipeline_time_secs)}")
        logger.info(f"    Total:       {_fmt_duration(result.total_time_secs)}")
        if result.success:
            logger.info(f"    RT factor:   {result.realtime_factor:.2f}x")
            logger.info(f"    Output size: {_fmt_size(result.output_size_bytes)}")
        else:
            logger.info(f"    Error:       {result.error_message}")

    logger.info(f"\n{sep}")
    logger.info("SUMMARY")
    logger.info(sep)
    logger.info(f"  Total videos:  {len(results)}")
    logger.info(f"  Successful:    {len(successful)}")
    logger.info(f"  Failed:        {len(failed)}")

    if successful:
        avg_rt = sum(r.realtime_factor for r in successful) / len(successful)
        total_dur = sum(r.video_duration_secs for r in successful)
        total_pipe = sum(r.pipeline_time_secs for r in successful)
        total_wall = sum(r.total_time_secs for r in successful)
        logger.info(f"  Avg RT factor: {avg_rt:.2f}x")
        logger.info(f"  Total input:   {_fmt_duration(total_dur)}")
        logger.info(f"  Total pipe:    {_fmt_duration(total_pipe)}")
        logger.info(f"  Total wall:    {_fmt_duration(total_wall)}")

    logger.info(f"{sep}\n")


def save_report(
    results: list[BatchResult],
    output_path: str,
) -> None:
    """Save batch processing results to a JSON file.

    Args:
        results (list[BatchResult]): Batch processing results.
        output_path (str): Path to the output JSON file.

    Examples:
        >>> save_report([result], "report.json")  # doctest: +SKIP
    """
    ensure_parent_dir(path=output_path)

    successful = [r for r in results if r.success]
    summary = {
        "total_videos": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_realtime_factor": (
            sum(r.realtime_factor for r in successful) / len(successful) if successful else 0.0
        ),
        "total_input_duration_secs": sum(r.video_duration_secs for r in successful),
        "total_diarization_time_secs": sum(r.diarization_time_secs for r in successful),
        "total_pipeline_time_secs": sum(r.pipeline_time_secs for r in successful),
        "total_wall_time_secs": sum(r.total_time_secs for r in successful),
    }

    data = {
        "results": [{**asdict(r), "realtime_factor": r.realtime_factor} for r in results],
        "summary": summary,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Report saved to: {output_path}")
