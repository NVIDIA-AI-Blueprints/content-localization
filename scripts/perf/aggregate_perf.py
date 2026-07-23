# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-config perf runs into a single comparison report.

Walks the output tree produced by ``scripts/perf/run_perf_matrix.sh`` and
merges every ``batch_processing_report.json`` (end-to-end timings) and S2S
latency JSON into one CSV + Markdown comparison, keyed by
``(config, diarization_mode, asset)``.

Derived columns:
  * ``s2s_contribution_secs`` — ``e2e(full) - e2e(bypass)`` per asset, an
    estimate of the S2S share of end-to-end latency.
  * the merged-per-speaker vs per-segment delta per ``(config, asset)``.

Pass ``--no-html`` to skip the interactive HTML dashboard (written by default).

Expected layout (per config: el | camb | bypass)::

    <in-dir>/<config>/combine/batch_processing_report.json
    <in-dir>/<config>/per_segment/batch_processing_report.json
    <in-dir>/<config>/s2s/<stem>.json            # el/camb only

Examples:
    $ python scripts/perf/aggregate_perf.py --in-dir outputs/perf
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys

# html_report lives in the same directory as this script; add the directory to
# sys.path so it can be imported regardless of the working directory.
sys.path.insert(0, os.path.dirname(__file__))
import html_report

# Human-readable backend label per config directory name.
_BACKEND_LABELS = {
    "el": "ElevenLabs",
    "camb": "Camb AI",
    "bypass": "Bypass S2S (ASD+LipSync)",
}
_DIAR_MODES = {
    "combine": "merged-per-speaker",
    "per_segment": "per-segment",
}


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not exist.

    Args:
        path (str): File path whose parent directory should exist.

    Examples:
        >>> _ensure_parent_dir("outputs/perf/perf_comparison.csv")  # doctest: +SKIP
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to float, falling back to *default* on bad input.

    Guards the downstream arithmetic (e.g. the S2S-contribution subtraction)
    against non-numeric values that may appear in a malformed report.

    Args:
        value (object): Value to coerce.
        default (float): Fallback when *value* is not numeric.

    Returns:
        float: The coerced value, or *default*.

    Examples:
        >>> _as_float("1.5")
        1.5
        >>> _as_float(None)
        0.0
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_json(path: str) -> dict | None:
    """Load a JSON file, returning ``None`` when it is absent.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict | None: Parsed JSON, or ``None`` if the file is missing or
            malformed (a warning is printed for malformed files so the
            aggregation continues over the remaining runs).

    Examples:
        >>> _load_json("missing.json") is None
        True
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"WARNING: skipping malformed JSON {path}: {exc}", file=sys.stderr)
        return None


def collect_rows(in_dir: str) -> list[dict]:
    """Collect per-(config, diarization_mode, asset) rows from a perf tree.

    Args:
        in_dir (str): Root directory holding ``<config>/<diar_mode>/`` runs.

    Returns:
        list[dict]: One row per processed asset, with timing and S2S fields.

    Examples:
        >>> rows = collect_rows("outputs/perf")  # doctest: +SKIP
    """
    rows: list[dict] = []
    for config in sorted(os.listdir(in_dir)):
        config_dir = os.path.join(in_dir, config)
        if not os.path.isdir(config_dir):
            continue

        # Standalone S2S latency, keyed by asset stem (el/camb only).
        s2s_by_stem: dict[str, dict] = {}
        s2s_dir = os.path.join(config_dir, "s2s")
        if os.path.isdir(s2s_dir):
            for fname in os.listdir(s2s_dir):
                if fname.endswith(".json"):
                    summary = _load_json(os.path.join(s2s_dir, fname))
                    if summary is not None:
                        s2s_by_stem[fname[: -len(".json")]] = summary

        for diar_mode, diar_label in _DIAR_MODES.items():
            mode_dir = os.path.join(config_dir, diar_mode)
            if not os.path.isdir(mode_dir):
                continue
            # Support both layouts: one report directly under the mode dir
            # (whole-directory batch run) and one report per asset subdir
            # (the one-video-per-process run that avoids the health-check race).
            report_paths = []
            direct = os.path.join(mode_dir, "batch_processing_report.json")
            if os.path.isfile(direct):
                report_paths.append(direct)
            report_paths += sorted(
                glob.glob(os.path.join(mode_dir, "*", "batch_processing_report.json"))
            )
            for report_path in report_paths:
                report = _load_json(report_path)
                if not isinstance(report, dict):
                    continue
                for result in report.get("results", []):
                    # Guard against a malformed entry so one bad row can't abort
                    # the whole aggregation.
                    if not isinstance(result, dict):
                        print(
                            f"WARNING: skipping non-dict result in {config}/{diar_mode}",
                            file=sys.stderr,
                        )
                        continue
                    stem = os.path.splitext(result.get("video_name", ""))[0]
                    s2s = s2s_by_stem.get(stem, {})
                    # Per-request ASD/LipSync FPS scraped into fps.json next to
                    # the report (only present for the one-video-per-process layout).
                    fps = _load_json(os.path.join(os.path.dirname(report_path), "fps.json")) or {}
                    lipsync_frames = fps.get("lipsync_frames")
                    e2e = _as_float(result.get("pipeline_time_secs"))
                    # Overall ASD+LipSync throughput = frames generated / pipeline
                    # wall time. Cleanest for bypass (pipeline is just ASD+LipSync);
                    # for el/camb the denominator also includes S2S.
                    overall_fps = (
                        round(lipsync_frames / e2e, 2)
                        if isinstance(lipsync_frames, int | float) and lipsync_frames and e2e
                        else None
                    )
                    width = result.get("video_width") or 0
                    height = result.get("video_height") or 0
                    resolution = f"{width}x{height}" if width and height else None
                    video_frame_count = result.get("video_frame_count")
                    # calc_fps: source video frames / wall-clock pipeline time.
                    # Measures true end-to-end throughput including gRPC/streaming
                    # overhead, for direct comparison with NIM-reported FPS.
                    calc_fps = (
                        round(video_frame_count / e2e, 2)
                        if isinstance(video_frame_count, int | float) and video_frame_count and e2e
                        else None
                    )
                    # Read diarization time from the named field; fall back to
                    # stage_timings dict for reports produced by older builds.
                    stage_timings = result.get("stage_timings") or {}
                    diarization_secs = result.get("diarization_time_secs") or _as_float(
                        stage_timings.get("diarization")
                    )
                    preprocess_secs = result.get("preprocess_time_secs") or _as_float(
                        stage_timings.get("preprocess")
                    )
                    rows.append(
                        {
                            "config": config,
                            "backend": _BACKEND_LABELS.get(config, config),
                            "diarization_mode": diar_label,
                            "asset": result.get("video_name", ""),
                            "resolution": resolution,
                            "duration_secs": _as_float(result.get("video_duration_secs")),
                            "video_frame_count": video_frame_count,
                            "e2e_pipeline_secs": e2e,
                            "preprocess_time_secs": preprocess_secs,
                            "diarization_time_secs": diarization_secs,
                            "realtime_factor": _as_float(result.get("realtime_factor")),
                            "success": result.get("success", False),
                            "s2s_wall_secs": s2s.get("wall_time_secs"),
                            "s2s_mean_per_chunk_latency": s2s.get("mean_per_chunk_latency"),
                            "s2s_is_realtime": s2s.get("is_realtime"),
                            "asd_fps": fps.get("asd_fps"),
                            "lipsync_fps": fps.get("lipsync_fps"),
                            "lipsync_frames": lipsync_frames,
                            "overall_asd_lipsync_fps": overall_fps,
                            "calc_fps": calc_fps,
                        }
                    )
    return rows


def add_s2s_contribution(rows: list[dict]) -> None:
    """Annotate rows with ``s2s_contribution_secs`` = e2e(full) - e2e(bypass).

    Uses the bypass run for the same ``(asset, diarization_mode)`` as the
    ASD+LipSync floor and subtracts it from each non-bypass config.

    Args:
        rows (list[dict]): Rows from :func:`collect_rows`, mutated in place.

    Examples:
        >>> rows = [
        ...     {
        ...         "config": "bypass",
        ...         "asset": "a.mp4",
        ...         "diarization_mode": "merged-per-speaker",
        ...         "e2e_pipeline_secs": 4.0,
        ...     },
        ...     {
        ...         "config": "el",
        ...         "asset": "a.mp4",
        ...         "diarization_mode": "merged-per-speaker",
        ...         "e2e_pipeline_secs": 10.0,
        ...     },
        ... ]
        >>> add_s2s_contribution(rows)
        >>> rows[1]["s2s_contribution_secs"]
        6.0
    """
    bypass_floor = {
        (r["asset"], r["diarization_mode"]): r["e2e_pipeline_secs"]
        for r in rows
        if r["config"] == "bypass"
    }
    for row in rows:
        floor = bypass_floor.get((row["asset"], row["diarization_mode"]))
        if row["config"] != "bypass" and floor is not None:
            row["s2s_contribution_secs"] = round(row["e2e_pipeline_secs"] - floor, 3)
        else:
            row["s2s_contribution_secs"] = None


_CSV_FIELDS = [
    "config",
    "backend",
    "diarization_mode",
    "asset",
    "resolution",
    "duration_secs",
    "preprocess_time_secs",
    "diarization_time_secs",
    "e2e_pipeline_secs",
    "realtime_factor",
    "s2s_contribution_secs",
    "s2s_wall_secs",
    "s2s_mean_per_chunk_latency",
    "s2s_is_realtime",
    "asd_fps",
    "lipsync_fps",
    "lipsync_frames",
    "video_frame_count",
    "overall_asd_lipsync_fps",
    "calc_fps",
    "success",
]


def build_summary_table(rows: list[dict]) -> str:
    """Build a per-config summary table over merged-per-speaker runs.

    Shows backend, asset count, success rate, average realtime factor, and
    average S2S contribution so the reader can compare configs at a glance
    without scanning every row.

    Args:
        rows (list[dict]): Annotated rows from :func:`collect_rows`.

    Returns:
        str: Markdown table (header + one row per config).

    Examples:
        >>> build_summary_table([])
        '_No data._'
    """
    combine_rows = [r for r in rows if r["diarization_mode"] == "merged-per-speaker"]
    if not combine_rows:
        return "_No data._"
    configs = sorted({r["config"] for r in combine_rows})
    headers = ["backend", "assets", "success", "avg_rt_factor", "avg_s2s_contribution_secs"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for config in configs:
        cfg_rows = [r for r in combine_rows if r["config"] == config]
        n = len(cfg_rows)
        successes = sum(1 for r in cfg_rows if r.get("success"))
        rt_vals = [r["realtime_factor"] for r in cfg_rows if r.get("realtime_factor")]
        avg_rt = round(sum(rt_vals) / len(rt_vals), 2) if rt_vals else None
        s2s_vals = [
            r["s2s_contribution_secs"]
            for r in cfg_rows
            if r.get("s2s_contribution_secs") is not None
        ]
        avg_s2s = round(sum(s2s_vals) / len(s2s_vals), 2) if s2s_vals else None
        backend = _BACKEND_LABELS.get(config, config)
        lines.append(f"| {backend} | {n} | {successes}/{n} | {_fmt(avg_rt)} | {_fmt(avg_s2s)} |")
    return "\n".join(lines)


def build_fps_table(rows: list[dict]) -> str:
    """Build the ASD + LipSync FPS comparison table as Markdown.

    Shows two FPS measurements side-by-side for each run:

    * **nim_lipsync_fps** — FPS reported by the LipSync NIM internally
      (scraped from Docker logs). Measures pure inference throughput.
    * **calc_fps** — ``source_frames / pipeline_wall_time``. Measures true
      end-to-end throughput including gRPC and streaming overhead.

    The ``nim/calc`` ratio reveals how much of the NIM's raw throughput is
    consumed by non-inference overhead; values near 100% mean the pipeline
    is nearly as fast as the NIM alone.

    Args:
        rows (list[dict]): Annotated rows from :func:`collect_rows`.

    Returns:
        str: Markdown table (header + rows), or a note if no FPS data exists.

    Examples:
        >>> build_fps_table([])
        '_No ASD/LipSync FPS captured._'
    """
    fps_rows = [
        r for r in rows if r.get("lipsync_fps") is not None or r.get("calc_fps") is not None
    ]
    if not fps_rows:
        return "_No ASD/LipSync FPS captured._"
    headers = [
        "config",
        "asset",
        "diarization_mode",
        "src_frames",
        "nim_asd_fps",
        "nim_lipsync_fps",
        "calc_fps",
        "nim/calc",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in sorted(fps_rows, key=lambda r: (r["asset"], r["config"], r["diarization_mode"])):
        nim = r.get("lipsync_fps")
        calc = r.get("calc_fps")
        ratio = (
            f"{nim / calc * 100:.0f}%"
            if isinstance(nim, int | float) and isinstance(calc, int | float) and calc
            else "-"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(r.get("config")),
                    _fmt(r.get("asset")),
                    _fmt(r.get("diarization_mode")),
                    _fmt(r.get("video_frame_count")),
                    _fmt(r.get("asd_fps")),
                    _fmt(nim),
                    _fmt(calc),
                    ratio,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(rows: list[dict], output_path: str) -> None:
    """Write the comparison rows to a CSV file.

    Args:
        rows (list[dict]): Annotated rows.
        output_path (str): Destination CSV path.

    Examples:
        >>> write_csv(rows, "outputs/perf/perf_comparison.csv")  # doctest: +SKIP
    """
    _ensure_parent_dir(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _CSV_FIELDS})


def _fmt(value: object) -> str:
    """Format a cell value for the Markdown table.

    Args:
        value (object): Cell value (number, bool, str, or None).

    Returns:
        str: Display string; floats to 2 decimals, ``None`` as ``"-"``.

    Examples:
        >>> _fmt(1.234)
        '1.23'
        >>> _fmt(None)
        '-'
    """
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown(rows: list[dict], output_path: str) -> None:
    """Write the comparison rows and granularity deltas as Markdown.

    Produces a structured report with a summary table, full per-asset
    end-to-end timings, diarization-granularity deltas, and GPU throughput,
    each preceded by an explanatory note for easier interpretation.

    Args:
        rows (list[dict]): Annotated rows.
        output_path (str): Destination Markdown path.

    Examples:
        >>> write_markdown(rows, "outputs/perf/perf_comparison.md")  # doctest: +SKIP
    """
    _ensure_parent_dir(output_path)
    configs_present = sorted({r["config"] for r in rows})
    n_assets = len({r["asset"] for r in rows if r["diarization_mode"] == "merged-per-speaker"})
    generated = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# CLBP Performance Report",
        "",
        f"Generated: {generated} | Configs: {', '.join(configs_present)} | Assets: {n_assets}",
        "",
        "> **RT factor** = pipeline time / video duration. "
        "Values below 1.0 indicate faster-than-real-time processing; "
        "lower is better. `s2s_contribution_secs` = e2e(full) - e2e(bypass), "
        "isolating the Speech-to-Speech share of pipeline latency.",
        "",
        "## Summary (merged-per-speaker diarization)",
        "",
        build_summary_table(rows),
        "",
    ]

    # Full per-asset table.
    e2e_headers = [
        "config",
        "asset",
        "resolution",
        "duration_secs",
        "diarization_mode",
        "preprocess_time_secs",
        "diarization_time_secs",
        "e2e_pipeline_secs",
        "realtime_factor",
        "s2s_contribution_secs",
        "s2s_wall_secs",
        "s2s_is_realtime",
        "success",
    ]
    lines += [
        "## End-to-End Performance",
        "",
        "All runs across both diarization modes. "
        "`s2s_wall_secs` and `s2s_is_realtime` are from the standalone S2S "
        "latency run (el/camb only).",
        "",
        "| " + " | ".join(e2e_headers) + " |",
        "| " + " | ".join("---" for _ in e2e_headers) + " |",
    ]
    for row in sorted(rows, key=lambda r: (r["asset"], r["config"], r["diarization_mode"])):
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in e2e_headers) + " |")

    # Diarization-granularity delta per (config, asset).
    lines += [
        "",
        "## Diarization Granularity Delta",
        "",
        "Difference in e2e pipeline time when using per-segment vs merged-per-speaker "
        "diarization. A positive delta means per-segment is slower.",
        "",
        "| config | asset | merged_secs | per_segment_secs | delta_secs |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        by_key.setdefault((row["config"], row["asset"]), {})[row["diarization_mode"]] = row[
            "e2e_pipeline_secs"
        ]
    for (config, asset), modes in sorted(by_key.items()):
        merged = modes.get("merged-per-speaker")
        per_seg = modes.get("per-segment")
        delta = round(per_seg - merged, 3) if (merged is not None and per_seg is not None) else None
        lines.append(f"| {config} | {asset} | {_fmt(merged)} | {_fmt(per_seg)} | {_fmt(delta)} |")

    # ASD + LipSync FPS (per-stage and overall throughput).
    lines += [
        "",
        "## GPU Throughput: ASD + LipSync",
        "",
        "`nim_lipsync_fps` is scraped from Docker logs (NIM-internal inference rate). "
        "`calc_fps` = source frames / pipeline wall time (includes gRPC + streaming overhead). "
        "`nim/calc` shows how much of the NIM's raw throughput is preserved end-to-end; "
        "values near 100% mean the pipeline adds negligible overhead.",
        "",
        build_fps_table(rows),
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    """Aggregate perf runs into CSV, Markdown, and (optionally) HTML reports.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="Aggregate perf-matrix runs.")
    parser.add_argument(
        "--in-dir",
        type=str,
        default="outputs/perf",
        help="Root directory of per-config perf runs (default: outputs/perf).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output path prefix (default: <in-dir>/perf_comparison).",
    )
    parser.add_argument(
        "--no-html",
        dest="html",
        action="store_false",
        default=True,
        help="Skip HTML dashboard generation.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        raise FileNotFoundError(f"Input directory not found: {args.in_dir}")

    rows = collect_rows(in_dir=args.in_dir)
    if not rows:
        print(f"No batch_processing_report.json found under {args.in_dir}")
        return
    add_s2s_contribution(rows=rows)

    out_prefix = args.out_prefix or os.path.join(args.in_dir, "perf_comparison")
    csv_path = f"{out_prefix}.csv"
    md_path = f"{out_prefix}.md"
    write_csv(rows=rows, output_path=csv_path)
    write_markdown(rows=rows, output_path=md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    if args.html:
        html_path = f"{out_prefix}.html"
        html_report.write_html(rows=rows, output_path=html_path)
        print(f"Wrote {html_path}")

    # Also print the ASD + LipSync FPS table to the console.
    print("\n## ASD + LipSync FPS\n")
    print(build_fps_table(rows))


if __name__ == "__main__":
    main()
