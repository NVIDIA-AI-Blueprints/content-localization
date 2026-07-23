# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate repeated CLBP perf runs into averaged side-by-side report inputs.

Each ``--dirs`` entry holds one machine's repeated-run outputs (``run1``,
``run2``, ...). The machine label shown in reports is the directory basename,
so name each directory after the machine that produced it.

Examples:
    $ source .venv/bin/activate
    $ python scripts/perf/aggregate_repeated_perf.py \
        --dirs outputs/perf_avg3/machine-a outputs/perf_avg3/machine-b \
        --out-dir outputs/combined-perf-report-avg3
"""

import argparse
import csv
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from statistics import median
from statistics import stdev

_RAW_FIELDS = [
    "machine",
    "run",
    "clip_length",
    "clip_sort_secs",
    "config",
    "diarization_mode",
    "input_variant",
    "source_video",
    "output_dir",
    "duration_secs",
    "video_frame_count",
    "preprocess_secs",
    "diarization_secs",
    "e2e_secs",
    "realtime_factor",
    "pipeline_fps",
    "asd_fps",
    "lipsync_fps",
    "lipsync_frames",
    "s2s_contribution_secs",
    "success",
]

_AVG_FIELDS = [
    "machine",
    "clip_length",
    "clip_sort_secs",
    "config",
    "diarization_mode",
    "input_variant",
    "run_count",
    "duration_secs_mean",
    "video_frame_count_mean",
    "preprocess_secs_mean",
    "diarization_secs_mean",
    "e2e_secs_mean",
    "e2e_secs_stddev",
    "realtime_factor_mean",
    "realtime_factor_stddev",
    "pipeline_fps_mean",
    "pipeline_fps_stddev",
    "asd_fps_mean",
    "asd_fps_stddev",
    "lipsync_fps_mean",
    "lipsync_fps_stddev",
    "lipsync_frames_mean",
    "s2s_contribution_secs_mean",
    "s2s_contribution_secs_stddev",
    "success_count",
]

# Per-machine metric columns in the side-by-side CSV, in output order. Each
# entry maps the column suffix to the averaged-row source field and an optional
# standard-deviation source field.
_SIDE_METRICS: list[tuple[str, str, str | None]] = [
    ("asd_fps", "asd_fps_mean", "asd_fps_stddev"),
    ("lipsync_fps", "lipsync_fps_mean", "lipsync_fps_stddev"),
    ("pipeline_fps", "pipeline_fps_mean", "pipeline_fps_stddev"),
    ("realtime_factor", "realtime_factor_mean", "realtime_factor_stddev"),
    ("e2e_secs", "e2e_secs_mean", "e2e_secs_stddev"),
    ("preprocess_secs", "preprocess_secs_mean", None),
    ("diarization_secs", "diarization_secs_mean", None),
    ("s2s_wall_secs", "s2s_contribution_secs_mean", "s2s_contribution_secs_stddev"),
    ("s2s_contribution_secs", "s2s_contribution_secs_mean", "s2s_contribution_secs_stddev"),
    ("video_frame_count", "video_frame_count_mean", None),
]

# Column suffixes that identify a per-machine column in a side-by-side CSV.
# Longest suffixes first so ``<label>_asd_fps_stddev`` resolves before
# ``<label>_asd_fps``.
_LABEL_COLUMN_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {"input_variant"}
        | {suffix for suffix, _, _ in _SIDE_METRICS}
        | {f"{suffix}_stddev" for suffix, _, stddev in _SIDE_METRICS if stddev},
        key=len,
        reverse=True,
    )
)

# Raw-row fields recoverable from one machine's side-by-side columns when a
# baseline CSV is folded in as run 0.
_BASELINE_NUMERIC_FIELDS = [
    "video_frame_count",
    "preprocess_secs",
    "diarization_secs",
    "e2e_secs",
    "realtime_factor",
    "pipeline_fps",
    "asd_fps",
    "lipsync_fps",
    "s2s_contribution_secs",
]


def side_by_side_fields(labels: list[str]) -> list[str]:
    """Return the side-by-side CSV field order for the given machine labels.

    Args:
        labels (list[str]): Machine labels in column order.

    Returns:
        list[str]: CSV field names.

    Examples:
        >>> side_by_side_fields(["m1"])[:4]
        ['clip_length', 'config', 'diarization_mode', 'm1_input_variant']
    """
    fields = ["clip_length", "config", "diarization_mode"]
    fields.extend(f"{label}_input_variant" for label in labels)
    fields.append("duration_secs")
    for suffix, _, stddev_field in _SIDE_METRICS:
        for label in labels:
            fields.append(f"{label}_{suffix}")
            if stddev_field is not None:
                fields.append(f"{label}_{suffix}_stddev")
    return fields


def machine_labels_from_fields(fieldnames: list[str]) -> list[str]:
    """Recover machine labels from side-by-side CSV column names.

    Args:
        fieldnames (list[str]): CSV header fields.

    Returns:
        list[str]: Machine labels in first-seen column order.

    Examples:
        >>> machine_labels_from_fields(["clip_length", "m1_asd_fps", "m2_asd_fps"])
        ['m1', 'm2']
    """
    labels: list[str] = []
    for field in fieldnames:
        for suffix in _LABEL_COLUMN_SUFFIXES:
            marker = f"_{suffix}"
            if field.endswith(marker) and len(field) > len(marker):
                label = field[: -len(marker)]
                if label not in labels:
                    labels.append(label)
                break
    return labels


_OUTLIER_MAD_Z_THRESHOLD = 3.5
_MIN_OUTLIER_SAMPLE_COUNT = 4


def _load_json(path: Path) -> dict:
    """Load a JSON object from disk.

    Args:
        path (Path): JSON path.

    Returns:
        dict: Parsed JSON object, or an empty dict when loading fails.

    Examples:
        >>> _load_json(Path("missing.json"))
        {}
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"WARNING: skipping malformed JSON {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _num(value: object) -> float | None:
    """Convert a value to float when possible.

    Args:
        value (object): Raw value.

    Returns:
        float | None: Numeric value, or ``None``.

    Examples:
        >>> _num("1.5")
        1.5
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _clip_seconds(label: str) -> float:
    """Return a sortable duration for a clip label.

    Args:
        label (str): Duration label.

    Returns:
        float: Seconds, or a high value for unknown labels.

    Examples:
        >>> _clip_seconds("18m48s")
        1128.0
    """
    text = label.strip().lower()
    if text.endswith("min"):
        return float(text[:-3]) * 60
    if text.endswith("s") and "m" not in text:
        return float(text[:-1])
    match = re.fullmatch(r"(\d+)m(\d+)s", text)
    if match:
        return float(match.group(1)) * 60 + float(match.group(2))
    return 1_000_000.0


def _run_index(run_dir: Path) -> int:
    """Extract the run number from a ``runN`` directory.

    Args:
        run_dir (Path): Run directory.

    Returns:
        int: Run index, or zero for unexpected names.

    Examples:
        >>> _run_index(Path("run3"))
        3
    """
    match = re.fullmatch(r"run(\d+)", run_dir.name)
    return int(match.group(1)) if match else 0


def _clip_label(stem: str) -> str:
    """Extract the duration label from a staged output stem.

    Args:
        stem (str): Output directory or video stem.

    Returns:
        str: Duration-only label.

    Examples:
        >>> _clip_label("10s_abcd_sample_aac_0010s")
        '10s'
    """
    return stem.split("_", maxsplit=1)[0]


def collect_rows(machine_dir: Path, machine: str, input_variant: str) -> list[dict[str, object]]:
    """Collect raw rows for one machine.

    Args:
        machine_dir (Path): Root containing ``runN/<config>/combine`` outputs.
        machine (str): Machine label, normally the directory basename.
        input_variant (str): Input normalization label for this machine.

    Returns:
        list[dict[str, object]]: One row per run/config/clip.

    Examples:
        >>> collect_rows(Path("missing"), "machine-a", "aac")
        []
    """
    rows: list[dict[str, object]] = []
    if not machine_dir.is_dir():
        return rows
    for run_dir in sorted(machine_dir.glob("run*"), key=_run_index):
        if not run_dir.is_dir():
            continue
        run = _run_index(run_dir)
        for config_dir in sorted(run_dir.iterdir()):
            if not config_dir.is_dir() or config_dir.name not in {"bypass", "el"}:
                continue
            combine_dir = config_dir / "combine"
            for report_path in sorted(combine_dir.glob("*/batch_processing_report.json")):
                run_out = report_path.parent
                if run_out.name == "_inputs":
                    continue
                report = _load_json(report_path)
                fps = _load_json(run_out / "fps.json")
                source_video_path = run_out / "source_video.txt"
                source_video = (
                    source_video_path.read_text(encoding="utf-8").strip()
                    if source_video_path.is_file()
                    else ""
                )
                for result in report.get("results", []):
                    if not isinstance(result, dict):
                        continue
                    stem = Path(str(result.get("video_name", run_out.name))).stem
                    clip = _clip_label(stem)
                    e2e_secs = _num(result.get("pipeline_time_secs"))
                    frame_count = _num(result.get("video_frame_count"))
                    pipeline_fps = (
                        round(frame_count / e2e_secs, 2)
                        if frame_count is not None and e2e_secs
                        else None
                    )
                    rows.append(
                        {
                            "machine": machine,
                            "run": run,
                            "clip_length": clip,
                            "clip_sort_secs": _clip_seconds(clip),
                            "config": config_dir.name,
                            "diarization_mode": "merged-per-speaker",
                            "input_variant": input_variant,
                            "source_video": source_video,
                            "output_dir": str(run_out),
                            "duration_secs": _num(result.get("video_duration_secs")),
                            "video_frame_count": frame_count,
                            "preprocess_secs": _num(result.get("preprocess_time_secs")),
                            "diarization_secs": _num(result.get("diarization_time_secs")),
                            "e2e_secs": e2e_secs,
                            "realtime_factor": _num(result.get("realtime_factor")),
                            "pipeline_fps": pipeline_fps,
                            "asd_fps": _num(fps.get("asd_fps")),
                            "lipsync_fps": _num(fps.get("lipsync_fps")),
                            "lipsync_frames": _num(fps.get("lipsync_frames")),
                            "s2s_contribution_secs": None,
                            "success": bool(result.get("success")),
                        }
                    )
    _add_s2s_contribution(rows)
    return rows


def collect_baseline_rows(path: Path, run: int = 0) -> list[dict[str, object]]:
    """Collect run rows from an existing side-by-side CSV.

    Machine labels are recovered from the CSV column names, so the baseline may
    contain any set of machines.

    Args:
        path (Path): Existing side-by-side CSV.
        run (int): Synthetic run index assigned to these rows.

    Returns:
        list[dict[str, object]]: Baseline rows, one per machine/config/clip.

    Examples:
        >>> collect_baseline_rows(Path("missing.csv"))
        []
    """
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        labels = machine_labels_from_fields(list(reader.fieldnames or []))
        for source in reader:
            if source.get("diarization_mode") != "merged-per-speaker":
                continue
            for machine in labels:
                row: dict[str, object] = {
                    "machine": machine,
                    "run": run,
                    "clip_length": source.get("clip_length", ""),
                    "clip_sort_secs": _clip_seconds(source.get("clip_length", "")),
                    "config": source.get("config", ""),
                    "diarization_mode": "merged-per-speaker",
                    "input_variant": source.get(f"{machine}_input_variant", ""),
                    "source_video": "baseline_side_by_side_csv",
                    "output_dir": str(path),
                    # The duration column is shared across machines in the
                    # side-by-side schema.
                    "duration_secs": _num(source.get("duration_secs")),
                    "lipsync_frames": None,
                    "success": True,
                }
                for field in _BASELINE_NUMERIC_FIELDS:
                    row[field] = _num(source.get(f"{machine}_{field}"))
                rows.append(row)
    return rows


def _add_s2s_contribution(rows: list[dict[str, object]]) -> None:
    """Annotate E2E rows with E2E minus bypass pipeline seconds.

    Floors are keyed per machine so mixed-machine row lists never subtract
    another machine's bypass floor.

    Args:
        rows (list[dict[str, object]]): Raw rows, mutated in place.

    Examples:
        >>> rows = [
        ...     {
        ...         "machine": "m1",
        ...         "run": 1,
        ...         "clip_length": "10s",
        ...         "config": "bypass",
        ...         "e2e_secs": 2.0,
        ...     },
        ...     {"machine": "m1", "run": 1, "clip_length": "10s", "config": "el", "e2e_secs": 5.0},
        ... ]
        >>> _add_s2s_contribution(rows)
        >>> rows[1]["s2s_contribution_secs"]
        3.0
    """
    floors = {
        (row.get("machine"), row["run"], row["clip_length"]): row.get("e2e_secs")
        for row in rows
        if row.get("config") == "bypass"
    }
    for row in rows:
        if row.get("config") != "el":
            continue
        floor = _num(floors.get((row.get("machine"), row["run"], row["clip_length"])))
        e2e_secs = _num(row.get("e2e_secs"))
        if floor is not None and e2e_secs is not None:
            row["s2s_contribution_secs"] = round(e2e_secs - floor, 3)


def _metric_values(rows: list[dict[str, object]], field: str) -> list[float]:
    """Return successful numeric values for one field.

    Args:
        rows (list[dict[str, object]]): Candidate rows.
        field (str): Field name.

    Returns:
        list[float]: Numeric values from successful rows.

    Examples:
        >>> _metric_values([{"success": True, "x": 1}], "x")
        [1.0]
    """
    values: list[float] = []
    for row in rows:
        if not row.get("success"):
            continue
        value = _num(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _drop_outliers(values: list[float]) -> list[float]:
    """Drop strong median absolute deviation outliers.

    Args:
        values (list[float]): Candidate values.

    Returns:
        list[float]: Values after outlier filtering.

    Examples:
        >>> _drop_outliers([1.0, 1.1, 0.9, 10.0])
        [1.0, 1.1, 0.9]
    """
    if len(values) < _MIN_OUTLIER_SAMPLE_COUNT:
        return values
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad == 0:
        non_zero = [deviation for deviation in deviations if deviation > 0]
        mad = min(non_zero) if non_zero else 0
    if mad == 0:
        return values
    filtered = [
        value for value in values if 0.6745 * abs(value - center) / mad <= _OUTLIER_MAD_Z_THRESHOLD
    ]
    return filtered or values


def _filtered_values(
    rows: list[dict[str, object]],
    field: str,
    drop_outliers: bool,
) -> list[float]:
    """Return successful values, optionally excluding outliers.

    Args:
        rows (list[dict[str, object]]): Candidate rows.
        field (str): Field name.
        drop_outliers (bool): Whether to apply outlier filtering.

    Returns:
        list[float]: Numeric values.

    Examples:
        >>> _filtered_values([{"success": True, "x": 1}], "x", True)
        [1.0]
    """
    values = _metric_values(rows, field)
    return _drop_outliers(values) if drop_outliers else values


def _mean(
    rows: list[dict[str, object]],
    field: str,
    drop_outliers: bool = False,
) -> float | None:
    """Return the mean of successful values for a field.

    Args:
        rows (list[dict[str, object]]): Candidate rows.
        field (str): Field name.

    Returns:
        float | None: Mean, or ``None``.

    Examples:
        >>> _mean([{"success": True, "x": 1}, {"success": True, "x": 3}], "x")
        2.0
    """
    values = _filtered_values(rows=rows, field=field, drop_outliers=drop_outliers)
    return mean(values) if values else None


def _stddev(
    rows: list[dict[str, object]],
    field: str,
    drop_outliers: bool = False,
) -> float | None:
    """Return sample standard deviation for successful values.

    Args:
        rows (list[dict[str, object]]): Candidate rows.
        field (str): Field name.

    Returns:
        float | None: Sample standard deviation, or ``None`` for fewer than two values.

    Examples:
        >>> _stddev([{"success": True, "x": 1}], "x") is None
        True
    """
    values = _filtered_values(rows=rows, field=field, drop_outliers=drop_outliers)
    return stdev(values) if len(values) > 1 else None


def average_rows(
    raw_rows: list[dict[str, object]],
    drop_outliers: bool = False,
) -> list[dict[str, object]]:
    """Average raw rows by machine, config, and clip.

    Args:
        raw_rows (list[dict[str, object]]): Raw repeated-run rows.

    Returns:
        list[dict[str, object]]: Averaged rows.

    Examples:
        >>> average_rows([])
        []
    """
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            row["machine"],
            row["clip_length"],
            row["config"],
            row["diarization_mode"],
        )
        grouped[key].append(row)

    averages: list[dict[str, object]] = []
    for key, rows in grouped.items():
        machine, clip, config, diarization_mode = key
        success_count = sum(1 for row in rows if row.get("success"))
        averages.append(
            {
                "machine": machine,
                "clip_length": clip,
                "clip_sort_secs": _clip_seconds(str(clip)),
                "config": config,
                "diarization_mode": diarization_mode,
                "input_variant": _input_variant_label(rows),
                "run_count": len(rows),
                "duration_secs_mean": _mean(rows, "duration_secs", drop_outliers),
                "video_frame_count_mean": _mean(rows, "video_frame_count", drop_outliers),
                "preprocess_secs_mean": _mean(rows, "preprocess_secs", drop_outliers),
                "diarization_secs_mean": _mean(rows, "diarization_secs", drop_outliers),
                "e2e_secs_mean": _mean(rows, "e2e_secs", drop_outliers),
                "e2e_secs_stddev": _stddev(rows, "e2e_secs", drop_outliers),
                "realtime_factor_mean": _mean(rows, "realtime_factor", drop_outliers),
                "realtime_factor_stddev": _stddev(rows, "realtime_factor", drop_outliers),
                "pipeline_fps_mean": _mean(rows, "pipeline_fps", drop_outliers),
                "pipeline_fps_stddev": _stddev(rows, "pipeline_fps", drop_outliers),
                "asd_fps_mean": _mean(rows, "asd_fps", drop_outliers),
                "asd_fps_stddev": _stddev(rows, "asd_fps", drop_outliers),
                "lipsync_fps_mean": _mean(rows, "lipsync_fps", drop_outliers),
                "lipsync_fps_stddev": _stddev(rows, "lipsync_fps", drop_outliers),
                "lipsync_frames_mean": _mean(rows, "lipsync_frames", drop_outliers),
                "s2s_contribution_secs_mean": _mean(
                    rows,
                    "s2s_contribution_secs",
                    drop_outliers,
                ),
                "s2s_contribution_secs_stddev": _stddev(
                    rows,
                    "s2s_contribution_secs",
                    drop_outliers,
                ),
                "success_count": success_count,
            }
        )
    return sorted(
        averages,
        key=lambda row: (row["clip_sort_secs"], str(row["config"]), str(row["machine"])),
    )


def _input_variant_label(rows: list[dict[str, object]]) -> str:
    """Return a compact input-variant label for a grouped set of rows.

    Args:
        rows (list[dict[str, object]]): Grouped rows.

    Returns:
        str: Single variant or a mixed-variant label.

    Examples:
        >>> _input_variant_label([{"input_variant": "aac"}, {"input_variant": "mp4"}])
        'mixed(aac,mp4)'
    """
    variants = sorted(
        {
            str(row.get("input_variant", "")).strip()
            for row in rows
            if str(row.get("input_variant", "")).strip()
        }
    )
    if not variants:
        return ""
    if len(variants) == 1:
        return variants[0]
    return f"mixed({','.join(variants)})"


def _fmt_csv(value: object) -> object:
    """Format values for stable CSV output.

    Args:
        value (object): Raw value.

    Returns:
        object: CSV-friendly value.

    Examples:
        >>> _fmt_csv(1.234567)
        '1.234567'
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return value


def side_by_side_rows(
    avg_rows: list[dict[str, object]],
    labels: list[str],
) -> list[dict[str, object]]:
    """Build the side-by-side CSV rows consumed by the HTML report.

    Args:
        avg_rows (list[dict[str, object]]): Averaged rows.
        labels (list[str]): Machine labels in column order.

    Returns:
        list[dict[str, object]]: Rows with one column group per machine label.

    Examples:
        >>> side_by_side_rows([], ["m1"])
        []
    """
    by_key = {
        (row["machine"], row["clip_length"], row["config"], row["diarization_mode"]): row
        for row in avg_rows
    }
    clips = sorted(
        {row["clip_length"] for row in avg_rows},
        key=lambda clip: _clip_seconds(str(clip)),
    )
    configs = ["bypass", "el"]
    rows: list[dict[str, object]] = []
    for clip in clips:
        for config in configs:
            machine_rows = {
                label: by_key.get((label, clip, config, "merged-per-speaker")) for label in labels
            }
            present = [row for row in machine_rows.values() if row is not None]
            if not present:
                continue
            side_row: dict[str, object] = {
                "clip_length": clip,
                "config": config,
                "diarization_mode": "merged-per-speaker",
                "duration_secs": _mean_of_rows(present, "duration_secs_mean"),
            }
            for label in labels:
                side_row[f"{label}_input_variant"] = _field(machine_rows[label], "input_variant")
            for suffix, mean_field, stddev_field in _SIDE_METRICS:
                for label in labels:
                    row = machine_rows[label]
                    side_row[f"{label}_{suffix}"] = _field(row, mean_field)
                    if stddev_field is not None:
                        side_row[f"{label}_{suffix}_stddev"] = _field(row, stddev_field)
            rows.append(side_row)
    return rows


def _field(row: dict[str, object] | None, name: str) -> object:
    """Return a row field or an empty cell.

    Args:
        row (dict[str, object] | None): Source row.
        name (str): Field name.

    Returns:
        object: Field value, or empty string.

    Examples:
        >>> _field({"x": 1}, "x")
        1
    """
    return "" if row is None else row.get(name, "")


def _mean_of_rows(rows: list[dict[str, object]], field: str) -> float | None:
    """Return the mean of one averaged field across machine rows.

    Args:
        rows (list[dict[str, object]]): Machine rows.
        field (str): Field to average.

    Returns:
        float | None: Mean value.

    Examples:
        >>> _mean_of_rows([{"x": 1}, {"x": 3}], "x")
        2.0
    """
    values = [_num(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def write_csv(rows: list[dict[str, object]], fields: list[str], path: Path) -> None:
    """Write rows to CSV.

    Args:
        rows (list[dict[str, object]]): Rows to write.
        fields (list[str]): CSV field order.
        path (Path): Destination path.

    Examples:
        >>> write_csv([], ["x"], Path("/tmp/empty.csv"))  # doctest: +SKIP
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt_csv(row.get(field)) for field in fields})


def _report_suffix(value: str) -> str:
    """Return a filesystem-safe report suffix.

    Args:
        value (str): Requested suffix.

    Returns:
        str: Sanitized suffix.

    Examples:
        >>> _report_suffix("4 run")
        '4_run'
    """
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return suffix.strip("._-") or "avg"


def _fmt_md(value: object) -> str:
    """Format a Markdown table cell.

    Args:
        value (object): Raw value.

    Returns:
        str: Cell text.

    Examples:
        >>> _fmt_md(1.234)
        '1.23'
    """
    if value in (None, ""):
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown(
    raw_rows: list[dict[str, object]],
    avg_rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
    path: Path,
    drop_outliers: bool = False,
) -> None:
    """Write a Markdown record of raw and averaged measurements.

    Machine labels for the side-by-side table are recovered from the
    side-by-side row columns.

    Args:
        raw_rows (list[dict[str, object]]): Individual run rows.
        avg_rows (list[dict[str, object]]): Averaged rows.
        side_rows (list[dict[str, object]]): Side-by-side averaged rows.
        path (Path): Destination path.

    Examples:
        >>> write_markdown([], [], [], Path("/tmp/report.md"))  # doctest: +SKIP
    """
    labels = machine_labels_from_fields(list(side_rows[0].keys())) if side_rows else []
    generated = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CLBP Perf analysis",
        "",
        f"Generated: {generated}",
        "",
        "This record averages successful repeated merged-per-speaker runs. "
        "Raw per-run rows are retained in `repeated_raw_runs.csv`.",
        "",
    ]
    if drop_outliers:
        lines.extend(
            [
                "Outlier policy: metric means and standard deviations exclude strong "
                "median absolute deviation outliers.",
                "",
            ]
        )
    lines.extend(["## Averaged Side-by-Side Metrics", ""])
    headers = ["clip_length", "config"]
    for metric in ("asd_fps", "lipsync_fps", "pipeline_fps", "e2e_secs"):
        headers.extend(f"{label}_{metric}" for label in labels)
    lines.extend(_md_table(side_rows, headers))
    lines.extend(["", "## Averaged Run Statistics", ""])
    avg_headers = [
        "machine",
        "clip_length",
        "config",
        "run_count",
        "success_count",
        "asd_fps_mean",
        "asd_fps_stddev",
        "lipsync_fps_mean",
        "lipsync_fps_stddev",
        "pipeline_fps_mean",
        "pipeline_fps_stddev",
        "e2e_secs_mean",
        "e2e_secs_stddev",
    ]
    lines.extend(_md_table(avg_rows, avg_headers))
    lines.extend(["", "## Raw Run Rows", ""])
    raw_headers = [
        "machine",
        "run",
        "clip_length",
        "config",
        "asd_fps",
        "lipsync_fps",
        "pipeline_fps",
        "e2e_secs",
        "realtime_factor",
        "success",
    ]
    lines.extend(_md_table(raw_rows, raw_headers))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_table(rows: list[dict[str, object]], headers: list[str]) -> list[str]:
    """Render a simple Markdown table.

    Args:
        rows (list[dict[str, object]]): Source rows.
        headers (list[str]): Header fields.

    Returns:
        list[str]: Markdown lines.

    Examples:
        >>> _md_table([], ["x"])[0]
        '| x |'
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_md(row.get(header)) for header in headers) + " |")
    return lines


def machine_label(directory: Path) -> str:
    """Return the machine label for one runs directory.

    Args:
        directory (Path): Machine runs directory.

    Returns:
        str: Filesystem- and CSV-safe label derived from the directory basename.

    Examples:
        >>> machine_label(Path("outputs/perf_x/machine a"))
        'machine_a'
    """
    label = re.sub(r"[^A-Za-z0-9.-]+", "_", directory.name.strip()).strip("._-")
    return label


def main() -> None:
    """Parse arguments and write repeated-run aggregate files.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="Aggregate repeated CLBP perf runs.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "One or more machine runs directories, each holding runN/ outputs. "
            "The directory basename is used as the machine label in reports."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--input-variant",
        default="aac",
        help="Input normalization label recorded for all machines (default: aac).",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Optional existing side-by-side CSV to include as run 0.",
    )
    parser.add_argument(
        "--drop-outliers",
        action="store_true",
        help="Exclude strong MAD outliers from metric means and standard deviations.",
    )
    parser.add_argument(
        "--report-suffix",
        default="avg3",
        help="Suffix for side-by-side CSV and Markdown report names (default: avg3).",
    )
    args = parser.parse_args()
    report_suffix = _report_suffix(args.report_suffix)

    # Fail fast on bad paths so a typo cannot produce an empty/partial report.
    for directory in args.dirs:
        if not directory.is_dir():
            parser.error(f"--dirs entry is not a directory: {directory}")
    if args.baseline_csv is not None and not args.baseline_csv.is_file():
        parser.error(f"--baseline-csv is not a file: {args.baseline_csv}")

    labels = [machine_label(directory) for directory in args.dirs]
    if any(not label for label in labels):
        parser.error("every --dirs entry needs a non-empty directory basename")
    if len(set(labels)) != len(labels):
        parser.error(f"machine labels from --dirs basenames must be unique, got: {labels}")

    raw_rows: list[dict[str, object]] = []
    for directory, label in zip(args.dirs, labels, strict=True):
        raw_rows += collect_rows(directory, machine=label, input_variant=args.input_variant)
    if args.baseline_csv is not None:
        raw_rows += collect_baseline_rows(path=args.baseline_csv)
    # Baseline CSVs may carry machines beyond --dirs; append them so their
    # columns are not silently dropped from the side-by-side output.
    for row in raw_rows:
        if row["machine"] not in labels:
            labels.append(str(row["machine"]))
    raw_rows = sorted(
        raw_rows,
        key=lambda row: (
            row["clip_sort_secs"],
            str(row["config"]),
            str(row["machine"]),
            int(row["run"]),
        ),
    )
    avg_rows = average_rows(raw_rows, drop_outliers=args.drop_outliers)
    side_rows = side_by_side_rows(avg_rows, labels=labels)

    write_csv(raw_rows, _RAW_FIELDS, args.out_dir / "repeated_raw_runs.csv")
    write_csv(avg_rows, _AVG_FIELDS, args.out_dir / "repeated_averages.csv")
    side_by_side_path = args.out_dir / f"side_by_side_{report_suffix}.csv"
    markdown_path = args.out_dir / f"combined_perf_report_{report_suffix}.md"
    write_csv(side_rows, side_by_side_fields(labels), side_by_side_path)
    write_markdown(
        raw_rows,
        avg_rows,
        side_rows,
        path=markdown_path,
        drop_outliers=args.drop_outliers,
    )
    print(f"Wrote {args.out_dir / 'repeated_raw_runs.csv'}")
    print(f"Wrote {args.out_dir / 'repeated_averages.csv'}")
    print(f"Wrote {side_by_side_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
