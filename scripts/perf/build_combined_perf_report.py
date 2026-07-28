# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the self-contained combined perf report HTML.

The side-by-side CSV comes from ``aggregate_repeated_perf.py``; machine labels
are recovered from its column names, so the report works for any machines and
any machine count.

Examples:
    $ source .venv/bin/activate
    $ python scripts/perf/build_combined_perf_report.py \
        --side-by-side-csv outputs/combined-perf-report/side_by_side_avg3.csv \
        --output-html outputs/combined-perf-report/combined_perf_report.html
"""

import argparse
import csv
import datetime
import html
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

# aggregate_repeated_perf lives in the same directory as this script; add the
# directory to sys.path so it can be imported regardless of the working
# directory.
sys.path.insert(0, os.path.dirname(__file__))
from aggregate_repeated_perf import machine_labels_from_fields

WALL_STDDEV_SECONDS_THRESHOLD = 10.0
CONFIG_LABELS = {
    "bypass": "Bypass S2S",
    "el": "E2E / ElevenLabs",
}

# Bar colors assigned to (config, machine) series in order; the palette wraps
# for large machine counts.
_SERIES_COLORS = [
    "#2563eb",
    "#16a34a",
    "#818cf8",
    "#86efac",
    "#f59e0b",
    "#ef4444",
    "#0ea5e9",
    "#a855f7",
]


def _machine_labels(
    rows: list[dict[str, str]],
    fieldnames: list[str] | None = None,
) -> list[str]:
    """Recover machine labels from side-by-side rows or CSV header fields.

    Args:
        rows (list[dict[str, str]]): Side-by-side CSV rows.
        fieldnames (list[str] | None): CSV header fields, used when provided so
            header-only CSVs still yield labels.

    Returns:
        list[str]: Machine labels in column order.

    Examples:
        >>> _machine_labels([])
        []
    """
    if fieldnames:
        return machine_labels_from_fields(fieldnames)
    if not rows:
        return []
    return machine_labels_from_fields(list(rows[0].keys()))


def _num(value: object) -> float | None:
    """Convert a CSV cell to float.

    Args:
        value (object): Raw value.

    Returns:
        float | None: Parsed number, or ``None`` when unavailable.

    Examples:
        >>> _num("1.25")
        1.25
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _fmt(value: object, digits: int = 2) -> str:
    """Format a number for report cells.

    Args:
        value (object): Value to render.
        digits (int): Decimal places for floats.

    Returns:
        str: Display value.

    Examples:
        >>> _fmt(1.234)
        '1.23'
    """
    number = _num(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _fmt_wall_time(value: object) -> str:
    """Format seconds as minutes:seconds.

    Args:
        value (object): Seconds.

    Returns:
        str: Wall time string.

    Examples:
        >>> _fmt_wall_time(90)
        '1:30'
    """
    seconds = _num(value)
    if seconds is None:
        return ""
    total = round(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _fmt_wall_stddev(value: object) -> str:
    """Format wall-time standard deviation compactly.

    Args:
        value (object): Seconds.

    Returns:
        str: Compact standard deviation string.

    Examples:
        >>> _fmt_wall_stddev(1.23)
        '1.2s'
    """
    seconds = _num(value)
    if seconds is None:
        return ""
    if seconds < WALL_STDDEV_SECONDS_THRESHOLD:
        return f"{seconds:.1f}s"
    return _fmt_wall_time(seconds)


def _stddev_value(row: dict[str, str] | None, field: str) -> float | None:
    """Return the standard deviation cell for a metric field.

    Args:
        row (dict[str, str] | None): Source row.
        field (str): Mean metric field.

    Returns:
        float | None: Standard deviation value.

    Examples:
        >>> _stddev_value({"x_stddev": "1.2"}, "x")
        1.2
    """
    if not row:
        return None
    return _num(row.get(f"{field}_stddev"))


def _clip_seconds(label: str) -> float:
    """Return a sorting value for a clip-length label.

    Args:
        label (str): Label such as ``10s``, ``1min``, or ``18m48s``.

    Returns:
        float: Sortable duration in seconds.

    Examples:
        >>> _clip_seconds("2min")
        120.0
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


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read side-by-side CSV rows and header fields.

    Args:
        path (Path): CSV path.

    Returns:
        tuple[list[dict[str, str]], list[str]]: CSV rows and header fields.

    Examples:
        >>> _read_rows(Path("missing.csv"))  # doctest: +SKIP
        ([], [])
    """
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _merged_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return merged-per-speaker rows sorted by clip length and config.

    Args:
        rows (list[dict[str, str]]): Side-by-side rows.

    Returns:
        list[dict[str, str]]: Filtered rows.

    Examples:
        >>> _merged_rows([])
        []
    """
    filtered = [row for row in rows if row.get("diarization_mode") == "merged-per-speaker"]
    return sorted(filtered, key=lambda row: (_clip_seconds(row["clip_length"]), row["config"]))


def _rows_by_clip(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    """Group rows by clip label and config.

    Args:
        rows (list[dict[str, str]]): Side-by-side rows.

    Returns:
        dict[str, dict[str, dict[str, str]]]: ``clip -> config -> row``.

    Examples:
        >>> _rows_by_clip([])
        {}
    """
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in _merged_rows(rows=rows):
        grouped.setdefault(row["clip_length"], {})[row["config"]] = row
    return dict(sorted(grouped.items(), key=lambda item: _clip_seconds(item[0])))


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean of values.

    Args:
        values (list[float]): Values.

    Returns:
        float | None: Mean, or ``None`` for an empty list.

    Examples:
        >>> _mean([1, 3])
        2.0
    """
    if not values:
        return None
    return sum(values) / len(values)


def _metric_values(
    grouped: dict[str, dict[str, dict[str, str]]],
    config: str,
    metric: str,
    label: str,
    min_seconds: float = 0.0,
) -> list[float]:
    """Collect numeric metric values for one config/machine.

    Args:
        grouped (dict[str, dict[str, dict[str, str]]]): Grouped rows.
        config (str): Config key.
        metric (str): Metric suffix, such as ``pipeline_fps``.
        label (str): Machine label.
        min_seconds (float): Minimum clip duration.

    Returns:
        list[float]: Numeric values.

    Examples:
        >>> _metric_values({}, "el", "pipeline_fps", "machine-a")
        []
    """
    values: list[float] = []
    field = f"{label}_{metric}"
    for clip, modes in grouped.items():
        if _clip_seconds(clip) < min_seconds:
            continue
        row = modes.get(config)
        value = _num(row.get(field)) if row else None
        if value is not None:
            values.append(value)
    return values


def _exec_summary(grouped: dict[str, dict[str, dict[str, str]]], labels: list[str]) -> str:
    """Render the executive summary section.

    Args:
        grouped (dict[str, dict[str, dict[str, str]]]): Grouped rows.
        labels (list[str]): Machine labels.

    Returns:
        str: HTML section.

    Examples:
        >>> _exec_summary({}, [])
        '<section class="summary-card"><h2>Executive Summary</h2>...'
    """
    longer_min = 60.0
    bullets: list[tuple[str, str, str]] = []
    for label in labels:
        lipsync = _mean(
            _metric_values(grouped, "bypass", "lipsync_fps", label, longer_min)
            + _metric_values(grouped, "el", "lipsync_fps", label, longer_min)
        )
        bypass_fps = _mean(_metric_values(grouped, "bypass", "pipeline_fps", label, longer_min))
        e2e_fps = _mean(_metric_values(grouped, "el", "pipeline_fps", label, longer_min))
        if None in (lipsync, bypass_fps, e2e_fps):
            continue
        bullets.append(
            (
                "#76b900",
                f"{html.escape(label)}: throughput on 1min and longer clips.",
                f"Pipeline FPS averages <strong>{bypass_fps:.1f}</strong> in Bypass "
                f"S2S and <strong>{e2e_fps:.1f}</strong> in E2E / ElevenLabs; LipSync "
                f"averages <strong>{lipsync:.1f} FPS</strong>.",
            )
        )
    bullets.append(
        (
            "#0066cc",
            "Short clips are overhead-bound.",
            "Use longer clips to judge steady-state throughput; short clips are "
            "more affected by setup, streaming, and service orchestration overhead.",
        )
    )
    bullets.append(
        (
            "#ff6600",
            "ASD FPS remains an anomaly check.",
            "The ASD values come from stage logs, so inspect ASD logs when they "
            "behave differently across inputs.",
        )
    )

    items = []
    for color, title, body in bullets:
        items.append(
            '<li><span class="dot" style="background:'
            f'{color}"></span><span><strong>{title}</strong> {body}</span></li>'
        )
    return (
        '<section class="summary-card"><h2>Executive Summary</h2>'
        f'<ul class="summary-list">{"".join(items)}</ul></section>'
    )


def _definitions_table() -> str:
    """Render notation and definitions.

    Returns:
        str: HTML table.

    Examples:
        >>> "Bypass S2S" in _definitions_table()
        True
    """
    rows = [
        (
            "Bypass S2S",
            "Pipeline mode where Speech-to-Speech is skipped and translated audio is "
            "provided directly to LipSync. ASD still runs.",
        ),
        (
            "E2E / ElevenLabs",
            "End-to-end pipeline where ElevenLabs Speech-to-Speech, ASD, and LipSync "
            "run through the controller.",
        ),
        (
            "Wall time",
            "Elapsed clock time for a stage or pipeline request. Lower is better.",
        ),
        ("Clip length", "Visible report label based only on video duration."),
        ("ASD FPS", "Active Speaker Detection FPS scraped from ASD NIM logs."),
        ("LipSync FPS", "End-to-end FPS reported by the LipSync NIM."),
        (
            "Pipeline FPS",
            "Source video frames divided by pipeline wall time. This is the main "
            "throughput metric.",
        ),
        (
            "Real-time factor",
            "Pipeline wall time divided by source video duration. Values below 1.0 "
            "are faster than real time.",
        ),
    ]
    body = "".join(
        f'<tr><td class="left">{html.escape(term)}</td><td>{html.escape(definition)}</td></tr>'
        for term, definition in rows
    )
    return (
        '<h2>Notation &amp; Definitions</h2><div class="table-wrap"><table>'
        '<thead><tr class="col-header"><th class="left">Term</th>'
        f"<th>Definition</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def _bar_title(clip: str, label: str, value: float, error: float | None) -> str:
    """Return tooltip text for a chart bar.

    Args:
        clip (str): Clip label.
        label (str): Series label.
        value (float): Mean value.
        error (float | None): Standard deviation.

    Returns:
        str: Tooltip text.

    Examples:
        >>> _bar_title("10s", "machine-a", 1.2, 0.1)
        '10s machine-a: 1.20 FPS +/- 0.10'
    """
    title = f"{clip} {label}: {value:.2f} FPS"
    if error is not None:
        title += f" +/- {error:.2f}"
    return title


def _error_bar_svg(
    x: float,
    bar_width: float,
    value: float,
    error: float | None,
    y_pos: Callable[[float], float],
) -> tuple[str, float]:
    """Render an error bar and return the preferred value-label y position.

    Args:
        x (float): Bar x position.
        bar_width (float): Bar width.
        value (float): Mean value.
        error (float | None): Standard deviation.
        y_pos (Callable[[float], float]): Value-to-y-coordinate mapper.

    Returns:
        tuple[str, float]: Error-bar SVG and label y position.

    Examples:
        >>> _error_bar_svg(1, 2, 3, None, lambda v: v)
        ('', -1)
    """
    label_y = y_pos(value) - 4
    if error is None:
        return "", label_y

    err_x = x + bar_width / 2
    low_y = y_pos(max(0.0, value - error))
    high_y = y_pos(value + error)
    label_y = high_y - 4
    cap = min(10.0, bar_width + 4)
    parts = [
        f'<line x1="{err_x:.1f}" x2="{err_x:.1f}" y1="{high_y:.1f}" '
        f'y2="{low_y:.1f}" stroke="#111827" stroke-width="1.2"></line>',
        f'<line x1="{err_x - cap / 2:.1f}" x2="{err_x + cap / 2:.1f}" '
        f'y1="{high_y:.1f}" y2="{high_y:.1f}" '
        'stroke="#111827" stroke-width="1.2"></line>',
        f'<line x1="{err_x - cap / 2:.1f}" x2="{err_x + cap / 2:.1f}" '
        f'y1="{low_y:.1f}" y2="{low_y:.1f}" '
        'stroke="#111827" stroke-width="1.2"></line>',
    ]
    return "".join(parts), label_y


def _axis_svg(
    max_value: float,
    width: int,
    right: int,
    left: int,
    y_pos: Callable[[float], float],
) -> str:
    """Render horizontal grid lines and y-axis labels.

    Args:
        max_value (float): Maximum chart value.
        width (int): SVG width.
        right (int): Right margin.
        left (int): Left margin.
        y_pos (Callable[[float], float]): Value-to-y-coordinate mapper.

    Returns:
        str: SVG axis markup.

    Examples:
        >>> "<line" in _axis_svg(10, 100, 10, 10, lambda v: v)
        True
    """
    parts = []
    for step in range(5):
        value = max_value * step / 4
        y = y_pos(value)
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"></line>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-family="Arial" font-size="11" fill="#64748b">'
            f"{value:.0f}</text>"
        )
    return "".join(parts)


def _legend_svg(series: list[tuple[str, str, str, str]], left: int) -> str:
    """Render the chart legend.

    Args:
        series (list[tuple[str, str, str, str]]): Chart series definitions.
        left (int): Left margin.

    Returns:
        str: SVG legend markup.

    Examples:
        >>> "m1" in _legend_svg([("bypass", "m1", "m1", "#000")], 10)
        True
    """
    parts = []
    legend_x = left
    for _, _, label, color in series:
        parts.append(f'<rect x="{legend_x}" y="20" width="10" height="10" fill="{color}"></rect>')
        parts.append(
            f'<text x="{legend_x + 14}" y="29" font-family="Arial" font-size="11" '
            f'fill="#334155">{html.escape(label)}</text>'
        )
        legend_x += 165
    return "".join(parts)


def _chart_series(labels: list[str]) -> list[tuple[str, str, str, str]]:
    """Build the chart series for the given machine labels.

    Args:
        labels (list[str]): Machine labels.

    Returns:
        list[tuple[str, str, str, str]]: ``(config, label, display, color)`` series.

    Examples:
        >>> _chart_series(["m1"])[0][2]
        'm1 Bypass S2S'
    """
    series: list[tuple[str, str, str, str]] = []
    for config in ("bypass", "el"):
        for label in labels:
            color = _SERIES_COLORS[len(series) % len(_SERIES_COLORS)]
            series.append((config, label, f"{label} {CONFIG_LABELS[config]}", color))
    return series


def _bar_svg(
    grouped: dict[str, dict[str, dict[str, str]]],
    labels: list[str],
    metric: str,
    title: str,
    y_label: str,
) -> str:
    """Render one inline SVG grouped bar chart.

    Args:
        grouped (dict[str, dict[str, dict[str, str]]]): Grouped rows.
        labels (list[str]): Machine labels.
        metric (str): Metric suffix.
        title (str): Chart title.
        y_label (str): Vertical axis label.

    Returns:
        str: SVG chart HTML.

    Examples:
        >>> _bar_svg({}, ["m1"], "pipeline_fps", "Pipeline", "FPS").startswith("<div")
        True
    """
    series = _chart_series(labels=labels)
    clips = list(grouped)
    values: list[float] = []
    for clip in clips:
        modes = grouped[clip]
        for config, label, _, _ in series:
            row = modes.get(config)
            field = f"{label}_{metric}"
            value = _num(row.get(field)) if row else None
            if value is not None:
                error = _stddev_value(row, field) or 0.0
                values.append(value + error)
    # Guard against an all-empty chart so y_pos never divides by zero.
    max_value = (max(values, default=0.0) or 1.0) * 1.18
    width = max(980, 120 + len(clips) * 116)
    height = 470
    left = 70
    right = 24
    top = 58
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / max(len(clips), 1)
    bar_width = min(18.0, group_width / (len(series) + 3))

    def y_pos(value: float) -> float:
        return top + plot_height - (value / max_value * plot_height)

    parts = [
        '<div class="chart-card"><h3>',
        html.escape(title),
        '</h3><div class="chart-image-wrap">',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="white"></rect>',
    ]
    parts.append(_axis_svg(max_value, width, right, left, y_pos))
    parts.append(_legend_svg(series, left))
    for clip_index, clip in enumerate(clips):
        cx = left + group_width * clip_index + group_width / 2
        base_x = cx - (bar_width * len(series) + 4 * (len(series) - 1)) / 2
        modes = grouped[clip]
        for series_index, (config, label, display, color) in enumerate(series):
            row = modes.get(config)
            field = f"{label}_{metric}"
            value = _num(row.get(field)) if row else None
            if value is None:
                continue
            error = _stddev_value(row, field)
            x = base_x + series_index * (bar_width + 4)
            y = y_pos(value)
            h = top + plot_height - y
            error_svg, label_y = _error_bar_svg(
                x=x,
                bar_width=bar_width,
                value=value,
                error=error,
                y_pos=y_pos,
            )
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{h:.1f}" rx="2" fill="{color}">'
                f"<title>{html.escape(_bar_title(clip, display, value, error))}</title>"
                "</rect>"
            )
            parts.append(error_svg)
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="10" '
                f'fill="#334155">{value:.2f}</text>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 34}" text-anchor="middle" '
            f'font-family="Arial" font-size="11" fill="#334155">{html.escape(clip)}</text>'
        )
    parts.append(
        f'<text x="24" y="{top + plot_height / 2:.1f}" '
        f'transform="rotate(-90 24 {top + plot_height / 2:.1f})" '
        'text-anchor="middle" font-family="Arial" font-size="12" '
        f'fill="#334155">{html.escape(y_label)}</text>'
    )
    parts.append("</svg></div></div>")
    return "".join(parts)


def _charts(grouped: dict[str, dict[str, dict[str, str]]], labels: list[str]) -> str:
    """Render the stacked chart section.

    Args:
        grouped (dict[str, dict[str, dict[str, str]]]): Grouped rows.
        labels (list[str]): Machine labels.

    Returns:
        str: Chart HTML.

    Examples:
        >>> "<svg" in _charts({}, ["m1"])
        True
    """
    chart_specs = [
        ("lipsync_fps", "LipSync FPS: Bypass S2S vs E2E / ElevenLabs"),
        ("pipeline_fps", "Pipeline FPS: Bypass S2S vs E2E / ElevenLabs"),
        ("asd_fps", "ASD FPS: Bypass S2S vs E2E / ElevenLabs"),
    ]
    charts = "".join(
        _bar_svg(grouped=grouped, labels=labels, metric=metric, title=title, y_label="FPS")
        for metric, title in chart_specs
    )
    return (
        '<h2>FPS Charts</h2><p class="note">Bars show the mean across repeated runs; '
        "error bars show sample standard deviation.</p>"
        f'<div class="chart-grid">{charts}</div>'
    )


def _cell(value: object, css_class: str = "") -> str:
    """Render a table cell.

    Args:
        value (object): Cell value.
        css_class (str): Optional class string.

    Returns:
        str: HTML table cell.

    Examples:
        >>> _cell("x")
        '<td>x</td>'
    """
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<td{class_attr}>{html.escape(str(value))}</td>"


def _metric_cell(row: dict[str, str] | None, field: str, css_class: str = "") -> str:
    """Render a numeric metric cell.

    Args:
        row (dict[str, str] | None): Source row.
        field (str): CSV field.
        css_class (str): Optional class string.

    Returns:
        str: HTML cell.

    Examples:
        >>> _metric_cell({"x": "1.2"}, "x")
        '<td>1.20</td>'
    """
    value = _fmt(row.get(field)) if row else ""
    stddev = _stddev_value(row, field)
    if value and stddev is not None:
        value = f"{value} +/- {stddev:.2f}"
    return _cell(value, css_class=css_class)


def _label_header_cells(labels: list[str], group_count: int) -> str:
    """Render machine-label header cells repeated for each metric group.

    The last label of every group except the final one gets the ``sep`` class,
    matching the vertical separators used in the body cells.

    Args:
        labels (list[str]): Machine labels.
        group_count (int): Number of metric groups.

    Returns:
        str: Header cell HTML.

    Examples:
        >>> _label_header_cells(["m1"], 2)
        '<th class="sep">m1</th><th>m1</th>'
    """
    cells = []
    for group_index in range(group_count):
        for label_index, label in enumerate(labels):
            is_sep = label_index == len(labels) - 1 and group_index < group_count - 1
            class_attr = ' class="sep"' if is_sep else ""
            cells.append(f"<th{class_attr}>{html.escape(label)}</th>")
    return "".join(cells)


def _metric_group_cells(
    row: dict[str, str] | None,
    labels: list[str],
    metrics: list[str],
    trailing_sep: bool,
) -> str:
    """Render body cells for one config's metric groups.

    Args:
        row (dict[str, str] | None): Side-by-side row for the config.
        labels (list[str]): Machine labels.
        metrics (list[str]): Metric suffixes, one group per metric.
        trailing_sep (bool): Whether the final group also ends with a separator.

    Returns:
        str: Body cell HTML.

    Examples:
        >>> _metric_group_cells(None, ["m1"], ["asd_fps"], trailing_sep=True)
        '<td class="sep"></td>'
    """
    cells = []
    for metric_index, metric in enumerate(metrics):
        for label_index, label in enumerate(labels):
            is_last_in_group = label_index == len(labels) - 1
            needs_sep = is_last_in_group and (trailing_sep or metric_index < len(metrics) - 1)
            cells.append(_metric_cell(row, f"{label}_{metric}", "sep" if needs_sep else ""))
    return "".join(cells)


def _fps_table(grouped: dict[str, dict[str, dict[str, str]]], labels: list[str]) -> str:
    """Render the per-clip FPS table.

    Args:
        grouped (dict[str, dict[str, dict[str, str]]]): Grouped rows.
        labels (list[str]): Machine labels.

    Returns:
        str: HTML table.

    Examples:
        >>> "Per-Clip FPS Metrics" in _fps_table({}, ["m1"])
        True
    """
    count = max(len(labels), 1)
    bypass_metrics = ["asd_fps", "lipsync_fps", "pipeline_fps"]
    e2e_metrics = ["asd_fps", "lipsync_fps", "pipeline_fps", "realtime_factor"]
    rows_html = []
    for clip, modes in grouped.items():
        rows_html.append(
            "<tr>"
            + _cell(clip, "left sep")
            + _metric_group_cells(
                row=modes.get("bypass"),
                labels=labels,
                metrics=bypass_metrics,
                trailing_sep=True,
            )
            + _metric_group_cells(
                row=modes.get("el"),
                labels=labels,
                metrics=e2e_metrics,
                trailing_sep=False,
            )
            + "</tr>"
        )
    metric_headers = "".join(
        f'<th colspan="{count}">{name}</th>'
        for name in (
            "ASD FPS",
            "LipSync FPS",
            "Pipeline FPS",
            "ASD FPS",
            "LipSync FPS",
            "Pipeline FPS",
            "Real-time Factor",
        )
    )
    return (
        '<h2>Per-Clip FPS Metrics</h2><p class="note">Each clip is one row. '
        "Bypass S2S and E2E / ElevenLabs are split into column groups, with "
        "ASD FPS, LipSync FPS, Pipeline FPS, and Real-time factor broken out "
        "per machine. Values are mean +/- sample standard deviation across "
        "repeated runs.</p>"
        '<div class="table-wrap"><table><thead>'
        '<tr class="group-header"><th rowspan="3" class="left sep">Clip</th>'
        f'<th colspan="{count * len(bypass_metrics)}">Bypass S2S (ASD + LipSync)</th>'
        f'<th colspan="{count * len(e2e_metrics)}">E2E / ElevenLabs</th></tr>'
        f'<tr class="group-header">{metric_headers}</tr>'
        '<tr class="col-header">'
        f"{_label_header_cells(labels=labels, group_count=len(bypass_metrics) + len(e2e_metrics))}"
        "</tr>"
        f"</thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def _wall_metric_cell(row: dict[str, str] | None, field: str, css_class: str = "") -> str:
    """Render a wall-time cell.

    Args:
        row (dict[str, str] | None): Source row.
        field (str): CSV field.
        css_class (str): Optional class string.

    Returns:
        str: HTML cell.

    Examples:
        >>> _wall_metric_cell({"x": "90"}, "x")
        '<td>1:30</td>'
    """
    value = _fmt_wall_time(row.get(field)) if row else ""
    stddev = _stddev_value(row, field)
    if value and stddev is not None:
        value = f"{value} +/- {_fmt_wall_stddev(stddev)}"
    return _cell(value, css_class=css_class)


def _wall_time_table(grouped: dict[str, dict[str, dict[str, str]]], labels: list[str]) -> str:
    """Render the per-clip wall-time table.

    Args:
        grouped (dict[str, dict[str, dict[str, str]]]): Grouped rows.
        labels (list[str]): Machine labels.

    Returns:
        str: HTML table.

    Examples:
        >>> "Wall Time" in _wall_time_table({}, ["m1"])
        True
    """
    count = max(len(labels), 1)
    # One (config, metric) pair per column group, in display order.
    groups = [("bypass", "e2e_secs"), ("el", "e2e_secs"), ("el", "s2s_wall_secs")]
    rows_html = []
    for clip, modes in grouped.items():
        cells = [_cell(clip, "left sep")]
        for group_index, (config, metric) in enumerate(groups):
            row = modes.get(config)
            for label_index, label in enumerate(labels):
                is_sep = label_index == len(labels) - 1 and group_index < len(groups) - 1
                cells.append(_wall_metric_cell(row, f"{label}_{metric}", "sep" if is_sep else ""))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<h2>Per-Clip Wall Time Metrics</h2><p class="note">Each clip matches '
        "the FPS table. Wall time is shown as minutes:seconds, with +/- sample "
        "standard deviation where repeated-run data is available.</p>"
        '<div class="table-wrap"><table><thead>'
        '<tr class="group-header"><th rowspan="2" class="left sep">Clip</th>'
        f'<th colspan="{count}">Bypass S2S: Pipeline Wall Time</th>'
        f'<th colspan="{count}">E2E / ElevenLabs: Pipeline Wall Time</th>'
        f'<th colspan="{count}">E2E / ElevenLabs: Added Speech-to-Speech Time</th></tr>'
        '<tr class="col-header">'
        f"{_label_header_cells(labels=labels, group_count=len(groups))}</tr>"
        f"</thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def _css() -> str:
    """Return report CSS.

    Returns:
        str: CSS.

    Examples:
        >>> ".chart-card" in _css()
        True
    """
    lines = [
        ":root{--green:#76b900;--dark:#1a1a2e;--border:#e0e0e0;",
        "--text:#1a1a1a;--muted:#666;--blue:#0066cc;--orange:#ff6600}",
        "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}",
        'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;',
        "background:#f5f6fa;color:var(--text);font-size:14px;line-height:1.5}",
        "header{background:var(--dark);color:white;padding:24px 40px;display:flex;",
        "align-items:center;gap:20px}",
        ".logo-nvidia{font-size:11px;letter-spacing:4px;text-transform:uppercase;",
        "color:var(--green);font-weight:700}",
        ".logo-title{font-size:20px;font-weight:700;color:white;margin-top:2px}",
        ".logo-sub{font-size:13px;color:#aaa;margin-top:2px}",
        ".badge{margin-left:auto;background:var(--green);color:#000;font-weight:700;",
        "padding:6px 14px;border-radius:4px;font-size:13px}",
        "main{max-width:1280px;margin:0 auto;padding:32px 24px}",
        "h2{font-size:17px;font-weight:700;color:var(--dark);margin:32px 0 10px;",
        "padding-bottom:7px;border-bottom:2px solid var(--green)}",
        "h3{font-size:14px;font-weight:600;color:var(--dark);margin-bottom:8px}",
        "p.note{color:var(--muted);font-size:12px;margin-top:8px;font-style:italic}",
        ".summary-card{background:white;border-radius:8px;border:1px solid var(--border);",
        "padding:24px 28px;margin-bottom:28px}",
        ".summary-card h2{margin-top:0;border-color:var(--orange)}",
        ".summary-list{list-style:none;padding:0}",
        ".summary-list li{display:flex;gap:12px;padding:10px 0;",
        "border-bottom:1px solid #f0f0f0;font-size:13.5px}",
        ".summary-list li:last-child{border-bottom:0}",
        ".dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:5px}",
        ".table-wrap{background:white;border-radius:8px;border:1px solid var(--border);",
        "overflow-x:auto;margin-bottom:24px}",
        "table{width:100%;border-collapse:collapse;font-size:13px}",
        "thead tr.group-header th{background:#2a2a40;color:#ccc;padding:6px 12px;",
        "font-size:11px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;",
        "text-align:center;border-right:1px solid #444}",
        "thead tr.group-header th:first-child{text-align:left}",
        "thead tr.col-header th{background:var(--dark);color:white;padding:9px 12px;",
        "text-align:right;font-weight:600;font-size:12px;white-space:nowrap;",
        "border-right:1px solid #333}",
        "thead tr.col-header th.left,thead tr.col-header th:first-child{text-align:left}",
        "thead tr.col-header th.sep{border-right:2px solid var(--green)}",
        "tbody tr{border-bottom:1px solid #f0f0f0}",
        "tbody tr:nth-child(odd){background:#fafbfc}",
        "tbody tr:hover{background:#f0f4ff}",
        "tbody td{padding:9px 12px;text-align:right;white-space:nowrap;",
        "border-right:1px solid #f0f0f0;font-variant-numeric:tabular-nums}",
        "tbody td:first-child,tbody td.left{text-align:left;font-weight:600}",
        "tbody td.sep{border-right:2px solid #d4edba}",
        ".chart-grid{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:24px}",
        "@media(max-width:900px){header{padding:20px 24px}.badge{display:none}}",
        ".chart-card{background:white;border-radius:8px;border:1px solid var(--border);",
        "padding:20px}",
        ".chart-image-wrap{background:white;overflow-x:auto}",
        ".chart-image-wrap svg{display:block;width:100%;height:auto;min-width:760px}",
        "footer{text-align:center;padding:20px;color:var(--muted);font-size:12px;",
        "border-top:1px solid var(--border);margin-top:40px}",
    ]
    return "".join(lines)


def build_html(
    rows: list[dict[str, str]],
    title: str = "CLBP Perf analysis",
    subtitle: str | None = None,
    fieldnames: list[str] | None = None,
) -> str:
    """Build the complete combined performance report HTML.

    Args:
        rows (list[dict[str, str]]): Side-by-side rows.
        title (str): Report title.
        subtitle (str | None): Header subtitle. A generated timestamp is appended.
        fieldnames (list[str] | None): CSV header fields, so machine labels
            survive header-only inputs.

    Returns:
        str: Self-contained HTML.

    Examples:
        >>> "CLBP Perf analysis" in build_html([])
        True
    """
    grouped = _rows_by_clip(rows=rows)
    labels = _machine_labels(rows=rows, fieldnames=fieldnames)
    generated = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    escaped_title = html.escape(title)
    header_subtitle = subtitle or "Repeated-run FPS report"
    body = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"<title>{escaped_title}</title><style>{_css()}</style></head><body>",
        "<header><div>",
        '<div class="logo-nvidia">NVIDIA</div>',
        f'<div class="logo-title">{escaped_title}</div>',
        f'<div class="logo-sub">{html.escape(header_subtitle)} - '
        f"{generated} - clip labels by length</div>",
        '</div><div class="badge">FPS</div></header><main>',
        _exec_summary(grouped=grouped, labels=labels),
        _definitions_table(),
        _charts(grouped=grouped, labels=labels),
        _fps_table(grouped=grouped, labels=labels),
        _wall_time_table(grouped=grouped, labels=labels),
        f"<footer>{escaped_title} - generated {generated}</footer>",
        "</main></body></html>",
    ]
    return "".join(body)


def main() -> None:
    """Parse arguments and write the combined report.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="Build the combined CLBP perf report HTML.")
    parser.add_argument(
        "--side-by-side-csv",
        type=Path,
        default=Path("outputs/combined-perf-report/side_by_side_avg3.csv"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/combined-perf-report/combined_perf_report.html"),
    )
    parser.add_argument("--title", default="CLBP Perf analysis")
    parser.add_argument("--subtitle", default=None)
    args = parser.parse_args()

    # Fail fast on a bad path so a typo cannot produce an empty report.
    if not args.side_by_side_csv.is_file():
        parser.error(f"--side-by-side-csv is not a file: {args.side_by_side_csv}")

    rows, fieldnames = _read_rows(path=args.side_by_side_csv)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(
        build_html(rows=rows, title=args.title, subtitle=args.subtitle, fieldnames=fieldnames),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
