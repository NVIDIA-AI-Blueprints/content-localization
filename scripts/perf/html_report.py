# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an HTML performance dashboard from aggregated perf rows (LEGACY).

.. note::

    This is the legacy single-machine report. It loads Chart.js from a CDN, so the
    output is NOT self-contained and requires internet access to render. The
    current, self-contained multi-machine report (inline SVG, no external scripts)
    is produced by ``build_combined_perf_report.py`` — prefer that for new work.

The dashboard replicates the visual design of the reference perf_report.html:
dark NVIDIA header, executive-summary bullets, notation table, per-video metrics
table with grouped column headers, two side-by-side Chart.js charts, and a footer.

Public API
----------
:func:`write_html` — the only entry point callers need.

Examples:
    >>> import html_report  # doctest: +SKIP
    >>> html_report.write_html(rows=rows, output_path="outputs/perf/report.html")
"""

import datetime
import json
import os
import re

# ── Label helpers ─────────────────────────────────────────────────────────────


def _make_label(asset: str) -> str:
    """Return a short human-readable label for a video asset filename.

    Strips the 8-character hex hash prefix added by ``run_perf_matrix.sh``
    (pattern ``[0-9a-f]{8}_``), then strips optional ``short_``, ``medium_``,
    or ``long_`` tag prefixes, replaces underscores with spaces, and title-cases
    the result. Truncated to 30 characters.

    Args:
        asset (str): Raw asset filename, e.g. ``"a1b2c3d4_short_my_clip.mp4"``.

    Returns:
        str: Display label, e.g. ``"My Clip"``.

    Examples:
        >>> _make_label("a1b2c3d4_short_my_clip.mp4")
        'My Clip'
        >>> _make_label("jensen_fireside.mp4")
        'Jensen Fireside'
    """
    # Strip file extension.
    stem = os.path.splitext(asset)[0]
    # Strip leading 8-char hex hash if present (e.g. "a1b2c3d4_...").
    stem = re.sub(r"^[0-9a-f]{8}_", "", stem)
    # Strip known duration-tag prefixes.
    stem = re.sub(r"^(?:short|medium|long)_", "", stem)
    label = stem.replace("_", " ").title()
    return label[:30]


# ── Config detection ──────────────────────────────────────────────────────────


def _detect_full_config(rows: list[dict]) -> str | None:
    """Return the first non-bypass config present in *rows*.

    Prefers ``"el"`` over ``"camb"``; returns ``None`` when no non-bypass
    config is found.

    Args:
        rows (list[dict]): Rows from ``aggregate_perf.collect_rows``.

    Returns:
        str | None: Config key (``"el"`` or ``"camb"``), or ``None``.

    Examples:
        >>> _detect_full_config([{"config": "bypass"}, {"config": "el"}])
        'el'
        >>> _detect_full_config([{"config": "bypass"}]) is None
        True
    """
    configs = {r["config"] for r in rows}
    for preferred in ("el", "camb"):
        if preferred in configs:
            return preferred
    # Fall back to any non-bypass config.
    for config in sorted(configs):
        if config != "bypass":
            return config
    return None


# ── Data point builder ────────────────────────────────────────────────────────


def _build_data_points(rows: list[dict], full_config: str) -> list[dict]:
    """Build per-asset chart/table data points from aggregated perf rows.

    Filters to merged-per-speaker diarization rows only, then pairs the
    ``bypass`` and ``full_config`` rows for each asset. Only assets where
    the full_config row exists are included.

    Result keys
    -----------
    ``label``, ``res``, ``duration``, ``s2s_wall``,
    ``bypass_e2e``, ``full_e2e``,
    ``bypass_asd``, ``full_asd``,
    ``bypass_ls``, ``full_ls``,
    ``bypass_calc``, ``full_calc``,
    ``frames``

    Args:
        rows (list[dict]): Annotated rows from ``aggregate_perf.collect_rows``.
        full_config (str): Config key for the full pipeline (e.g. ``"el"``).

    Returns:
        list[dict]: Per-asset data points sorted by ``duration`` ascending.

    Examples:
        >>> _build_data_points([], "el")
        []
    """
    # Index merged-per-speaker rows by (config, asset).
    combined = [r for r in rows if r.get("diarization_mode") == "merged-per-speaker"]
    by_key: dict[tuple[str, str], dict] = {(r["config"], r["asset"]): r for r in combined}
    assets = {r["asset"] for r in combined}
    points: list[dict] = []
    for asset in sorted(assets):
        full_row = by_key.get((full_config, asset))
        # Skip assets without a full-config row — bypass-only data is incomplete.
        if full_row is None:
            continue
        bypass_row = by_key.get(("bypass", asset))
        bypass_e2e = bypass_row["e2e_pipeline_secs"] if bypass_row else None
        bypass_asd = bypass_row.get("asd_fps") if bypass_row else None
        bypass_ls = bypass_row.get("lipsync_fps") if bypass_row else None
        bypass_calc = bypass_row.get("calc_fps") if bypass_row else None
        points.append(
            {
                "label": _make_label(asset),
                "res": full_row.get("resolution") or "",
                "duration": full_row.get("duration_secs") or 0,
                "s2s_wall": full_row.get("s2s_wall_secs"),
                "bypass_e2e": bypass_e2e,
                "full_e2e": full_row["e2e_pipeline_secs"],
                "bypass_asd": bypass_asd,
                "full_asd": full_row.get("asd_fps"),
                "bypass_ls": bypass_ls,
                "full_ls": full_row.get("lipsync_fps"),
                "bypass_calc": bypass_calc,
                "full_calc": full_row.get("calc_fps"),
                "frames": full_row.get("video_frame_count"),
            }
        )
    points.sort(key=lambda d: d["duration"])
    return points


# ── Executive summary bullets ─────────────────────────────────────────────────


def _exec_bullets(
    data_points: list[dict],
    full_label: str,
) -> list[tuple[str, str, str]]:
    """Compute up to 4 executive-summary bullet points from chart data.

    Each returned tuple is ``(color_hex, title_html, body_html)``.

    Bullet logic
    ------------
    1. **Green** — overall average real-time factor for bypass and full pipeline.
    2. **Green** — list of videos where S2S is absorbed (long content,
       ``|full_e2e - bypass_e2e| <= 1.5 s`` and ``duration >= 30 s``).
    3. **Orange** — videos where ``(full_e2e - bypass_e2e) > 12 s``
       (short-clip S2S impact); omitted when no such videos exist.
    4. **Blue** — average LipSync FPS comparison (bypass vs full);
       only included when FPS data is present.

    Args:
        data_points (list[dict]): Output of :func:`_build_data_points`.
        full_label (str): Human-readable backend name, e.g. ``"ElevenLabs"``.

    Returns:
        list[tuple[str, str, str]]: Up to 4 ``(color, title, body)`` bullets.

    Examples:
        >>> _exec_bullets([], "ElevenLabs")
        []
    """
    if not data_points:
        return []

    bullets: list[tuple[str, str, str]] = []

    def _rt(e2e: float | None, dur: float) -> float | None:
        """Compute real-time factor; returns None when inputs are unavailable."""
        if e2e is None or not dur:
            return None
        return round(e2e / dur, 2)

    # Bullet 1 — average RT factor.
    bypass_rts = [
        _rt(d["bypass_e2e"], d["duration"])
        for d in data_points
        if d.get("bypass_e2e") is not None and d["duration"]
    ]
    full_rts = [
        _rt(d["full_e2e"], d["duration"])
        for d in data_points
        if d.get("full_e2e") is not None and d["duration"]
    ]
    if bypass_rts and full_rts:
        avg_bypass = round(sum(bypass_rts) / len(bypass_rts), 2)
        avg_full = round(sum(full_rts) / len(full_rts), 2)
        title = "Average real-time factors across all benchmark videos."
        body = (
            f"Bypass S2S (ASD+LipSync only): <strong>{avg_bypass}&times;</strong> avg RT. "
            f"{full_label} full pipeline: <strong>{avg_full}&times;</strong> avg RT. "
            "Values below 1.0 indicate faster-than-real-time processing."
        )
        bullets.append(("#76b900", title, body))

    # Bullet 2 — S2S absorbed by long content.
    absorbed = [
        d["label"]
        for d in data_points
        if (
            d.get("bypass_e2e") is not None
            and d.get("full_e2e") is not None
            and abs(d["full_e2e"] - d["bypass_e2e"]) <= 1.5
            and d["duration"] >= 30
        )
    ]
    if absorbed:
        names = ", ".join(absorbed)
        title = f"S2S overhead absorbed by long content (&ge;30&nbsp;s): {len(absorbed)} video(s)."
        body = (
            f"For these videos the {full_label} S2S step runs concurrently with "
            f"ASD+LipSync and adds &lt;1.5&nbsp;s to end-to-end time: {names}."
        )
        bullets.append(("#76b900", title, body))

    # Bullet 3 — short-clip S2S impact (optional).
    impacted = [
        (d["label"], round(d["full_e2e"] - d["bypass_e2e"], 1))
        for d in data_points
        if (
            d.get("bypass_e2e") is not None
            and d.get("full_e2e") is not None
            and (d["full_e2e"] - d["bypass_e2e"]) > 12
        )
    ]
    if impacted:
        detail = "; ".join(f"{lbl} (+{delta:.1f}s)" for lbl, delta in impacted)
        title = "Short clips are most impacted by S2S fixed overhead."
        body = (
            f"Videos where {full_label} adds &gt;12&nbsp;s: {detail}. "
            "Bypass S2S keeps these at or near real-time."
        )
        bullets.append(("#ff6600", title, body))

    # Bullet 4 — LipSync FPS comparison (optional).
    bypass_ls_vals = [d["bypass_ls"] for d in data_points if d.get("bypass_ls") is not None]
    full_ls_vals = [d["full_ls"] for d in data_points if d.get("full_ls") is not None]
    if bypass_ls_vals and full_ls_vals:
        avg_bls = round(sum(bypass_ls_vals) / len(bypass_ls_vals), 1)
        avg_fls = round(sum(full_ls_vals) / len(full_ls_vals), 1)
        pct = round((avg_bls - avg_fls) / avg_fls * 100) if avg_fls else None
        title = f"LipSync FPS: Bypass S2S vs {full_label} Full Pipeline."
        pct_str = f" ({pct}% higher)" if pct is not None else ""
        body = (
            f"Bypass S2S averages <strong>{avg_bls} FPS</strong>{pct_str} vs "
            f"<strong>{avg_fls} FPS</strong> in the {full_label} full pipeline. "
            "Without S2S contending for pipeline bandwidth, LipSync throughput increases."
        )
        bullets.append(("#0066cc", title, body))

    return bullets


# ── Static JS (no f-string — braces must not be escaped) ─────────────────────

_STATIC_JS = """\
const labels = DATA.map(d => d.label);

const GREEN   = "rgba(118,185,0,0.85)";
const GREEN_B = "rgba(118,185,0,1)";
const BLUE    = "rgba(0,102,204,0.75)";
const BLUE_B  = "rgba(0,102,204,1)";
const ORANGE  = "rgba(255,102,0,0.8)";

Chart.defaults.font.family =
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 14;

// ── BUILD TABLE ──────────────────────────────────────────────────────────────
const tbody = document.getElementById("mainTable");
const addCell = (tr, value, className = "") => {
  const td = document.createElement("td");
  if (className) {
    td.className = className;
  }
  td.textContent = value;
  tr.appendChild(td);
};
DATA.forEach(d => {
  const fmt = v => (v == null ? "-" : v.toFixed(1));
  const tr = document.createElement("tr");
  addCell(tr, d.label);
  addCell(tr, `${d.duration.toFixed(1)}s`, "dur");
  addCell(tr, d.res || "-", "dur sep");
  addCell(tr, `${fmt(d.s2s_wall)}s`);
  addCell(tr, `${fmt(d.bypass_e2e)}s`, "sep");
  addCell(tr, fmt(d.bypass_asd));
  addCell(tr, fmt(d.full_asd), "sep");
  addCell(tr, fmt(d.bypass_ls));
  addCell(tr, fmt(d.full_ls), "sep");
  addCell(tr, fmt(d.bypass_calc));
  addCell(tr, fmt(d.full_calc));
  tbody.appendChild(tr);
});

// ── CHART: e2e FPS ───────────────────────────────────────────────────────────
new Chart(document.getElementById("chartE2E"), {
  type: "bar",
  data: {
    labels,
    datasets: [
      {
        label: "Bypass S2S",
        data: DATA.map(d => d.bypass_calc),
        backgroundColor: GREEN,
        borderColor: GREEN_B,
        borderWidth: 1,
        barPercentage: 0.45,
      },
      {
        label: FULL_LABEL + " (Full Pipeline)",
        data: DATA.map(d => d.full_calc),
        backgroundColor: BLUE,
        borderColor: BLUE_B,
        borderWidth: 1,
        barPercentage: 0.45,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    scales: {
      x: { ticks: { font: { size: 11 } } },
      y: { title: { display: true, text: "e2e FPS" }, min: 0 },
    },
  },
});

// ── CHART: Pipeline Time ─────────────────────────────────────────────────────
new Chart(document.getElementById("chartPipe"), {
  type: "bar",
  data: {
    labels,
    datasets: [
      {
        label: "S2S Wall Time (" + FULL_LABEL + ")",
        data: DATA.map(d => d.s2s_wall),
        backgroundColor: ORANGE,
        borderColor: "rgba(255,102,0,1)",
        borderWidth: 1,
        barPercentage: 0.3,
      },
      {
        label: "ASD+LipSync Time (Bypass S2S)",
        data: DATA.map(d => d.bypass_e2e),
        backgroundColor: GREEN,
        borderColor: GREEN_B,
        borderWidth: 1,
        barPercentage: 0.3,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    scales: {
      x: { ticks: { font: { size: 11 } } },
      y: { title: { display: true, text: "Wall Time (s)" }, min: 0 },
    },
  },
  plugins: [{
    id: "durLine",
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      ctx.save();
      ctx.strokeStyle = "#cc0000";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      DATA.forEach((d, i) => {
        const xPx = scales.x.getPixelForValue(i);
        const yPx = scales.y.getPixelForValue(d.duration);
        ctx.beginPath();
        ctx.moveTo(xPx - 16, yPx);
        ctx.lineTo(xPx + 16, yPx);
        ctx.stroke();
      });
      ctx.fillStyle = "#cc0000";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(
        "— video duration",
        chartArea.right - 110,
        chartArea.top + 14,
      );
      ctx.restore();
    },
  }],
});
"""

# ── Backend display names ─────────────────────────────────────────────────────

_BACKEND_DISPLAY: dict[str, str] = {
    "el": "ElevenLabs",
    "camb": "Camb AI",
}


def _backend_display_name(config: str) -> str:
    """Return a human-readable backend name for *config*.

    Args:
        config (str): Config directory key (``"el"``, ``"camb"``, etc.).

    Returns:
        str: Display name.

    Examples:
        >>> _backend_display_name("el")
        'ElevenLabs'
        >>> _backend_display_name("custom")
        'custom'
    """
    return _BACKEND_DISPLAY.get(config, config)


# ── Notation definitions ──────────────────────────────────────────────────────


def _notation_rows(full_label: str) -> list[tuple[str, str]]:
    """Return (term, definition) pairs for the Notation & Definitions table.

    Args:
        full_label (str): Human-readable backend name, e.g. ``"ElevenLabs"``.

    Returns:
        list[tuple[str, str]]: Ordered list of (term, HTML-safe definition) pairs.

    Examples:
        >>> rows = _notation_rows("ElevenLabs")
        >>> rows[0][0]
        'Full Pipeline'
    """
    return [
        (
            "Full Pipeline",
            f"Full end-to-end pipeline: {full_label} S2S speech translation runs "
            "concurrently with ASD and LipSync. The translated audio feeds into "
            "LipSync to produce the final dubbed video.",
        ),
        (
            "Bypass S2S",
            "Pipeline mode where the Speech-to-Speech (S2S) translation step is "
            "skipped entirely. Only ASD and LipSync run. Used to isolate the cost "
            "of ASD+LipSync independently from S2S, and as the baseline for "
            "measuring S2S overhead.",
        ),
        (
            "S2S (Speech-to-Speech)",
            f"The translation service that converts source-language audio into "
            f"target-language audio, here backed by {full_label}. The S2S wall "
            "time is the end-to-end latency for the backend to return the "
            "translated audio track.",
        ),
        (
            "ASD (Active Speaker Detection)",
            "NVIDIA NIM that detects which speaker is visible and active in each "
            "video frame, producing per-frame speaker bounding boxes used by LipSync.",
        ),
        (
            "LipSync",
            "NVIDIA NIM that re-animates the speaker&#39;s lips in each video frame "
            "to match the translated audio. Runs on GPU and is the primary compute "
            "bottleneck in Bypass S2S mode.",
        ),
        (
            "ASD FPS",
            "Frames per second reported internally by the ASD NIM (scraped from "
            "Docker logs). Measures raw inference throughput of the ASD model on "
            "the GPU, not including gRPC or streaming overhead.",
        ),
        (
            "LipSync FPS",
            "Frames per second reported internally by the LipSync NIM (scraped from "
            "Docker logs). Measures raw inference throughput of the LipSync model on "
            "the GPU, not including gRPC or streaming overhead.",
        ),
        (
            "e2e FPS",
            "End-to-end frames per second = source video frame count &divide; "
            "total pipeline wall time. Unlike NIM FPS, this includes all overhead: "
            "gRPC transport, preprocessing, audio sync waits, and queuing. "
            "It is the true throughput observable from outside the pipeline.",
        ),
        (
            "ASD+LipSync Time (Bypass S2S)",
            "Total wall-clock time for the Bypass S2S pipeline to complete — "
            "i.e., the time for ASD and LipSync to process the entire video with "
            "no S2S step. This is the floor latency achievable by the GPU pipeline alone.",
        ),
        (
            "S2S Wall Time",
            f"Total wall-clock time for {full_label} to translate the audio track. "
            "This runs concurrently with ASD+LipSync, so for long videos it is "
            "fully absorbed by the pipeline and adds no extra latency.",
        ),
    ]


# ── HTML assembly ─────────────────────────────────────────────────────────────


def write_html(rows: list[dict], output_path: str) -> None:
    """Generate an HTML performance dashboard and write it to disk (LEGACY, not self-contained).

    Reads aggregated perf rows (from ``aggregate_perf.collect_rows``), detects
    the full-pipeline config, pairs it with the bypass rows, computes executive
    summary bullets, and emits a single HTML file with embedded CSS, inline
    Chart.js data, and a CDN script tag for Chart.js 4.4.4.

    The HTML requires internet access to render charts (Chart.js is loaded from
    ``cdn.jsdelivr.net``). Open the file in any modern browser.

    Args:
        rows (list[dict]): Annotated rows from ``aggregate_perf.collect_rows``.
        output_path (str): Destination ``.html`` path. Parent directories are
            created if they do not exist.

    Examples:
        >>> write_html(rows=[], output_path="/tmp/report.html")  # doctest: +SKIP
    """
    full_config = _detect_full_config(rows=rows)
    full_label = _backend_display_name(full_config) if full_config else "Full Pipeline"
    data_points = _build_data_points(
        rows=rows,
        full_config=full_config or "el",
    )
    bullets = _exec_bullets(data_points=data_points, full_label=full_label)
    generated = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_assets = len(data_points)
    configs_present = sorted({r["config"] for r in rows})

    # Serialize data as JS constants — injected before the static JS block.
    js_init = f"const DATA = {json.dumps(data_points, indent=2)};\n"
    js_init += f"const FULL_LABEL = {json.dumps(full_label)};\n"

    h: list[str] = []

    # ── HTML head ─────────────────────────────────────────────────────────────
    h.append("<!DOCTYPE html>")
    h.append('<html lang="en">')
    h.append("<head>")
    h.append('<meta charset="UTF-8">')
    h.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    h.append(f"<title>CLBP Performance Report — {full_label}</title>")
    h.append(
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>'
    )
    h.append("<style>")
    h.append("  :root {")
    h.append("    --green: #76b900; --dark: #1a1a2e; --border: #e0e0e0;")
    h.append("    --text: #1a1a1a; --muted: #666;")
    h.append("  }")
    h.append("  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }")
    h.append("  body {")
    h.append('    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;')
    h.append("    background: #f5f6fa; color: var(--text);")
    h.append("    font-size: 14px; line-height: 1.5;")
    h.append("  }")
    h.append("  header {")
    h.append("    background: var(--dark); color: white;")
    h.append("    padding: 24px 40px; display: flex; align-items: center; gap: 20px;")
    h.append("  }")
    h.append(
        "  .logo-nvidia { font-size: 11px; letter-spacing: 4px; text-transform: uppercase;"
        " color: var(--green); font-weight: 700; }"
    )
    h.append("  .logo-title { font-size: 20px; font-weight: 700; color: white; margin-top: 2px; }")
    h.append("  .logo-sub { font-size: 13px; color: #aaa; margin-top: 2px; }")
    h.append(
        "  .badge { margin-left: auto; background: var(--green); color: #000;"
        " font-weight: 700; padding: 6px 14px; border-radius: 4px; font-size: 13px; }"
    )
    h.append("  main { max-width: 1280px; margin: 0 auto; padding: 32px 24px; }")
    h.append("  h2 {")
    h.append("    font-size: 17px; font-weight: 700; color: var(--dark);")
    h.append("    margin: 32px 0 10px; padding-bottom: 7px;")
    h.append("    border-bottom: 2px solid var(--green);")
    h.append("  }")
    h.append("  h3 { font-size: 14px; font-weight: 600; color: var(--dark); margin-bottom: 8px; }")
    h.append(
        "  p.note { color: var(--muted); font-size: 12px; margin-top: 8px; font-style: italic; }"
    )
    h.append(
        "  .legend-strip { display: flex; gap: 20px; align-items: center;"
        " margin-bottom: 16px; font-size: 13px; }"
    )
    h.append(
        "  .legend-dot { width: 12px; height: 12px; border-radius: 50%;"
        " display: inline-block; margin-right: 5px; }"
    )
    h.append(
        "  .table-wrap { background: white; border-radius: 8px;"
        " border: 1px solid var(--border); overflow-x: auto; margin-bottom: 24px; }"
    )
    h.append("  table { width: 100%; border-collapse: collapse; font-size: 13px; }")
    h.append("  thead tr.group-header th {")
    h.append("    background: #2a2a40; color: #ccc; padding: 6px 12px;")
    h.append("    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
    h.append("    text-transform: uppercase; text-align: center;")
    h.append("    border-right: 1px solid #444;")
    h.append("  }")
    h.append("  thead tr.group-header th:first-child { text-align: left; }")
    h.append("  thead tr.col-header th {")
    h.append("    background: var(--dark); color: white; padding: 9px 12px;")
    h.append("    text-align: right; font-weight: 600; font-size: 12px;")
    h.append("    white-space: nowrap; border-right: 1px solid #333;")
    h.append("  }")
    h.append(
        "  thead tr.col-header th:first-child, thead tr.col-header th.left { text-align: left; }"
    )
    h.append("  thead tr.col-header th.sep { border-right: 2px solid #76b900; }")
    h.append("  tbody tr { border-bottom: 1px solid #f0f0f0; }")
    h.append("  tbody tr:nth-child(odd) { background: #fafbfc; }")
    h.append("  tbody tr:hover { background: #f0f4ff; }")
    h.append("  tbody td {")
    h.append("    padding: 9px 12px; text-align: right;")
    h.append("    white-space: nowrap; border-right: 1px solid #f0f0f0;")
    h.append("    font-variant-numeric: tabular-nums;")
    h.append("  }")
    h.append("  tbody td:first-child { text-align: left; font-weight: 600; }")
    h.append("  tbody td.sep { border-right: 2px solid #d4edba; }")
    h.append("  tbody td.dur { color: var(--muted); font-weight: 400; }")
    h.append(
        "  .chart-grid { display: grid; grid-template-columns: 1fr 1fr;"
        " gap: 20px; margin-bottom: 24px; }"
    )
    h.append("  @media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }")
    h.append(
        "  .chart-card { background: white; border-radius: 8px;"
        " border: 1px solid var(--border); padding: 20px; }"
    )
    h.append("  .canvas-wrap { position: relative; height: 300px; }")
    h.append("  footer {")
    h.append("    text-align: center; padding: 20px;")
    h.append("    color: var(--muted); font-size: 12px;")
    h.append("    border-top: 1px solid var(--border); margin-top: 40px;")
    h.append("  }")
    h.append("</style>")
    h.append("</head>")
    h.append("<body>")
    h.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    h.append("<header>")
    h.append("  <div>")
    h.append('    <div class="logo-nvidia">NVIDIA</div>')
    h.append('    <div class="logo-title">CLBP &mdash; Content Localization Blueprint</div>')
    h.append(
        f'    <div class="logo-sub">Performance Report &mdash; {full_label} backend'
        f" &mdash; {generated} &mdash; Merged-per-speaker diarization</div>"
    )
    h.append("  </div>")
    h.append(f'  <div class="badge">{full_label}</div>')
    h.append("</header>")
    h.append("")
    h.append("<main>")
    h.append("")

    # ── Executive Summary ─────────────────────────────────────────────────────
    h.append(
        '<section style="background:white;border-radius:8px;border:1px solid var(--border);'
        'padding:24px 28px;margin-bottom:28px;">'
    )
    h.append('  <h2 style="margin-top:0;border-color:#ff6600;">Executive Summary</h2>')
    h.append('  <ul style="list-style:none;padding:0;">')
    for color, title, body in bullets:
        h.append(
            '    <li style="display:flex;gap:12px;padding:10px 0;'
            'border-bottom:1px solid #f0f0f0;font-size:13.5px;">'
        )
        h.append(
            f'      <span style="width:8px;height:8px;border-radius:50%;'
            f'background:{color};flex-shrink:0;margin-top:5px;"></span>'
        )
        h.append(f"      <span><strong>{title}</strong> {body}</span>")
        h.append("    </li>")
    if not bullets:
        h.append(
            '    <li style="padding:10px 0;font-size:13.5px;color:var(--muted);">'
            "No data available.</li>"
        )
    h.append("  </ul>")
    h.append("</section>")
    h.append("")

    # ── Notation & Definitions ────────────────────────────────────────────────
    h.append("<h2>Notation &amp; Definitions</h2>")
    h.append('<div class="table-wrap">')
    h.append("<table>")
    h.append("  <thead>")
    h.append('    <tr class="col-header">')
    h.append('      <th class="left" style="width:220px">Term</th>')
    h.append('      <th class="left">Definition</th>')
    h.append("    </tr>")
    h.append("  </thead>")
    h.append("  <tbody>")
    for term, defn in _notation_rows(full_label=full_label):
        h.append("    <tr>")
        h.append(f'      <td style="text-align:left;font-weight:700">{term}</td>')
        h.append(f'      <td style="text-align:left;white-space:normal">{defn}</td>')
        h.append("    </tr>")
    h.append("  </tbody>")
    h.append("</table>")
    h.append("</div>")
    h.append("")

    # ── Per-Video Metrics Table ───────────────────────────────────────────────
    h.append("<h2>Per-Video Metrics (merged-per-speaker)</h2>")
    h.append("")
    h.append('<div class="legend-strip">')
    h.append(
        '  <span><span class="legend-dot" style="background:#76b900"></span>'
        "Bypass S2S = ASD + LipSync only (no S2S)</span>"
    )
    h.append(
        f'  <span><span class="legend-dot" style="background:#0066cc"></span>'
        f"Full Pipeline = {full_label} S2S + ASD + LipSync</span>"
    )
    h.append(
        '  <span style="color:var(--muted)">e2e FPS = source frames &divide; '
        "pipeline wall time</span>"
    )
    h.append("</div>")
    h.append("")
    h.append('<div class="table-wrap">')
    h.append("<table>")
    h.append("  <thead>")
    h.append('    <tr class="group-header">')
    h.append('      <th rowspan="2" style="vertical-align:middle">Video</th>')
    h.append('      <th rowspan="2" style="vertical-align:middle;text-align:right">Duration</th>')
    h.append(
        '      <th rowspan="2" style="vertical-align:middle;text-align:right;'
        'border-right:2px solid #76b900">Resolution</th>'
    )
    h.append('      <th colspan="2" style="border-right:2px solid #76b900">Pipeline Time (s)</th>')
    h.append('      <th colspan="2" style="border-right:2px solid #76b900">ASD FPS</th>')
    h.append('      <th colspan="2" style="border-right:2px solid #76b900">LipSync FPS</th>')
    h.append('      <th colspan="2">e2e FPS</th>')
    h.append("    </tr>")
    h.append('    <tr class="col-header">')
    h.append(f"      <th>S2S ({full_label} wall)</th>")
    h.append(
        '      <th class="sep" style="border-left:2px solid #76b900">'
        "ASD+LipSync<br>(Bypass S2S)</th>"
    )
    h.append("      <th>Bypass S2S</th>")
    h.append('      <th class="sep">Full Pipeline</th>')
    h.append("      <th>Bypass S2S</th>")
    h.append('      <th class="sep">Full Pipeline</th>')
    h.append("      <th>Bypass S2S</th>")
    h.append("      <th>Full Pipeline</th>")
    h.append("    </tr>")
    h.append("  </thead>")
    h.append('  <tbody id="mainTable"></tbody>')
    h.append("</table>")
    h.append("</div>")
    h.append("")

    # ── Charts ────────────────────────────────────────────────────────────────
    h.append("<h2>Charts</h2>")
    h.append("")
    h.append('<div class="chart-grid">')
    h.append('  <div class="chart-card">')
    h.append("    <h3>e2e FPS &mdash; Bypass S2S vs Full Pipeline</h3>")
    h.append('    <div class="canvas-wrap"><canvas id="chartE2E"></canvas></div>')
    h.append(
        '    <p class="note">Source frames &divide; total pipeline wall time. '
        "Measures true end-to-end throughput including all overhead.</p>"
    )
    h.append("  </div>")
    h.append('  <div class="chart-card">')
    h.append("    <h3>Pipeline Time &mdash; S2S Wall vs ASD+LipSync (Bypass S2S)</h3>")
    h.append('    <div class="canvas-wrap"><canvas id="chartPipe"></canvas></div>')
    h.append(
        '    <p class="note">Orange = S2S wall time; green = ASD+LipSync time '
        "(Bypass S2S). Red tick = video duration (real-time boundary).</p>"
    )
    h.append("  </div>")
    h.append("</div>")
    h.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    h.append("</main>")
    h.append("")
    h.append("<footer>")
    configs_str = ", ".join(configs_present)
    h.append(
        f"  CLBP Performance Report &mdash; {full_label} backend &mdash; {generated}"
        f" &mdash; {n_assets} asset(s) &bull; configs: {configs_str}"
        " &bull; Merged-per-speaker diarization"
    )
    h.append("</footer>")
    h.append("")

    # ── Inline JS ─────────────────────────────────────────────────────────────
    h.append("<script>")
    h.append(js_init)
    h.append(_STATIC_JS)
    h.append("</script>")
    h.append("</body>")
    h.append("</html>")

    html_str = "\n".join(h) + "\n"

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_str)
