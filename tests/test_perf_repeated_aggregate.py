# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the machine-generic repeated-run perf aggregation and report."""

from pathlib import Path

import pytest

from scripts.perf.aggregate_repeated_perf import collect_baseline_rows
from scripts.perf.aggregate_repeated_perf import machine_label
from scripts.perf.aggregate_repeated_perf import machine_labels_from_fields
from scripts.perf.aggregate_repeated_perf import side_by_side_fields
from scripts.perf.aggregate_repeated_perf import side_by_side_rows
from scripts.perf.aggregate_repeated_perf import write_csv
from scripts.perf.build_combined_perf_report import _chart_series
from scripts.perf.build_combined_perf_report import _machine_labels
from scripts.perf.build_combined_perf_report import build_html


def _avg_row(machine: str, config: str, asd_fps: float) -> dict[str, object]:
    """Build one averaged row for *machine*/*config* with distinct metrics."""
    return {
        "machine": machine,
        "clip_length": "10s",
        "clip_sort_secs": 10.0,
        "config": config,
        "diarization_mode": "merged-per-speaker",
        "input_variant": "aac",
        "run_count": 3,
        "duration_secs_mean": 10.0,
        "video_frame_count_mean": 300.0,
        "preprocess_secs_mean": 0.5,
        "diarization_secs_mean": 0.2,
        "e2e_secs_mean": 2.0,
        "e2e_secs_stddev": 0.1,
        "realtime_factor_mean": 0.2,
        "realtime_factor_stddev": 0.01,
        "pipeline_fps_mean": 150.0,
        "pipeline_fps_stddev": 5.0,
        "asd_fps_mean": asd_fps,
        "asd_fps_stddev": 1.0,
        "lipsync_fps_mean": 50.0,
        "lipsync_fps_stddev": 2.0,
        "lipsync_frames_mean": 300.0,
        "s2s_contribution_secs_mean": 3.0 if config == "el" else None,
        "s2s_contribution_secs_stddev": 0.2 if config == "el" else None,
        "success_count": 3,
    }


@pytest.mark.unit
class TestMachineLabels:
    """Label derivation from directories and CSV headers."""

    def test_machine_label_sanitizes_basename(self) -> None:
        """Whitespace collapses to underscores; safe characters survive."""
        assert machine_label(Path("outputs/perf_x/machine a")) == "machine_a"
        assert machine_label(Path("outputs/perf_x/gpu-server.1")) == "gpu-server.1"

    def test_labels_recovered_from_fields_in_order(self) -> None:
        """Labels come back in first-seen column order."""
        fields = side_by_side_fields(["m1", "m2"])
        assert machine_labels_from_fields(fields) == ["m1", "m2"]

    def test_labels_with_underscores_and_stddev_columns(self) -> None:
        """Longest-suffix matching keeps underscored labels intact."""
        fields = ["clip_length", "gpu_a_asd_fps_stddev", "gpu_a_asd_fps"]
        assert machine_labels_from_fields(fields) == ["gpu_a"]


@pytest.mark.unit
class TestSideBySide:
    """Side-by-side schema and row construction."""

    def test_fields_start_with_shared_columns(self) -> None:
        """Shared columns lead, then per-machine input variants and duration."""
        fields = side_by_side_fields(["m1", "m2"])
        assert fields[:6] == [
            "clip_length",
            "config",
            "diarization_mode",
            "m1_input_variant",
            "m2_input_variant",
            "duration_secs",
        ]
        assert "m1_pipeline_fps_stddev" in fields
        assert "m2_video_frame_count" in fields

    def test_rows_key_metrics_by_label(self) -> None:
        """Each machine's averaged metrics land under its own columns."""
        avg_rows = [
            _avg_row(machine="m1", config="bypass", asd_fps=40.0),
            _avg_row(machine="m2", config="bypass", asd_fps=44.0),
            _avg_row(machine="m1", config="el", asd_fps=41.0),
        ]
        rows = side_by_side_rows(avg_rows, labels=["m1", "m2"])
        assert len(rows) == 2
        bypass = rows[0]
        assert bypass["m1_asd_fps"] == 40.0
        assert bypass["m2_asd_fps"] == 44.0
        assert bypass["duration_secs"] == 10.0
        el = rows[1]
        assert el["m1_asd_fps"] == 41.0
        # m2 has no el row, so its cells are blank rather than dropped.
        assert el["m2_asd_fps"] == ""

    def test_baseline_round_trip(self, tmp_path: Path) -> None:
        """A written side-by-side CSV folds back in as run-0 raw rows."""
        avg_rows = [
            _avg_row(machine="m1", config="bypass", asd_fps=40.0),
            _avg_row(machine="m2", config="bypass", asd_fps=44.0),
        ]
        labels = ["m1", "m2"]
        path = tmp_path / "side_by_side.csv"
        write_csv(side_by_side_rows(avg_rows, labels=labels), side_by_side_fields(labels), path)
        baseline = collect_baseline_rows(path=path)
        assert {row["machine"] for row in baseline} == {"m1", "m2"}
        assert all(row["run"] == 0 for row in baseline)
        by_machine = {row["machine"]: row for row in baseline}
        assert by_machine["m1"]["asd_fps"] == 40.0
        assert by_machine["m2"]["asd_fps"] == 44.0
        assert by_machine["m1"]["duration_secs"] == 10.0


@pytest.mark.unit
class TestReportBuilder:
    """Machine-generic HTML report generation."""

    def _side_rows(self, labels: list[str]) -> list[dict[str, str]]:
        """Build side-by-side rows (as CSV-shaped strings) for *labels*."""
        avg_rows = [
            _avg_row(machine=label, config=config, asd_fps=40.0 + index)
            for index, label in enumerate(labels)
            for config in ("bypass", "el")
        ]
        return [
            {key: "" if value in (None, "") else str(value) for key, value in row.items()}
            for row in side_by_side_rows(avg_rows, labels=labels)
        ]

    def test_labels_discovered_from_rows(self) -> None:
        """The report derives machine labels from the CSV columns."""
        rows = self._side_rows(["gpu-alpha", "gpu-beta"])
        assert _machine_labels(rows) == ["gpu-alpha", "gpu-beta"]

    def test_chart_series_covers_configs_per_label(self) -> None:
        """Two configs per machine, in bypass-then-el order."""
        series = _chart_series(["m1", "m2"])
        assert [entry[2] for entry in series] == [
            "m1 Bypass S2S",
            "m2 Bypass S2S",
            "m1 E2E / ElevenLabs",
            "m2 E2E / ElevenLabs",
        ]

    def test_two_machine_report_renders_labels(self) -> None:
        """Both machine labels appear; no hardcoded machine names or warnings."""
        html_text = build_html(rows=self._side_rows(["gpu-alpha", "gpu-beta"]))
        assert "gpu-alpha" in html_text
        assert "gpu-beta" in html_text
        assert "warn-cell" not in html_text
        assert "33 FPS" not in html_text
        assert "Per-Clip FPS Metrics" in html_text
        assert "Per-Clip Wall Time Metrics" in html_text

    def test_single_machine_report_degrades_gracefully(self) -> None:
        """One machine renders a complete report without comparison columns."""
        html_text = build_html(rows=self._side_rows(["solo"]))
        assert "solo" in html_text
        assert "Executive Summary" in html_text
