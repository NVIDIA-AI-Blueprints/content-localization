# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/perf/aggregate_perf.py."""

import csv
import json
from pathlib import Path

import pytest

from scripts.perf import html_report
from scripts.perf.aggregate_perf import add_s2s_contribution
from scripts.perf.aggregate_perf import build_fps_table
from scripts.perf.aggregate_perf import collect_rows
from scripts.perf.aggregate_perf import write_csv
from scripts.perf.aggregate_perf import write_markdown


def _write_report(
    path: Path,
    video_name: str,
    duration: float,
    pipeline: float,
    rt: float,
) -> None:
    """Write a minimal batch_processing_report.json at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "video_name": video_name,
                        "video_duration_secs": duration,
                        "pipeline_time_secs": pipeline,
                        "realtime_factor": rt,
                        "success": True,
                    }
                ],
                "summary": {},
            }
        )
    )


@pytest.fixture
def perf_tree(tmp_path: Path) -> Path:
    """Build an el + bypass perf tree with both diarization modes."""
    _write_report(
        path=tmp_path / "el" / "combine" / "batch_processing_report.json",
        video_name="a.mp4",
        duration=17.0,
        pipeline=10.0,
        rt=0.59,
    )
    _write_report(
        path=tmp_path / "el" / "per_segment" / "batch_processing_report.json",
        video_name="a.mp4",
        duration=17.0,
        pipeline=11.5,
        rt=0.68,
    )
    _write_report(
        path=tmp_path / "bypass" / "combine" / "batch_processing_report.json",
        video_name="a.mp4",
        duration=17.0,
        pipeline=4.0,
        rt=0.24,
    )
    _write_report(
        path=tmp_path / "bypass" / "per_segment" / "batch_processing_report.json",
        video_name="a.mp4",
        duration=17.0,
        pipeline=4.3,
        rt=0.25,
    )
    s2s_dir = tmp_path / "el" / "s2s"
    s2s_dir.mkdir(parents=True)
    (s2s_dir / "a.json").write_text(
        json.dumps(
            {
                "asset": "a.wav",
                "duration_secs": 17.0,
                "wall_time_secs": 6.2,
                "mean_per_chunk_latency": 0.8,
                "is_realtime": True,
            }
        )
    )
    return tmp_path


@pytest.mark.unit
class TestCollectRows:
    """Test cases for ``collect_rows``."""

    def test_collects_all_configs_and_modes(self, perf_tree):
        """One row per (config, diarization_mode, asset)."""
        rows = collect_rows(in_dir=str(perf_tree))
        keys = {(r["config"], r["diarization_mode"]) for r in rows}
        assert keys == {
            ("el", "merged-per-speaker"),
            ("el", "per-segment"),
            ("bypass", "merged-per-speaker"),
            ("bypass", "per-segment"),
        }

    def test_joins_s2s_latency_by_stem(self, perf_tree):
        """S2S latency JSON is matched to the e2e rows by asset stem."""
        rows = collect_rows(in_dir=str(perf_tree))
        el_rows = [r for r in rows if r["config"] == "el"]
        assert all(r["s2s_wall_secs"] == 6.2 for r in el_rows)
        assert all(r["s2s_is_realtime"] is True for r in el_rows)

    def test_reads_per_asset_subdir_layout(self, tmp_path: Path):
        """Reports nested one-per-asset under the mode dir are all collected."""
        # One-video-per-process layout: <config>/<mode>/<stem>/report.json
        for stem, pipe in (("a", 5.0), ("b", 9.0)):
            _write_report(
                path=tmp_path / "el" / "combine" / stem / "batch_processing_report.json",
                video_name=f"{stem}.mp4",
                duration=10.0,
                pipeline=pipe,
                rt=pipe / 10.0,
            )
        rows = collect_rows(in_dir=str(tmp_path))
        assets = {r["asset"] for r in rows}
        assert assets == {"a.mp4", "b.mp4"}

    def test_skips_malformed_reports(self, tmp_path: Path):
        """Malformed JSON and non-dict report shapes are skipped, not fatal."""
        # Malformed JSON in one config.
        bad = tmp_path / "el" / "combine" / "batch_processing_report.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("NOT JSON")
        # A valid-JSON-but-wrong-shape report (a list) in another config.
        wrong = tmp_path / "camb" / "combine" / "batch_processing_report.json"
        wrong.parent.mkdir(parents=True)
        wrong.write_text("[1, 2, 3]")
        # A good report so aggregation still yields a row.
        _write_report(
            path=tmp_path / "bypass" / "combine" / "batch_processing_report.json",
            video_name="a.mp4",
            duration=17.0,
            pipeline=4.0,
            rt=0.24,
        )

        rows = collect_rows(in_dir=str(tmp_path))
        # Only the good report contributes a row; the bad ones are skipped.
        assert len(rows) == 1
        assert rows[0]["config"] == "bypass"

    def test_non_numeric_timing_is_coerced(self, tmp_path: Path):
        """A non-numeric pipeline_time_secs is coerced so arithmetic is safe."""
        bad = tmp_path / "el" / "combine" / "batch_processing_report.json"
        bad.parent.mkdir(parents=True)
        bad.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "video_name": "a.mp4",
                            "video_duration_secs": 17.0,
                            "pipeline_time_secs": "oops",
                            "realtime_factor": None,
                            "success": True,
                        }
                    ]
                }
            )
        )
        rows = collect_rows(in_dir=str(tmp_path))
        add_s2s_contribution(rows=rows)  # must not raise on the coerced 0.0
        assert rows[0]["e2e_pipeline_secs"] == 0.0
        assert rows[0]["realtime_factor"] == 0.0


@pytest.mark.unit
class TestS2SContribution:
    """Test cases for ``add_s2s_contribution``."""

    def test_subtracts_bypass_floor(self, perf_tree):
        """s2s_contribution = e2e(full) - e2e(bypass) for the same mode."""
        rows = collect_rows(in_dir=str(perf_tree))
        add_s2s_contribution(rows=rows)
        el_combine = next(
            r for r in rows if r["config"] == "el" and r["diarization_mode"] == "merged-per-speaker"
        )
        assert el_combine["s2s_contribution_secs"] == pytest.approx(6.0)
        bypass = next(r for r in rows if r["config"] == "bypass")
        assert bypass["s2s_contribution_secs"] is None


@pytest.mark.unit
class TestWriters:
    """Test cases for CSV and Markdown writers."""

    def test_csv_and_markdown_written(self, perf_tree, tmp_path):
        """Both report files are produced with the expected content."""
        rows = collect_rows(in_dir=str(perf_tree))
        add_s2s_contribution(rows=rows)

        csv_path = tmp_path / "out.csv"
        md_path = tmp_path / "out.md"
        write_csv(rows=rows, output_path=str(csv_path))
        write_markdown(rows=rows, output_path=str(md_path))

        with open(csv_path, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 4
        assert "s2s_contribution_secs" in csv_rows[0]

        md = md_path.read_text()
        assert "# CLBP Performance Report" in md
        assert "Diarization Granularity Delta" in md


@pytest.mark.unit
class TestFpsAggregation:
    """Test cases for ASD/LipSync FPS aggregation."""

    def test_overall_fps_and_table(self, tmp_path: Path):
        """fps.json next to a report yields overall FPS = frames / pipeline."""
        d = tmp_path / "bypass" / "combine" / "a"
        _write_report(
            path=d / "batch_processing_report.json",
            video_name="a.mp4",
            duration=41.0,
            pipeline=90.0,
            rt=2.2,
        )
        (d / "fps.json").write_text(
            json.dumps(
                {"asd_frames": 1800, "asd_fps": 27.8, "lipsync_frames": 1800, "lipsync_fps": 24.8}
            )
        )
        rows = collect_rows(in_dir=str(tmp_path))
        assert len(rows) == 1
        # overall = 1800 / 90.0 = 20.0
        assert rows[0]["overall_asd_lipsync_fps"] == pytest.approx(20.0)
        assert rows[0]["lipsync_fps"] == 24.8
        table = build_fps_table(rows)
        # Columns are now nim_lipsync_fps and calc_fps; 24.8 is the NIM value.
        assert "nim_lipsync_fps" in table
        assert "24.80" in table

    def test_no_fps_data(self, tmp_path: Path):
        """Without fps.json, FPS fields are None and the table notes it."""
        _write_report(
            path=tmp_path / "bypass" / "combine" / "a" / "batch_processing_report.json",
            video_name="a.mp4",
            duration=41.0,
            pipeline=90.0,
            rt=2.2,
        )
        rows = collect_rows(in_dir=str(tmp_path))
        assert rows[0]["overall_asd_lipsync_fps"] is None
        assert build_fps_table(rows) == "_No ASD/LipSync FPS captured._"


@pytest.mark.unit
class TestHtmlReport:
    """Test cases for scripts/perf/html_report.py."""

    def test_metric_rows_use_text_content(self, tmp_path: Path):
        """Metric table rows must not interpolate labels into innerHTML."""
        asset = "evil_<img src=x onerror=alert(1)>.mp4"
        rows = [
            {
                "config": "el",
                "asset": asset,
                "diarization_mode": "merged-per-speaker",
                "resolution": "1920x1080",
                "duration_secs": 1.0,
                "video_frame_count": 30,
                "e2e_pipeline_secs": 2.0,
                "s2s_wall_secs": 1.0,
                "asd_fps": 10.0,
                "lipsync_fps": 11.0,
                "calc_fps": 15.0,
                "success": True,
            },
            {
                "config": "bypass",
                "asset": asset,
                "diarization_mode": "merged-per-speaker",
                "resolution": "1920x1080",
                "duration_secs": 1.0,
                "video_frame_count": 30,
                "e2e_pipeline_secs": 1.5,
                "s2s_wall_secs": None,
                "asd_fps": 12.0,
                "lipsync_fps": 13.0,
                "calc_fps": 20.0,
                "success": True,
            },
        ]
        report_path = tmp_path / "report.html"

        html_report.write_html(rows=rows, output_path=str(report_path))

        html = report_path.read_text()
        if "tr.innerHTML" in html:
            pytest.fail("Metric rows still use innerHTML")
        if 'document.createElement("td")' not in html:
            pytest.fail("Metric rows do not create table cells explicitly")
        if "td.textContent = value" not in html:
            pytest.fail("Metric rows do not assign cell text with textContent")


@pytest.mark.unit
class TestPerfRunnerScript:
    """Text-level regressions for scripts/perf/run_perf_matrix.sh."""

    def test_nim_fps_missing_patterns_are_non_fatal(self):
        script = Path("scripts/perf/run_perf_matrix.sh").read_text()

        if "grep -oE '[0-9.]+' || true" not in script:
            pytest.fail("FPS extraction does not tolerate missing NIM log FPS")
        if "grep -oE '[0-9]+' || true" not in script:
            pytest.fail("Frame extraction does not tolerate missing NIM log frames")

    def test_s2s_archive_variable_is_not_local_at_top_level(self):
        script = Path("scripts/perf/run_perf_matrix.sh").read_text()

        if "local s2s_artifact_dir=" in script:
            pytest.fail("S2S artifact variable is still declared local at top level")
        if 's2s_artifact_dir="${ARTIFACTS_DIR}/${CONFIG}/s2s"' not in script:
            pytest.fail("S2S artifact variable assignment is missing")


if __name__ == "__main__":
    pytest.main([__file__])
