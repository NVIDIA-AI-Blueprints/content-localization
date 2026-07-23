# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency analysis module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from client.s2s.latency_analysis import _percentile
from client.s2s.latency_analysis import calculate_per_chunk_latencies
from client.s2s.latency_analysis import plot_latency
from client.s2s.latency_analysis import write_latency_json


@pytest.mark.unit
class TestCalculateLatency:
    """Test cases for calculate_per_chunk_latencies function."""

    def test_calculate_latency_basic(self):
        """Test basic latency calculation."""
        # Create mock ledgers
        input_ledger = {
            0: 1000.0,  # chunk_id: timestamp
            1: 1001.0,
            2: 1002.0,
        }

        output_ledger = {
            0: 1000.5,  # chunk_id: timestamp
            1: 1001.5,
            2: 1002.5,
        }

        # Calculate latency
        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        # Check results
        assert len(latency_data) == 3
        assert latency_data[0] == 0.5  # 1000.5 - 1000.0
        assert latency_data[1] == 0.5  # 1001.5 - 1001.0
        assert latency_data[2] == 0.5  # 1002.5 - 1002.0

    def test_calculate_latency_different_timings(self):
        """Test latency calculation with different timings."""
        input_ledger = {
            0: 1000.0,
            1: 1001.0,
            2: 1002.0,
        }

        output_ledger = {
            0: 1000.2,  # Faster processing
            1: 1001.8,  # Slower processing
            2: 1002.1,  # Normal processing
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 3
        assert latency_data[0] == pytest.approx(0.2)
        assert latency_data[1] == pytest.approx(0.8)
        assert latency_data[2] == pytest.approx(0.1)

    def test_calculate_latency_missing_chunks(self):
        """Test latency calculation with missing chunks."""
        input_ledger = {
            0: 1000.0,
            1: 1001.0,
            2: 1002.0,
        }

        output_ledger = {
            0: 1000.5,
            # Missing chunk 1
            2: 1002.5,
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        # Should only include chunks that exist in both ledgers
        assert len(latency_data) == 2
        assert latency_data[0] == 0.5  # chunk 0
        assert latency_data[1] == 0.5  # chunk 2

    def test_calculate_latency_empty_ledgers(self):
        """Test latency calculation with empty ledgers."""
        input_ledger = {}
        output_ledger = {}

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 0

    def test_calculate_latency_one_empty_ledger(self):
        """Test latency calculation with one empty ledger."""
        input_ledger = {0: 1000.0, 1: 1001.0}
        output_ledger = {}

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 0

    def test_calculate_latency_negative_latency(self):
        """Test latency calculation with negative latency (sink before source)."""
        input_ledger = {
            0: 1000.0,
        }

        output_ledger = {
            0: 999.5,  # Sink timestamp before source timestamp
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 1
        assert latency_data[0] == -0.5  # Negative latency

    def test_calculate_latency_large_numbers(self):
        """Test latency calculation with large timestamp values."""
        input_ledger = {
            0: 1640995200.0,  # Unix timestamp
            1: 1640995201.0,
        }

        output_ledger = {
            0: 1640995200.5,
            1: 1640995201.5,
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 2
        assert latency_data[0] == 0.5
        assert latency_data[1] == 0.5


@pytest.mark.unit
class TestPlotLatencyAnalysis:
    """Test cases for plot_latency function."""

    def test_plot_latency_basic(self):
        """Test basic latency plotting."""
        # Create sample latency data
        output_stream_latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
        per_chunk_latencies = [0.05, 0.1, 0.08, 0.12, 0.09]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_empty_data(self):
        """Test latency plotting with empty data."""
        output_stream_latencies = []
        per_chunk_latencies = []
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created even with empty data
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_single_point(self):
        """Test latency plotting with single data point."""
        output_stream_latencies = [0.5]
        per_chunk_latencies = [0.2]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_large_dataset(self):
        """Test latency plotting with large dataset."""
        # Create larger dataset
        np.random.seed(42)  # For reproducible results
        output_stream_latencies = np.random.normal(0.2, 0.05, 100).tolist()
        per_chunk_latencies = np.random.normal(0.1, 0.02, 100).tolist()
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_negative_values(self):
        """Test latency plotting with negative values."""
        output_stream_latencies = [-0.1, 0.2, -0.15, 0.3, -0.25]
        per_chunk_latencies = [-0.05, 0.1, -0.08, 0.12, -0.09]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_extreme_values(self):
        """Test latency plotting with extreme values."""
        output_stream_latencies = [0.001, 10.0, 0.0001, 100.0]
        per_chunk_latencies = [0.0005, 5.0, 0.00005, 50.0]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    @patch("client.s2s.latency_analysis.plt.savefig")
    def test_plot_latency_savefig_called(self, mock_savefig):
        """Test that plt.savefig is called correctly."""
        output_stream_latencies = [0.1, 0.2, 0.15]
        per_chunk_latencies = [0.05, 0.1, 0.08]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that savefig was called
            mock_savefig.assert_called_once()

    @patch("client.s2s.latency_analysis.plt.close")
    def test_plot_latency_plt_close_called(self, mock_close):
        """Test that plt.close is called correctly."""
        output_stream_latencies = [0.1, 0.2, 0.15]
        per_chunk_latencies = [0.05, 0.1, 0.08]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that plt.close was called
            mock_close.assert_called_once()

    def test_plot_latency_creates_directory(self):
        """Test that plot_latency creates output directory if it doesn't exist."""
        output_stream_latencies = [0.1, 0.2, 0.15]
        per_chunk_latencies = [0.05, 0.1, 0.08]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a nested directory path that doesn't exist
            nested_dir = Path(temp_dir) / "nested" / "subdirectory"
            output_path = nested_dir / "latency_plot.png"

            # Test that the function creates the directory and saves the plot
            plot_latency(
                output_stream_latencies,
                per_chunk_latencies,
                chunk_size_secs,
                str(output_path),
            )

            # Check that the directory was created and the plot file exists
            assert nested_dir.exists()
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_statistics_calculation(self):
        """Test that latency plotting handles statistics correctly."""
        output_stream_latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
        per_chunk_latencies = [0.05, 0.1, 0.08, 0.12, 0.09]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # The function should handle the statistics internally
            assert output_path.exists()
            assert output_path.stat().st_size > 0


@pytest.mark.unit
class TestPercentile:
    """Test cases for the ``_percentile`` helper."""

    def test_empty_returns_zero(self):
        """An empty list yields 0.0 rather than raising."""
        assert _percentile(values=[], pct=95.0) == 0.0

    def test_single_value(self):
        """A single sample is its own percentile."""
        assert _percentile(values=[3.0], pct=95.0) == 3.0

    def test_interpolated_median(self):
        """The 50th percentile interpolates between the middle samples."""
        assert _percentile(values=[1.0, 2.0, 3.0, 4.0], pct=50.0) == pytest.approx(2.5)

    def test_p95_high_end(self):
        """A high percentile lands near the top of the range."""
        values = [float(i) for i in range(1, 101)]
        assert _percentile(values=values, pct=95.0) == pytest.approx(95.05)

    def test_out_of_bounds_raises(self):
        """A pct outside [0, 100] raises, even for empty input."""
        with pytest.raises(ValueError, match="pct must be in"):
            _percentile(values=[1.0, 2.0], pct=150.0)
        with pytest.raises(ValueError, match="pct must be in"):
            _percentile(values=[], pct=-1.0)


@pytest.mark.unit
class TestLatencyJsonArg:
    """Test cases for the ``--latency-json`` CLI flag."""

    def test_default_value(self):
        """The flag defaults to outputs/s2s_latency.json."""
        from client.s2s.args import argsfactory

        args = argsfactory().parse_args([])
        assert args.latency_json == "outputs/s2s_latency.json"

    def test_override(self):
        """A user-supplied path overrides the default."""
        from client.s2s.args import argsfactory

        args = argsfactory().parse_args(["--latency-json", "custom.json"])
        assert args.latency_json == "custom.json"


@pytest.mark.unit
class TestWriteLatencyJson:
    """Test cases for ``write_latency_json``."""

    def test_writes_summary_with_stats(self):
        """The summary captures means, p95s, and metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "s2s_latency.json"
            summary = write_latency_json(
                per_chunk_latencies=[0.4, 0.6],
                output_stream_latencies=[0.9, 1.1],
                chunk_size_secs=1.0,
                is_realtime=False,
                output_path=str(output_path),
                asset="a.wav",
                duration_secs=17.0,
                wall_time_secs=6.5,
            )

            assert output_path.exists()
            on_disk = json.loads(output_path.read_text())
            assert on_disk == summary
            assert summary["mean_per_chunk_latency"] == pytest.approx(0.5)
            assert summary["mean_output_stream_latency"] == pytest.approx(1.0)
            assert summary["num_chunks"] == 2
            assert summary["is_realtime"] is False
            assert summary["asset"] == "a.wav"
            assert summary["duration_secs"] == 17.0
            assert summary["wall_time_secs"] == 6.5

    def test_handles_empty_latencies(self):
        """Empty latency lists produce zeroed stats, not errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "s2s_latency.json"
            summary = write_latency_json(
                per_chunk_latencies=[],
                output_stream_latencies=[],
                chunk_size_secs=1.0,
                is_realtime=True,
                output_path=str(output_path),
            )
            assert summary["mean_per_chunk_latency"] == 0.0
            assert summary["p95_output_stream_latency"] == 0.0
            assert summary["num_chunks"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
