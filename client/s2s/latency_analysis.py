# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latency analysis functions for S2S client."""

import json

import matplotlib.pyplot as plt

from client.common.paths import ensure_parent_dir


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct* percentile of *values* via linear interpolation.

    Args:
        values (list[float]): Numeric samples (need not be sorted).
        pct (float): Percentile in the range ``[0, 100]``.

    Returns:
        float: The interpolated percentile, or ``0.0`` for empty input.

    Raises:
        ValueError: If *pct* is outside the ``[0, 100]`` range.

    Examples:
        >>> _percentile([1.0, 2.0, 3.0, 4.0], 50.0)
        2.5
    """
    # Validate the requested percentile before the empty-input short-circuit
    # so a bad pct is always surfaced, regardless of the sample count.
    if pct < 0.0 or pct > 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def write_latency_json(
    per_chunk_latencies: list[float],
    output_stream_latencies: list[float],
    chunk_size_secs: float,
    is_realtime: bool,
    output_path: str,
    asset: str | None = None,
    duration_secs: float | None = None,
    wall_time_secs: float | None = None,
) -> dict[str, object]:
    """Write a machine-readable S2S latency summary to JSON.

    Aggregates the per-chunk and output-stream latency lists into mean
    and p95 statistics and persists them alongside run metadata so a
    downstream aggregator can ingest the numbers without scraping logs.

    Args:
        per_chunk_latencies (list[float]): Input-to-output latency per chunk.
        output_stream_latencies (list[float]): Gap between consecutive
            output chunks (the real-time-relevant series).
        chunk_size_secs (float): Streaming chunk size in seconds.
        is_realtime (bool): Whether every output-stream gap stayed under
            ``chunk_size_secs``.
        output_path (str): Destination JSON path.
        asset (str | None): Input asset label (e.g. file path).
        duration_secs (float | None): Input audio duration in seconds.
        wall_time_secs (float | None): Total streaming wall-clock time.

    Returns:
        dict: The summary that was written to disk.

    Examples:
        >>> summary = write_latency_json(
        ...     per_chunk_latencies=[0.5, 0.6],
        ...     output_stream_latencies=[0.9],
        ...     chunk_size_secs=1.0,
        ...     is_realtime=True,
        ...     output_path="outputs/s2s_latency.json",
        ... )  # doctest: +SKIP
    """
    summary = {
        "asset": asset,
        "duration_secs": duration_secs,
        "chunk_size_secs": chunk_size_secs,
        "wall_time_secs": wall_time_secs,
        "num_chunks": len(per_chunk_latencies),
        "mean_per_chunk_latency": (
            sum(per_chunk_latencies) / len(per_chunk_latencies) if per_chunk_latencies else 0.0
        ),
        "p95_per_chunk_latency": _percentile(values=per_chunk_latencies, pct=95.0),
        "mean_output_stream_latency": (
            sum(output_stream_latencies) / len(output_stream_latencies)
            if output_stream_latencies
            else 0.0
        ),
        "p95_output_stream_latency": _percentile(values=output_stream_latencies, pct=95.0),
        "is_realtime": is_realtime,
    }

    ensure_parent_dir(path=output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def calculate_per_chunk_latencies(input_ledger: dict, output_ledger: dict) -> list:
    """Calculate the latencies between the input and output chunks in dictionary format.

    The key is the chunk id, the value is the timestamp of the output chunk for the ledgers.
    The latencies are calculated as the difference between the output chunk timestamp
    and the input chunk timestamp as registered by the ledgers.
    Accomodations are made for the case where the input and output
    ledgers are not of the same length.

    Args:
        input_ledger (dict): The ledger of the input chunks.
        output_ledger (dict): The ledger of the output chunks.

    Returns:
        list: A list of latencies for each chunk.
    """
    latencies = []
    if not input_ledger or not output_ledger:
        return latencies

    max_chunk_id = min(max(input_ledger.keys()), max(output_ledger.keys()))
    for chunk_id in range(max_chunk_id + 1):
        if chunk_id in input_ledger and chunk_id in output_ledger:
            input_timestamp = input_ledger[chunk_id]
            output_timestamp = output_ledger[chunk_id]
            latencies.append(output_timestamp - input_timestamp)
    return latencies


def calculate_output_stream_latencies(input_ledger: dict, output_ledger: dict) -> list:
    """Calculate the latencies between the output + 1 frame and output chunks in dictionary format.

    The key is the chunk id, the value is the timestamp of the output chunk for the ledgers.
    The latencies are calculated as the difference between the output chunk timestamps
    of the current and the next chunk of audio.

    For real-time, we want this latency to be less than chunk size.

    Args:
        input_ledger (dict): The ledger of the input chunks.
        output_ledger (dict): The ledger of the output chunks.

    Returns:
        list: A list of latencies for each chunk.
    """
    latencies = []
    if not input_ledger or not output_ledger:
        return latencies

    max_chunk_id = min(max(input_ledger.keys()), max(output_ledger.keys()))
    for chunk_id in range(max_chunk_id):
        if chunk_id in output_ledger and chunk_id + 1 in output_ledger:
            latencies.append(output_ledger[chunk_id + 1] - output_ledger[chunk_id])
    return latencies


def plot_latency(
    output_stream_latencies: list,
    per_chunk_latencies: list,
    chunk_size_secs: float,
    output_path: str,
) -> None:
    """Plot both output stream and per-chunk latencies on dual y-axes.

    Args:
        output_stream_latencies (list): List of output stream latency values.
        per_chunk_latencies (list): List of per-chunk latency values.
        chunk_size_secs (float): Chunk size in seconds for reference line.
        output_path (str): Path to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot output stream latencies on left y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("Chunk Index")
    ax1.set_ylabel("Output Stream Latency (seconds)", color=color1)
    line1 = ax1.plot(
        output_stream_latencies,
        color=color1,
        label="Output Stream Latency",
        linewidth=2,
        alpha=0.8,
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create second y-axis for per-chunk latencies
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Per-Chunk Latency (seconds)", color=color2)
    line2 = ax2.plot(
        per_chunk_latencies, color=color2, label="Per-Chunk Latency", linewidth=2, alpha=0.8
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add chunk size reference line on both axes
    line3 = ax1.axhline(
        y=chunk_size_secs,
        color="r",
        linestyle="--",
        label="Real-time latency bound",
        alpha=0.7,
    )

    # Combine legends from both axes
    lines = line1 + line2 + [line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Speech-to-Speech Latency Analysis")
    ax1.grid(True, alpha=0.3)

    # Create output directory if it doesn't exist
    ensure_parent_dir(path=output_path)

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
