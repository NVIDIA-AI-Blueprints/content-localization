# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared wall-clock timing utilities for client pipeline stages.

All clients use :class:`StageTimer` to instrument their stages uniformly.
At DEBUG level each stage emits a start and a completion line; at INFO level
:meth:`StageTimer.log_summary` emits a single line with all stage times and
the total.

Standard stage names used across clients:

* ``health_check`` — gRPC health probe(s)
* ``setup`` — source creation, config wiring, channel/stub setup
* ``inference`` — active gRPC streaming (request send + response drain)
* ``cleanup`` — closing sources, channels, and joining threads

Batch-processing uses domain-specific stages in addition to the above:

* ``preprocess`` — video-to-WAV extraction (ffmpeg)
* ``diarization`` — speaker-diarization API call + JSON parse
* ``pipeline`` — controller gRPC call (maps to ``inference`` in other clients)
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from common.base_utils import logger


@dataclass
class StageTiming:
    """Elapsed time for a single named pipeline stage.

    Attributes:
        name: Stage identifier (e.g. ``"health_check"``, ``"inference"``).
        elapsed_secs: Wall-clock seconds the stage took.
    """

    name: str
    elapsed_secs: float


class StageTimer:
    """Wall-clock timer for named pipeline stages.

    Use the :meth:`stage` context manager to bracket each logical step.
    On entry a DEBUG line is emitted; on exit a second DEBUG line records
    the elapsed time.  Call :meth:`log_summary` once at the end of a run to
    emit a single INFO line with all stage times and the total.

    The :meth:`as_dict` snapshot is serialisable and can be embedded directly
    in JSON reports so downstream tools (e.g. ``aggregate_perf.py``) consume
    timing data without scraping logs.

    Examples:
        >>> timer = StageTimer()
        >>> with timer.stage("health_check"):
        ...     pass
        >>> with timer.stage("inference"):
        ...     pass
        >>> timer.as_dict()  # doctest: +SKIP
        {'health_check': 0.0, 'inference': 0.0}
        >>> timer.total_secs() >= 0.0
        True
    """

    def __init__(self) -> None:
        self._completed: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager that times a named stage.

        Logs ``[{name}] starting`` at DEBUG on entry and
        ``[{name}] done in {elapsed:.1f}s`` at DEBUG on exit.

        Args:
            name (str): Stage label used in log messages and :meth:`as_dict`.

        Yields:
            None

        Examples:
            >>> timer = StageTimer()
            >>> with timer.stage("setup"):
            ...     pass  # timed work here
        """
        logger.debug(f"[{name}] starting")
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_secs = time.time() - start_time
            self._completed.append(StageTiming(name=name, elapsed_secs=elapsed_secs))
            logger.debug(f"[{name}] done in {elapsed_secs:.1f}s")

    def as_dict(self) -> dict[str, float]:
        """Return stage timings as ``{name: elapsed_secs}``.

        Preserves insertion order (Python 3.7+).

        Returns:
            dict[str, float]: Mapping of stage name to elapsed seconds.

        Examples:
            >>> StageTimer().as_dict()
            {}
        """
        return {t.name: t.elapsed_secs for t in self._completed}

    def total_secs(self) -> float:
        """Sum of all completed stage durations.

        Returns:
            float: Total elapsed seconds across all recorded stages.

        Examples:
            >>> StageTimer().total_secs()
            0.0
        """
        return sum(t.elapsed_secs for t in self._completed)

    def log_summary(self, label: str = "") -> None:
        """Emit a single INFO line listing every stage time and the grand total.

        Args:
            label (str): Optional prefix (e.g. video name) prepended before
                ``timings:``.

        Examples:
            >>> StageTimer().log_summary(label="[video.mp4]")  # doctest: +SKIP
            INFO  [video.mp4] timings: total=0.0s
        """
        parts = [f"{t.name}={t.elapsed_secs:.1f}s" for t in self._completed]
        body = ", ".join(parts) + f", total={self.total_secs():.1f}s" if parts else "total=0.0s"
        prefix = f"{label} " if label else ""
        logger.info(f"{prefix}timings: {body}")
