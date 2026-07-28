# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared filesystem path helpers for the client applications."""

import os


def ensure_parent_dir(path: str | os.PathLike) -> None:
    """Create the parent directory of *path* if it does not exist.

    A bare filename has no directory component, and ``os.makedirs("")``
    raises ``FileNotFoundError`` — this helper only creates the parent
    directory when the path actually contains one, so output paths like
    ``"output.mp4"`` and ``"outputs/output.mp4"`` are both accepted.

    Args:
        path (str | os.PathLike): File path whose parent directory should
            exist. May be a bare filename.

    Returns:
        None

    Examples:
        >>> ensure_parent_dir(path="outputs/report/summary.json")  # doctest: +SKIP
        >>> ensure_parent_dir(path="summary.json")  # no-op, current directory
    """
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
