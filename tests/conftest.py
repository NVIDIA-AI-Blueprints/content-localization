# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test configuration and fixtures."""

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure src/ is on the Python path for test imports."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    yield


# Markers that satisfy the project rule "every test must declare a marker".
_REQUIRED_MARKERS = frozenset(
    {
        "unit",
        "integration",
        "system",
        "acceptance",
        "docs",
        "skipduringci",
        "pleasefixme",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fail collection if any test lacks one of the required pytest markers.

    ``--strict-markers`` only rejects *unregistered* marker names; it does not
    require a marker to be *present*. This hook enforces the project rule that
    every test declares one of the registered markers, so marker-based selection
    (for example ``-m unit`` or ``-m 'not skipduringci'``) stays meaningful.

    Args:
        items (list[pytest.Item]): The collected test items.

    Raises:
        pytest.UsageError: If any collected item has no required marker.

    Examples:
        >>> # Invoked automatically by pytest during collection.
        >>> pytest_collection_modifyitems(items=[])  # doctest: +SKIP
    """
    unmarked = [
        item.nodeid
        for item in items
        if not _REQUIRED_MARKERS.intersection(marker.name for marker in item.iter_markers())
    ]
    if unmarked:
        listing = "\n  ".join(unmarked)
        raise pytest.UsageError(
            "These tests have no required pytest marker "
            f"(add one of {sorted(_REQUIRED_MARKERS)}):\n  {listing}"
        )
