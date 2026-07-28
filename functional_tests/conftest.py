# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the live-service functional test suite."""

import os
import sys
from pathlib import Path

import pytest

# Make project imports (common.*, client.*, generated protos) resolvable even
# when PYTHONPATH is not exported, matching the layout the test modules expect.
# Insertion order is reversed so the final search order is root, src, client,
# protos/generated — src must precede client so ``common`` resolves to
# src/common rather than the client/common package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in reversed(
    (
        _PROJECT_ROOT,
        _PROJECT_ROOT / "src",
        _PROJECT_ROOT / "client",
        _PROJECT_ROOT / "protos" / "generated",
    )
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Path bootstrap above must run before project imports resolve.
from common.health import check_service_health

# Paths the client entrypoints need on PYTHONPATH when spawned as
# subprocesses; mirrors the sys.path bootstrap above.
_SUBPROCESS_PYTHONPATH = (
    _PROJECT_ROOT,
    _PROJECT_ROOT / "src",
    _PROJECT_ROOT / "client",
    _PROJECT_ROOT / "protos" / "generated",
)


@pytest.fixture(scope="session")
def client_subprocess_env() -> dict[str, str]:
    """Environment for spawning client entrypoints as subprocesses.

    The sys.path bootstrap above only fixes imports for the pytest
    process itself; spawned clients would otherwise depend on the
    caller's PYTHONPATH (easily clobbered, e.g. by sourcing an ``.env``
    that sets its own). Prepending the project paths makes the suite
    self-contained regardless of the invoking shell.

    Returns:
        dict[str, str]: A copy of ``os.environ`` with the project paths
            prepended to ``PYTHONPATH``.

    Examples:
        >>> # In a test:
        >>> # subprocess.run(cmd, env=client_subprocess_env, ...)
    """
    env = dict(os.environ)
    project_paths = os.pathsep.join(str(path) for path in _SUBPROCESS_PYTHONPATH)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_paths}{os.pathsep}{existing}" if existing else project_paths
    return env


# Markers that satisfy the project rule "every test must declare a marker",
# restricted to the marker set registered in functional_tests/pytest.ini.
_REQUIRED_MARKERS = frozenset(
    {
        "functional",
        "slow",
        "integration",
        "deployment",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fail collection if any test lacks one of the required pytest markers.

    ``--strict-markers`` only rejects *unregistered* marker names; it does not
    require a marker to be *present*. This hook enforces the project rule that
    every functional test declares one of the registered markers, so
    marker-based selection (for example ``-m functional``) stays meaningful.

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


@pytest.fixture(scope="module", autouse=True)
def _require_live_services(request: pytest.FixtureRequest) -> None:
    """Skip or fail every test in a module whose required services are down.

    Each functional test module declares the gRPC services it exercises in a
    module-level ``REQUIRED_SERVICES`` tuple of ``(name, address)`` pairs.
    This fixture probes each address with the standard gRPC health protocol.
    When a service is unreachable, the whole module is skipped by default so
    functional runs on machines without a live deployment report skips
    instead of failures. With ``--require-services`` (release validation),
    unreachable services fail the run instead — otherwise a run in which
    every test skipped would still exit successfully.

    Args:
        request (pytest.FixtureRequest): The fixture request, used to read
            the ``REQUIRED_SERVICES`` attribute of the requesting module and
            the ``--require-services`` option.

    Examples:
        >>> # Declared in a test module:
        >>> REQUIRED_SERVICES = (("ASD", "localhost:50055"),)  # doctest: +SKIP
    """
    strict = request.config.getoption("--require-services")
    services: tuple[tuple[str, str], ...] = getattr(request.module, "REQUIRED_SERVICES", ())
    for service_name, address in services:
        try:
            check_service_health(server=address)
        # Any probe failure (connection refused, DNS, RPC error) means the
        # deployment is not available on this machine.
        except Exception as exc:
            message = f"{service_name} service is not reachable at {address}: {exc}"
            if strict:
                pytest.fail(reason=message, pytrace=False)
            pytest.skip(message)


def pytest_addoption(parser):
    group = parser.getgroup("language")
    default_source_language = os.environ.get("TEST_SOURCE_LANGUAGE")
    default_target_language = os.environ.get("TEST_TARGET_LANGUAGE")
    default_audio_format = os.environ.get("TEST_AUDIO_FORMAT")
    default_diarization_file = os.environ.get("TEST_DIARIZATION_FILE")
    default_diarization_format = os.environ.get("TEST_DIARIZATION_FORMAT", "elevenlabs-scribe")
    group.addoption(
        "--source-language",
        action="store",
        default=default_source_language,
        help=(
            "Source language code to use in functional tests. "
            "Uses TEST_SOURCE_LANGUAGE if set, otherwise client defaults."
        ),
    )
    group.addoption(
        "--target-language",
        action="store",
        default=default_target_language,
        help=(
            "Target language code to use in functional tests. "
            "Uses TEST_TARGET_LANGUAGE if set, otherwise client defaults."
        ),
    )
    group.addoption(
        "--audio-format",
        action="store",
        default=default_audio_format,
        help=(
            "Audio format for outputs in functional tests: wav or mp3. "
            "Uses TEST_AUDIO_FORMAT if set, otherwise client defaults."
        ),
    )
    group.addoption(
        "--diarization-file",
        action="store",
        default=default_diarization_file,
        help=(
            "Path to diarization JSON file for functional tests. "
            "Uses TEST_DIARIZATION_FILE if set, otherwise tests use their hardcoded default."
        ),
    )
    group.addoption(
        "--diarization-format",
        action="store",
        default=default_diarization_format,
        help=(
            "Diarization format for functional tests (e.g. elevenlabs-scribe, camb). "
            "Uses TEST_DIARIZATION_FORMAT if set, otherwise elevenlabs-scribe."
        ),
    )

    services_group = parser.getgroup("live-services")
    services_group.addoption(
        "--require-services",
        action="store_true",
        default=os.environ.get("TEST_REQUIRE_SERVICES") == "1",
        help=(
            "Fail instead of skip when a module's required live services are "
            "unreachable. Use in release validation so a run where every test "
            "skipped cannot report success. Uses TEST_REQUIRE_SERVICES=1 if set."
        ),
    )


@pytest.fixture
def source_language(pytestconfig):
    return pytestconfig.getoption("--source-language")


@pytest.fixture
def target_language(pytestconfig):
    return pytestconfig.getoption("--target-language")


@pytest.fixture
def audio_format(pytestconfig):
    value = pytestconfig.getoption("--audio-format")
    if value is None:
        return None
    return str(value).lower()


@pytest.fixture
def diarization_file(pytestconfig):
    return pytestconfig.getoption("--diarization-file")


@pytest.fixture
def diarization_format(pytestconfig):
    return pytestconfig.getoption("--diarization-format")
