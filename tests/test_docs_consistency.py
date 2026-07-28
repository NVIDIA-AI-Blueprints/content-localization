# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation/code consistency guards.

These tests keep the documentation from silently drifting away from the code:

* the generated CLI reference page matches the client ``argsfactory`` parsers,
* every environment variable documented in ``configuration.rst`` actually
  exists in the config files, the compose file, or the source, and
* the ASD speaker-info CSV schema and the LipSync RPC name in the docs match
  the code.
"""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from client.asd.response_writers import SPEAKER_INFO_CSV_FIELDNAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
GENERATOR_PATH = REPO_ROOT / "scripts" / "docs" / "generate_cli_reference.py"

# Matches ``ALL_CAPS_WITH_UNDERSCORES`` tokens (environment-variable style)
# inside RST inline literals. The required underscore keeps plain values such
# as ``INFO`` or ``WAV`` from being treated as environment variables.
_ENV_VAR_PATTERN = re.compile(r"``([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)``")


def _read(path: Path) -> str:
    """Read a UTF-8 text file and return its contents."""
    return path.read_text(encoding="utf-8")


def _load_generator() -> ModuleType:
    """Import the standalone CLI-reference generator script as a module.

    Returns:
        ModuleType: The loaded ``generate_cli_reference`` module.
    """
    spec = importlib.util.spec_from_file_location("generate_cli_reference", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load CLI reference generator at {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.docs
def test_cli_reference_page_is_up_to_date() -> None:
    """The committed CLI reference page must match freshly generated output."""
    generator = _load_generator()
    expected = generator.render_cli_reference()
    actual = _read(DOCS_SOURCE / "cli_reference.rst")
    assert actual == expected, (
        "docs/source/cli_reference.rst is stale. Regenerate with "
        "`python scripts/docs/generate_cli_reference.py`."
    )


@pytest.mark.docs
def test_documented_env_vars_exist_in_code_or_config() -> None:
    """Env vars documented in configuration.rst must exist somewhere real."""
    documented = set(_ENV_VAR_PATTERN.findall(_read(DOCS_SOURCE / "configuration.rst")))
    assert documented, "No environment variables were found in configuration.rst"

    corpus_paths = [
        *sorted((REPO_ROOT / "configs").glob("*.env")),
        REPO_ROOT / "docker-compose.yml",
        *sorted((REPO_ROOT / "src").rglob("*.py")),
        *sorted((REPO_ROOT / "client").rglob("*.py")),
    ]
    corpus = "\n".join(_read(path) for path in corpus_paths if path.exists())

    missing = sorted(var for var in documented if var not in corpus)
    assert not missing, (
        "configuration.rst documents environment variables not found in "
        f"configs/, docker-compose.yml, or source: {missing}"
    )


@pytest.mark.docs
def test_speaker_info_csv_schema_matches_docs() -> None:
    """Documented ASD speaker-info CSV columns must match the code constant."""
    asd_readme = _read(REPO_ROOT / "client" / "asd" / "README.md")
    client_doc = _read(DOCS_SOURCE / "client.rst")
    for column in SPEAKER_INFO_CSV_FIELDNAMES:
        boundary = re.compile(rf"\b{re.escape(column)}\b")
        assert boundary.search(asd_readme), f"ASD README is missing CSV column '{column}'"
        assert boundary.search(client_doc), f"client.rst is missing CSV column '{column}'"


@pytest.mark.docs
def test_lipsync_rpc_name_in_docs_matches_code() -> None:
    """Docs must describe the LipSync RPC as ``Lipsync`` (matching nims.py)."""
    nims_source = _read(REPO_ROOT / "src" / "common" / "nims.py")
    assert ".Lipsync(" in nims_source, "Expected LipsyncHandle to call the Lipsync RPC"

    for doc_name in ("common.rst", "client_types.rst"):
        doc_text = _read(DOCS_SOURCE / doc_name)
        assert "AnimateResponse" not in doc_text, f"{doc_name} still references AnimateResponse"
        assert "Animate(" not in doc_text, f"{doc_name} still references the Animate( RPC"
    assert "Lipsync" in _read(DOCS_SOURCE / "common.rst")
