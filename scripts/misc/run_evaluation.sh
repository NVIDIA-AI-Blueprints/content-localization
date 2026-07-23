#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the batch processing pipeline against a directory of videos.
#
# Activates the virtual environment, sets PYTHONPATH, and forwards all
# arguments to the batch processing Python module.
#
# Usage:
#   ./scripts/misc/run_evaluation.sh --input-dir <path> [options]
#
# Examples:
#   ./scripts/misc/run_evaluation.sh --input-dir assets/
#   ./scripts/misc/run_evaluation.sh --input-dir assets/ --target-language fr
#   ./scripts/misc/run_evaluation.sh --input-dir /data/my_videos --output-dir outputs/eval_run_1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# ── Activate virtual environment ──────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/client:${REPO_ROOT}/protos/generated"

# ── Run the batch processing pipeline ─────────────────────────────────
python -m client.batch_processing.app "$@"
