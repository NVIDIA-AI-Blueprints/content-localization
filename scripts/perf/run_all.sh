#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Turnkey driver: run the full performance matrix end-to-end and emit the
# aggregated report. Runs every cell as a one-at-a-time controller request
# (the controller is single-worker by design).
#
# By default it manages the docker stack so el/camb use the right S2S backend:
#   1. bring up the ElevenLabs stack  -> run `el` and `bypass`
#   2. switch S2S to Camb             -> run `camb`
#   3. aggregate -> outputs/perf/perf_comparison.{csv,md}
#
# Pass --skip-docker if you manage the services yourself (then only the backend
# currently running is meaningful for el/camb).
#
# Usage:
#   bash scripts/perf/run_all.sh                 # full matrix, manage docker
#   bash scripts/perf/run_all.sh --skip-docker   # assume services already up
#   bash scripts/perf/run_all.sh --configs "el bypass"   # subset

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- Defaults (override via flags / env) ---
CONFIGS="el bypass camb"
OUT_DIR="outputs/perf"
TARGET_LANGUAGE="de"
MANIFEST="scripts/perf/assets.manifest"
PROFILE="${PERF_COMPOSE_PROFILE:-controller-third-party-s2s}"
EL_ENV="${PERF_EL_ENV:-configs/elevenlabs.env}"
CAMB_ENV="${PERF_CAMB_ENV:-configs/camb.env}"
BASE_ENV="${PERF_BASE_ENV:-.env}"
SKIP_DOCKER=0
HEALTH_TIMEOUT_SECS="${PERF_HEALTH_TIMEOUT_SECS:-300}"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIGS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --target-language) TARGET_LANGUAGE="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --skip-docker) SKIP_DOCKER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# --- helpers --------------------------------------------------------------

# Block until controller + NIM gRPC health all report SERVING. Safe to probe
# here because nothing is serving a request yet (fresh bring-up / idle).
wait_healthy() {
    echo ">> waiting for services to be healthy (timeout ${HEALTH_TIMEOUT_SECS}s)..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/protos/generated" \
    HEALTH_TIMEOUT_SECS="$HEALTH_TIMEOUT_SECS" python3 - <<'PY'
import os, time, grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
targets = [("controller","localhost:50056"),("s2s","localhost:50050"),
           ("asd","localhost:50055"),("lipsync","localhost:50054")]
deadline = time.time() + float(os.environ.get("HEALTH_TIMEOUT_SECS", "300"))
pending = dict(targets)
while pending and time.time() < deadline:
    for name, addr in list(pending.items()):
        ch = grpc.insecure_channel(addr)
        try:
            r = health_pb2_grpc.HealthStub(ch).Check(
                health_pb2.HealthCheckRequest(service=""), timeout=5.0)
            if r.status == health_pb2.HealthCheckResponse.SERVING:
                print(f"   {name} SERVING"); pending.pop(name)
        except grpc.RpcError:
            pass
        finally:
            ch.close()
    if pending:
        time.sleep(5)
if pending:
    raise SystemExit(f"services not healthy in time: {list(pending)}")
print("   all services healthy")
PY
}

bring_up() {  # $@ = extra env files; brings up the whole profile
    local env_args=(--env-file "$BASE_ENV")
    for e in "$@"; do env_args+=(--env-file "$e"); done
    echo ">> docker compose --profile ${PROFILE} ${env_args[*]} up -d"
    docker compose --profile "$PROFILE" "${env_args[@]}" up -d
}

run_cfg() {  # $1 = el|camb|bypass
    echo ""
    echo "############################################################"
    echo "## config: $1"
    echo "############################################################"
    bash scripts/perf/run_perf_matrix.sh \
        --config "$1" \
        --out-dir "$OUT_DIR" \
        --manifest "$MANIFEST" \
        --target-language "$TARGET_LANGUAGE"
}

# --- run ------------------------------------------------------------------

rm -rf "$OUT_DIR"

needs_el=0; needs_camb=0
for c in $CONFIGS; do
    [[ "$c" == "el" || "$c" == "bypass" ]] && needs_el=1
    [[ "$c" == "camb" ]] && needs_camb=1
done

# Phase 1: ElevenLabs-backed configs (el, bypass).
if [[ "$needs_el" -eq 1 ]]; then
    [[ "$SKIP_DOCKER" -eq 0 ]] && { bring_up "$EL_ENV"; wait_healthy; }
    for c in $CONFIGS; do
        [[ "$c" == "el" || "$c" == "bypass" ]] && run_cfg "$c"
    done
fi

# Phase 2: Camb-backed config. Switch the S2S service to Camb first.
if [[ "$needs_camb" -eq 1 ]]; then
    [[ "$SKIP_DOCKER" -eq 0 ]] && { bring_up "$CAMB_ENV"; wait_healthy; }
    run_cfg "camb"
fi

# Phase 3: aggregate into the final report.
echo ""
echo ">> aggregating report..."
# shellcheck disable=SC1091
source .venv/bin/activate
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/protos/generated" \
    python scripts/perf/aggregate_perf.py --in-dir "$OUT_DIR"

echo ""
echo ">> DONE. Report:"
echo "   ${OUT_DIR}/perf_comparison.md"
echo "   ${OUT_DIR}/perf_comparison.csv"
