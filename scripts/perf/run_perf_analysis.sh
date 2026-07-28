#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run repeated merged-per-speaker CLBP perf measurements.
#
# This is intentionally narrower than run_perf_matrix.sh: it skips per-segment
# and standalone S2S. Run this script on one target machine at a time, writing
# each machine's runs to a directory named after that machine;
# aggregate_repeated_perf.py --dirs uses the directory basenames as the machine
# labels when combining independently collected outputs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS="el,bypass"
CONTROLLER_SERVER="localhost:50056"
RUNS=3
RUN_START=1
MANIFEST=""
OUT_DIR=""
MACHINE_LABEL=""
CANONICAL_DIARIZATION_DIR=""
TARGET_LANGUAGE="de"
SOURCE_LANGUAGE="en"
CHUNK_SIZE_AUDIO_SECS="1.0"
ASD_CONTAINER="${ASD_CONTAINER:-asd}"
LIPSYNC_CONTAINER="${LIPSYNC_CONTAINER:-lipsync}"

usage() {
    cat <<'EOF'
Usage: run_perf_analysis.sh --manifest PATH --out-dir PATH [options]

  --manifest PATH                  Required. Duration-tagged asset manifest.
  --out-dir PATH                   Required. Output root for repeated runs.
  --machine-label LABEL            Optional. Machine label recorded in machine_info.json.
  --runs N                         Number of runs (default: 3).
  --run-start N                    First run index to write (default: 1).
  --configs LIST                   Comma-separated configs (default: el,bypass).
  --canonical-diarization-dir PATH Optional {tag}.json diarization files to reuse.
  --controller-server HOST:PORT    Controller address (default: localhost:50056).
  --target-language LANG           Target language (default: de).
  --source-language LANG           Source language (default: en).
EOF
}

require_value() {
    local flag="$1" value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        echo "Error: $flag requires a value" >&2
        usage
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) require_value "$1" "${2-}"; MANIFEST="$2"; shift 2 ;;
        --out-dir) require_value "$1" "${2-}"; OUT_DIR="$2"; shift 2 ;;
        --machine-label) require_value "$1" "${2-}"; MACHINE_LABEL="$2"; shift 2 ;;
        --runs) require_value "$1" "${2-}"; RUNS="$2"; shift 2 ;;
        --run-start) require_value "$1" "${2-}"; RUN_START="$2"; shift 2 ;;
        --configs) require_value "$1" "${2-}"; CONFIGS="$2"; shift 2 ;;
        --canonical-diarization-dir)
            require_value "$1" "${2-}"
            CANONICAL_DIARIZATION_DIR="$2"
            shift 2
            ;;
        --controller-server) require_value "$1" "${2-}"; CONTROLLER_SERVER="$2"; shift 2 ;;
        --target-language) require_value "$1" "${2-}"; TARGET_LANGUAGE="$2"; shift 2 ;;
        --source-language) require_value "$1" "${2-}"; SOURCE_LANGUAGE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$MANIFEST" || -z "$OUT_DIR" ]]; then
    echo "Error: --manifest and --out-dir are required" >&2
    usage
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: manifest not found: $MANIFEST" >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/client:${REPO_ROOT}/protos/generated${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$OUT_DIR"
if [[ -n "$MACHINE_LABEL" ]]; then
    json_machine="${MACHINE_LABEL//\\/\\\\}"
    json_machine="${json_machine//\"/\\\"}"
    json_host="$(hostname -f 2>/dev/null || hostname)"
    json_host="${json_host//\\/\\\\}"
    json_host="${json_host//\"/\\\"}"
    cat > "${OUT_DIR}/machine_info.json" <<EOF
{"machine_label": "${json_machine}", "host": "${json_host}", "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
fi
STAGING_DIR="${OUT_DIR}/staging"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

declare -a STAGED_TAGS=()
declare -a STAGED_PATHS=()

echo ">> Staging manifest videos from ${MANIFEST}"
while IFS= read -r raw_line; do
    line="${raw_line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    if [[ "$line" == *$'\t'* ]]; then
        tag="${line%%$'\t'*}"
        video_path="${line##*$'\t'}"
    else
        video_path="$line"
        tag="$(basename "${video_path%.*}")"
    fi
    if [[ ! -f "$video_path" ]]; then
        echo "WARNING: asset missing, skipping: $video_path" >&2
        continue
    fi
    abs_path="$(readlink -f "$video_path")"
    path_hash="$(printf '%s' "$abs_path" | sha1sum | cut -c1-8)"
    safe_tag="${tag//[^A-Za-z0-9._-]/_}"
    link_name="${safe_tag}_${path_hash}_$(basename "$video_path")"
    ln -sf "$abs_path" "${STAGING_DIR}/${link_name}"
    STAGED_TAGS+=("$tag")
    STAGED_PATHS+=("${STAGING_DIR}/${link_name}")
    echo "   staged: ${link_name}"
done < "$MANIFEST"

if [[ "${#STAGED_PATHS[@]}" -eq 0 ]]; then
    echo "Error: no videos staged from manifest" >&2
    exit 1
fi

_nim_fps() {
    local container="$1" since="$2" log_path="$3" fps frames
    if ! command -v docker >/dev/null 2>&1; then
        echo "null null"
        return
    fi
    docker logs --since "$since" "$container" > "$log_path" 2>&1 || true
    fps="$(grep -oE 'FPS: [0-9.]+' "$log_path" | tail -1 | grep -oE '[0-9.]+' || true)"
    frames="$(
        grep -oE 'Total frames processed: [0-9]+' "$log_path" \
            | tail -1 | grep -oE '[0-9]+' || true
    )"
    echo "${frames:-null} ${fps:-null}"
}

capture_nim_fps() {
    local run_out="$1" since="$2" af afps lf lfps
    read -r af afps <<<"$(_nim_fps "$ASD_CONTAINER" "$since" "${run_out}/asd_docker_since_request.log")"
    read -r lf lfps <<<"$(_nim_fps "$LIPSYNC_CONTAINER" "$since" "${run_out}/lipsync_docker_since_request.log")"
    cat > "${run_out}/fps.json" <<EOF
{"asd_frames": ${af}, "asd_fps": ${afps}, "lipsync_frames": ${lf}, "lipsync_fps": ${lfps}}
EOF
}

run_one() {
    local run_idx="$1" config="$2" tag="$3" video="$4"
    local s2s_service="EL_DUBBING"
    local bypass_flags=()
    if [[ "$config" == "bypass" ]]; then
        bypass_flags+=(--bypass-s2s)
    elif [[ "$config" != "el" ]]; then
        echo "Error: unsupported config for repeated combine run: $config" >&2
        exit 1
    fi

    local stem one_in run_out req_start canonical
    stem="$(basename "${video%.*}")"
    run_out="${OUT_DIR}/run${run_idx}/${config}/combine/${stem}"
    one_in="${OUT_DIR}/run${run_idx}/${config}/combine/_inputs/${stem}"
    mkdir -p "$one_in" "$run_out"
    ln -sf "$(readlink -f "$video")" "${one_in}/$(basename "$video")"

    if [[ -n "$CANONICAL_DIARIZATION_DIR" ]]; then
        canonical="${CANONICAL_DIARIZATION_DIR}/${tag}.json"
        if [[ -f "$canonical" ]]; then
            mkdir -p "${run_out}/diarization"
            cp "$canonical" "${run_out}/diarization/${stem}.json"
        else
            echo "WARNING: canonical diarization missing for tag ${tag}: ${canonical}" >&2
        fi
    fi

    echo ">> run${run_idx}: ${config}/combine/${stem}"
    req_start="$(date +%s)"
    printf '%s\n' "$req_start" > "${run_out}/request_start_epoch.txt"
    printf '%s\n' "$(readlink -f "$video")" > "${run_out}/source_video.txt"

    python -m client.batch_processing.app \
        --input-dir "$one_in" \
        --output-dir "$run_out" \
        --controller-server "$CONTROLLER_SERVER" \
        --s2s-service "$s2s_service" \
        --source-language "$SOURCE_LANGUAGE" \
        --target-language "$TARGET_LANGUAGE" \
        --chunk-size-audio-secs "$CHUNK_SIZE_AUDIO_SECS" \
        "${bypass_flags[@]}"

    capture_nim_fps "$run_out" "$req_start"
}

IFS=',' read -r -a CONFIG_ARRAY <<<"$CONFIGS"
RUN_END=$((RUN_START + RUNS - 1))
for run_idx in $(seq "$RUN_START" "$RUN_END"); do
    for config in "${CONFIG_ARRAY[@]}"; do
        config="${config//[[:space:]]/}"
        [[ -z "$config" ]] && continue
        for idx in "${!STAGED_PATHS[@]}"; do
            run_one "$run_idx" "$config" "${STAGED_TAGS[$idx]}" "${STAGED_PATHS[$idx]}"
        done
    done
done

echo ">> repeated combine runs complete: ${OUT_DIR}"
