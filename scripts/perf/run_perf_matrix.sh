#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the end-to-end performance matrix for ONE S2S backend configuration.
#
# The S2S backend (ElevenLabs vs Camb) is selected server-side via the
# S2S_SERVICE env / docker profile, so this script measures whatever backend is
# currently running. Invoke it once per backend, then merge with aggregate_perf.py:
#
#   # with the ElevenLabs S2S service up:
#   scripts/perf/run_perf_matrix.sh --config el
#   scripts/perf/run_perf_matrix.sh --config bypass   # backend-independent
#   # restart the S2S service with configs/camb.env, then:
#   scripts/perf/run_perf_matrix.sh --config camb
#   python scripts/perf/aggregate_perf.py --in-dir outputs/perf
#
# For each config it runs client.batch_processing.app over the manifest videos
# twice (merged-per-speaker and per-segment diarization). For el/camb it also
# runs client.s2s.app per asset to capture standalone S2S latency. GPU clients
# (ASD/LipSync) run strictly sequentially — nothing here is backgrounded.

set -euo pipefail

# Run from the repo root so the relative defaults below (.venv, .env, the
# default manifest) and the `python -m client.*` PYTHONPATH all resolve,
# regardless of the caller's working directory. scripts/perf/ is two levels
# below the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- Defaults ---
CONFIG=""
MANIFEST="scripts/perf/assets.manifest"
OUT_DIR="outputs/perf"
# Output videos and S2S audio are copied here after each run for accuracy
# review. Unlike OUT_DIR this directory is never wiped on re-runs.
ARTIFACTS_DIR="outputs/perf_artifacts"
# Directory of {stem}.wav pre-translated audio files for bypass-S2S runs.
# When empty, source audio is used as a perf timing stand-in instead.
TRANSLATED_AUDIO_DIR=""
TARGET_LANGUAGE="de"
SOURCE_LANGUAGE="en"
CONTROLLER_SERVER="localhost:50056"
S2S_SERVER="localhost:50050"
CHUNK_SIZE_AUDIO_SECS="1.0"
# Container names used to scrape per-request ASD/LipSync FPS from NIM logs.
ASD_CONTAINER="${ASD_CONTAINER:-asd}"
LIPSYNC_CONTAINER="${LIPSYNC_CONTAINER:-lipsync}"

usage() {
    cat <<'EOF'
Usage: run_perf_matrix.sh --config {el|camb|bypass} [options]

  --config              Required. el | camb | bypass (selects the run label and
                        diarization provider; bypass skips S2S).
  --manifest PATH       Asset manifest (default: scripts/perf/assets.manifest).
  --out-dir PATH        Output root (default: outputs/perf).
  --artifacts-dir PATH  Directory for archived output videos and S2S audio used
                        for accuracy review (default: outputs/perf_artifacts).
                        Never wiped on re-runs.
  --translated-audio-dir PATH
                        Directory of {stem}.wav pre-translated audio files for
                        bypass-S2S runs. When omitted, source audio is used as
                        a perf timing stand-in (ASD+LipSync latency only).
                        Ignored unless --config bypass.
  --target-language L   Target language (default: de).
  --source-language L   Source language (default: en).
  --controller-server   Controller gRPC address (default: localhost:50056).
  --s2s-server          S2S gRPC address (default: localhost:50050).
EOF
}

# Reject a missing/option-shaped value before assigning (safe under set -u).
require_value() {
    local flag="$1" value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        echo "Error: $flag requires a value" >&2
        usage
        exit 1
    fi
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) require_value "--config" "${2-}"; CONFIG="$2"; shift 2 ;;
        --manifest) require_value "--manifest" "${2-}"; MANIFEST="$2"; shift 2 ;;
        --out-dir) require_value "--out-dir" "${2-}"; OUT_DIR="$2"; shift 2 ;;
        --target-language) require_value "--target-language" "${2-}"; TARGET_LANGUAGE="$2"; shift 2 ;;
        --source-language) require_value "--source-language" "${2-}"; SOURCE_LANGUAGE="$2"; shift 2 ;;
        --artifacts-dir) require_value "--artifacts-dir" "${2-}"; ARTIFACTS_DIR="$2"; shift 2 ;;
        --translated-audio-dir) require_value "--translated-audio-dir" "${2-}"; TRANSLATED_AUDIO_DIR="$2"; shift 2 ;;
        --controller-server) require_value "--controller-server" "${2-}"; CONTROLLER_SERVER="$2"; shift 2 ;;
        --s2s-server) require_value "--s2s-server" "${2-}"; S2S_SERVER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required" >&2
    usage
    exit 1
fi

# Map config -> S2S service label (controls diarization provider) and bypass flag.
case "$CONFIG" in
    el)     S2S_SERVICE="EL_DUBBING";   BYPASS=0; S2S_AUDIO_EXT="mp3" ;;
    camb)   S2S_SERVICE="CAMB_DUBBING"; BYPASS=0; S2S_AUDIO_EXT="mp3" ;;
    bypass) S2S_SERVICE="EL_DUBBING";   BYPASS=1; S2S_AUDIO_EXT="mp3" ;;
    *) echo "Error: --config must be el, camb, or bypass (got '$CONFIG')" >&2; exit 1 ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: manifest not found: $MANIFEST" >&2
    exit 1
fi

if [[ -n "$TRANSLATED_AUDIO_DIR" && "$BYPASS" -eq 0 ]]; then
    echo "WARNING: --translated-audio-dir is ignored without --config bypass" >&2
fi

# --- Environment: activate the project venv and export .env ---
# The clients need the venv on PATH and API keys (ELEVENLABS_API_KEY,
# CAMB_API_KEY) exported. Activate/source only when present so an already-
# prepared shell is left untouched.
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

# Export PYTHONPATH *after* sourcing .env: the clients import from src/, client/,
# and protos/generated/ directly (no `src.` prefix), and .env may define its own
# (stale) PYTHONPATH that would otherwise shadow these roots.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/client:${REPO_ROOT}/protos/generated${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_DIR="${OUT_DIR}/${CONFIG}"
STAGING_DIR="${CONFIG_DIR}/staging"
# Reset the whole per-config output tree so a re-run never mixes in stale
# artifacts (symlinks, preprocessed audio, reports, or S2S JSON) from assets
# that have since been removed from the manifest.
rm -rf "$CONFIG_DIR"
mkdir -p "$STAGING_DIR"

# --- Stage manifest videos into a single directory of symlinks ---
# batch_processing scans one --input-dir, but the manifest may reference assets
# spread across the repo (assets/, outputs/perf_segments/...). Symlinks unify
# them without copying multi-hundred-MB files.
echo ">> Staging manifest videos from ${MANIFEST}"
while IFS= read -r raw_line; do
    line="${raw_line%%#*}"                          # strip inline comments
    line="${line#"${line%%[![:space:]]*}"}"         # trim leading whitespace
    line="${line%"${line##*[![:space:]]}"}"         # trim trailing whitespace
    [[ -z "$line" ]] && continue
    # Manifest line is either "tag<TAB>path" or just "path".
    video_path="${line##*$'\t'}"
    if [[ ! -f "$video_path" ]]; then
        echo "   WARNING: asset missing, skipping: $video_path" >&2
        continue
    fi
    abs_path="$(readlink -f "$video_path")"
    # A short hash of the absolute path guarantees uniqueness (two entries with
    # the same basename or the same tag can't collide) and keeps staged names
    # comparable across runs, independent of manifest ordering. A readable tag
    # prefix is added when the manifest line carries one.
    path_hash="$(printf '%s' "$abs_path" | sha1sum | cut -c1-8)"
    if [[ "$line" == *$'\t'* ]]; then
        tag="${line%%$'\t'*}"
        tag="${tag//[^A-Za-z0-9._-]/_}"               # sanitize for filenames
        link_name="${tag}_${path_hash}_$(basename "$video_path")"
    else
        link_name="${path_hash}_$(basename "$video_path")"
    fi
    ln -sf "$abs_path" "${STAGING_DIR}/${link_name}"
    echo "   staged: ${link_name}"
done < "$MANIFEST"

if [[ -z "$(ls -A "$STAGING_DIR" 2>/dev/null)" ]]; then
    echo "Error: no videos staged from manifest; nothing to run." >&2
    exit 1
fi

# No inter-request settle: each video runs as its own process that fully exits
# (closing its gRPC channel) before the next starts, and the controller closes
# its downstream NIM channels after every request. Strictly one-by-one needs no
# artificial delay.

# Scrape the frame count + FPS a NIM logged since $2 (Unix epoch). Because
# requests run strictly one-by-one, the entries since the request started belong
# to that request. Echoes "<frames> <fps>", or "null null" if unavailable.
_nim_fps() {
    local container="$1" since="$2" logs fps frames
    command -v docker >/dev/null 2>&1 || { echo "null null"; return; }
    logs="$(docker logs --since "$since" "$container" 2>&1)" || { echo "null null"; return; }
    fps="$(
        printf '%s\n' "$logs" | grep -oE 'FPS: [0-9.]+' | tail -1 | grep -oE '[0-9.]+' || true
    )"
    frames="$(
        printf '%s\n' "$logs" \
            | grep -oE 'Total frames processed: [0-9]+' \
            | tail -1 | grep -oE '[0-9]+' || true
    )"
    echo "${frames:-null} ${fps:-null}"
}

# Write {asd_frames,asd_fps,lipsync_frames,lipsync_fps}.json for one request.
capture_nim_fps() {
    local out_dir="$1" since="$2" af afps lf lfps
    read -r af afps <<<"$(_nim_fps "$ASD_CONTAINER" "$since")"
    read -r lf lfps <<<"$(_nim_fps "$LIPSYNC_CONTAINER" "$since")"
    cat > "${out_dir}/fps.json" <<EOF
{"asd_frames": ${af}, "asd_fps": ${afps}, "lipsync_frames": ${lf}, "lipsync_fps": ${lfps}}
EOF
}

# --- Run the e2e matrix: both diarization granularities, ONE video per process ---
run_batch() {
    local diar_mode="$1" extra_flag="$2"
    echo ""
    echo "============================================================"
    echo ">> e2e | config=${CONFIG} | diarization=${diar_mode}"
    echo "============================================================"
    local bypass_flags=()
    if [[ "$BYPASS" -eq 1 ]]; then
        bypass_flags+=(--bypass-s2s)
        if [[ -n "$TRANSLATED_AUDIO_DIR" ]]; then
            bypass_flags+=(--translated-audio-dir "$TRANSLATED_AUDIO_DIR")
        fi
    fi
    local video stem one_in run_out
    for video in "$STAGING_DIR"/*; do
        [[ -e "$video" ]] || continue
        stem="$(basename "${video%.*}")"
        # Each video gets its own single-entry input dir + output dir so it runs
        # as a fully isolated client invocation (separate process, own channel).
        one_in="${CONFIG_DIR}/${diar_mode}/_inputs/${stem}"
        run_out="${CONFIG_DIR}/${diar_mode}/${stem}"
        mkdir -p "$one_in"
        ln -sf "$(readlink -f "$video")" "${one_in}/$(basename "$video")"
        echo ">> run: ${CONFIG}/${diar_mode}/${stem}"
        local req_start
        req_start="$(date +%s)"
        python -m client.batch_processing.app \
            --input-dir "$one_in" \
            --output-dir "$run_out" \
            --controller-server "$CONTROLLER_SERVER" \
            --s2s-service "$S2S_SERVICE" \
            --source-language "$SOURCE_LANGUAGE" \
            --target-language "$TARGET_LANGUAGE" \
            --chunk-size-audio-secs "$CHUNK_SIZE_AUDIO_SECS" \
            "${bypass_flags[@]}" \
            ${extra_flag:+$extra_flag}
        # Capture the ASD/LipSync FPS this request logged (one-by-one => the
        # entries since req_start are this request's).
        capture_nim_fps "$run_out" "$req_start"
        # Archive output video for accuracy verification using the original
        # source filename so it is human-readable regardless of the staged name.
        local out_mp4="${run_out}/${stem}_${TARGET_LANGUAGE}.mp4"
        if [[ -f "$out_mp4" ]]; then
            local orig_name artifact_dir
            orig_name="$(basename "$(readlink -f "$video")")"
            artifact_dir="${ARTIFACTS_DIR}/${CONFIG}/${diar_mode}"
            mkdir -p "$artifact_dir"
            cp "$out_mp4" "${artifact_dir}/${orig_name%.*}_${TARGET_LANGUAGE}.mp4"
            echo "   archived: ${artifact_dir}/${orig_name%.*}_${TARGET_LANGUAGE}.mp4"
        fi
    done
}

run_batch "combine" ""
run_batch "per_segment" "--diarization-chunked-per-segment"

# --- Standalone S2S latency (el/camb only; bypass has no S2S) ---
if [[ "$BYPASS" -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo ">> standalone S2S latency | config=${CONFIG}"
    echo "============================================================"
    S2S_OUT="${CONFIG_DIR}/s2s"
    mkdir -p "$S2S_OUT"
    # Reuse the WAVs the combine runs extracted (one per-asset preprocessed dir).
    for wav in "${CONFIG_DIR}"/combine/*/preprocessed/*.wav; do
        [[ -e "$wav" ]] || continue
        stem="$(basename "${wav%.wav}")"
        echo "   S2S: ${stem}"
        python -m client.s2s.app \
            --s2s-server "$S2S_SERVER" \
            --input-audio "$wav" \
            --output-audio "${S2S_OUT}/${stem}_out.${S2S_AUDIO_EXT}" \
            --source-language "$SOURCE_LANGUAGE" \
            --target-language "$TARGET_LANGUAGE" \
            --chunk-size-audio-secs "$CHUNK_SIZE_AUDIO_SECS" \
            --latency-plot "${S2S_OUT}/${stem}_latency.png" \
            --latency-json "${S2S_OUT}/${stem}.json"
    done
    # Archive S2S audio output for accuracy review.
    s2s_artifact_dir="${ARTIFACTS_DIR}/${CONFIG}/s2s"
    mkdir -p "$s2s_artifact_dir"
    find "$S2S_OUT" -name "*_out.*" -exec cp {} "$s2s_artifact_dir/" \;
    echo "   S2S audio archived to ${s2s_artifact_dir}/"
fi

echo ""
echo ">> Config ${CONFIG} done. Results under ${CONFIG_DIR}/"

# Refresh the aggregated report from every config present under OUT_DIR so a
# single run always leaves an up-to-date report behind.
echo ">> aggregating report from ${OUT_DIR}/ ..."
python scripts/perf/aggregate_perf.py --in-dir "${OUT_DIR}" || \
    echo "WARNING: aggregation failed (other configs may not be present yet)"
echo ">> report: ${OUT_DIR}/perf_comparison.md (+ .csv)"
