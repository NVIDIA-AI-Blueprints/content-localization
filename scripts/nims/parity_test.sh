#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Parity test: run Controller, Direct, and Batch Processing clients on all
# videos and compare outputs.

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_video_dir> [output_dir] [target_language]}"
OUTPUT_DIR="${2:-outputs/parity_test_full}"
TARGET_LANG="${3:-es}"

BATCH_RUN="$OUTPUT_DIR/batch_run"
CTRL_RUN="$OUTPUT_DIR/controller_run"
DIRECT_RUN="$OUTPUT_DIR/direct_run"
BATCH_INPUT="$OUTPUT_DIR/batch_input"
REPORT="$OUTPUT_DIR/parity_report.txt"

# ── Environment ──────────────────────────────────────────────────────────────
source .venv/bin/activate
set -a && source .env && set +a
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"

# ── Prepare directories ─────────────────────────────────────────────────────
mkdir -p "$BATCH_RUN" "$CTRL_RUN" "$DIRECT_RUN" "$BATCH_INPUT"

for f in "$INPUT_DIR"/*.mp4; do
    cp -n "$f" "$BATCH_INPUT/" 2>/dev/null || true
done

VIDEOS=("$BATCH_INPUT"/*.mp4)
NUM_VIDEOS=${#VIDEOS[@]}
echo "=== PARITY TEST: $NUM_VIDEOS videos, target=$TARGET_LANG ==="
echo ""

# ── Phase 1: Batch Processing ───────────────────────────────────────────────
echo "========================================"
echo "PHASE 1: Batch Processing ($NUM_VIDEOS videos)"
echo "========================================"
python -m client.batch_processing.app \
    --input-dir "$BATCH_INPUT" \
    --output-dir "$BATCH_RUN" \
    --target-language "$TARGET_LANG"
echo ""

# ── Phase 2: Controller Client ──────────────────────────────────────────────
echo "========================================"
echo "PHASE 2: Controller Client ($NUM_VIDEOS videos)"
echo "========================================"
for video_path in "${VIDEOS[@]}"; do
    stem=$(basename "$video_path" .mp4)
    wav_path="$BATCH_RUN/preprocessed/${stem}.wav"
    diar_path="$BATCH_RUN/diarization/${stem}.json"
    output_path="$CTRL_RUN/${stem}_${TARGET_LANG}.mp4"

    if [ ! -f "$wav_path" ]; then
        echo "[SKIP] $stem: no preprocessed WAV (batch may have failed)"
        continue
    fi

    echo "[CONTROLLER] $stem ..."
    diar_args=""
    if [ -f "$diar_path" ]; then
        diar_args="--diarization-file $diar_path --diarization-format elevenlabs-scribe"
    fi

    python -m client.controller.app \
        --input-audio "$wav_path" \
        --input-mp4 "$video_path" \
        --output-mp4 "$output_path" \
        --target-language "$TARGET_LANG" \
        $diar_args \
        2>&1 || echo "[FAIL] Controller failed for $stem"
    echo ""
done

# ── Phase 3: Direct Client ──────────────────────────────────────────────────
echo "========================================"
echo "PHASE 3: Direct Client ($NUM_VIDEOS videos)"
echo "========================================"
for video_path in "${VIDEOS[@]}"; do
    stem=$(basename "$video_path" .mp4)
    wav_path="$BATCH_RUN/preprocessed/${stem}.wav"
    diar_path="$BATCH_RUN/diarization/${stem}.json"
    output_mp4="$DIRECT_RUN/${stem}_${TARGET_LANG}.mp4"
    output_mp3="$DIRECT_RUN/${stem}_${TARGET_LANG}.mp3"

    if [ ! -f "$wav_path" ]; then
        echo "[SKIP] $stem: no preprocessed WAV (batch may have failed)"
        continue
    fi

    echo "[DIRECT] $stem ..."
    diar_args=""
    if [ -f "$diar_path" ]; then
        diar_args="--diarization-file $diar_path --diarization-format elevenlabs-scribe"
    fi

    python -m client.direct.app \
        --input-audio "$wav_path" \
        --input-mp4 "$video_path" \
        --output-mp4 "$output_mp4" \
        --output-audio "$output_mp3" \
        --target-language "$TARGET_LANG" \
        $diar_args \
        2>&1 || echo "[FAIL] Direct failed for $stem"
    echo ""
done

# ── Phase 4: Comparison Report ───────────────────────────────────────────────
echo "========================================"
echo "PHASE 4: Comparison Report"
echo "========================================"

{
    printf "%-40s | %-6s | %-6s | %-6s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s\n" \
        "Video" "Batch" "Ctrl" "Direct" \
        "B-Dur" "C-Dur" "D-Dur" "B-Size" "C-Size" "D-Size"
    printf "%s\n" "$(printf '%0.s-' {1..120})"

    pass=0
    fail=0

    for video_path in "${VIDEOS[@]}"; do
        stem=$(basename "$video_path" .mp4)
        batch_mp4="$BATCH_RUN/${stem}_${TARGET_LANG}.mp4"
        ctrl_mp4="$CTRL_RUN/${stem}_${TARGET_LANG}.mp4"
        direct_mp4="$DIRECT_RUN/${stem}_${TARGET_LANG}.mp4"

        b_status="MISS"; c_status="MISS"; d_status="MISS"
        [ -f "$batch_mp4" ] && b_status="OK"
        [ -f "$ctrl_mp4" ] && c_status="OK"
        [ -f "$direct_mp4" ] && d_status="OK"

        b_dur="-"; c_dur="-"; d_dur="-"
        if [ -f "$batch_mp4" ]; then
            b_dur=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$batch_mp4" 2>/dev/null || echo "-")
        fi
        if [ -f "$ctrl_mp4" ]; then
            c_dur=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$ctrl_mp4" 2>/dev/null || echo "-")
        fi
        if [ -f "$direct_mp4" ]; then
            d_dur=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$direct_mp4" 2>/dev/null || echo "-")
        fi

        b_size="-"; c_size="-"; d_size="-"
        [ -f "$batch_mp4" ] && b_size=$(du -h "$batch_mp4" | cut -f1)
        [ -f "$ctrl_mp4" ] && c_size=$(du -h "$ctrl_mp4" | cut -f1)
        [ -f "$direct_mp4" ] && d_size=$(du -h "$direct_mp4" | cut -f1)

        orig_dur=$(ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$video_path" 2>/dev/null || echo "0")

        if [ "$b_status" = "OK" ] && [ "$c_status" = "OK" ] && [ "$d_status" = "OK" ]; then
            b_r=$(printf "%.1f" "$b_dur" 2>/dev/null || echo "0")
            c_r=$(printf "%.1f" "$c_dur" 2>/dev/null || echo "0")
            d_r=$(printf "%.1f" "$d_dur" 2>/dev/null || echo "0")
            o_r=$(printf "%.1f" "$orig_dur" 2>/dev/null || echo "0")
            if [ "$b_r" = "$o_r" ] && [ "$c_r" = "$o_r" ] && [ "$d_r" = "$o_r" ]; then
                pass=$((pass + 1))
            else
                fail=$((fail + 1))
            fi
        else
            fail=$((fail + 1))
        fi

        printf "%-40s | %-6s | %-6s | %-6s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s\n" \
            "$stem" "$b_status" "$c_status" "$d_status" \
            "${b_dur:0:10}" "${c_dur:0:10}" "${d_dur:0:10}" \
            "$b_size" "$c_size" "$d_size"
    done

    echo ""
    echo "========================================"
    echo "PARITY SUMMARY: $pass PASS, $fail FAIL out of $NUM_VIDEOS videos"
    echo "========================================"
} | tee "$REPORT"

echo ""
echo "Report saved to: $REPORT"
