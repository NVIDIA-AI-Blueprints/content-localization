#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all clients against the sample video.
# Usage: bash scripts/misc/run_all_clients.sh [OPTIONS] [OUTPUT_DIR] [TARGET_IP]
#
# Options:
#   --no-background-audio        skip background audio in LipSync
#   --diarization-format FORMAT  diarization format (default: elevenlabs-scribe)
#   --diarization-file   FILE    path to diarization JSON (default: assets/diarization.json)
#   --source-language    LANG    source language code or ID (default: en)
#   --target-language    LANG    target language code or ID (default: de)
#   --s2s-service        SVC     S2S service type: EL_DUBBING or CAMB_DUBBING (default: EL_DUBBING)
#
# Positional:
#   OUTPUT_DIR  output directory (default: outputs)
#   TARGET_IP   server IP/hostname (default: localhost)

set -euo pipefail

# Parse flags
NO_BACKGROUND_AUDIO=false
DIARIZATION_FORMAT="elevenlabs-scribe"
DIARIZATION_FILE="assets/diarization.json"
SOURCE_LANGUAGE="en"
TARGET_LANGUAGE="de"
S2S_SERVICE="EL_DUBBING"

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --no-background-audio) NO_BACKGROUND_AUDIO=true; shift ;;
        --diarization-format)  DIARIZATION_FORMAT="$2";  shift 2 ;;
        --diarization-file)    DIARIZATION_FILE="$2";    shift 2 ;;
        --source-language)     SOURCE_LANGUAGE="$2";     shift 2 ;;
        --target-language)     TARGET_LANGUAGE="$2";     shift 2 ;;
        --s2s-service)         S2S_SERVICE="$2";         shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source .venv/bin/activate
set -a && source .env && set +a
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"

OUT="${1:-outputs}"
TARGET="${2:-localhost}"

AUDIO="assets/sample_audio.wav"
VIDEO_NON_STREAMABLE="assets/sample_video.mp4"
VIDEO="assets/sample_video_streamable.mp4"
DIARIZATION="${DIARIZATION_FILE}"

mkdir -p "$OUT"

run_step() {
    local label="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "================================================================"
    local start=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - start ))
    echo ">>> $label completed in ${elapsed}s"
    echo ""
}

# --- Generate Streamable Video ---
run_step "Generate Streamable Video" \
    sh scripts/misc/convert_to_streamable_mp4.sh "$VIDEO_NON_STREAMABLE" "$VIDEO"

# --- Generate Audio ---
run_step "Generate Audio" \
    sh scripts/misc/extract_audio_from_videos.sh "$VIDEO" "$AUDIO"

# --- Run Diarization ---
if [ "$S2S_SERVICE" = "CAMB_DUBBING" ]; then
    run_step "Diarization (CambAI)" \
        python scripts/camb/diarize.py \
            --input-file "$AUDIO" \
            --output-file "$DIARIZATION"
else
    run_step "Diarization (ElevenLabs)" \
        python scripts/elevenlabs/diarize.py \
            --input-file "$AUDIO" \
            --output-file "$DIARIZATION"
fi

# --- S2S Client ---
run_step "S2S Client" \
    python client/s2s/app.py \
        --s2s-server "$TARGET:50050" \
        --input-audio "$AUDIO" \
        --output-audio "$OUT/sample_audio_output_s2s_client.mp3" \
        --latency-plot "$OUT/s2s_latency.png" \
        --source-language "$SOURCE_LANGUAGE" --target-language "$TARGET_LANGUAGE"

# --- ASD Client ---
run_step "ASD Client" \
    python client/asd/app.py \
        --asd-server "$TARGET:50055" \
        --input-mp4 "$VIDEO" \
        --input-audio "$AUDIO" \
        --output-speaker-info "assets/asd_speaker_info_from_asd.csv" \
        --diarization-file "$DIARIZATION" \
        --diarization-format "$DIARIZATION_FORMAT"

# --- LipSync Client (uses S2S output + ASD speaker info) ---
# Background audio skipped for standalone LipSync: the source WAV (16 kHz) and
# the S2S output (44.1 kHz MP3) have different sample rates. The controller and
# direct clients handle resampling internally.
run_step "LipSync Client" \
    python client/lipsync/app.py \
        --lipsync-server "$TARGET:50054" \
        --input-mp4 "$VIDEO" \
        --input-audio "$OUT/sample_audio_output_s2s_client.mp3" \
        --speaker-info-input "assets/asd_speaker_info_from_asd.csv" \
        --output-mp4 "$OUT/lipsync_output.mp4" \
        --lipsync-input-audio-codec MP3

# --- Controller Client ---
run_step "Controller Client" \
    python client/controller/app.py \
        --controller-server "$TARGET:50056" \
        --input-audio "$AUDIO" \
        --input-mp4 "$VIDEO" \
        --output-mp4 "$OUT/controller_output.mp4" \
        --diarization-file "$DIARIZATION" \
        --diarization-format "$DIARIZATION_FORMAT" \
        --source-language "$SOURCE_LANGUAGE" --target-language "$TARGET_LANGUAGE"

# --- Direct Client ---
run_step "Direct Client" \
    python client/direct/app.py \
        --s2s-server "$TARGET:50050" \
        --asd-server "$TARGET:50055" \
        --lipsync-server "$TARGET:50054" \
        --input-audio "$AUDIO" \
        --input-mp4 "$VIDEO" \
        --output-mp4 "$OUT/direct_output.mp4" \
        --output-audio "$OUT/direct_audio_output.mp3" \
        --diarization-file "$DIARIZATION" \
        --diarization-format "$DIARIZATION_FORMAT" \
        --source-language "$SOURCE_LANGUAGE" --target-language "$TARGET_LANGUAGE"

# --- Batch Processing Client ---
run_step "Batch Processing Client" \
    python client/batch_processing/app.py \
        --controller-server "$TARGET:50056" \
        --input-dir "assets" \
        --output-dir "$OUT/batch_processing" \
        --source-language "$SOURCE_LANGUAGE" \
        --target-language "$TARGET_LANGUAGE" \
        --s2s-service "$S2S_SERVICE"

# --- Summary ---
echo ""
echo "================================================================"
echo "  ALL CLIENTS COMPLETE - Output Summary"
echo "================================================================"
echo ""
echo "Outputs:"
ls -lh "$OUT/"
