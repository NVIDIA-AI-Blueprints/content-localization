#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the full ElevenLabs functional test suite and all 6 clients.
#
# Steps:
#   1. Stop any existing compose stack (clean slate)
#   2. Re-create input assets: streamable video, audio, ElevenLabs diarization
#   3. Start docker compose with elevenlabs.env (LIPSYNC_DEBUG_MODE=1)
#   4. Run all functional tests with ElevenLabs settings
#   5. Run all 6 clients, writing outputs to outputs/elevenlabs/
#   6. Stop docker compose (also runs on EXIT via trap)
#
# Prerequisites:
#   ELEVENLABS_API_KEY must be set in .env or the environment.
#   .venv/ must exist (run scripts/misc/setup_env.sh first if needed).
#
# Usage:
#   bash scripts/functional_tests/run_elevenlabs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

ENV_FILE="configs/elevenlabs.env"
OUTPUT_DIR="outputs/elevenlabs"
DIARIZATION_FILE="assets/diarization.json"
COMPOSE_PROFILE="controller-third-party-s2s"

source .venv/bin/activate
set -a && source .env && set +a
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"

mkdir -p "$OUTPUT_DIR"

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

cleanup() {
    echo ""
    echo ">>> Stopping docker compose stack..."
    docker compose \
        --profile "$COMPOSE_PROFILE" \
        --env-file "$ENV_FILE" \
        --env-file .env \
        down || true
}
trap cleanup EXIT

echo "================================================================"
echo "  ElevenLabs Full Functional Test Run"
echo "================================================================"

# 1. Stop any previous stack
run_step "Stop Previous Stack" \
    docker compose \
        --profile "$COMPOSE_PROFILE" \
        --env-file "$ENV_FILE" \
        --env-file .env \
        down

# 2a. Re-create streamable video
run_step "Generate Streamable Video" \
    sh scripts/misc/convert_to_streamable_mp4.sh \
        "assets/sample_video.mp4" \
        "assets/sample_video_streamable.mp4"

# 2b. Re-create audio extracted from video
run_step "Extract Audio" \
    sh scripts/misc/extract_audio_from_videos.sh \
        "assets/sample_video_streamable.mp4" \
        "assets/sample_audio.wav"

# 2c. Re-create ElevenLabs diarization
run_step "ElevenLabs Diarization" \
    python scripts/elevenlabs/diarize.py \
        --input-file "assets/sample_audio.wav" \
        --output-file "$DIARIZATION_FILE"

# 3. Start docker compose and block until all service healthchecks pass
run_step "Start Docker Compose (ElevenLabs, LipSync debug mode)" \
    docker compose \
        --profile "$COMPOSE_PROFILE" \
        --env-file "$ENV_FILE" \
        --env-file .env \
        up --build --wait

# 4. Run all functional tests with ElevenLabs settings
run_step "Functional Tests (ElevenLabs)" \
    python -m pytest functional_tests/ -v --tb=short \
        --diarization-file "$DIARIZATION_FILE" \
        --diarization-format elevenlabs-scribe \
        --source-language en \
        --target-language de \
        --audio-format mp3

# 5. Run all 6 clients, writing outputs to outputs/elevenlabs/
run_step "All Clients (ElevenLabs)" \
    bash scripts/misc/run_all_clients.sh \
        --diarization-format elevenlabs-scribe \
        --diarization-file "$DIARIZATION_FILE" \
        --source-language en \
        --target-language de \
        --s2s-service EL_DUBBING \
        "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "  ElevenLabs Run Complete"
echo "  Outputs: $OUTPUT_DIR"
echo "================================================================"
ls -lh "$OUTPUT_DIR/"
