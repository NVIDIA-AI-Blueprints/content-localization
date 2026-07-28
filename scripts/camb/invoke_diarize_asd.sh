#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end Camb AI diarization + ASD pipeline.
# 1. Run scripts/camb/diarize.py on the sample audio → diarization JSON
# 2. Run ASD client with Camb AI diarization → speaker info CSV
#
# Requires: CAMB_API_KEY env var, ASD NIM on localhost:50055

set -euo pipefail

INPUT_AUDIO="assets/sample_audio.wav"
INPUT_VIDEO="assets/sample_video_streamable.mp4"
OUTPUT_DIR="outputs/camb_outputs"
DIARIZATION_FILE="${OUTPUT_DIR}/sample_diarization_camb.json"
ASD_OUTPUT="${OUTPUT_DIR}/asd_output.csv"

mkdir -p "${OUTPUT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"

echo "=== Step 1: Camb AI Diarization ==="
python3 scripts/camb/diarize.py \
  --input-file "${INPUT_AUDIO}" \
  --output-file "${DIARIZATION_FILE}" \
  --language-id 1

echo ""
echo "=== Step 2: ASD with Camb AI Diarization ==="
python3 -m client.asd.app \
  --input-mp4 "${INPUT_VIDEO}" \
  --input-audio "${INPUT_AUDIO}" \
  --diarization-file "${DIARIZATION_FILE}" \
  --diarization-format camb \
  --output-speaker-info "${ASD_OUTPUT}"

echo ""
echo "=== Done ==="
echo "Diarization: ${DIARIZATION_FILE}"
echo "ASD output:  ${ASD_OUTPUT}"
