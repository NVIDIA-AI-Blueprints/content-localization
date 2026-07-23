#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs ElevenLabs S2S dubbing pipeline.
# Assumes all files are present locally and ELEVENLABS_API_KEY is set.

set -euo pipefail

if [ ! -f .venv/bin/activate ]; then
    echo "Error: .venv/bin/activate not found. Create the project venv before running this script." >&2
    exit 1
fi

mkdir -p outputs
source .venv/bin/activate

python3 scripts/elevenlabs/s2s_infer.py \
   --source-language-code en \
   --target-language-code es \
   --input-file assets/sample_audio.wav \
   --output-file outputs/sample_audio_es.wav \
   --source-transcript-output-file outputs/sample_audio_source_transcript.json \
   --target-transcript-output-file outputs/sample_audio_es_transcript.json \
   --transcript-format json
