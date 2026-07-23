#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs CAMB direct dubbing pipeline.
# Assumes CAMB_API_KEY is set in the environment.
#
# CAMB's dubbing API only emits diarization for the target language. The
# JSON written by --target-transcript-output-file is therefore in the
# target language; for source-language diarization (e.g. ASD input), run
# scripts/camb/diarize.py against the original audio instead.

set -euo pipefail

mkdir -p outputs
source .venv/bin/activate

python3 scripts/camb/s2s_infer.py \
  --source-language 1 \
  --target-language 54 \
  --input-file assets/sample_audio.wav \
  --output-file outputs/camb_output_es.mp3 \
  --target-transcript-output-file outputs/camb_output_es_transcript.json \
  --transcript-format json

# Alternative: use a publicly accessible URL instead of a local file.
# python3 scripts/camb/s2s_infer.py \
#   --source-language 1 \
#   --target-language 54 \
#   --input-url "https://example.com/input_audio_or_video.mp3" \
#   --output-file outputs/camb_output_es.mp3
