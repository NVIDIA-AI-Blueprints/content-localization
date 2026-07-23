#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Resolve the NGC API key up front so we fail with a clear preflight error
# instead of letting the container start and die on an empty credential.
API_KEY="${LIPSYNC_API_KEY:-${NGC_API_KEY:-}}"
if [[ -z "${API_KEY}" ]]; then
  echo "Missing API key: set LIPSYNC_API_KEY or NGC_API_KEY." >&2
  exit 1
fi

export CONTAINER_ID="lipsync"
export NIM_TAG="1.3.0"
export LIPSYNC_IMAGE_DEFAULT="nvcr.io/nim/nvidia/lipsync:${NIM_TAG}"
export LIPSYNC_NIM_HTTP_API_PORT="${LIPSYNC_NIM_HTTP_API_PORT:-8004}"
export LIPSYNC_NIM_GRPC_API_PORT="${LIPSYNC_NIM_GRPC_API_PORT:-50054}"
export LIPSYNC_NIM_METRICS_PORT="${LIPSYNC_NIM_METRICS_PORT:-9002}"
export LIPSYNC_MODEL_MOUNT_PATH="${LIPSYNC_MODEL_MOUNT_PATH:-$(pwd)/volumes/models/lipsync}"
export LIPSYNC_NIM_TAGS_SELECTOR="${LIPSYNC_NIM_TAGS_SELECTOR:-language=de}"

mkdir -p "${LIPSYNC_MODEL_MOUNT_PATH}"

docker run -it --rm --name="${CONTAINER_ID}" \
   --runtime=nvidia \
   --gpus all \
   --shm-size=8GB \
   --ipc=host \
   -e NGC_API_KEY="${API_KEY}" \
   -e NIM_HTTP_API_PORT="${LIPSYNC_NIM_HTTP_API_PORT}" \
   -e NIM_GRPC_API_PORT="${LIPSYNC_NIM_GRPC_API_PORT}" \
   -e NIM_TAGS_SELECTOR="${LIPSYNC_NIM_TAGS_SELECTOR}" \
   -e NIM_MAX_CONCURRENCY_PER_GPU=1 \
   -e NIM_CACHE_DIR=/opt/nim/.cache \
   -e LIPSYNC_DEBUG_MODE="${LIPSYNC_DEBUG_MODE:-0}" \
   -e AI4M_LOG_LEVEL="${LIPSYNC_LOG_LEVEL:-INFO}" \
   -e NIM_LOG_LEVEL="${LIPSYNC_LOG_LEVEL:-INFO}" \
   -e LOG_LEVEL="${LIPSYNC_LOG_LEVEL:-INFO}" \
   -p "${LIPSYNC_NIM_HTTP_API_PORT}:${LIPSYNC_NIM_HTTP_API_PORT}" \
   -p "${LIPSYNC_NIM_GRPC_API_PORT}:${LIPSYNC_NIM_GRPC_API_PORT}" \
   -p "${LIPSYNC_NIM_METRICS_PORT}:${LIPSYNC_NIM_METRICS_PORT}" \
   -v "${LIPSYNC_MODEL_MOUNT_PATH}:/opt/nim/.cache:rw" \
   "${LIPSYNC_IMAGE:-${LIPSYNC_IMAGE_DEFAULT}}"
