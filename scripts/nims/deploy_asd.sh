#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Resolve the NGC API key up front so we fail with a clear preflight error
# instead of letting the container start and die on an empty credential.
API_KEY="${ASD_API_KEY:-${NGC_API_KEY:-}}"
if [[ -z "${API_KEY}" ]]; then
  echo "Missing API key: set ASD_API_KEY or NGC_API_KEY." >&2
  exit 1
fi

export CONTAINER_ID="asd"
export NIM_TAG="1.1.0"
export ASD_IMAGE_DEFAULT="nvcr.io/nim/nvidia/active-speaker-detection:${NIM_TAG}"
export ASD_NIM_HTTP_API_PORT="${ASD_NIM_HTTP_API_PORT:-8005}"
export ASD_GRPC_API_PORT="${ASD_GRPC_API_PORT:-50055}"
export ASD_MODEL_MOUNT_PATH="${ASD_MODEL_MOUNT_PATH:-$(pwd)/volumes/models/asd}"

mkdir -p "${ASD_MODEL_MOUNT_PATH}"

docker run -it --rm --name="${CONTAINER_ID}" \
   --runtime=nvidia \
   --gpus all \
   --shm-size=4GB \
   -e NGC_API_KEY="${API_KEY}" \
   -e NIM_HTTP_API_PORT="${ASD_NIM_HTTP_API_PORT}" \
   -e NIM_GRPC_API_PORT="${ASD_GRPC_API_PORT}" \
   -e AI4M_LOG_LEVEL="${ASD_LOG_LEVEL:-INFO}" \
   -e NIM_LOG_LEVEL="${ASD_LOG_LEVEL:-INFO}" \
   -e MAXINE_MAX_CONCURRENCY_PER_GPU=1 \
   -e LOG_LEVEL="${ASD_LOG_LEVEL:-INFO}" \
   -p "${ASD_NIM_HTTP_API_PORT}:${ASD_NIM_HTTP_API_PORT}" \
   -p "${ASD_GRPC_API_PORT}:${ASD_GRPC_API_PORT}" \
   -v "${ASD_MODEL_MOUNT_PATH}:/opt/nim/.cache:rw" \
   "${ASD_IMAGE:-${ASD_IMAGE_DEFAULT}}"
