#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "Starting Controller Service..."

# Opt-in remote debugging: debugpy is not shipped in the production image,
# so install it at container start only when CONTROLLER_VS_CODE_DEBUG=1
# (requires network access; pinned for reproducibility). Publish the debug
# port with docker-compose.debug.yml when using this mode.
if [ "${CONTROLLER_VS_CODE_DEBUG}" = 1 ]; then
    echo "Starting in DEBUG mode with debugpy."
    if ! python -c "import debugpy" 2>/dev/null; then
        echo "Installing debugpy for the opt-in debug workflow..."
        # python -m pip guarantees the install targets the same interpreter
        # that runs the service (and the import check above).
        python -m pip install --no-cache-dir debugpy==1.8.21
    fi
    DEBUG_CMD="python -m debugpy --listen 0.0.0.0:${CONTROLLER_DEBUG_PORT} --wait-for-client"
else
    DEBUG_CMD="python"
fi

# Build the command with required arguments
CMD="${DEBUG_CMD} /opt/controller/controller_service/entrypoint.py \
    --service-uri controller:${CONTROLLER_GRPC_API_PORT} \
    --max-concurrency ${CONTROLLER_MAX_CONCURRENCY} \
    --concurrency-mode ${CONTROLLER_GRPC_CONCURRENCY_MODE} \
    --threads-per-process ${CONTROLLER_GRPC_THREADS_PER_PROCESS} \
    --s2s-server ${S2S_SERVER} \
    --lipsync-server ${LIPSYNC_SERVER}"

# ASD server is optional — when provided, requests can use ASD;
# when omitted, only bypass_asd=True requests are supported
if [ -n "${ASD_SERVER}" ]; then
    CMD="${CMD} --asd-server ${ASD_SERVER}"
    echo "ASD server configured: ${ASD_SERVER}"
fi

# Execute the command
exec $CMD
