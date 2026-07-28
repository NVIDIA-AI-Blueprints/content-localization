#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "Triggering Entrypoint: $S2S_SERVICE"
if [ "$S2S_SERVICE" = "EL_DUBBING" ]; then
    python /opt/s2s/s2s_service/entrypoint.py el_dubbing \
        --service-uri "$S2S_SERVER" \
        --max-concurrency "$S2S_MAX_CONCURRENCY" \
        --concurrency-mode "$S2S_GRPC_CONCURRENCY_MODE" \
        --threads-per-process "$S2S_GRPC_THREADS_PER_PROCESS" \
        --sample-rate-hz "$S2S_SAMPLE_RATE_HZ" \
        --message-size "$S2S_MESSAGE_SIZE" \
        --default-source-language "$S2S_DEFAULT_SOURCE_LANGUAGE" \
        --default-target-language "$S2S_DEFAULT_TARGET_LANGUAGE" \
        --audio-format "MP3"
elif [ "$S2S_SERVICE" = "CAMB_DUBBING" ]; then
    python /opt/s2s/s2s_service/entrypoint.py camb_dubbing \
        --service-uri "$S2S_SERVER" \
        --max-concurrency "$S2S_MAX_CONCURRENCY" \
        --concurrency-mode "$S2S_GRPC_CONCURRENCY_MODE" \
        --threads-per-process "$S2S_GRPC_THREADS_PER_PROCESS" \
        --sample-rate-hz "$S2S_SAMPLE_RATE_HZ" \
        --message-size "$S2S_MESSAGE_SIZE" \
        --default-source-language "$S2S_DEFAULT_SOURCE_LANGUAGE" \
        --default-target-language "$S2S_DEFAULT_TARGET_LANGUAGE" \
        --audio-format "MP3"
else
    echo "Entrypoint: Invalid service: $S2S_SERVICE"
    exit 1
fi
