#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration settings for gRPC services."""

import os
from typing import List

# GRPC server Configuration
AI4M_DEFAULT_SERVICE_GRPC_HOST: str = "0.0.0.0"
AI4M_DEFAULT_SERVICE_GRPC_PORT: int = 8001
AI4M_DEFAULT_SERVICE_GRPC_URI: str = (
    f"{AI4M_DEFAULT_SERVICE_GRPC_HOST}:{AI4M_DEFAULT_SERVICE_GRPC_PORT}"
)
AI4M_DEFAULT_MAX_CONCURRENCY: int = 5
AI4M_DEFAULT_MESSAGE_SIZE: int = 64 * 1024  # 64KB in bytes

# File size limit (only when explicitly set by user)
_max_file_size_env = os.getenv("AI4M_MAX_INPUT_FILE_SIZE_MB")
AI4M_MAX_INPUT_FILE_SIZE_MB: int = int(_max_file_size_env) if _max_file_size_env else None


# Logging Configuration
AI4M_DEFAULT_LOG_LEVEL: str = "info"
AI4M_VALID_LOG_LEVELS: List[str] = ["notset", "debug", "info", "warning", "error", "critical"]
AI4M_DEFAULT_LOG_DETAILED: bool = False
