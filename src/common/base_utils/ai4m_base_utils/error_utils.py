#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error handling utilities for gRPC services."""


class FileSizeError(Exception):
    """Exception raised when a file exceeds the maximum allowed size."""


class SSLConfigurationError(Exception):
    """Exception raised when there are issues with SSL configuration."""


class ServiceConfigurationError(Exception):
    """Exception raised when there are issues with service configuration."""
