# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base utilities — re-exports vendored ai4m_base_utils for the project.

The vendored source at ``base_utils/ai4m_base_utils/`` is kept **unmodified** so
upstream updates can be dropped in without conflict.  A ``sys.path`` shim lets
the vendored package's internal imports (``from ai4m_base_utils.config import …``)
resolve as top-level, while consumers import everything through this wrapper::

    from common.base_utils import logger
    from common.base_utils import GRPCServiceBase
"""

import sys
from pathlib import Path

# Allow vendored ai4m_base_utils internal imports to resolve as top-level
sys.path.insert(0, str(Path(__file__).parent))

# --- Re-export public API -------------------------------------------------- #

from ai4m_base_utils.auth import Auth
from ai4m_base_utils.config import AI4M_DEFAULT_MESSAGE_SIZE
from ai4m_base_utils.error_utils import FileSizeError
from ai4m_base_utils.error_utils import ServiceConfigurationError
from ai4m_base_utils.error_utils import SSLConfigurationError
from ai4m_base_utils.file_utils import FileUtils
from ai4m_base_utils.grpc_service import GRPCServiceBase
from ai4m_base_utils.hooks import BaseHooks
from ai4m_base_utils.hooks import CleanupHooks
from ai4m_base_utils.hooks import MonitoringHooks
from ai4m_base_utils.logger import logger

__all__ = [
    "AI4M_DEFAULT_MESSAGE_SIZE",
    "Auth",
    "BaseHooks",
    "CleanupHooks",
    "FileSizeError",
    "FileUtils",
    "GRPCServiceBase",
    "MonitoringHooks",
    "SSLConfigurationError",
    "ServiceConfigurationError",
    "logger",
]
