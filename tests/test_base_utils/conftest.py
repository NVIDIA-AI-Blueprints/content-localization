# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and shared fixtures for base_utils tests."""

import tempfile
from collections.abc import Iterator

import pytest

from common.base_utils import GRPCServiceBase


@pytest.fixture
def temp_dir() -> Iterator[str]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


# Concrete implementation for testing the abstract base class
class ConcreteGRPCService(GRPCServiceBase):
    """Concrete implementation of GRPCServiceBase for testing."""

    def add_servicer_to_server(self, server):
        """Mock implementation of abstract method."""
        self._servicer_added = True


@pytest.fixture
def concrete_grpc_service():
    """Create a concrete GRPC service instance for testing."""
    return ConcreteGRPCService()
