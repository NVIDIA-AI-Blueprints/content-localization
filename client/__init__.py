# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All client packages.

This package contains client implementations for the Content Localization Blueprint services.
It provides clients for:
- Controller Service (orchestrates all services)
- Direct Service clients (individual service access)
- ASD (Active Speaker Detection) Service
- S2S (Speech-to-Speech) Service
- LipSync Service
- Batch Processing (runs the pipeline on every video in a directory)

Shared client helpers and source/sink simulators live under the ``common``
package (``common.health``, ``common.media``, ``common.tls``,
``common.proto_utils``, ``common.context``, ``common.source_sink``).

Client Types
============

Controller Client
-----------------
The main client that connects to the Controller Service, which orchestrates
all downstream services (S2S, ASD, LipSync).

Direct Clients
--------------
Individual clients that connect directly to specific services:
- ASD client for speaker detection
- S2S client for speech-to-speech translation
- LipSync client for lip synchronization

Batch Processing Client
-----------------------
Runs the end-to-end pipeline on every video in a directory and produces
a timing report.
"""

__version__ = "1.1.0"
