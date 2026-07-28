# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S2S (Speech-to-Speech) client package.

This package provides a client for the S2S service, which translates and synthesizes
audio content using ElevenLabs or CambAI backends.

Components
==========

- app: Main application module with S2S client implementation
- args: Command-line argument parsing and configuration
- latency_analysis: Tools for analyzing and measuring latency

Usage
=====

.. code-block:: python

   from client.s2s.app import main
   from client.s2s.args import argsfactory
   from client.s2s.latency_analysis import calculate_latencies

   # Parse command line arguments
   args = argsfactory()

   # Run the S2S client
   main(args)

Features
========

- Real-time speech-to-speech translation
- Support for ElevenLabs and CambAI backends
- Multiple language support
- Voice customization options
- Latency analysis and monitoring
- Streaming audio processing
"""
