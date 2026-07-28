# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source and sink classes for testing and development.

This package collects everything that **produces** bytes for a downstream
consumer (sources) or **captures** bytes from a producer (sinks). The
transport-agnostic file-IO helpers sit at this package level; concrete
transport-coupled helpers live in per-transport subpackages
(:mod:`common.source_sink.grpc`).

Per the project convention, this ``__init__.py`` does **not** re-export
anything; import each class from its concrete module:

.. code-block:: python

   from common.source_sink.base import BaseFileSimulator
   from common.source_sink.file import FileSourceSimulator
   from common.source_sink.grpc.audio import AudioSourceSimulator, AudioSinkSimulator
   from common.source_sink.grpc.video import VideoSourceSimulator, VideoSinkSimulator

Module catalogue
================

- :mod:`base`: Abstract base for file-iterator simulators (``BaseFileSimulator``).
- :mod:`file`: Generic file-as-byte-iterator source simulator.
- :mod:`grpc`: File-based source / sink simulators that produce gRPC proto
  request types (audio + video). Consumed by the gRPC clients under
  ``client/{controller,direct,s2s,asd,lipsync,batch_processing}``.
"""
