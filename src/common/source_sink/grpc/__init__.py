# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC-coupled source/sink helpers.

This subpackage holds the file-based source/sink simulators that produce or
consume gRPC proto request types (``SpeechToSpeechRequest``,
``DetectActiveSpeakerRequest``, …) for the gRPC clients under
``client/{controller,direct,s2s,asd,lipsync,batch_processing}``. The
transport-agnostic file-IO base classes live one level up at
``common/source_sink/base.py`` and ``common/source_sink/file.py``.

Per project convention, this ``__init__.py`` does **not** re-export
anything; import classes from their concrete modules:

.. code-block:: python

   from common.source_sink.grpc.audio import (
       AudioSourceSimulator,
       AudioSinkSimulator,
       simulated_audio_chunk_generator,
       simulated_audio_chunk_generator_raw,
   )
   from common.source_sink.grpc.video import (
       VideoSourceSimulator,
       VideoSinkSimulator,
       video_chunk_generator,
       simulated_asd_video_chunk_generator,
       simulated_video_chunk_generator_raw,
   )
"""
