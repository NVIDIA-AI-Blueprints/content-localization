# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package for the main S2S service.

Here is the flow of the S2S service:

.. code-block:: text

    Client
    |
    | 1. gRPC StreamSpeechToSpeech(request_iterator)
    v
    S2SServiceServicer (from service.py)
    |
    | 2. Extract request_id, wrap iterator
    | 3. Call: service.infer(request_iterator, context, request_id)
    v
    S2SService (abstract, implemented by ELDubbingService or
    CambDubbingService)
    |
    |--[ElevenLabs Path]------------------------------------------|
    |                                                             |
    | 4a. ELDubbingService.infer()                                |
    |    |                                                        |
    |    | 5a. Extract config from first request                   |
    |    |    (source_language, target_language)                   |
    |    |    v                                                   |
    |    | 6a. self.download_input_audio()                         |
    |    |    (collects all audio, writes temp WAV)                |
    |    |    v                                                   |
    |    | 7a. self._impl()                                        |
    |    |    |-- background thread: create_dub_from_file(),        |
    |    |    |   download MP3, enqueue chunks                     |
    |    |    `-- main thread: dequeue chunks, send keep-alive     |
    |    |    v                                                   |
    |    | 8a. yield SpeechToSpeechResponse(audio_data, ...)       |
    |    |                                                        |
    |<--------------------------------------------------------|
    |                                                             |
    |--[CambAI Path]----------------------------------------------|
    |                                                             |
    | 4b. CambDubbingService.infer()                              |
    |    |                                                        |
    |    | 5b. Extract config from first request                   |
    |    |    (source_language, target_language)                   |
    |    |    v                                                   |
    |    | 6b. self.download_input_audio()                         |
    |    |    (collects all audio, writes temp WAV)                |
    |    |    v                                                   |
    |    | 7b. self._impl()                                        |
    |    |    |-- background thread: create CambAI dub,            |
    |    |    |   download MP3 (alt format), enqueue chunks       |
    |    |    `-- main thread: dequeue chunks, send keep-alive     |
    |    |    v                                                   |
    |    | 8b. yield SpeechToSpeechResponse(audio_data, ...)       |
    |    |<--------------------------------------------------------|
    |<-------------------------------------------------------------|
    |
    v
    Client receives streaming SpeechToSpeechResponse messages
"""

__version__ = "1.1.0"

# Modules available for import:
# - service: Abstract base class for S2S services
# - entrypoint: Main entry point for the S2S service
# - el_utils: ElevenLabs utilities and dubbing service
# - camb_utils: CambAI utilities and dubbing service
# - segmentizer: Text segmentation utilities
