.. _s2s_service:

===========
S2S Service
===========

The Speech-to-Speech (S2S) service provides the speech translation stage of
the blueprint. It collects the source-language audio stream to a temporary
file, submits it to a third-party dubbing API (ElevenLabs or CambAI), keeps
the gRPC stream alive with keepalive messages while the dubbing job runs, and
streams the dubbed target-language audio back to client applications.

.. contents::
   :local:
   :depth: 1

Overview
========

The service is implemented in `s2s_service.service` and exposes a gRPC
``StreamSpeechToSpeech`` endpoint. Requests arrive as a stream of
``SpeechToSpeechRequest`` messages containing both configuration metadata and
audio chunks. Responses are streamed as ``SpeechToSpeechResponse`` messages so
clients can play translated audio as soon as it is ready.

Key Responsibilities
====================

- Manage streaming request lifecycles and map each client stream to a unique
  request identifier for observability.
- Validate runtime configuration such as supported languages, sample rates, and
  audio formats before handing work to concrete implementations.
- Delegate the dubbing work to backend-specific implementations that drive
  the ElevenLabs or CambAI dubbing APIs.
- Support expanded ElevenLabs dubbing parameters including ``num_speakers``,
  ``drop_background_audio``, ``use_profanity_filter``, ``target_accent``,
  ``highest_resolution``, ``watermark``, and ``dubbing_studio``.
  Voice cloning is always enabled.
- Report audio format metadata (``audio_format``, ``audio_sample_rate``,
  ``audio_num_channels``) in responses so downstream services can adapt.
- Surface gRPC errors and logging details that help diagnose failures across
  the multi-stage pipeline.

Primary Modules
===============

- ``s2s_service.service``: Abstract service base class plus the
  ``S2SServiceServicer`` gRPC adapter used by the deployment entrypoint.
- ``s2s_service.segmentizer``: Text segmentation utilities that preserve
  sentence boundaries. Not used by the dubbing backends.
- ``s2s_service.el_utils`` and ``s2s_service.camb_utils``: Concrete helpers
  for integrating with ElevenLabs and CambAI backends, respectively. The
  ElevenLabs dubbing service (``el_utils.dubbing``) processes audio on a
  background thread with keep-alive support and exposes the full set of
  ElevenLabs dubbing API parameters.
- ``s2s_service.entrypoint``: Argument parsing and server bootstrap logic for
  starting the S2S container.
