.. _controller_service:

==================
Controller Service
==================

The Controller service orchestrates the end-to-end Content Localization
workflow. It receives client requests, coordinates downstream services, and
returns localized media streams that combine translated audio, active-speaker
regions, and lip-synced video.

.. contents::
   :local:
   :depth: 1

Overview
========

The service is implemented in ``controller_service.service`` and exposes the
``StreamContentLocalization`` gRPC endpoint defined in the controller
protobuf. The CLI entrypoint in ``controller_service.entrypoint`` starts a
``ControllerService`` instance and a matching ``ControllerServiceServicer``
(from ``controller_service.servicer``) that bridges gRPC requests into the
business logic.

The ``ContentLocalizationDeserializer`` distributes incoming request packets
into typed buffers: ``audio_buffer`` (2 queues, for S2S and ASD),
``video_buffer`` (2 queues, for ASD and LipSync), ``diarization_buffer``
(1 queue, for ASD), ``translated_audio_buffer`` (1 queue, for LipSync when
S2S is bypassed), ``controller_config_buffer`` (1 queue),
``background_audio_buffer`` (1 queue, for LipSync), and per-service config
buffers (``s2s_config_buffer``, ``asd_config_buffer``,
``lipsync_config_buffer``). Downstream client threads consume from these
buffers concurrently.

Supporting Modules
==================

- ``controller_service.servicer``: gRPC servicer that receives the client
  request stream, delegates to ``ControllerService.infer``, and maps
  pipeline exceptions to specific gRPC status codes.
- ``controller_service.pipeline``: Thread-orchestration functions that launch
  the S2S, ASD, and LipSync client threads, drain unused buffer queues,
  stream responses, and clean up pipeline threads.
- ``controller_service.config``: Per-request dataclasses bundling the
  effective pipeline configuration and downstream service handles.
- ``controller_service.stream_adapters``: Adapts client request streams into
  iterators suitable for downstream service request pipelines. Merges video,
  audio, and optional diarization streams into ASD requests, and video, audio,
  and speaker info streams into LipSync requests. Includes
  ``translated_audio_to_lipsync_audio`` for routing pre-translated audio
  directly to LipSync when S2S is bypassed.
- ``controller_service.deserializer``: Splits the incoming
  ``ContentLocalizationRequest`` stream into audio, video, diarization, and
  config buffers used by multi-threaded processing.
- ``controller_service.conversions``: Utility helpers for request and response
  payload transformations (S2S, ASD video/audio/diarization, LipSync).
  Includes ``to_lipsync_translated_audio`` for converting pre-translated
  audio into LipSync input format.
- ``controller_service.constants``: Shared configuration keys and audio format
  defaults.
- ``controller_service.entrypoint``: CLI bootstrap that wires configuration and
  starts the gRPC server.
- ``controller_service.helpers``: Helper functions for controller service
  operations including service creation, health checking, and thread
  management.

S2S Bypass Mode
===============

When the client sends a ``ContentLocalizationConfig`` with ``bypass_s2s``
set to ``true``, the controller skips the S2S client thread entirely.
Instead of translating audio through the S2S service, pre-translated
audio from the ``translated_audio_buffer`` is routed directly to LipSync
via the ``translated_audio_to_lipsync_audio`` stream adapter. This
adapter calls ``to_lipsync_translated_audio`` from the conversions module
to convert each ``ContentLocalizationRequest.translated_audio_data``
payload into a ready-to-send ``LipsyncRequest`` message.

Clients enable this mode by passing ``--translated-audio`` with a path
to a pre-translated audio file (WAV or MP3). The Controller and Direct
clients both support this flag.

