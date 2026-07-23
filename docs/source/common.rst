.. _common:

======
Common
======

The ``common`` package (``src/common/``) contains the shared abstractions
that every service in the pipeline depends on: thread-safe buffers,
background-thread deserializers, gRPC client wrappers, and audio I/O
helpers.

.. contents::
   :local:
   :depth: 1

Buffers (``common.buffers``)
============================

``Buffer[T]`` is a thread-safe producer-consumer queue with multi-queue
fan-out. When created with ``num_queues=2``, every ``put()`` copies the
item into both queues. This is how the same audio stream feeds the S2S
service (queue 0) and the ASD service (queue 1) concurrently.

Key methods: ``put(item)``, ``get(consumer_id)``, ``done`` (property to
signal the producer has finished), ``is_exhausted(consumer_id)``
(returns ``True`` when the queue is empty and ``done`` is set).

``RequestIteratorFromBuffer[ReqT]`` wraps a ``Buffer`` as a standard
Python iterator, draining a specific consumer queue until exhausted. It
is the bridge between buffers and gRPC request streams.

Deserializer (``common.deserializer``)
======================================

``Deserializer[T]`` runs a daemon thread that reads items from an
iterator (typically an incoming gRPC stream) and distributes each item
to one or more buffers. Subclasses implement two hooks:

- ``_distribute(request)`` -- routes a single item to the appropriate
  buffer(s) based on which fields are populated.
- ``_on_complete()`` -- called when the stream ends; marks all buffers
  as done.

The Controller's ``ContentLocalizationDeserializer`` extends this to
split ``ContentLocalizationRequest`` packets into ``audio_buffer``,
``video_buffer``, ``diarization_buffer``, ``background_audio_buffer``,
``translated_audio_buffer``, ``controller_config_buffer``, and
per-service config buffers.

Handles (``common.handles``)
============================

Client-side handles for reaching peer services:

``GRPCServiceHandle`` names a remote gRPC endpoint and probes it using
the standard ``grpc_health.v1`` health-checking protocol via
``is_healthy()``. Timeout is controlled by the
``HEALTH_CHECK_TIMEOUT`` environment variable. A ``from_string(url)``
factory method parses ``host:port`` strings.

Inference Service (``common.service``)
======================================

``GRPCInferenceHandle`` extends ``GRPCServiceHandle`` with channel and
stub management (``connect()``/``close()``), and exposes an abstract
``get_response_iterator(request_iterator)`` that subclasses implement
to call a specific RPC.

Concrete implementations live in ``common.nims``:

- ``SpeechToSpeechHandle`` -- calls ``StreamSpeechToSpeech``
- ``ActiveSpeakerDetectionHandle`` -- calls ``DetectActiveSpeaker``
- ``LipsyncHandle`` -- calls ``Lipsync``

Clients (``common.clients``)
=============================

``Client[ReqT, RespT]`` is an abstract class that reads requests from
an iterator, sends them through a ``GRPCInferenceHandle``, and writes
responses into an output ``Buffer``. It sets ``output_buffer.done = True``
automatically when the stream finishes. Subclasses implement
``_impl()`` for service-specific logic (e.g., filtering keepalive
messages).

Concrete implementations in ``common.nims``:

- ``SpeechToSpeechClient`` -- streams S2S responses, skips keepalives
- ``ActiveSpeakerDetectionClient`` -- streams ASD results, filters
  keepalive and config-only messages
- ``LipsyncClient`` -- streams LipSync responses, skips keepalives

Audio Utilities (``common.audio_utils``)
========================================

- ``download_audio_file_from_iterator(chunks, file_path)`` -- writes
  streaming audio chunks to a raw binary file. Handles both objects
  with an ``audio_data`` attribute and plain ``bytes``.
- ``write_wav_iterator_to_file(chunks, file_path, ...)`` -- writes
  streaming audio into a properly formatted WAV file with correct
  headers (sample rate, width, channels).
- ``create_wav_header(n_channels, sample_width, frame_rate, n_frames)``
  -- generates a standalone WAV file header as bytes.
