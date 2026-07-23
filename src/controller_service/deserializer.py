# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-specific deserializer for ``ContentLocalizationRequest`` streams.

The :class:`ContentLocalizationDeserializer` runs on a background thread,
reads the incoming gRPC request stream, and distributes each packet into
the appropriate output buffers:

* Audio packets               -->  ``audio_buffer`` (2 consumer queues,
  copy-on-put)
* S2S-config packets          -->  ``s2s_config_buffer`` (1 consumer queue)
* ASD-config packets          -->  ``asd_config_buffer`` (1 consumer queue)
* LipSync-config packets      -->  ``lipsync_config_buffer`` (1 consumer queue)
* Video packets               -->  ``video_buffer`` (2 consumer queues,
  copy-on-put)
* Diarization packets         -->  ``diarization_buffer`` (1 consumer queue)
* Background audio packets    -->  ``background_audio_buffer`` (1 consumer
  queue, LipSync only)
* Translated audio packets    -->  ``translated_audio_buffer`` (1 consumer
  queue, bypasses S2S)
* Controller config packets   -->  ``controller_config_buffer`` (1 consumer
  queue)

Callers construct ``RequestIteratorFromBuffer`` instances on the buffers
to feed downstream client threads.

Example::

    from common.buffers import RequestIteratorFromBuffer

    deserializer = ContentLocalizationDeserializer(request_iterator)
    deserializer.start(request_id="req-1")

    s2s_cfg_iter = RequestIteratorFromBuffer(
        deserializer.s2s_config_buffer,
        consumer_id=0,
    )
    s2s_audio_iter = RequestIteratorFromBuffer(
        deserializer.audio_buffer,
        consumer_id=AudioQueueConsumer.S2S,
    )
    asd_video_iter = RequestIteratorFromBuffer(
        deserializer.video_buffer,
        consumer_id=VideoQueueConsumer.ASD,
    )
    lipsync_v_iter = RequestIteratorFromBuffer(
        deserializer.video_buffer,
        consumer_id=VideoQueueConsumer.LIPSYNC,
    )
    diarization_iter = RequestIteratorFromBuffer(
        deserializer.diarization_buffer,
        consumer_id=DiarizationQueueConsumer.ASD,
    )
    ctrl_cfg_iter = RequestIteratorFromBuffer(
        deserializer.controller_config_buffer,
        consumer_id=0,
    )
    translated_iter = RequestIteratorFromBuffer(
        deserializer.translated_audio_buffer,
        consumer_id=0,
    )
"""

from collections.abc import Iterator
from copy import deepcopy

from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest

from common.base_utils import logger
from common.buffers import Buffer
from common.deserializer import Deserializer

# Cap on distinct conflicting request_ids remembered for warning dedup, so a
# client stamping a fresh id on every chunk cannot grow the set unboundedly.
_MAX_WARNED_REQUEST_IDS = 100


class VideoQueueConsumer:
    """Consumer IDs for the video_buffer (num_queues=2)."""

    ASD = 0
    LIPSYNC = 1


class AudioQueueConsumer:
    """Consumer IDs for the audio_buffer (num_queues=2)."""

    S2S = 0
    ASD = 1


class DiarizationQueueConsumer:
    """Consumer IDs for the diarization_buffer (num_queues=1)."""

    ASD = 0


class ContentLocalizationDeserializer(Deserializer[ContentLocalizationRequest]):
    """Splits a ``ContentLocalizationRequest`` stream into audio, video,
    and diarization output buffers for downstream services.

    Audio and video buffers use ``num_queues=2`` with ``copy_func=deepcopy``
    so that a single ``put()`` fans out to both consumer queues with isolated
    copies. The ``s2s_config_buffer`` and diarization buffer use ``num_queues=1``.

    Attributes:
        audio_buffer: Buffer holding audio packets (2 queues).
        s2s_config_buffer: Buffer holding S2S-config packets (1 queue).
        asd_config_buffer: Buffer holding ASD-config packets (1 queue).
        lipsync_config_buffer: Buffer holding LipSync-config packets (1 queue).
        video_buffer: Buffer holding video packets (2 queues).
        diarization_buffer: Buffer holding diarization packets (1 queue).
        background_audio_buffer: Buffer holding background audio packets
            (1 queue, consumed by LipSync only).
        translated_audio_buffer: Buffer holding pre-translated audio packets
            (1 queue, bypasses S2S when present).
        controller_config_buffer: Buffer holding controller config packets
            (1 queue, e.g. bypass_s2s flag).
        client_request_id: First client-supplied request_id seen on the
            stream; echoed in responses. Empty when the client sends none.
            Differing ids arriving later are ignored with a warning.
    """

    def __init__(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
    ) -> None:
        """Initialise the deserializer and create output buffers.

        Args:
            request_iterator: The upstream gRPC request stream.
        """
        super().__init__(request_iterator)

        self.audio_buffer: Buffer[ContentLocalizationRequest] = Buffer(
            num_queues=2, copy_func=deepcopy
        )
        self.s2s_config_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)
        self.asd_config_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)
        self.lipsync_config_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)
        self.video_buffer: Buffer[ContentLocalizationRequest] = Buffer(
            num_queues=2, copy_func=deepcopy
        )
        self.diarization_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)
        self.background_audio_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)
        self.translated_audio_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)
        self.controller_config_buffer: Buffer[ContentLocalizationRequest] = Buffer(num_queues=1)

        # First client-supplied request_id seen on the stream; echoed in
        # responses so clients can correlate them with their requests.
        self.client_request_id: str = ""

        # Conflicting ids already warned about. Clients may stamp every
        # chunk, so warning once per distinct id keeps logs bounded; the
        # cap keeps the set bounded against ever-changing ids.
        self._warned_request_ids: set[str] = set()

        logger.debug("ContentLocalizationDeserializer initialised")

    def named_buffers(self) -> dict[str, Buffer[ContentLocalizationRequest]]:
        """Return the output buffers keyed by their attribute names.

        Returns:
            dict[str, Buffer[ContentLocalizationRequest]]: Mapping of buffer
                name to buffer, covering every buffer this deserializer
                routes into.

        Examples:
            >>> sorted(deserializer.named_buffers())[:2]  # doctest: +SKIP
            ['asd_config_buffer', 'audio_buffer']
        """
        return {
            "audio_buffer": self.audio_buffer,
            "s2s_config_buffer": self.s2s_config_buffer,
            "asd_config_buffer": self.asd_config_buffer,
            "lipsync_config_buffer": self.lipsync_config_buffer,
            "video_buffer": self.video_buffer,
            "diarization_buffer": self.diarization_buffer,
            "background_audio_buffer": self.background_audio_buffer,
            "translated_audio_buffer": self.translated_audio_buffer,
            "controller_config_buffer": self.controller_config_buffer,
        }

    # -- routing logic ---------------------------------------------------

    def _capture_request_id(self, request: ContentLocalizationRequest) -> None:
        """Record the first client-supplied request_id seen on the stream.

        Later requests carrying a different id are ignored so responses stay
        correlated with a single id; each distinct conflicting id is logged
        once as a warning.

        Args:
            request: The incoming gRPC request packet.

        Returns:
            None

        Examples:
            >>> deserializer._capture_request_id(request=request)  # doctest: +SKIP
        """
        if not request.HasField("request_id"):
            return
        if not self.client_request_id:
            self.client_request_id = request.request_id
        elif (
            request.request_id != self.client_request_id
            and request.request_id not in self._warned_request_ids
            and len(self._warned_request_ids) < _MAX_WARNED_REQUEST_IDS
        ):
            self._warned_request_ids.add(request.request_id)
            logger.warning(
                f"Ignoring request_id {request.request_id!r} received mid-stream; "
                f"responses keep the first id {self.client_request_id!r}"
            )

    def _distribute(self, request: ContentLocalizationRequest) -> None:
        """Route a single ``ContentLocalizationRequest`` to the correct buffers.

        * ``audio_data`` -> ``audio_buffer``
        * ``s2s_config`` -> ``s2s_config_buffer``
        * ``asd_config`` -> ``asd_config_buffer``
        * ``lipsync_config`` -> ``lipsync_config_buffer``
        * ``video_file_data`` -> ``video_buffer`` (fans out to both consumers)
        * ``diarization_info`` -> ``diarization_buffer``
        * ``background_audio_data`` -> ``background_audio_buffer``
        * ``translated_audio_data`` -> ``translated_audio_buffer``
        * ``controller_config`` -> ``controller_config_buffer``

        Args:
            request: The incoming gRPC request packet.
        """
        self._capture_request_id(request=request)

        if request.WhichOneof("payload") is None:
            logger.warning("Request message carries no payload field; ignoring")
            return

        if request.HasField("s2s_config"):
            self.s2s_config_buffer.put(request)
            logger.debug("Deserialized s2s_config packet into s2s_config_buffer")

        if request.HasField("asd_config"):
            self.asd_config_buffer.put(request)
            logger.debug("Deserialized asd_config packet into asd_config_buffer")

        if request.HasField("lipsync_config"):
            self.lipsync_config_buffer.put(request)
            logger.debug("Deserialized lipsync_config packet into lipsync_config_buffer")

        if request.HasField("audio_data"):
            self.audio_buffer.put(request)
            logger.debug("Deserialized audio packet into audio_buffer")

        if request.HasField("video_file_data"):
            self.video_buffer.put(request)
            logger.debug("Deserialized video packet into video_buffer (2 consumers)")

        if request.HasField("diarization_info"):
            self.diarization_buffer.put(request)
            logger.debug("Deserialized diarization packet into diarization_buffer")

        if request.HasField("background_audio_data"):
            self.background_audio_buffer.put(request)
            logger.debug("Deserialized background_audio packet into background_audio_buffer")

        if request.HasField("translated_audio_data"):
            self.translated_audio_buffer.put(request)
            logger.debug("Deserialized translated_audio packet into translated_audio_buffer")

        if request.HasField("controller_config"):
            self.controller_config_buffer.put(request)
            logger.debug("Deserialized controller_config packet into controller_config_buffer")

    def _on_complete(self) -> None:
        """Mark all output buffers as done."""
        self.s2s_config_buffer.done = True
        self.asd_config_buffer.done = True
        self.lipsync_config_buffer.done = True
        self.video_buffer.done = True
        self.diarization_buffer.done = True
        self.audio_buffer.done = True
        self.background_audio_buffer.done = True
        self.translated_audio_buffer.done = True
        self.controller_config_buffer.done = True
        logger.debug("ContentLocalizationDeserializer: all buffers marked done")
