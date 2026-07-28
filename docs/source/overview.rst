.. _overview:

Overview
========

The Content Localization Blueprint is an AI-powered microservices
system that takes video with speech in one language and produces
a new video where the speech is translated, the speakers' lip
movements are re-synced to match the new audio, and the correct
speaker's face is identified for each frame. It is an automated
dubbing pipeline built on NVIDIA AI services and gRPC streaming.

.. mermaid::

   flowchart TB
       subgraph clients [Client Layer]
           controllerClient[Controller Client]
           directClient[Direct Client]
           individualClients[Individual Clients]
       end

       subgraph controllerLayer [Controller Service Layer]
           controllerService[ControllerService]
           requestProcessing[Multi-Threaded Processing]
       end

       subgraph aiServices [AI Service Layer]
           s2sService[S2S Service]
           asdService[ASD Service]
           lipService[LipSync Service]
       end

       subgraph externalServices [External Services]
           elevenLabs[ElevenLabs]
           cambAi[CambAI]
       end

       controllerClient --> controllerService
       directClient --> s2sService
       directClient --> asdService
       directClient --> lipService
       individualClients --> s2sService
       individualClients --> asdService
       individualClients --> lipService

       controllerService --> requestProcessing
       controllerService --> s2sService
       controllerService --> asdService
       controllerService --> lipService

       s2sService --> elevenLabs
       s2sService --> cambAi

AI Services
-----------

The system orchestrates three AI services through a central
Controller:

**Speech-to-Speech (S2S)** translates audio from one language
to another. Two backends are supported:

- **ElevenLabs**: Cloud-based dubbing API with voice cloning,
  multi-speaker detection, and profanity filtering. Produces
  MP3 output.
- **CambAI**: Cloud-based dubbing API for translated speech.
  Produces MP3 output for downstream LipSync.

**Active Speaker Detection (ASD)** analyzes video and audio
together to identify which face is speaking in each frame.
It returns per-frame bounding boxes, speaker IDs, and
``is_speaking`` flags. ASD is optional -- when disabled, the
LipSync NIM uses its own internal face detection.

**LipSync** takes the original video, the translated audio,
and (optionally) speaker face info from ASD, then re-renders
the video so that lip movements match the new audio.

Core Abstractions
-----------------

Four classes in ``src/common/`` form the backbone of the system:

**Buffer** (``common.buffers``) is a thread-safe
producer-consumer queue with multi-queue fan-out. When created
with ``num_queues=2``, every ``put()`` copies the item to both
queues. This is how the same audio stream feeds both the S2S
service (queue 0) and the ASD service (queue 1).
``RequestIteratorFromBuffer`` wraps a buffer as a standard
Python iterator.

**Deserializer** (``common.deserializer``) runs a background
daemon thread that reads from an iterator (e.g., a gRPC stream)
and calls ``_distribute()`` on each item. Subclasses decide
which buffer gets which item. When the stream ends,
``_on_complete()`` marks all buffers done.

**GRPCInferenceHandle** (``common.service``) is an abstract
gRPC client wrapper that manages a channel and stub. Concrete
subclasses (``SpeechToSpeechHandle``,
``ActiveSpeakerDetectionHandle``, ``LipsyncHandle``) each know
which RPC to call.

**Client** (``common.clients``) reads requests from an iterator,
calls a ``GRPCInferenceHandle``, and writes responses to an
output buffer. Concrete implementations
(``SpeechToSpeechClient``, ``ActiveSpeakerDetectionClient``,
``LipsyncClient``) stream responses from the service into the
buffer, skipping keep-alive messages.

Multi-Threaded Pipeline
-----------------------

The Controller's ``_controller_impl`` method orchestrates the
full pipeline. Here is how a request flows through the system:

.. mermaid::

   flowchart TB
       requestStream[ContentLocalizationRequest stream]
       requestStream --> deserializer[ContentLocalizationDeserializer]
       deserializer --> audioBuffer[audio_buffer]
       deserializer --> videoBuffer[video_buffer]
       deserializer --> diarizationBuffer[diarization_buffer]
       deserializer --> bgAudioBuffer[background_audio_buffer]
       deserializer --> translatedAudioBuffer[translated_audio_buffer]

       audioBuffer -.-> s2sReq[to_s2s_request optional]
       s2sReq --> s2sService[S2SService]
       s2sService --> s2sResp[s2s_output_buffer]
       s2sResp --> audioAdapter[s2s_audio_to_lipsync_audio]

       videoBuffer --> asdVideoReq[to_asd_video_data]
       audioBuffer --> asdAudioReq[to_asd_audio_data]
       diarizationBuffer --> asdDiarizationReq[to_asd_diarization_data]
       asdVideoReq --> asdRequestGen[asd_request_generator]
       asdAudioReq --> asdRequestGen
       asdDiarizationReq --> asdRequestGen
       asdRequestGen --> asdService[ASDService optional]
       asdService --> asdResp[asd_output_buffer]
       asdResp --> roiAdapter[asd_response_to_lipsync_speaker_info]

       videoBuffer --> videoAdapter[to_lipsync_video]
       bgAudioBuffer --> bgAudioAdapter[to_lipsync_background_audio]
       translatedAudioBuffer --> translatedAdapter[translated_audio_to_lipsync_audio]
       videoAdapter --> lipsyncGen[lipsync_request_generator]
       audioAdapter --> lipsyncGen
       roiAdapter --> lipsyncGen
       bgAudioAdapter --> lipsyncGen
       translatedAdapter --> lipsyncGen

       lipsyncGen --> lipsyncService[LipSyncService]
       lipsyncService --> lipsyncResp[lipsync_output_buffer]
       lipsyncResp --> response[ContentLocalizationResponse stream]

1. The **ContentLocalizationDeserializer** (background thread)
   receives the incoming gRPC stream and routes each packet to
   the correct buffer based on its field: audio goes to
   ``audio_buffer`` (2 queues), video to ``video_buffer``
   (2 queues), diarization to ``diarization_buffer``, and
   configs to their respective buffers.
2. The **S2S client thread** (optional -- skipped when
   ``bypass_s2s`` is set) reads audio from
   ``audio_buffer[0]``, sends it to the S2S service, and puts
   translated audio into ``s2s_output_buffer``. When bypassed,
   pre-translated audio from ``translated_audio_buffer`` is
   routed directly to LipSync.
3. The **ASD client thread** (optional) reads video from
   ``video_buffer[0]``, audio from ``audio_buffer[1]``, and
   diarization data, sends them to the ASD NIM, and puts
   per-frame speaker info into ``asd_output_buffer``.
4. The **LipSync client thread** reads video from
   ``video_buffer[1]``, translated audio from
   ``s2s_output_buffer``, and speaker info from
   ``asd_output_buffer``. Stream adapters
   (``s2s_audio_to_lipsync_audio``,
   ``asd_response_to_lipsync_speaker_info``) convert between
   service-specific protobuf formats.
5. The **main thread** reads lip-synced video from
   ``lipsync_output_buffer`` and streams it back to the client.

gRPC Protocol
-------------

Eight ``.proto`` files define the API:

- ``nvidia.ai4m.controller.v1`` -- Controller service, the main
  entry point
- ``nvidia.ai4m.s2s.v1`` -- Speech-to-Speech with ElevenLabs-specific
  config fields
- ``nvidia.ai4m.activespeakerdetection.v1`` -- ASD NIM
- ``nvidia.ai4m.lipsync.v1`` -- LipSync NIM with speaker info per
  frame
- ``nvidia.ai4m.audio.v1`` -- Shared audio types (codecs, configs)
- ``nvidia.ai4m.video.v1`` -- Shared video types (encoding)
- ``nvidia.ai4m.common.v1`` -- Shared types (BoundingBox)
- ``health.proto`` -- gRPC health checking

The key data flow through the controller proto:

- **Request**: ``ContentLocalizationRequest`` carries optional
  fields for ``audio_data``, ``video_file_data``,
  ``s2s_config``, ``asd_config``, ``lipsync_config``,
  ``controller_config``, ``translated_audio_data``,
  ``background_audio_data``, and ``diarization_info`` -- all
  multiplexed on the same stream.
- **Response**: ``ContentLocalizationResponse`` carries
  ``video_file_data`` (the lip-synced output video).

Client Types
------------

Several standalone clients cover different use cases:

- **Controller** (``client/controller/app.py``): Full end-to-end
  pipeline via the Controller service. Recommended for
  production.
- **Direct** (``client/direct/app.py``): Calls S2S, ASD, and
  LipSync directly (no Controller). Best for development.
- **S2S** (``client/s2s/app.py``): Audio translation only.
- **ASD** (``client/asd/app.py``): Speaker detection only.
- **LipSync** (``client/lipsync/app.py``): Lip sync only.
- **Batch Processing** (``client/batch_processing/app.py``): Batch process
  a directory of videos.

See :doc:`client_types` for a detailed guide.

Getting Started
---------------

1. Set up your environment: :doc:`setup`
2. Deploy services: :doc:`deployment`
3. Run a client: :doc:`client`

For a quick demo with a web UI, see :doc:`demo_app`.
