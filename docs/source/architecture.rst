.. _architecture:

Architecture Overview
=====================

The Content Localization Blueprint follows a microservices
architecture that orchestrates multiple AI services to provide
end-to-end content localization. It is designed for flexibility
and scalability across diverse deployment scenarios, balancing
throughput with low latency while keeping operational complexity
manageable.

Design Rationale: Streaming API
-------------------------------

Every service in the blueprint, including the Controller, exposes a
synchronous bidirectional streaming gRPC API. This is a deliberate
design choice: audio and video flow through the pipeline as chunked
streams, so a single RPC carries a request's full media exchange with
natural flow control on each hop. Long-running work — third-party
dubbing jobs can take many minutes — rides the same stream, with
keepalive messages maintaining the connection while the job runs.
Each request opens its own channels to the downstream services, and
failures propagate as typed errors mapped to specific gRPC status
codes at a single abort point, so a failure anywhere in the pipeline
reaches the caller deterministically.

System Architecture
-------------------

At a high level, the system comprises a Controller service that
coordinates requests and responses, three AI services (S2S, ASD,
and LipSync) that perform the core processing, and client
applications that interact with the pipeline.

.. mermaid::

   flowchart TB
       subgraph clients [Client Layer]
           controllerClient[Controller Client]
           directClient[Direct Client]
           individualClients[Individual Clients]
       end

       subgraph controllerLayer [Controller Service]
           controllerService[ControllerService]
           deserializer[ContentLocalizationDeserializer]
           audioBuffer[audio_buffer]
           videoBuffer[video_buffer]
           diarizationBuffer[diarization_buffer]
           bgAudioBuffer[background_audio_buffer]
           translatedAudioBuffer[translated_audio_buffer]
           controllerConfigBuffer[controller_config_buffer]
           s2sThread[S2S Client Thread]
           asdThread[ASD Client Thread]
           lipsyncThread[LipSync Client Thread]
       end

       subgraph aiServices [AI Services]
           s2sService[Speech-to-Speech Service]
           asdService[Active Speaker Detection NIM]
           lipService[LipSync NIM]
       end

       subgraph s2sBackends [S2S Backends]
           elevenLabs[ElevenLabs Dubbing API]
           cambAi[CambAI Dubbing API]
       end

       controllerClient --> controllerService
       directClient --> s2sService
       directClient --> asdService
       directClient --> lipService
       individualClients --> s2sService
       individualClients --> asdService
       individualClients --> lipService

       controllerService --> deserializer
       deserializer --> audioBuffer
       deserializer --> videoBuffer
       deserializer --> diarizationBuffer
       deserializer --> bgAudioBuffer
       deserializer --> translatedAudioBuffer
       deserializer --> controllerConfigBuffer

       audioBuffer -.-> s2sThread
       s2sThread -.-> s2sService

       audioBuffer --> asdThread
       videoBuffer --> asdThread
       diarizationBuffer --> asdThread
       asdThread --> asdService

       videoBuffer --> lipsyncThread
       s2sService -.-> lipsyncThread
       asdService --> lipsyncThread
       bgAudioBuffer --> lipsyncThread
       translatedAudioBuffer --> lipsyncThread
       lipsyncThread --> lipService

       lipService --> controllerService

       s2sService --> elevenLabs
       s2sService --> cambAi

Controller Service
------------------

The Controller service is the central orchestrator. It manages
incoming client requests, coordinates interactions with S2S,
ASD, and LipSync, and streams results back to clients. The
controller uses a multi-threaded architecture where a
``ContentLocalizationDeserializer`` runs on a background thread,
consuming the incoming gRPC stream and distributing packets into
typed buffers (``audio_buffer``, ``video_buffer``,
``diarization_buffer``, ``background_audio_buffer``,
``translated_audio_buffer``, ``controller_config_buffer``,
and per-service config buffers).
Downstream client threads consume from these buffers
concurrently.

.. mermaid::

   flowchart TB
       clientApp[ClientApplication]
       clientApp --> servicer[ControllerServiceServicer]
       servicer --> controller[ControllerService]
       controller --> impl[_controller_impl]

       impl --> deserializer[ContentLocalizationDeserializer]
       deserializer --> audioBuffer[audio_buffer]
       deserializer --> videoBuffer[video_buffer]
       deserializer --> diarizationBuffer[diarization_buffer]
       deserializer --> bgAudioBuf[background_audio_buffer]
       deserializer --> translatedAudioBuf[translated_audio_buffer]
       deserializer --> controllerConfigBuf[controller_config_buffer]

       audioBuffer -.-> s2sThread[S2S client thread optional]
       s2sThread --> s2sClient[SpeechToSpeechClient]
       s2sClient --> s2sServer[S2SService]
       s2sServer --> s2sOut[s2s_output_buffer]

       videoBuffer --> asdThread[ASD client thread]
       audioBuffer --> asdThread
       diarizationBuffer --> asdThread
       asdThread --> asdClient[ActiveSpeakerDetectionClient]
       asdClient --> asdServer[ASDService]
       asdServer --> asdOut[asd_output_buffer]

       videoBuffer --> videoIter[to_lipsync_video]
       s2sOut --> audioAdapter[s2s_audio_to_lipsync_audio]
       asdOut --> roiAdapter[asd_response_to_lipsync_speaker_info]
       bgAudioBuf --> bgAudioIter[to_lipsync_background_audio]
       translatedAudioBuf --> translatedAdapter[translated_audio_to_lipsync_audio]

       videoIter --> requestGen[lipsync_request_generator]
       audioAdapter --> requestGen
       roiAdapter --> requestGen
       bgAudioIter --> requestGen
       translatedAdapter --> requestGen

       requestGen --> lipsyncThread[LipSync client thread]
       lipsyncThread --> lipsyncClient[LipsyncClient]
       lipsyncClient --> lipsyncServer[LipSyncService]
       lipsyncServer --> lipsyncOut[lipsync_output_buffer]

       lipsyncOut --> controller
       controller --> servicer
       servicer --> clientApp

Controller Data Flow
--------------------

The following diagram shows the detailed data flow through
the controller pipeline, including the stream adapter
functions that transform data between services.

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

Request Processing Sequence
---------------------------

The following sequence diagram shows the temporal ordering
of operations in the multi-threaded pipeline.

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant S as ControllerService
       participant D as ContentLocalizationDeserializer
       participant ST as S2S Client Thread
       participant AT as ASD Client Thread
       participant LT as LipSync Client Thread
       participant S2S as S2SService
       participant ASD as ASDService
       participant L as LipSyncService

       C->>S: StreamContentLocalization(requestIterator)
       S->>D: start(requestIterator)
       opt S2S enabled (no bypass)
           S->>ST: start()
       end
       opt ASD enabled
           S->>AT: start()
       end
       S->>LT: start()

       loop Continuous processing
           opt S2S enabled (no bypass)
               D->>ST: audio_buffer requests
               ST->>S2S: stream speech requests
               S2S-->>ST: speech responses
           end

           opt S2S bypassed
               Note over D,LT: translated_audio_buffer → LipSync directly
           end

           opt ASD enabled
               D->>AT: video_buffer requests
               D->>AT: audio_buffer requests
               D->>AT: diarization_buffer requests
               AT->>ASD: stream speaker requests
               ASD-->>AT: speaker responses
           end

           D->>LT: video_buffer requests
           ST->>LT: adapted audio stream (or translated audio)
           AT->>LT: adapted speaker info stream
           LT->>L: stream LipsyncRequest
           L-->>S: LipsyncResponse
           S-->>C: ContentLocalizationResponse
       end

       S->>D: stop()
       opt S2S enabled (no bypass)
           S->>ST: join()
       end
       opt ASD enabled
           S->>AT: join()
       end
       S->>LT: join()

AI Services
-----------

Three AI services implement the core functionality. The
Speech-to-Speech (S2S) service translates and synthesizes audio
with streaming support, multi-language coverage, and voice
cloning. It can run with ElevenLabs backends (with expanded
parameters such as ``num_speakers``, ``drop_background_audio``,
``use_profanity_filter``, ``target_accent``, and
``highest_resolution``) or CambAI dubbing, accepting common audio
formats and producing translated audio for downstream services.
The Active Speaker Detection (ASD) service identifies
the speaker in each scene to be lipsynced, accepts video, audio,
and optional diarization input for speaker-aware detection, and
returns per-frame speaker info with confidence scores; it can be
disabled when speaker tracking is not required (such as in clear
single-speaker use cases). The LipSync service aligns mouth
movements to the translated audio and offers flexible encoding
and multiple output formats, consuming video, translated audio,
and optional speaker info inputs to produce synchronized video.

Client Applications
-------------------

Clients come in three styles.

1. The Controller client uses a single gRPC connection to the
   controller for turnkey end-to-end processing and is the
   preferred option for production deployments.
2. The Direct client connects to each service independently,
   enabling custom orchestration and deep monitoring -- well
   suited to development and performance work.
3. Individual clients focus on single services (S2S, ASD, or
   LipSync) for testing, debugging, or minimal integrations.

A typical pipeline begins when the client sends audio and video
to the system. The S2S service translates speech to the target
language, ASD (optionally) identifies speaking faces, and
LipSync aligns the translated audio with the video. The
resulting stream is then returned to the client. Both the
Controller and Direct clients support S2S bypass via
``--translated-audio``, which routes pre-translated audio
directly to LipSync and skips S2S entirely. Exact data
paths vary by client type: the Controller client delegates all
orchestration to the controller, the Direct client manages
interactions explicitly, and Individual clients operate one
service at a time.

See :doc:`client_types` for a detailed guide to each client
type.
