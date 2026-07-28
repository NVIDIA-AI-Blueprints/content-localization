.. _client_types:

Client Types
============

Three client types target different points on the control-vs-simplicity
spectrum:

* **Controller Client** -- one gRPC connection to the Controller service,
  which handles all orchestration internally. Best for production.
* **Direct Client** -- separate gRPC connections to S2S, ASD, and LipSync.
  Gives full pipeline control; best for development and performance tuning.
* **Individual Clients** -- standalone S2S, ASD, or LipSync clients for
  testing a single service in isolation.



Controller Client
-----------------

The Controller Client provides a simplified interface to the Content Localization Blueprint through a single gRPC connection to the Controller service.

.. mermaid::

   sequenceDiagram
       participant CC as ControllerClient
       participant CS as ControllerService
       participant S2S as S2SService
       participant ASD as ASDNIM
       participant LS as LipSyncService

       CC->>CS: Check()
       CS->>S2S: Check()
       CS->>ASD: Check()
       CS->>LS: Check()

       CC->>CS: StreamContentLocalization(requestIterator)
       Note over CS: Controller starts background threads

       CS->>S2S: StreamSpeechToSpeech(audioChunks)
       S2S-->>CS: SpeechToSpeechResponse(audioData)

       opt ASD NIM enabled
           CS->>ASD: StreamSpeakerDetection(videoChunks)
           ASD-->>CS: SpeakerDetectionResponse(roiData)
       end

       CS->>LS: Lipsync(videoData,audioData,roiData)
       LS-->>CS: LipsyncResponse(videoFileData)
       CS-->>CC: ContentLocalizationResponse(videoFileData)


.. mermaid::

   flowchart LR
       subgraph controllerClient [ControllerClient]
           controllerMain["client.controller.app.main()"]
           controllerArgsFactory["client.controller.args.argsfactory()"]
       end

       subgraph directClient [DirectClient]
           directMain["client.direct.app.main()"]
           directArgsFactory["client.direct.args.argsfactory()"]
       end

       subgraph individualClients [IndividualClients]
           s2sMain["client.s2s.app.main()"]
           s2sArgsFactory["client.s2s.args.argsfactory()"]
           lipsyncMain["client.lipsync.app.main()"]
           lipsyncArgsFactory["client.lipsync.args.argsfactory()"]
           asdMain["client.asd.app.main()"]
           asdArgsFactory["client.asd.args.argsfactory()"]
       end

       subgraph simulators [Simulators]
           audioSource[AudioSourceSimulator]
           videoSource[VideoSourceSimulator]
           fileSource[FileSourceSimulator]
           audioSink[AudioSinkSimulator]
           videoSink[VideoSinkSimulator]
       end

       subgraph stubs [GrpcStubs]
           controllerStub[ContentLocalizationControllerStub]
           s2sStub[SpeechToSpeechStub]
           asdStub[SpeakerDetectionStub]
           lipStub[LipSyncServiceStub]
       end

       controllerMain --> controllerArgsFactory
       controllerMain --> audioSource
       controllerMain --> videoSource
       controllerMain --> controllerStub

       directMain --> directArgsFactory
       directMain --> audioSource
       directMain --> videoSource
       directMain --> s2sStub
       directMain --> asdStub
       directMain --> lipStub

       s2sMain --> s2sArgsFactory
       s2sMain --> audioSource
       s2sMain --> s2sStub
       lipsyncMain --> lipsyncArgsFactory
       lipsyncMain --> videoSource
       lipsyncMain --> lipStub
       asdMain --> asdArgsFactory
       asdMain --> videoSource
       asdMain --> asdStub

       controllerStub --> controllerSvc[ControllerService]
       s2sStub --> s2sSvc[S2SService]
       asdStub --> asdSvc[ASDNIM]
       lipStub --> lipSvc[LipSyncService]

The Controller Client consists of:

* **``client.controller.app.main()``**: CLI entry point
* **``client.controller.args.argsfactory()``**: Command-line argument parser factory
* **Source Simulators**: Audio and video file handling
* **gRPC Stub**: Communication with Controller service

Entry point imports:

.. code-block:: python

   from controller.app import main
   from controller.args import argsfactory

The Controller client supports optional diarization pass-through for
speaker-aware dubbing, background audio mixing, and S2S bypass via
``--translated-audio`` (pre-translated audio goes directly to LipSync).

Example Usage
~~~~~~~~~~~~~

.. code-block:: bash

   python client/controller/app.py \
       --input-audio assets/audio.wav \
       --input-mp4 assets/video.mp4 \
       --output-mp4 outputs/video_output.mp4

Configuration
~~~~~~~~~~~~~

.. code-block:: bash

   python client/controller/app.py \
       --controller-server localhost:50056 \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4

Background Audio
""""""""""""""""

Pass background audio through the pipeline so LipSync mixes it into
the output video:

.. code-block:: bash

   python client/controller/app.py \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4 \
       --background-audio-input background_music.wav \
       --lipsync-background-audio-volume 0.3

The codec is auto-detected from the file extension (``.wav`` or
``.mp3``).  Override with ``--lipsync-background-audio-codec``.

Translated Audio (S2S Bypass)
"""""""""""""""""""""""""""""

Provide pre-translated audio to bypass the S2S service entirely.
The translated audio is sent directly to LipSync:

.. code-block:: bash

   python client/controller/app.py \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4 \
       --translated-audio assets/translated.mp3

The translated audio file can be WAV or MP3 format.

ASD Bypass
""""""""""

Skip the Active Speaker Detection service so that LipSync uses its
internal face detection instead. This is useful when diarization data
is unavailable or speaker-aware lip sync is not needed.

ASD bypass is **auto-enabled** when no ``--diarization-file`` is
provided. To enable it explicitly:

.. code-block:: bash

   python client/controller/app.py \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4 \
       --bypass-asd

When ``--bypass-asd`` is active:

* The ASD NIM service does not need to be running.
* LipSync uses internal face detection (no speaker bounding boxes).
* ``--lipsync-is-speaker-info-provided`` is forced to ``False``.
* Diarization-related arguments are ignored.

**Using pre-computed ASD output** — If you want speaker-aware lip sync
without running ASD inside the controller pipeline, run ASD as a
standalone step first, then feed its output into the LipSync client:

.. code-block:: bash

   # 1. Run ASD standalone to produce speaker info CSV
   python client/asd/app.py \
       --input-mp4 input.mp4 \
       --input-audio input.wav \
       --output-speaker-info assets/asd_speaker_info.csv

   # 2. Run LipSync standalone with the pre-computed speaker info
   python client/lipsync/app.py \
       --input-mp4 input.mp4 \
       --input-audio translated_audio.mp3 \
       --speaker-info-input assets/asd_speaker_info.csv \
       --lipsync-is-speaker-info-provided \
       --lipsync-input-audio-codec MP3 \
       --output-mp4 outputs/lipsync_output.mp4

Direct Client
-------------

The Direct Client provides full control over service interactions through multiple gRPC connections to individual services.

.. mermaid::

   flowchart LR
       directMain["client.direct.app.main()"] --> directArgsFactory["client.direct.args.argsfactory()"]
       directMain --> orchestration[ServiceOrchestration]
       directMain --> monitoring[PerformanceMonitoring]
       directMain --> audioSource[AudioSourceSimulator]
       directMain --> videoSource[VideoSourceSimulator]
       directMain --> audioSink[AudioSinkSimulator]
       directMain --> videoSink[VideoSinkSimulator]

       directMain --> s2sConn[S2SConnection]
       directMain --> asdConn[ASDNIMConnection]
       directMain --> lipConn[LipSyncConnection]

       s2sConn --> s2sService[S2SService]
       asdConn --> asdService[ASDNIM]
       lipConn --> lipService[LipSyncService]

       s2sService --> elevenLabs[ElevenLabs]
       s2sService --> cambAi[CambAI]

       audioSource --> s2sService
       videoSource --> asdService
       s2sService --> lipService
       asdService --> lipService
       videoSource --> lipService
       lipService --> videoSink

       monitoring --> s2sConn
       monitoring --> asdConn
       monitoring --> lipConn


The Direct Client consists of:

* **``client.direct.app.main()``**: CLI entry point
* **``client.direct.args.argsfactory()``**: Command-line argument parser factory
* **Service Orchestration**: Custom orchestration logic
* **Source Simulators**: Audio and video file handling

Entry point imports:

.. code-block:: python

   from direct.app import main
   from direct.args import argsfactory

The Direct client also supports S2S bypass via ``--translated-audio``.

Example Usage
~~~~~~~~~~~~~

.. code-block:: bash

   python client/direct/app.py \
       --input-audio assets/audio.wav \
       --input-mp4 assets/video.mp4 \
       --output-mp4 outputs/video_output.mp4

Configuration
~~~~~~~~~~~~~

.. code-block:: bash

   python client/direct/app.py \
       --s2s-server localhost:50050 \
       --asd-server localhost:50055 \
       --lipsync-server localhost:50054 \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4

Background Audio
""""""""""""""""

.. code-block:: bash

   python client/direct/app.py \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4 \
       --background-audio-input background_music.mp3 \
       --lipsync-background-audio-volume 0.5

Translated Audio (S2S Bypass)
"""""""""""""""""""""""""""""

Provide pre-translated audio to bypass the S2S service entirely.
The translated audio is sent directly to LipSync:

.. code-block:: bash

   python client/direct/app.py \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4 \
       --translated-audio assets/translated.mp3

When using ``--translated-audio``, the S2S service is not required.

ASD Bypass
""""""""""

Skip the Active Speaker Detection service so that LipSync uses its
internal face detection. Works the same as in the Controller client:

.. code-block:: bash

   python client/direct/app.py \
       --input-audio input.wav \
       --input-mp4 input.mp4 \
       --output-mp4 outputs/output.mp4 \
       --bypass-asd

ASD bypass is **auto-enabled** when no ``--diarization-file`` is
provided. When active, the ASD NIM does not need to be running and
``--lipsync-is-speaker-info-provided`` is forced to ``False``.

Individual Clients
------------------

Individual Clients provide focused functionality for specific services, allowing you to test and develop individual components.

S2S Client
~~~~~~~~~~

The S2S Client provides focused functionality for
speech-to-speech processing.

.. mermaid::

   flowchart LR
       s2sApp[S2SApp] --> args[S2SArgsFactory]
       s2sApp --> audioSource[AudioSourceSimulator]
       s2sApp --> audioSink[AudioSinkSimulator]
       s2sApp --> s2sStub[SpeechToSpeechStub]
       s2sApp --> latency[LatencyAnalysis]
       s2sApp --> health[ServiceHealthCheck]
       s2sStub --> s2sService[SpeechToSpeechService]
       s2sStub --> s2sReq[SpeechToSpeechRequest]
       s2sStub --> s2sResp[SpeechToSpeechResponse]

Features
~~~~~~~~

* Audio input/output processing with streaming
* Multiple source and target languages
* Voice selection and parameter tuning
* Latency analysis and throughput metrics
* WAV/MP3 format support
* ElevenLabs or CambAI backend

Example Usage
~~~~~~~~~~~~~

.. code-block:: bash

   python client/s2s/app.py \
       --input-audio assets/audio.wav \
       --output-audio outputs/audio_output.wav

LipSync Client
~~~~~~~~~~~~~~

The LipSync Client provides focused functionality for lip
synchronization.

.. mermaid::

   flowchart LR
       lipApp[LipSyncApp] --> lipArgs[LipSyncArgsFactory]
       lipApp --> videoSource[VideoSourceSimulator]
       lipApp --> videoSink[VideoSinkSimulator]
       lipApp --> audioReader[AudioFileReader]
       lipApp --> roiReader[ROIFileReader]
       lipApp --> lipStub[LipSyncServiceStub]
       lipStub --> lipService[LipSyncService]
       lipStub --> lipReq[LipsyncRequest]
       lipStub --> lipResp[LipsyncResponse]
       lipReq --> lipConfig[LipsyncConfig]
       lipReq --> lipInput[LipsyncInputData]

Features
~~~~~~~~

* Video input/output processing
* Audio and video synchronization
* Speaker info (from ASD) or auto face detection
* Optional background audio mixing
* Configurable output encoding and quality settings

Example Usage
~~~~~~~~~~~~~

.. code-block:: bash

   python client/lipsync/app.py \
       --lipsync-server localhost:50054 \
       --input-mp4 assets/video.mp4 \
       --input-audio assets/audio.wav \
       --output-mp4 outputs/video_output.mp4

With background audio:

.. code-block:: bash

   python client/lipsync/app.py \
       --lipsync-server localhost:50054 \
       --input-mp4 assets/video.mp4 \
       --input-audio assets/audio.wav \
       --output-mp4 outputs/video_output.mp4 \
       --background-audio-input background_music.wav \
       --lipsync-background-audio-volume 0.4

ASD Client
~~~~~~~~~~

The ASD Client provides focused functionality for active
speaker detection.

.. mermaid::

   flowchart LR
       asdApp[ASDApp] --> asdArgs[ASDArgsFactory]
       asdApp --> asdConfig[ASDConfig]
       asdApp --> videoSource[VideoSourceSimulator]
       asdApp --> audioSource[AudioSourceSimulator]
       asdApp --> diarizationLoader[DiarizationLoader]
       asdApp --> roiWriter[ROIOutputWriter]
       asdApp --> asdStub[ActiveSpeakerDetectionStub]
       diarizationLoader --> flatFormat[FlatJSON]
       diarizationLoader --> elevenLabsFormat[ElevenLabsJSON]
       diarizationLoader --> elevenLabsDubbingApiFormat[ElevenLabsDubbingAPIJSON]
       diarizationLoader --> cambFormat[CambAIJSON]
       asdStub --> asdService[ActiveSpeakerDetectionService]
       asdStub --> asdReq[DetectActiveSpeakerRequest]
       asdStub --> asdResp[DetectActiveSpeakerResponse]

Features
~~~~~~~~

* Video and audio input processing
* Speaker detection with optional diarization data
* Multi-format diarization loading (flat, ElevenLabs STT, ElevenLabs
  Dubbing API, ElevenLabs Studio, and CambAI JSON) via ``--diarization-file``
* Per-frame speaker info output in CSV format
* Configurable audio source (``--asd-audio-source-config``)
* Adjustable speaker detection threshold
  (``--asd-speaker-detection-threshold``)
* Configuration via ``ASDConfig`` dataclass with validation

Example Usage
~~~~~~~~~~~~~

.. code-block:: bash

   python client/asd/app.py \
       --input-mp4 assets/video.mp4 \
       --output-speaker-info assets/speaker_info.csv

Batch Processing Client
-----------------------

The Batch Processing Client runs the end-to-end content localization
pipeline on every MP4 video in a directory and produces a timing report.
It reuses the controller client's request generator and response writer,
adding automatic video discovery, preprocessing (audio extraction via
``ffmpeg``), ElevenLabs diarization, and per-video reporting.

.. mermaid::

   flowchart LR
       A["Input Directory (.mp4 files)"] --> B[Discover Videos]
       B --> C["Preprocess (ffmpeg)"]
       C --> D[Controller gRPC Pipeline]
       D --> E[Collect BatchResult]
       E --> F["Report (console + JSON)"]

Entry point imports:

.. code-block:: python

   from client.batch_processing.app import main
   from client.batch_processing.args import argsfactory

It wraps the controller client pipeline, processing videos sequentially
and collecting per-video timing metrics (preprocess time, pipeline time,
real-time factor).

Example Usage
~~~~~~~~~~~~~

.. code-block:: bash

   ./scripts/misc/run_evaluation.sh --input-dir assets/

Or directly:

.. code-block:: bash

   python -m client.batch_processing.app \
       --input-dir assets/ \
       --output-dir outputs/batch_processing \
       --target-language fr

See :ref:`batch_processing` for full documentation.

Shared CLI Arguments
--------------------

Background Audio Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

These arguments are available on the Controller, Direct, and
standalone LipSync clients:

``--background-audio-input FILE``
    Path to a WAV or MP3 file containing background audio (music,
    ambient sound) to mix into the output video.

``--lipsync-background-audio-volume FLOAT``
    Volume level for the background audio track (0.0 -- 1.0).
    Omit to use the LipSync NIM default.

``--lipsync-background-audio-codec {WAV,MP3}``
    Override the background audio codec.  When omitted the codec
    is auto-detected from the file extension.

Translated Audio Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

These arguments are available on the Controller and Direct
clients:

``--translated-audio FILE``
    Path to a pre-translated audio file (WAV or MP3). When
    provided, the S2S service is bypassed and this audio is
    sent directly to LipSync for lip synchronization.

Bypass ASD Arguments
~~~~~~~~~~~~~~~~~~~~

These arguments are available on the Controller and Direct clients:

``--bypass-asd``
    Skip the Active Speaker Detection service. When active, LipSync
    uses its internal face detection instead of speaker bounding boxes
    from ASD. The ASD NIM does not need to be running.

    ASD bypass is **auto-enabled** when no ``--diarization-file`` is
    provided. When active, ``--lipsync-is-speaker-info-provided`` is
    forced to ``False`` and diarization-related arguments are ignored.

    For speaker-aware lip sync without ASD in the pipeline, run ASD
    standalone first (``python client/asd/app.py``) to produce a
    speaker info CSV, then use it with the LipSync client via
    ``--speaker-info-input``.

ASD Tuning Arguments
~~~~~~~~~~~~~~~~~~~~

These arguments are available on clients that use ASD (Controller,
Direct, standalone ASD):

``--asd-audio-source-config {unspecified,separate_stream,embedded_in_video}``
    How audio is supplied to the ASD NIM.  ``separate_stream``
    (default when unspecified) means audio arrives in its own
    data messages; ``embedded_in_video`` means the ASD NIM should
    extract audio from the video stream.

``--asd-speaker-detection-threshold FLOAT``
    Confidence threshold for speaker detection (0.0 -- 1.0).
    Omit to use the NIM default (0.5986).

Client Selection Guide
----------------------

Choose the appropriate client type based on your requirements:

+------------------+------------------+------------------+------------------+
| Requirement      | Controller       | Direct           | Individual       |
+==================+==================+==================+==================+
| Complexity       | Low              | High             | Low              |
+------------------+------------------+------------------+------------------+
| Control          | Limited          | Full             | Service-specific |
+------------------+------------------+------------------+------------------+
| Maintenance      | Low              | High             | Low              |
+------------------+------------------+------------------+------------------+
| Use Cases        | Production       | Development      | Testing          |
+------------------+------------------+------------------+------------------+
| Integration      | Simple           | Complex          | Simple           |
+------------------+------------------+------------------+------------------+
