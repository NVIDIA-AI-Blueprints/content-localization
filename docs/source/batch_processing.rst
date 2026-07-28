.. _batch_processing:

Batch Processing
================

The ``client.batch_processing`` package provides a batch processing
harness that runs the end-to-end content localization pipeline on every
MP4 video in a directory and produces a timing report. Use it to measure
throughput, real-time factors, and pipeline latency across a corpus of
videos.

.. mermaid::

   flowchart LR
       A["Input Directory (.mp4 files)"] --> B[Discover Videos]
       B --> C["Preprocess (ffmpeg)"]
       C --> D[Controller gRPC Pipeline]
       D --> E[Collect BatchResult]
       E --> F["Report (console + JSON)"]

----

Quick Start
-----------

**Prerequisites:**

* The controller service (and its downstream S2S, ASD, LipSync services)
  must be running. See :ref:`deployment <deployment>` for instructions.
* ``ffmpeg`` and ``ffprobe`` must be installed and available on ``PATH``.

**Step 1: Start the Services**

.. code-block:: bash

   docker compose --profile controller-third-party-s2s \
       --env-file configs/elevenlabs.env --env-file .env up --build

**Step 2: Run the Batch Processing Pipeline**

Use the convenience shell script, which activates the virtual
environment and sets ``PYTHONPATH`` automatically:

.. code-block:: bash

   ./scripts/misc/run_evaluation.sh --input-dir assets/

Or run the Python module directly:

.. code-block:: bash

   source .venv/bin/activate
   export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"
   python -m client.batch_processing.app --input-dir assets/

**Step 3: Review the Report**

After processing completes, a summary is printed to the console and a
JSON report is saved to the output directory (default:
``outputs/batch_processing/batch_processing_report.json``).

----

How It Works
------------

For each ``.mp4`` file discovered in ``--input-dir``, the tool performs
the following steps:

1. **Preprocess** -- Extracts 16 kHz mono WAV audio via ``ffmpeg`` and
   probes the video duration via ``ffprobe``. Extracted audio is saved
   to ``{output-dir}/preprocessed/``.

2. **Pipeline** -- Creates ``AudioSourceSimulator`` and
   ``VideoSourceSimulator`` objects from the WAV and MP4 files, then
   streams them to the controller gRPC service using the same request
   generator and response writer as the controller client.

3. **Report** -- Collects per-video timing metrics (preprocess time,
   pipeline time, total wall-clock time, real-time factor, output file
   size) and aggregates them into a summary.

----

Configuration
-------------

Batch Processing Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--input-dir``
     - *(required)*
     - Directory containing input MP4 video files.
   * - ``--output-dir``
     - ``outputs/batch_processing``
     - Directory for output files (translated videos, reports).
   * - ``--controller-server``
     - ``localhost:50056``
     - Controller gRPC server address (``host:port``).
   * - ``--chunk-size-audio-secs``
     - ``1.0``
     - Audio chunk size in seconds for streaming.
   * - ``--chunk-size-video-bytes``
     - ``1048576`` (1 MB)
     - Video chunk size in bytes for streaming.
   * - ``--s2s-service``
     - ``EL_DUBBING``
     - S2S backend (``EL_DUBBING`` or ``CAMB_DUBBING``); also selects the
       diarization provider.
   * - ``--bypass-s2s``
     - ``False``
     - Skip S2S and feed pre-translated audio directly to LipSync (ASD still
       runs). See ``--translated-audio-dir``.
   * - ``--translated-audio-dir``
     - ``None``
     - Directory of pre-translated audio files named ``{video_stem}.wav`` or
       ``{video_stem}.mp3``, used when ``--bypass-s2s`` is set. When omitted,
       each video's own extracted source audio is used as a timing stand-in.
   * - ``--diarization-chunked-per-segment``
     - ``False``
     - Stream one diarization chunk per source segment instead of merging
       consecutive same-speaker segments.

S2S Arguments
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--source-language``
     - ``en``
     - Source language code for speech-to-speech translation.
   * - ``--target-language``
     - ``de``
     - Target language code for speech-to-speech translation.
   * - ``--voice-name``
     - ``None``
     - Voice name for TTS (optional; service uses default if unset).
   * - ``--elevenlabs-num-speakers``
     - ``0``
     - Number of speakers for ElevenLabs dubbing. 0 = auto-detect.
   * - ``--elevenlabs-drop-background-audio``
     - ``False``
     - Drop background audio from the final dub.
   * - ``--elevenlabs-use-profanity-filter``
     - ``False``
     - Censor profanities in transcripts (Beta).
   * - ``--elevenlabs-target-accent``
     - ``None``
     - Accent to apply when selecting voices (Experimental).
   * - ``--elevenlabs-highest-resolution``
     - ``False``
     - Use the highest resolution available.
   * - ``--elevenlabs-watermark``
     - ``False``
     - Apply watermark to the output.
   * - ``--elevenlabs-dubbing-studio``
     - ``False``
     - Prepare dub for edits in dubbing studio.

ASD Arguments
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--asd-input-audio-codec``
     - ``WAV``
     - Audio codec for ASD input (``WAV`` or ``MP3``).
   * - ``--asd-input-video-codec``
     - ``None``
     - Video codec for ASD input (``H264`` or unspecified).

LipSync Arguments
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--lipsync-input-audio-codec``
     - ``MP3``
     - Audio codec for LipSync input (``WAV`` or ``MP3``).
   * - ``--lipsync-extend-audio``
     - ``unspecified``
     - How to handle video longer than audio.
   * - ``--lipsync-extend-video``
     - ``unspecified``
     - How to handle audio longer than video.
   * - ``--lipsync-output-bitrate-mbps``
     - ``20``
     - Output video bitrate in Mbps.
   * - ``--lipsync-output-idr-interval``
     - ``8``
     - Output video IDR (keyframe) interval.
   * - ``--lipsync-head-movement-speed``
     - ``None``
     - Head movement speed: 0 = static/slow, 1 = fast (optional).
   * - ``--lipsync-output-audio-codec``
     - ``None``
     - Output audio codec (``WAV`` or ``MP3``, optional).
   * - ``--lipsync-is-speaker-info-provided``
     - ``False``
     - Whether speaker bounding boxes are provided from ASD.

----

Output
------

Console Report
~~~~~~~~~~~~~~

After all videos are processed, a formatted summary is printed:

.. code-block:: text

   ========================================================================
   BATCH PROCESSING REPORT
   ========================================================================

     [OK] video_01.mp4
       Duration:    1m 30.0s
       Preprocess:  2.3s
       Pipeline:    45.7s
       Total:       48.0s
       RT factor:   0.51x
       Output size: 12.3 MB

     [FAIL] video_02.mp4
       Duration:    0.0s
       Preprocess:  0.0s
       Pipeline:    0.0s
       Total:       1.2s
       Error:       Connection refused

   ========================================================================
   SUMMARY
   ========================================================================
     Total videos:  2
     Successful:    1
     Failed:        1
     Avg RT factor: 0.51x
     Total input:   1m 30.0s
     Total pipe:    45.7s
     Total wall:    48.0s
   ========================================================================

The **real-time factor** (RT factor) is ``pipeline_time / video_duration``.
A value below 1.0 means the pipeline processes faster than real-time.

JSON Report
~~~~~~~~~~~

A machine-readable JSON report is saved to
``{output-dir}/batch_processing_report.json``:

.. code-block:: json

   {
     "results": [
       {
         "video_name": "video_01.mp4",
         "video_duration_secs": 90.0,
         "preprocess_time_secs": 2.3,
         "pipeline_time_secs": 45.7,
         "total_time_secs": 48.0,
         "output_path": "outputs/batch_processing/video_01_de.mp4",
         "output_size_bytes": 12902400,
         "success": true,
         "error_message": null,
         "realtime_factor": 0.51
       }
     ],
     "summary": {
       "total_videos": 1,
       "successful": 1,
       "failed": 0,
       "avg_realtime_factor": 0.51,
       "total_input_duration_secs": 90.0,
       "total_pipeline_time_secs": 45.7,
       "total_wall_time_secs": 48.0
     }
   }

Output Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   outputs/batch_processing/
   ├── preprocessed/
   │   ├── video_01.wav           # Extracted audio (16 kHz mono)
   │   └── video_02.wav
   ├── video_01_de.mp4            # Translated output video
   ├── video_02_de.mp4
   └── batch_processing_report.json  # JSON report

----

Examples
--------

Translate all videos to French:

.. code-block:: bash

   ./scripts/misc/run_evaluation.sh \
       --input-dir /data/my_videos \
       --target-language fr

Specify a custom output directory and controller address:

.. code-block:: bash

   ./scripts/misc/run_evaluation.sh \
       --input-dir /data/my_videos \
       --output-dir outputs/eval_run_1 \
       --controller-server 10.0.0.5:50056

Adjust chunking parameters for large videos:

.. code-block:: bash

   ./scripts/misc/run_evaluation.sh \
       --input-dir /data/my_videos \
       --chunk-size-audio-secs 2.0 \
       --chunk-size-video-bytes 2097152

----

Module Reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - ``client.batch_processing.app``
     - Main entry point: discover, preprocess, run, and report.
   * - ``client.batch_processing.args``
     - Argument parser with batch-processing-specific and NIM config args.
   * - ``client.batch_processing.preprocessing``
     - ffmpeg/ffprobe wrappers for audio extraction and duration probing.
   * - ``client.batch_processing.runner``
     - ``run_single_video()`` — streams one video through the controller.
   * - ``client.batch_processing.report``
     - ``BatchResult`` dataclass, ``print_report()``, and ``save_report()``.

----

Troubleshooting
---------------

ffmpeg / ffprobe Not Found
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``RuntimeError: ffprobe failed`` or ``Audio extraction failed``

**Solution:** Install ffmpeg:

.. code-block:: bash

   # Ubuntu / Debian
   sudo apt-get install -y ffmpeg

   # Verify
   ffmpeg -version
   ffprobe -version

Controller Service Unreachable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``grpc._channel._InactiveRpcError`` or health check failure

**Solution:**

1. Verify the controller service is running:

   .. code-block:: bash

      docker compose ps

2. Confirm the ``--controller-server`` address matches the running
   service (default: ``localhost:50056``).

3. If running on a remote host, ensure the port is accessible.

No Videos Found
~~~~~~~~~~~~~~~

**Symptom:** ``No video files found in <dir>``

**Solution:**

* Verify the ``--input-dir`` path exists and contains ``.mp4`` files.
* Only ``.mp4`` files are discovered; other formats (e.g., ``.avi``,
  ``.mkv``) must be converted first. Use
  ``scripts/misc/convert_to_streamable_mp4.sh`` to convert.

Pipeline Failure for a Single Video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``[video_name.mp4] FAILED: <error>``

The pipeline continues processing remaining videos when one fails.
Failed videos appear as ``[FAIL]`` in the report with the error message.
Common causes include non-streamable MP4 files or unsupported codecs.
