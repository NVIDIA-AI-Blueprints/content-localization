.. _utilities:

Utility Scripts
===============

The ``scripts/`` directory contains various utility scripts to help with development,
deployment, and testing.

Deployment Scripts
------------------

These scripts download models and start individual services for verification before full deployment.

deploy_lipsync.sh
~~~~~~~~~~~~~~~~~

Deploy LipSync service.

.. code-block:: bash

   ./scripts/nims/deploy_lipsync.sh

**Features:**

* Downloads LipSync models to ``volumes/models/lipsync/``
* Starts LipSync container on ports 8004 (HTTP) and 50054 (gRPC)
* Requires ``LIPSYNC_API_KEY`` environment variable

deploy_asd.sh
~~~~~~~~~~~~~

Deploy Active Speaker Detection (ASD) NIM container.

.. code-block:: bash

   ./scripts/nims/deploy_asd.sh

**Features:**

* Deploys ASD NIM container with GPU support
* Configures HTTP port (``ASD_NIM_HTTP_API_PORT``, default 8005) and gRPC port
  (``ASD_GRPC_API_PORT``, default 50055)
* Mounts model cache at ``volumes/models/asd/``
* Requires ``ASD_API_KEY`` environment variable

Development Scripts
-------------------

setup_env.sh
~~~~~~~~~~~~

Complete automated setup of the development environment.

.. code-block:: bash

   # Full setup including Docker and GPU drivers
   ./scripts/misc/setup_env.sh

   # Development setup (adds lint and pre-commit tools)
   ./scripts/misc/setup_env.sh --dev

   # Skip Docker and GPU driver installation
   ./scripts/misc/setup_env.sh --no-docker --no-gpu --dev

**Options:**

* ``--no-docker`` — Skip Docker and NVIDIA Container Toolkit installation
* ``--no-gpu`` — Skip NVIDIA GPU driver and CUDA toolkit installation
* ``--dev`` — Install development dependencies (lint, pre-commit)
* ``--docs`` — Install documentation build dependencies

**Process:**

1. Installs system packages (build tools, ffmpeg, etc.)
2. Installs Python 3.12
3. Installs ``uv`` package manager
4. Creates virtual environment and installs dependencies via ``uv sync``
5. Generates gRPC/protobuf Python code
6. Installs pre-commit hooks (with ``--dev``)
7. Installs Docker and NVIDIA GPU drivers (unless skipped)
8. Creates ``.env`` template if missing

**Requirements:**

* Ubuntu 22.04 or 24.04 recommended
* Internet connection for package downloads

copy_docker_logs.sh
~~~~~~~~~~~~~~~~~~~

Copy logs from Docker containers to local files for debugging and sharing.

.. code-block:: bash

   # Copy all service logs
   ./scripts/misc/copy_docker_logs.sh

   # Copy specific service logs
   ./scripts/misc/copy_docker_logs.sh s2s
   ./scripts/misc/copy_docker_logs.sh controller

**Output:** Logs saved to ``./logs/`` directory with filenames like ``s2s.log``,
``controller.log``, etc.

Media Processing Scripts
------------------------

convert_to_streamable_mp4.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts video files to MP4 format suitable for streaming with the ``faststart`` flag.

.. code-block:: bash

   ./scripts/misc/convert_to_streamable_mp4.sh input.mp4
   # Output: input_streamable.mp4

**Features:**

* Automatically installs ffmpeg if not present
* Copies video/audio streams without re-encoding
* Adds ``faststart`` flag for progressive download
* Supports various input formats (avi, mkv, mp4, etc.)

extract_audio_from_videos.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch extract audio from all video files in a directory.

.. code-block:: bash

   # Basic usage with defaults (16kHz, mono, WAV)
   ./scripts/misc/extract_audio_from_videos.sh videos/ audio/

   # Custom parameters (44.1kHz, stereo, MP3)
   ./scripts/misc/extract_audio_from_videos.sh videos/ audio/ 44100 2 mp3

**Arguments:**

* ``input_dir`` - Directory containing video files (required)
* ``output_dir`` - Directory to save audio files (required)
* ``sample_rate`` - Sample rate in Hz (default: 16000)
* ``channels`` - Audio channels: 1=mono, 2=stereo (default: 1)
* ``format`` - Output format: wav, mp3, flac (default: wav)

**Features:**

* Processes multiple video formats (mp4, avi, mkv, mov, webm)
* Configurable sample rate and channels
* Progress tracking and error reporting
* Creates output directory if needed

Evaluation Scripts
------------------

run_evaluation.sh
~~~~~~~~~~~~~~~~~

Run the batch processing pipeline against a directory of videos. This script
activates the virtual environment, sets ``PYTHONPATH``, and forwards all
arguments to the ``client.batch_processing.app`` Python module.

.. code-block:: bash

   ./scripts/misc/run_evaluation.sh --input-dir assets/

**Features:**

* Activates the ``.venv`` virtual environment automatically
* Configures ``PYTHONPATH`` for all project source roots
* Forwards all CLI arguments to ``python -m client.batch_processing.app``

**Arguments:**

All arguments are forwarded to the batch processing module. Required:

* ``--input-dir`` - Directory containing input MP4 video files

See :ref:`batch_processing` for the full list of configuration options
and detailed usage instructions.

**Examples:**

.. code-block:: bash

   # Basic usage
   ./scripts/misc/run_evaluation.sh --input-dir assets/

   # Translate to French
   ./scripts/misc/run_evaluation.sh --input-dir assets/ --target-language fr

   # Custom output directory
   ./scripts/misc/run_evaluation.sh --input-dir /data/my_videos --output-dir outputs/eval_run_1

Diarization Scripts
-------------------

These scripts generate diarization data (speaker segmentation) from audio files,
producing JSON files that can be passed to the ASD client or Controller client
via ``--diarization-file``.

Select the matching client parser with ``--diarization-format``. ElevenLabs
Scribe output uses ``elevenlabs-scribe``, ElevenLabs Dubbing Transcript API
JSON uses ``elevenlabs-dubbing-api``, and CAMB JSON output uses ``camb``.
See :doc:`diarization_formats` for a full comparison.

elevenlabs/diarize.py
~~~~~~~~~~~~~~~~~~~~~

Generate diarization data using the ElevenLabs Speech-to-Text (Scribe) API.
Outputs native ElevenLabs STT JSON format.

.. code-block:: bash

   ELEVENLABS_API_KEY=<key> python scripts/elevenlabs/diarize.py \
       --input-file audio.wav \
       --output-file diarization.json

**Arguments:**

* ``--input-file`` - Path to audio file (WAV, MP3, etc.) (required)
* ``--output-file`` - Path to output JSON file (default: ``diarization.json``)
* ``--language-code`` - Language code (default: auto-detect)
* ``--max-speakers`` - Maximum number of speakers (default: model default)
* ``--model-id`` - Scribe model ID (default: ``scribe_v2``)
* ``--tag-audio-events`` - Tag audio events such as (laughter) or (footsteps) (flag, default off)

**Requirements:** ``ELEVENLABS_API_KEY`` environment variable.

camb/diarize.py
~~~~~~~~~~~~~~~

Generate diarization data using the Camb AI Transcription API.
Outputs native Camb AI transcription JSON with word-level timestamps.

.. code-block:: bash

   CAMB_API_KEY=<key> python scripts/camb/diarize.py \
       --input-file audio.wav \
       --output-file diarization.json

**Arguments:**

* ``--input-file`` - Path to audio file (WAV, MP3, etc.) (required)
* ``--output-file`` - Path to output JSON file (default: ``diarization.json``)
* ``--language-id`` - Camb AI numeric language ID (default: ``1`` for English)

**Requirements:** ``CAMB_API_KEY`` environment variable.

camb/audio_isolation.py
~~~~~~~~~~~~~~~~~~~~~~~

Separate foreground (speech/vocals) from background using Camb's **audio separation**
API

.. code-block:: bash

   CAMB_API_KEY=<key> python scripts/camb/audio_isolation.py \
       -i noisy_recording.wav -o voice_only.wav

   python scripts/camb/audio_isolation.py -i mix.mp3 -o fg.wav --background-output bg.wav

**Requirements:** ``CAMB_API_KEY``, supported input formats per Camb docs (AAC/FLAC/MP3/WAV).

Standalone Dubbing Scripts
--------------------------

These scripts perform end-to-end dubbing outside of the gRPC service pipeline,
using cloud dubbing APIs directly.

elevenlabs/s2s_infer.py
~~~~~~~~~~~~~~~~~~~~~~~

Invoke ElevenLabs end-to-end dubbing for local media files. Extracts audio from
video, submits a dubbing request, and downloads the translated audio.

.. code-block:: bash

   ELEVENLABS_API_KEY=<key> python scripts/elevenlabs/s2s_infer.py \
       --input-file assets/sample_audio.wav \
       --source-language-code en \
       --target-language-code es \
       --output-file output.wav \
       --source-transcript-output-file source_transcript.json \
       --target-transcript-output-file target_transcript.json \
       --transcript-format json

Optional transcript outputs support ``json``, ``srt``, and ``webvtt`` formats;
``vtt`` is accepted as an alias for ``webvtt``. JSON transcripts from this
script can be passed to clients with the ``elevenlabs-dubbing-api``
diarization format.

**Requirements:** ``ELEVENLABS_API_KEY`` environment variable and ``ffmpeg`` installed.

camb/s2s_infer.py
~~~~~~~~~~~~~~~~~

Invoke CAMB end-to-end dubbing for local or URL-based media. Submits a dubbing
request, polls for completion, and downloads the translated audio.

CAMB.AI uses **integer language IDs** (e.g. ``1`` = English, ``54`` = Spanish).
To get the full mapping, query the CambAI API or see the
`source languages <https://docs.camb.ai/api-reference/endpoint/get-source-languages>`_
and `target languages <https://docs.camb.ai/api-reference/endpoint/get-target-languages>`_
docs.

.. code-block:: bash

   CAMB_API_KEY=<key> python scripts/camb/s2s_infer.py \
       --input-file assets/sample_audio.wav \
       --source-language 1 \
       --target-language 54 \
       --output-file output.mp3 \
       --target-transcript-output-file target_transcript.json \
       --transcript-format json

Optional transcript outputs support ``json``, ``txt``, ``srt``, and ``vtt``,
and are always in the **target** language: CAMB's dubbing API does not
expose a source-language transcript. JSON writes the target-language
diarized transcript from CAMB's ``dub-result`` payload and can be passed
to clients with ``--diarization-format camb`` only when target-language
diarization is acceptable. Use ``scripts/camb/diarize.py`` (Camb AI
Transcription API) when source-language diarization is required (for
example, as ASD input).

**Requirements:** ``CAMB_API_KEY`` environment variable.

elevenlabs/invoke_e2e.sh / camb/invoke_e2e.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convenience wrappers around ``scripts/elevenlabs/s2s_infer.py`` and
``scripts/camb/s2s_infer.py`` with example parameters. Edit the scripts to
change input files and language codes.

.. code-block:: bash

   ./scripts/elevenlabs/invoke_e2e.sh
   ./scripts/camb/invoke_e2e.sh

Script Best Practices
---------------------

**For Deployment Scripts:**

* Each script runs in interactive mode and will occupy your terminal
* Run each script in a separate terminal or stop (Ctrl+C) before running the next
* These scripts are for **verification only** - use docker compose for production deployments

**For Development:**

* Always check prerequisites before running scripts
* Review script output for errors or warnings
* Keep environment variables properly configured in ``.env`` file
