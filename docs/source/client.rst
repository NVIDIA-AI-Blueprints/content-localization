.. _client:

Client Package Documentation
============================

The client package provides a comprehensive implementation for interacting with the Content Localization services. It includes multiple client applications for different use cases, along with modular components for audio/video processing, request generation, response handling, and latency analysis.

Overview
--------

The client package is organized into several key components:

- **Client Applications**: Specialized clients for different workflows
- **Source Simulators**: Audio and video input/output simulators
- **Utilities**: Helper functions for file handling, service health checks, and more
- **Analysis Tools**: Latency calculation and visualization
- **Configuration**: Command-line argument parsing per client

Available Clients
-----------------
1. Controller Client (Recommended for production workflows)
2. Direct Client (Full control over individual service communication)
3. Individual Clients (S2S, ASD, LipSync).

Client Selection Guide
----------------------

Choose the right client for your use case:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Use Case
     - Recommended Client
     - Description
   * - **Complete Pipeline (Simplified)**
     - ``controller/``
     - Single service orchestration with minimal configuration; ideal for production workflows.
   * - **Complete Pipeline (Full Control)**
     - ``direct/``
     - Direct service communication with complete pipeline control for detailed monitoring.
   * - **Audio Translation Only**
     - ``s2s/``
     - Speech-to-speech translation with performance analysis and latency monitoring.
   * - **Lip Sync Only**
     - ``lipsync/``
     - Lip synchronization with advanced encoding options and speaker info support.
   * - **Speaker Detection Only**
     - ``asd/``
     - Active speaker detection and speaker info generation.
   * - **Production Testing**
     - ``controller/``
     - Orchestrated end-to-end workflow for production use.
   * - **Development/Debugging**
     - ``direct/``
     - Detailed control and monitoring for development.

Prerequisites
-------------

Before running any client:

1. **Services Running**: Ensure required services are running via Docker Compose

   - Controller Service (port 50056) - for controller client
   - S2S Service (port 50050) - for direct/S2S clients
   - LipSync Service (port 50054) - for direct/LipSync clients
   - ASD NIM Service (port 50055) - for direct/ASD clients

2. **Virtual Environment**: Activate the project virtual environment

   .. code-block:: bash

       source .venv/bin/activate

3. **Input Files**: Prepare required input files

   - Audio files in WAV format (MP3 for output)
   - Video files in MP4 format (streamable MP4 recommended)
   - Speaker info files in CSV format (optional for LipSync)

4. **Convert Videos**: If needed, convert videos to streamable format

   .. code-block:: bash

       scripts/misc/convert_to_streamable_mp4.sh input.mp4

Starting Services
~~~~~~~~~~~~~~~~~

**For Controller Client**:

.. code-block:: bash

    docker compose --profile controller-third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build

**For Direct Client**:

.. code-block:: bash

    docker compose --profile third-party-s2s-asd-lipsync --env-file configs/elevenlabs.env --env-file .env up --build

**For Individual Services**:

.. code-block:: bash

    # S2S service only
    docker compose --profile third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build

    # LipSync service only
    docker compose --profile lipsync --env-file configs/elevenlabs.env --env-file .env up --build

    # ASD NIM service only
    docker compose --profile asd --env-file configs/elevenlabs.env --env-file .env up --build

Configuration Options
---------------------

.. note::

   The lists below are a curated summary. For the complete, always-current set
   of options for every client (generated directly from the ``argsfactory``
   argument parsers), see :doc:`cli_reference`.

Controller Client Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--controller-server``: Controller service endpoint (default: localhost:50056)
- ``--input-audio``: Input audio file path (default: assets/sample_audio.wav)
- ``--input-mp4``: Input MP4 video file path (default: assets/sample_video_streamable.mp4)
- ``--chunk-size-audio-secs``: Audio chunk duration in seconds (default: 1.0)
- ``--chunk-size-video-bytes``: Video chunk size in bytes (default: 1 MB)
- ``--output-mp4``: Output MP4 video file path (default: outputs/controller_output.mp4)
- ``--source-language``: Source language code (default: en)
- ``--target-language``: Target language code (default: de)
- ``--voice-name``: Voice name for TTS (optional)
- ``--diarization-file``: Path to JSON diarization file for speaker segments (optional)
- ``--diarization-format``: Format of the diarization file (choices: flat, elevenlabs-scribe, elevenlabs-dubbing-api, elevenlabs-studio, camb; default: elevenlabs-scribe). See :doc:`diarization_formats` for a comparison.
- ``--bypass-asd``: Bypass ASD NIM (Active Speaker Detection); LipSync uses internal face detection
- ``--background-audio-input``: Path to background audio file for mixing (optional)
- ``--translated-audio``: Path to pre-translated audio file to bypass S2S (optional)
- ``--diarization-rows-per-chunk``: Number of diarization segment rows per chunk. Use -1 to send all segments in one message (default: 10)

.. note::

   The Controller client also accepts all S2S (``--elevenlabs-*``, ``--camb-*``),
   ASD (``--asd-*``), and LipSync (``--lipsync-*``) config arguments.
   Run ``python client/controller/app.py --help`` for the full list.

Direct Client Arguments
~~~~~~~~~~~~~~~~~~~~~~~

- ``--s2s-server``: S2S service endpoint (default: localhost:50050)
- ``--lipsync-server``: LipSync service endpoint (default: localhost:50054)
- ``--asd-server``: ASD NIM service endpoint (default: localhost:50055)
- ``--input-audio``: Input audio file path (default: assets/sample_audio.wav)
- ``--output-audio``: Output audio file path (default: outputs/sample_audio_output.mp3)
- ``--translated-audio``: Path to pre-translated audio file to bypass S2S (optional)
- ``--input-mp4``: Input MP4 video file path (default: assets/sample_video_streamable.mp4)
- ``--output-mp4``: Output MP4 video file path (default: outputs/direct_output.mp4)
- ``--chunk-size-audio-secs``: Audio chunk duration in seconds (default: 1.0)
- ``--chunk-size-video-bytes``: Video chunk size in bytes (default: 1 MB)
- ``--bypass-asd``: Bypass ASD NIM (Active Speaker Detection); LipSync uses internal face detection
- ``--diarization-file``: Path to JSON diarization file for speaker segments (optional)
- ``--diarization-format``: Format of the diarization file (choices: flat, elevenlabs-scribe, elevenlabs-dubbing-api, elevenlabs-studio, camb; default: elevenlabs-scribe). See :doc:`diarization_formats` for a comparison.
- ``--background-audio-input``: Path to background audio file for mixing (optional)

.. note::

   The Direct client also accepts all S2S (``--source-language``, ``--target-language``,
   ``--voice-name``, ``--elevenlabs-*``, ``--camb-*``), ASD (``--asd-*``), and LipSync
   (``--lipsync-*``) config arguments.
   Run ``python client/direct/app.py --help`` for the full list.

S2S Client Arguments
~~~~~~~~~~~~~~~~~~~~

- ``--s2s-server``: S2S service endpoint (default: localhost:50050)
- ``--input-audio``: Input audio file path (default: assets/sample_audio.wav)
- ``--output-audio``: Output audio file path, WAV or MP3 (default: outputs/sample_audio_output.mp3)
- ``--chunk-size-audio-secs``: Audio chunk duration in seconds (default: 1.0)
- ``--latency-plot``: Path for latency analysis plot (default: outputs/latency.png)
- ``--source-language``: Source language code (default: en)
- ``--target-language``: Target language code (default: de)
- ``--voice-name``: Voice name for TTS (optional, auto-extracted for zero-shot TTS)
- ``--elevenlabs-num-speakers``: Number of speakers for ElevenLabs dubbing, 0 = auto-detect (default: 0)
- ``--elevenlabs-drop-background-audio``: Drop background audio from the final dub (flag)
- ``--elevenlabs-use-profanity-filter``: Censor profanities in transcripts (flag, beta)
- ``--elevenlabs-target-accent``: Accent for voice selection (optional, experimental)
- ``--elevenlabs-highest-resolution``: Use highest resolution available (flag)
- ``--elevenlabs-watermark``: Apply watermark to output (flag)
- ``--elevenlabs-dubbing-studio``: Prepare dub for edits in dubbing studio (flag)

ASD Client Arguments
~~~~~~~~~~~~~~~~~~~~

- ``--asd-server``: ASD NIM service endpoint (default: localhost:50055)
- ``--input-mp4``: Input MP4 video file path (default: assets/sample_video_streamable.mp4)
- ``--input-audio``: Input audio file path (default: assets/sample_audio.wav)
- ``--chunk-size-video-bytes``: Video chunk size in bytes (default: 1 MB)
- ``--chunk-size-audio-secs``: Audio chunk duration in seconds (default: 1.0)
- ``--output-speaker-info``: Output speaker info CSV file path (default: assets/asd_speaker_info.csv)
- ``--diarization-file``: Path to JSON diarization file for speaker segments (optional)
- ``--diarization-format``: Format of the diarization file (choices: flat, elevenlabs-scribe, elevenlabs-dubbing-api, elevenlabs-studio, camb; default: elevenlabs-scribe). See :doc:`diarization_formats` for a comparison.
- ``--asd-input-audio-codec``: Audio codec for ASD input (choices: WAV, MP3; default: WAV)
- ``--asd-input-video-codec``: Video codec for ASD input (choices: H264; optional)
- ``--asd-audio-source-config``: Audio source mode (choices: unspecified, separate_stream, embedded_in_video; default: unspecified)
- ``--asd-speaker-detection-threshold``: Confidence threshold for speaker detection (0.0-1.0; default: 0.5986)

LipSync Client Arguments
~~~~~~~~~~~~~~~~~~~~~~~~

- ``--lipsync-server``: gRPC service endpoint (default: 127.0.0.1:50054; ``--target`` is a deprecated alias)
- ``--input-mp4``: Input video file path (MP4 format; ``--video-input`` is a deprecated alias)
- ``--input-audio``: Input audio file path (``--audio-input`` is a deprecated alias)
- ``--speaker-info-input``: Speaker info CSV file path (optional)
- ``--background-audio-input``: Background audio file (WAV or MP3) for mixing (optional)
- ``--output-mp4``: Output video file path (default: outputs/lipsync_output.mp4; ``--output`` is a deprecated alias)
- ``--lipsync-input-audio-codec``: Audio codec for input (choices: WAV, MP3; default: MP3)
- ``--lipsync-extend-audio``: Audio extension handling (choices: unspecified, silence; default: unspecified)
- ``--lipsync-extend-video``: Video extension handling (choices: unspecified, forward, reverse; default: unspecified)
- ``--lipsync-output-bitrate-mbps``: Output video bitrate in Mbps (default: 20)
- ``--lipsync-output-idr-interval``: IDR keyframe interval (default: 8)
- ``--lipsync-head-movement-speed``: Head movement speed: 0 for static/slow, 1 for fast (optional)
- ``--lipsync-output-audio-codec``: Output audio codec (choices: WAV, MP3; optional)
- ``--lipsync-is-speaker-info-provided``: Flag indicating speaker bounding boxes are provided (from ASD)
- ``--lipsync-background-audio-volume``: Background audio volume 0.0-1.0 (default: 1.0)
- ``--lipsync-background-audio-codec``: Background audio codec override (choices: WAV, MP3; optional)
- ``--lipsync-lossless``: Enable lossless video encoding (flag)
- ``--lipsync-custom-encoding-params``: Custom encoding parameters in JSON format (optional)
- ``--ssl-mode``: SSL mode for secure communication (choices: DISABLED, MTLS, TLS; default: DISABLED)
- ``--ssl-key``: Path to SSL private key
- ``--ssl-cert``: Path to SSL certificate chain
- ``--ssl-root-cert``: Path to SSL root certificate

Performance Considerations
--------------------------

Chunk Sizes
~~~~~~~~~~~

- **Audio chunks**: 0.5-2 seconds recommended (default: 1.0 second)
  
  - Smaller chunks (0.5s): Lower latency, higher overhead
  - Larger chunks (2s): Higher throughput, increased latency
  
- **Video chunks**: 32KB-2MB recommended (default: 1 MB)
  
  - Smaller chunks (32KB-64KB): Better for low-bandwidth scenarios
  - Larger chunks (1-2MB): Better throughput for high-bandwidth scenarios

Streaming Mode
~~~~~~~~~~~~~~

The LipSync NIM uses bidirectional streaming gRPC to send video
and audio bytes. Using a streamable MP4 (moov atom at the start)
is recommended for best performance but not required:

- Streamable MP4 reduces memory usage and enables earlier inference
- Convert to streamable format: ``scripts/misc/convert_to_streamable_mp4.sh``
- Non-streamable MP4 files are also accepted

Latency Optimization
~~~~~~~~~~~~~~~~~~~~

- Monitor latency plots (S2S and Direct clients) to identify bottlenecks
- Adjust chunk sizes based on network conditions and use case
- Use appropriate video encoding settings (bitrate, lossless)
- Consider using speaker info data to focus processing on relevant areas
- For lowest latency: Use Controller client with optimized chunk sizes

Quick Start
-----------

Setup
~~~~~

Clone the repository, install dependencies, and point your ``PYTHONPATH`` to the project root, ``src``, ``client``, and generated protos.

.. code-block:: bash

   git clone <repository-url>
   cd repo-dir
   uv sync
   source .venv/bin/activate
   export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated:${PYTHONPATH}"

Basic Usage
~~~~~~~~~~~

Controller Client
^^^^^^^^^^^^^^^^^

The Controller client provides a streamlined single-entry-point pipeline:

- **Purpose**: Complete content localization through orchestrated services
- **Services**: Controller service (which orchestrates S2S, ASD NIM, and LipSync)
- **Use Case**: Production workflows with simplified service management

.. code-block:: bash

   # Default settings
   python client/controller/app.py

.. code-block:: bash

   # Custom input/output files and languages
   python client/controller/app.py \
       --input-audio assets/audio.wav \
       --input-mp4 assets/video.mp4 \
       --output-mp4 outputs/video_output.mp4 \
       --source-language en \
       --target-language de

Direct Client
^^^^^^^^^^^^^

The Direct client provides a direct interface to the individual services:

- **Purpose**: Full control over the end-to-end pipeline
- **Services**: Direct communication with S2S, ASD NIM, and LipSync services
- **Use Case**: Development, testing, and detailed performance monitoring

.. code-block:: bash

   # Default settings
   python client/direct/app.py

.. code-block:: bash

   # Custom input/output files and languages
   python client/direct/app.py \
       --input-audio assets/audio.wav \
       --output-audio outputs/audio_output.mp3 \
       --input-mp4 assets/video.mp4 \
       --output-mp4 outputs/video_output.mp4 \
       --source-language en \
       --target-language de

S2S Processing
--------------

The S2S client provides focused functionality for speech-to-speech processing:

- **Purpose**: Audio translation and synthesis
- **Services**: S2S service (which handles audio translation and synthesis)
- **Use Case**: Development, testing, and detailed performance monitoring

1. **Start with default settings**:

   .. code-block:: bash

       # Defaults
       python client/s2s/app.py

2. **Use custom input files**:

   .. code-block:: bash

       python client/s2s/app.py \
           --input-audio assets/audio.wav \
           --output-audio outputs/audio_output.wav

LipSync Processing
------------------

For streaming LipSync, point to the service, provide video and audio, and optionally include
speaker info data for focused processing.

- **Purpose**: Lip synchronization
- **Services**: LipSync service (which handles lip synchronization)
- **Use Case**: Development, testing, and detailed performance monitoring

.. code-block:: bash

   # Basic
   python client/lipsync/app.py \
       --lipsync-server 127.0.0.1:50054 \
       --input-mp4 input.mp4 \
       --input-audio input.wav \
       --output-mp4 output.mp4

.. code-block:: bash

   # With speaker info from ASD
   python client/lipsync/app.py \
       --lipsync-server 127.0.0.1:50054 \
       --input-mp4 input.mp4 \
       --input-audio input.wav \
       --speaker-info-input speaker_info.csv \
       --output-mp4 output.mp4

Client Types
------------

For a detailed guide to each client type (Controller, Direct,
and Individual), see :doc:`client_types`.

Troubleshooting
---------------

Service Connection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

Cannot Connect to a Service
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**:
- Connection refused errors
- gRPC timeout errors
- Service health check failures

**Solutions**:

1. **Check if service is running**:

   .. code-block:: bash

       # Check if port is listening
       netstat -tuln | grep 50050
       
       # Or use ss command
       ss -tuln | grep 50050

2. **Verify service endpoint**:

   .. code-block:: python

       from common.health import check_service_health

       # Test connection
       is_healthy = check_service_health("localhost:50050")
       print(f"S2S service healthy: {is_healthy}")

3. **Verify Docker container status**:

   .. code-block:: bash

       # Check if containers are running
       docker ps | grep s2s
       
       # Check container logs
       docker logs <container_name>

File Format Issues
~~~~~~~~~~~~~~~~~~

Unsupported Audio Format
~~~~~~~~~~~~~~~~~~~~~~~~

- Only WAV and MP3 audio formats are supported
- ElevenLabs produces MP3 output by default.
- File validation failures

**Solutions**:

1. **Convert to WAV format**:

   .. code-block:: bash

       # Convert MP3 to WAV
       ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 -ac 1 output.wav
       
       # Convert other formats
       ffmpeg -i input.m4a -acodec pcm_s16le -ar 16000 -ac 1 output.wav

2. **Verify WAV parameters**:
   - Sample rate: 16000 Hz (recommended)
   - Bit depth: 16-bit
   - Channels: 1 (mono) or 2 (stereo)
   - Format: PCM

Unsupported Video Format
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**:
- "Only MP4 video format is supported" error
- Video processing failures

**Solutions**:

1. **Convert to MP4 format**:

   .. code-block:: bash

       # Convert to MP4
       ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
       
       # Convert to streamable MP4
       ffmpeg -i input.mp4 -movflags +faststart output_streamable.mp4

2. **Check video properties**:

   .. code-block:: bash

       # Check video properties
       ffprobe input.mp4
       
       # Check if video is streamable
       ffprobe -v quiet -print_format json -show_format input.mp4

3. **Verify MP4 requirements**:
   - Container: MP4
   - Codec: H.264 (recommended)
   - Audio: AAC or MP3
   - Streamable: moov atom at start

Non-Streamable Video
~~~~~~~~~~~~~~~~~~~~

**Symptoms**:

- ``Video streamable: False`` in client DEBUG logs (no error is raised)
- Higher memory usage and delayed inference start; non-streamable MP4
  files are still accepted

**Solutions**:

1. **Convert to streamable format**:

   .. code-block:: bash

       # Make video streamable using helper script
       ./scripts/misc/convert_to_streamable_mp4.sh input.mp4

2. **Check streamability**:

   .. code-block:: python

       from common.media import check_streamable

       is_streamable = check_streamable("input.mp4")
       print(f"Video is streamable: {is_streamable}")

Performance Issues
~~~~~~~~~~~~~~~~~~

High Latency
~~~~~~~~~~~~

**Symptoms**:
- Slow processing
- High latency values in analysis
- Not real-time performance

**Solutions**:

1. **Reduce chunk sizes**:

    .. code-block:: bash

        # Use smaller audio chunks
        python client/controller/app.py --chunk-size-audio-secs 0.05
        
        # Use smaller video chunks
        python client/controller/app.py --chunk-size-video-bytes 32768

2. **Check network conditions**:
   - Ensure low network latency
   - Use local services when possible
   - Check bandwidth availability

3. **Optimize video encoding**:

   .. code-block:: bash

        # Use lower bitrate for faster processing
        python client/lipsync/app.py --lipsync-output-bitrate-mbps 2

Memory Issues
~~~~~~~~~~~~~

**Symptoms**:
- Out of memory errors
- High memory usage
- System slowdown

**Solutions**:

1. **Reduce chunk sizes**:

    .. code-block:: bash

        # Smaller chunks use less memory
        python client/controller/app.py --chunk-size-video-bytes 16384

2. **Process in batches**:
   - Split large files into smaller segments
   - Process files sequentially

File I/O Issues
~~~~~~~~~~~~~~~

Permission Denied
~~~~~~~~~~~~~~~~~

**Symptoms**:
- "Permission denied" errors
- Cannot write output files

**Solutions**:

1. **Check file permissions**:

    .. code-block:: bash

        # Check directory permissions
        ls -la outputs/
        
        # Fix permissions
        chmod 755 outputs/
        chmod 644 outputs/*.wav

2. **Create output directories**:

    .. code-block:: bash

        # Create output directory
        mkdir -p outputs/
        
        # Ensure write permissions
        chmod 755 outputs/

3. **Check disk space**:

    .. code-block:: bash

        # Check available disk space
        df -h
        
        # Check directory size
        du -sh outputs/

Speaker Info File Issues
~~~~~~~~~~~~~~~~~~~~~~~~

Invalid Speaker Info Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**:
- Speaker info parsing errors
- Incorrect bounding box coordinates

**Solutions**:

1. **Check speaker info file format**:

    .. code-block:: bash

        # Check CSV format
        head -5 speaker_info.csv
        
        # Verify column structure. Expected header:
        # frame_id,x,y,width,height,diarized_speaker_id,face_id,is_speaking,face_detection_confidence

2. **Validate speaker info coordinates**:
   - Ensure coordinates are within video dimensions
   - Check for negative values
   - Verify coordinate format (pixels)

3. **Create sample speaker info file**:

    .. code-block:: python

        import csv
        
        # Create sample speaker info file
        with open('sample_speaker_info.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'x', 'y', 'width', 'height',
                'diarized_speaker_id', 'face_id', 'is_speaking',
                'face_detection_confidence',
            ])
            # Add one row per video frame
            for frame in range(100):
                writer.writerow([frame, 100, 100, 200, 200, 0, 0, True, 0.99])

SSL/TLS Issues
~~~~~~~~~~~~~~

SSL Certificate Errors
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**:
- SSL certificate validation failures
- TLS handshake errors

**Solutions**:

1. **Use insecure mode for testing**:

    .. code-block:: bash

        # Disable SSL for local testing
        python client/lipsync/app.py --ssl-mode DISABLED

2. **Check certificate paths**:

    .. code-block:: bash

        # Verify certificate files exist
        ls -la certs/

        # Check certificate validity
        openssl x509 -in certs/client.pem -text -noout

3. **Generate self-signed certificates** (for testing):

    .. code-block:: bash

        # Generates a dev CA plus per-service server certs and an mTLS
        # client cert into ./certs (development/testing only)
        bash scripts/misc/generate_dev_certs.sh
