.. _client_troubleshooting:

Client Troubleshooting Guide
============================

This guide helps you resolve common issues when using the client package.

Service Connection Issues
-------------------------

Cannot Connect to a Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
------------------

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
------------------

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
---------------

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
------------------------

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
--------------

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

Python Environment Issues
-------------------------

Import Errors
~~~~~~~~~~~~~

**Symptoms**:
- Module not found errors
- Import failures

**Solutions**:

1. **Set Python path**:

    .. code-block:: bash

        # Set PYTHONPATH
        export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated:${PYTHONPATH}"

        # Or add to .bashrc (replace /path/to/content-localization with your repo path)
        echo 'export PYTHONPATH="/home/user/content-localization:/home/user/content-localization/src:/home/user/content-localization/client:/home/user/content-localization/protos/generated:${PYTHONPATH}"' >> ~/.bashrc

2. **Install dependencies**:

    .. code-block:: bash

        # Install project dependencies
        uv sync --extra test --extra lint --extra docs

3. **Check Python version**:

    .. code-block:: bash

        # Verify Python version
        python --version
        
        # Should be 3.12 or later

4. **Use virtual environment**:

    .. code-block:: bash

        # Create virtual environment
        uv venv --python 3.12
        source .venv/bin/activate

        # Install dependencies
        uv sync --extra test --extra lint --extra docs

gRPC Issues
-----------

gRPC Version Conflicts
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**:
- gRPC import errors
- Version compatibility issues

**Solutions**:

1. **Regenerate protobuf files**:

    .. code-block:: bash

        # Regenerate all protobuf files with the canonical script
        bash protos/generate_protos.sh

2. **Check protobuf version**:

    .. code-block:: bash

        # Check protobuf version
        pip show protobuf
        
        # Update if needed
        pip install --upgrade protobuf

Debugging Tips
--------------

Enable Verbose Logging
~~~~~~~~~~~~~~~~~~~~~~
Change logging level to DEBUG in all services in the docker compose file.

Common Error Messages
---------------------

**Connection refused**: Service not running - Start services, check ports

**File not found**: Missing input file - Verify file path and existence

**Permission denied**: File/directory permissions - Fix permissions or create directories

**Unsupported format**: Wrong file format - Convert to supported format

**Video streamable: False**: DEBUG log, not an error - Convert with movflags +faststart
for better performance

**gRPC timeout**: Network/service issues - Check network, increase timeout

**Out of memory**: Large files/chunks - Reduce chunk sizes, use streaming

Getting Help
------------

If you're still experiencing issues:

1. **Check the logs**: Look at service and client logs for detailed error messages
2. **Verify setup**: Ensure all prerequisites are met

3. **Test with sample files**: Use the provided sample assets
4. **Check documentation**: Review the main client documentation

For more detailed information, see the main :doc:`client` documentation.
