# Direct Client

This client provides a direct end-to-end pipeline implementation that orchestrates multiple services without using the Controller service.

## Overview

The Direct client processes both audio and video input and coordinates directly with:
- **S2S (Speech-to-Speech)** - Audio translation service
- **ASD NIM (Active Speaker Detection)** - Speaker detection and speaker info generation service
- **LipSync** - Lip synchronization service

The client manages the complete pipeline flow: S2S → LipSync → ASD, outputting a processed video file with lip-synchronized speech in the target language.

## Architecture

Unlike the Controller client, the Direct client:
- **Direct Service Communication**: Communicates directly with each service (S2S, ASD NIM, LipSync)
- **Pipeline Orchestration**: Manages the flow between services without a central orchestrator
- **Streaming Coordination**: Handles streaming between services with proper buffering
- **Error Management**: Provides comprehensive error handling across all services

## Usage

### Basic Usage

```bash
python client/direct/app.py
```

This will use the default settings:
- Input audio: `assets/sample_audio.wav`
- Input video: `assets/sample_video_streamable.mp4`
- S2S server: `localhost:50050`
- LipSync server: `localhost:50054`
- ASD server: `localhost:50055`
- Output video: `outputs/direct_output.mp4`

### Custom Parameters

```bash
python client/direct/app.py \
    --s2s-server localhost:50050 \
    --lipsync-server localhost:50054 \
    --asd-server localhost:50055 \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/direct_output.mp4 \
    --chunk-size-audio-secs 0.5 \
    --chunk-size-video-bytes 32768
```

### S2S Bypass (Pre-Translated Audio)

```bash
python client/direct/app.py \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/my_direct_pipeline_video.mp4 \
    --translated-audio assets/es.wav
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--s2s-server` | `localhost:50050` | Address and port of the S2S gRPC service |
| `--lipsync-server` | `localhost:50054` | Address and port of the LipSync gRPC service |
| `--asd-server` | `localhost:50055` | Address and port of the ASD NIM gRPC service |
| `--input-audio` | `assets/sample_audio.wav` | Path to input audio file (WAV format) |
| `--input-mp4` | `assets/sample_video_streamable.mp4` | Path to input video file (MP4; streamable MP4 recommended) |
| `--chunk-size-audio-secs` | `1` | Audio chunk size for streaming in seconds |
| `--chunk-size-video-bytes` | `1048576` (1 MB) | Video chunk size for streaming in bytes |
| `--output-mp4` | `outputs/direct_output.mp4` | Path to output video file (MP4 format) |
| `--bypass-asd` | `False` | Bypass ASD NIM and use LipSync's internal face detection |
| `--translated-audio` | `None` | Path to pre-translated audio file (WAV or MP3). Bypasses S2S and sends audio directly to LipSync |

## Pipeline Flow

The Direct client implements the following pipeline:

```
Input Audio/Video
  ├→ S2S Service → Translated Audio ──→ LipSync Service → Output Video
  └→ ASD NIM Service → Speaker Info ──↗
```

With `--translated-audio`, the S2S step is bypassed:

```
Input Video + Translated Audio ──→ LipSync Service → Output Video
ASD NIM Service → Speaker Info ──↗
```

### Step-by-Step Process

1. **Input Processing**: Reads audio and video files in chunks
2. **S2S Translation**: Sends audio to S2S service for translation (skipped with `--translated-audio`)
3. **LipSync Processing**: Sends translated audio + video to LipSync service
4. **ASD Processing**: Sends video to ASD NIM for speaker info generation
5. **Output Generation**: Receives processed video and writes to output file

## Key Features

### Direct Service Coordination
- **No Central Orchestrator**: Communicates directly with each service
- **Streaming Management**: Handles streaming between services with buffering
- **Error Propagation**: Manages errors across the entire pipeline
- **Service Independence**: Each service can be configured independently

### Performance Optimization
- **Parallel Processing**: Services can process data concurrently where possible
- **Buffering**: Implements buffering between service stages
- **Chunk-based Streaming**: Configurable chunk sizes for optimal performance
- **Memory Management**: Efficient memory usage for large files

### Error Handling
- **Service Failures**: Graceful handling of individual service failures
- **Network Issues**: Robust network error recovery
- **Data Validation**: Input/output data validation
- **Logging**: Comprehensive logging for debugging

## Requirements

- Python 3.12+
- gRPC
- All three services (S2S, LipSync, ASD NIM) must be running and accessible (S2S is optional when using `--translated-audio`)
- Input audio must be in WAV format
- Input video must be in streamable MP4 format

## Service Dependencies

The Direct client requires all three services to be running:

1. **S2S Service** (default: `localhost:50050`)
   - Handles audio translation
   - Must be accessible and healthy
   - Optional when using `--translated-audio` (S2S bypass)

2. **LipSync Service** (default: `localhost:50054`)
   - Handles lip synchronization
   - Requires translated audio and video input

3. **ASD NIM Service** (default: `localhost:50055`)
   - Handles speaker detection and speaker info generation
   - Optional (can be bypassed with `--bypass-asd`)

## Error Handling

The client includes comprehensive error handling for:
- gRPC connection issues with any service
- Service unavailability or health issues
- File I/O errors
- Invalid input formats
- Pipeline coordination errors
- Memory and resource issues

## Integration

This client is designed for:
- **Direct Pipeline Control**: Full control over the service pipeline
- **Custom Orchestration**: Custom logic for service coordination
- **Performance Testing**: Direct measurement of service performance
- **Debugging**: Detailed visibility into each service interaction

## Troubleshooting

### Common Issues

1. **Service not found**: Ensure all required services are running on specified ports
2. **Pipeline coordination errors**: Check service health and network connectivity
3. **Video format issues**: Convert your video to streamable MP4 format
4. **Memory issues**: Try reducing chunk sizes if processing large files
5. **Network timeouts**: Check network connectivity between services

### Service-Specific Issues

1. **S2S service errors**: Check audio format and S2S service logs
2. **LipSync service errors**: Verify video format and LipSync service configuration
3. **ASD NIM errors**: Check video format and ASD NIM logs

### Keep-Alive Handling

The direct client properly handles keep-alive responses from the S2S service:
- Keep-alive responses are automatically filtered out when creating audio iterators for LipSync
- This prevents `GeneratorExit` errors that can occur when keep-alive responses are processed as audio data
- The client logs keep-alive responses for debugging: `lipsync | received keep-alive from S2S, skipping`

If you encounter issues with keep-alive handling, ensure the client is using the latest version with proper keep-alive filtering.

## Example Workflow

```bash
# 1. Start all required services
docker compose --profile third-party-s2s-asd-lipsync --env-file configs/elevenlabs.env --env-file .env up --build

# 2. Run the direct client
python client/direct/app.py \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/my_direct_pipeline_video.mp4

# 3. Check the output
ls -la outputs/my_direct_pipeline_video.mp4
```
The Direct client provides complete control over the content localization pipeline, allowing for custom orchestration and detailed performance monitoring.
