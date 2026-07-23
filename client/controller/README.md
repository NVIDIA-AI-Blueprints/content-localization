# Controller Client

This client provides a standalone interface for the Controller service, which orchestrates the complete content localization pipeline.

## Overview

The Controller client processes both audio and video input and sends them to the Controller service, which coordinates:
- **S2S (Speech-to-Speech)** - Audio translation
- **ASD NIM (Active Speaker Detection)** - Speaker detection and speaker info generation  
- **LipSync** - Lip synchronization with translated audio

The client outputs a processed video file with lip-synchronized speech in the target language.

## Usage

### Basic Usage

```bash
python client/controller/app.py
```

This will use the default settings:
- Input audio: `assets/sample_audio.wav`
- Input video: `assets/sample_video_streamable.mp4`
- Controller server: `localhost:50056`
- Output video: `outputs/controller_output.mp4`
- Audio chunk size: 1 second
- Video chunk size: 1 MB

### Custom Parameters

```bash
python client/controller/app.py \
    --controller-server localhost:50056 \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/controller_output.mp4 \
    --chunk-size-audio-secs 0.5 \
    --chunk-size-video-bytes 32768
```

### S2S Bypass (Pre-Translated Audio)

```bash
python client/controller/app.py \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/my_translated_video.mp4 \
    --translated-audio assets/es.wav
```

### ASD Bypass (Skip Active Speaker Detection)

When ASD is bypassed, the controller skips the Active Speaker Detection service
and LipSync uses its internal face detection instead. This is useful when you
don't have diarization data or don't need speaker-aware lip sync.

ASD bypass is **auto-enabled** when no `--diarization-file` is provided.
To explicitly enable it:

```bash
python client/controller/app.py \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/my_translated_video.mp4 \
    --bypass-asd
```

When `--bypass-asd` is active:
- The ASD NIM service does not need to be running
- LipSync uses internal face detection (no speaker bounding boxes)
- `--lipsync-is-speaker-info-provided` is forced to `False`
- Diarization-related arguments are ignored

### Using Pre-Computed ASD Output with LipSync

If you want speaker-aware lip sync without running ASD inside the controller
pipeline, you can run ASD as a standalone step first, then feed its output
into the LipSync client directly:

```bash
# 1. Run ASD standalone to produce speaker info CSV
python client/asd/app.py \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/sample_audio.wav \
    --output-speaker-info assets/asd_speaker_info.csv

# 2. Run LipSync standalone with the pre-computed speaker info
python client/lipsync/app.py \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/translated_audio.mp3 \
    --speaker-info-input assets/asd_speaker_info.csv \
    --lipsync-is-speaker-info-provided \
    --lipsync-input-audio-codec MP3 \
    --output-mp4 outputs/lipsync_output.mp4
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--controller-server` | `localhost:50056` | Address and port of the controller gRPC service |
| `--request-id` | generated UUID4 | Correlation id stamped on every request message and echoed by the controller in its responses |
| `--input-audio` | `assets/sample_audio.wav` | Path to input audio file (WAV format) |
| `--input-mp4` | `assets/sample_video_streamable.mp4` | Path to input video file (MP4; streamable MP4 recommended) |
| `--chunk-size-audio-secs` | `1` | Audio chunk size for streaming in seconds |
| `--chunk-size-video-bytes` | `1048576` (1 MB) | Video chunk size for streaming in bytes |
| `--output-mp4` | `outputs/controller_output.mp4` | Path to output video file (MP4 format) |
| `--bypass-asd` | `False` | Bypass ASD NIM (Active Speaker Detection); LipSync uses internal face detection |
| `--translated-audio` | `None` | Path to pre-translated audio file (WAV or MP3). Bypasses S2S and sends audio directly to LipSync |
| `--diarization-rows-per-chunk` | `10` | Number of diarization segment rows per chunk. Use -1 to send all segments in one message |

## How It Works

1. **Input Processing**: The client reads audio and video files in chunks
2. **Request Generation**: Creates `ContentLocalizationRequest` objects containing both audio and video data
3. **Service Communication**: Streams requests to the Controller service via gRPC
4. **S2S Bypass** (optional): When `--translated-audio` is provided, the pre-translated audio is sent directly to LipSync, skipping S2S entirely
5. **ASD Bypass** (optional): When `--bypass-asd` is set (or no `--diarization-file` is provided), ASD is skipped and LipSync uses internal face detection
6. **Pipeline Orchestration**: The Controller service coordinates:
   - S2S service for audio translation (skipped when using `--translated-audio`)
   - ASD NIM for speaker detection and speaker info generation (skipped when `--bypass-asd`)
   - LipSync service for lip synchronization
7. **Output Generation**: Receives processed video chunks and writes them to the output file

## Requirements

- Python 3.12+
- gRPC
- The Controller service must be running and accessible
- Input audio must be in WAV format
- Input video must be in streamable MP4 format
- Translated audio (when using `--translated-audio`) can be WAV or MP3 format

## Error Handling

The client includes proper error handling for:
- gRPC connection issues
- File I/O errors
- Service unavailability
- Invalid input formats

## Integration

This client is designed to work with the complete content localization pipeline. It provides a single entry point for:
- Audio translation between languages
- Speaker detection and speaker info generation
- Lip synchronization with translated audio

The output video contains the original video with lip movements synchronized to the translated audio.

## Troubleshooting

1. **Service not found**: Ensure the Controller service is running on the specified port
2. **Video format issues**: Convert your video to streamable MP4 format using the provided script
3. **Permission errors**: Ensure you have write permissions for the output directory
4. **Memory issues**: Try reducing the chunk sizes if processing large files
5. **Audio format issues**: Ensure input audio is in WAV format
6. **S2S still running**: When using `--translated-audio`, the S2S container stays up but is not called. Rebuild the Docker image to pick up the new controller image

## Example Workflow

```bash
# 1. Start the services
docker compose --profile controller-third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build

# 2. Run the controller client
python client/controller/app.py \
    --input-audio assets/sample_audio.wav \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --output-mp4 outputs/my_translated_video.mp4

# 3. Check the output
ls -la outputs/my_translated_video.mp4
```

The output video will contain the original video with lip movements synchronized to the translated audio from the S2S service.
