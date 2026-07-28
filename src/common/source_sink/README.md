# Source / Sink helpers

This directory contains shared source and sink utilities used by all client
applications for file I/O. Transport-agnostic abstract bases (`base.py`,
`file.py`) sit at this directory level; transport-coupled concrete classes
live in per-transport subpackages:

- `grpc/` — file-based audio + video simulators that produce gRPC proto
  request types (consumed by all gRPC clients in `controller/`, `direct/`,
  `s2s/`, `asd/`, `lipsync/`, `batch_processing/`).

## Overview

The source/sink helpers provide standardized interfaces for reading and writing
audio and video files across all client applications. They handle:
- **Audio Processing**: WAV and MP3 file reading/writing
- **Video Processing**: MP4 file reading/writing with streaming support
- **Chunk-based Processing**: Efficient handling of large files
- **Format Validation**: Input/output format verification

## Components

### Core Modules

#### `base.py`
**Base classes for file simulators**
- `BaseFileSimulator` - Abstract base class for file simulators
- Common functionality for file operations
- Error handling and validation utilities

#### `file.py`
**Generic file source simulator**
- `FileSourceSimulator` - Streams any file as raw bytes in fixed-size chunks
- Treats files as opaque byte streams without format parsing
- Useful for non-WAV audio formats (e.g., MP3) or any binary file
- Compatible with `AudioSourceSimulator` interface via `read()` method

#### `grpc/audio.py`
**Audio source and sink simulators (gRPC-coupled)**
- `AudioSourceSimulator` - Reads audio files (WAV/MP3) in chunks
- `AudioSinkSimulator` - Writes audio data to WAV files
- `simulated_audio_chunk_generator` - Yields `SpeechToSpeechRequest`
  proto objects suitable for gRPC streaming clients
- Audio format conversion and validation
- Chunk-based streaming support

#### `grpc/video.py`
**Video source and sink simulators (gRPC-coupled)**
- `VideoSourceSimulator` - Reads video files (MP4) in chunks
- `VideoSinkSimulator` - Writes video data to MP4 files
- `video_chunk_generator`, `simulated_asd_video_chunk_generator` - Yield
  proto request objects for S2S / ASD gRPC clients
- Streaming video support with proper metadata handling
- Video format validation and conversion

## Usage

### Audio Processing

```python
from common.source_sink.grpc.audio import AudioSinkSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator

# Read audio file
audio_source = AudioSourceSimulator(file_path="input.wav")
for chunk in audio_source.read(chunk_duration_secs=0.128):
    process_audio(chunk)

# Write audio file
audio_sink = AudioSinkSimulator(file_path="output.wav")
audio_sink.write(wave_bytes=audio_chunk)
```

### File Processing (Non-WAV Audio)

```python
from common.source_sink.file import FileSourceSimulator

# Stream any file as raw bytes (useful for MP3 or other formats)
source = FileSourceSimulator(file_path="translated.mp3")
for chunk in source.read():
    process_chunk(chunk)
```

The Controller client uses `FileSourceSimulator` internally for streaming
non-WAV translated audio files when `--translated-audio` is provided.

### Video Processing

```python
from common.source_sink.grpc.video import VideoSinkSimulator
from common.source_sink.grpc.video import VideoSourceSimulator

# Read video file
video_source = VideoSourceSimulator(file_path="input.mp4")
for chunk in video_source.read(chunk_size=64 * 1024):
    process_video(chunk)

# Write video file
video_sink = VideoSinkSimulator(file_path="output.mp4")
video_sink.write(video_bytes=video_chunk)
video_sink.flush()
```

## Features

### Audio Support
- **Input Formats**: WAV, MP3
- **Output Format**: WAV
- **Chunk-based Processing**: Configurable chunk sizes
- **Format Validation**: Automatic format detection and validation
- **Metadata Handling**: Preserves audio metadata

### Video Support
- **Input Format**: MP4 (streamable preferred)
- **Output Format**: MP4
- **Streaming Support**: Optimized for streaming video files
- **Metadata Preservation**: Maintains video metadata and properties
- **Chunk-based Processing**: Efficient handling of large video files

### Common Features
- **Error Handling**: Local file existence and basic format validation
- **Memory Efficiency**: Chunk-based processing for large files
- **Format Validation**: Input/output format verification where implemented
- **Logging**: Runtime status messages for long streaming operations

## Integration

The source simulators are used by all client applications:

- **Controller Client**: Audio and video processing for complete pipeline;
  `FileSourceSimulator` for non-WAV translated audio
- **Direct Client**: Audio and video processing for direct service communication
- **S2S Client**: Audio processing for speech-to-speech translation
- **LipSync Client**: Audio and video processing for lip synchronization
- **ASD Client**: Video processing for speaker detection

## File Format Requirements

### Audio Files
- **WAV**: Uncompressed PCM audio (recommended)
- **MP3**: Compressed audio (supported for input)
- **Sample Rates**: 8kHz, 16kHz, 44.1kHz, 48kHz
- **Channels**: Mono or stereo

### Video Files
- **MP4**: H.264 encoded video (streamable preferred)
- **Resolution**: Any standard resolution
- **Frame Rate**: Any standard frame rate
- **Streaming**: `moov` atom at start of file for optimal streaming

## Performance Considerations

### Chunk Size Optimization
- **Audio**: 0.5-2 seconds for optimal processing
- **Video**: 32KB-128KB for optimal streaming
- **Memory Usage**: Larger chunks use more memory but reduce overhead

### File Format Optimization
- **Streamable MP4**: Use `ffmpeg -movflags +faststart` for optimal streaming
- **WAV Audio**: Uncompressed for best quality and processing speed
- **File Size**: Consider chunk sizes based on available memory

## Error Handling

The simulators include basic local-file error handling for:
- **File Not Found**: Missing input files and missing output directories
- **Format Errors**: Invalid file format detection where implemented
- **I/O Errors**: Disk read/write errors surfaced to callers
- **Validation Errors**: Input/output path validation

## Troubleshooting

### Common Issues

1. **File not found**: Check file paths and permissions
2. **Format errors**: Verify file format and convert if necessary
3. **Memory issues**: Reduce chunk sizes for large files
4. **Streaming issues**: Convert video to streamable format
5. **Permission errors**: Check read/write permissions

### Performance Issues

1. **Slow processing**: Optimize chunk sizes for your use case
2. **High memory usage**: Reduce chunk sizes or use streaming
3. **File format issues**: Convert to recommended formats

## Example Workflows

### Audio Processing Pipeline
```python
from common.source_sink.grpc.audio import AudioSourceSimulator, AudioSinkSimulator

# Process audio file
source = AudioSourceSimulator("input.wav")
sink = AudioSinkSimulator("output.wav")

for chunk in source:
    # Process audio chunk (e.g., translate with S2S)
    processed_chunk = process_audio_chunk(chunk)
    sink.write_audio_data(processed_chunk)
```

### Video Processing Pipeline
```python
from common.source_sink.grpc.video import VideoSourceSimulator, VideoSinkSimulator

# Process video file
source = VideoSourceSimulator("input.mp4")
sink = VideoSinkSimulator("output.mp4")

for chunk in source:
    # Process video chunk (e.g., lip sync with LipSync)
    processed_chunk = process_video_chunk(chunk)
    sink.write_video_data(processed_chunk)
```

The source/sink helpers provide a shared foundation for client applications, with consistent
file I/O behavior across the content localization pipeline.
