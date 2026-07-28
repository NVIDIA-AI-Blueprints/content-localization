# LipSync Client

This client provides a standalone interface for the LipSync NIM service, which synchronizes lip movements in video to match a given audio track.

## Overview

The LipSync client sends video and audio data to the LipSync NIM via gRPC and receives a processed video with lip movements synchronized to the provided audio. It supports:

- Bidirectional streaming gRPC (always streaming)
- Speaker info from ASD (Active Speaker Detection) for multi-speaker scenarios
- Background audio mixing
- Lossy and lossless video encoding
- Audio/video duration mismatch handling (extend audio or extend video)

## Usage

### Basic Usage

```bash
python client/lipsync/app.py
```

This will use the default settings:
- Input video: `assets/sample_video_streamable.mp4`
- Input audio: `assets/sample_audio.wav`
- LipSync server: `127.0.0.1:50054`
- Output video: `outputs/lipsync_output.mp4`
- Input audio codec: MP3
- Output bitrate: 20 Mbps
- IDR interval: 8 frames

### Custom Parameters

The LipSync NIM uses bidirectional streaming gRPC.
Streamable MP4 (moov atom at the start) is recommended for best performance but not required.

```bash
python client/lipsync/app.py \
    --lipsync-server 127.0.0.1:50054 \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/sample_audio.wav \
    --output-mp4 outputs/lipsync_output.mp4 \
    --lipsync-output-bitrate-mbps 30 \
    --lipsync-output-idr-interval 16 \
    --lipsync-extend-video reverse \
    --lipsync-extend-audio silence
```

### With Speaker Info (from ASD)

When speaker bounding boxes are available (e.g., from the ASD NIM), provide them via a CSV file. This enables multi-speaker lip synchronization.

```bash
python client/lipsync/app.py \
    --lipsync-server 127.0.0.1:50054 \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/sample_audio.wav \
    --speaker-info-input assets/asd_speaker_info.csv \
    --lipsync-is-speaker-info-provided
```

### With Background Audio

Mix background audio (music, ambient sound) into the output video:

```bash
python client/lipsync/app.py \
    --lipsync-server 127.0.0.1:50054 \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/sample_audio.wav \
    --background-audio-input assets/background.mp3 \
    --lipsync-background-audio-volume 0.5 \
    --output-mp4 outputs/lipsync_output.mp4
```

### Lossless Encoding

```bash
python client/lipsync/app.py \
    --lipsync-server 127.0.0.1:50054 \
    --lipsync-lossless
```

## Command Line Arguments

### Connection

| Argument | Default | Description |
|----------|---------|-------------|
| `--lipsync-server` | `127.0.0.1:50054` | IP:port of the LipSync gRPC service (`--target` is a deprecated alias) |
| `--ssl-mode` | `DISABLED` | SSL mode (`DISABLED`, `MTLS`, or `TLS`) |
| `--ssl-key` | `../ssl_key/ssl_key_client.pem` | Path to SSL private key |
| `--ssl-cert` | `../ssl_key/ssl_cert_client.pem` | Path to SSL certificate chain |
| `--ssl-root-cert` | `../ssl_key/ssl_ca_cert.pem` | Path to SSL root certificate |

### Input / Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-mp4` | `assets/sample_video_streamable.mp4` | Path to input video file (MP4; `--video-input` is a deprecated alias) |
| `--input-audio` | `assets/sample_audio.wav` | Path to input audio file (WAV or MP3; `--audio-input` is a deprecated alias) |
| `--speaker-info-input` | `None` | Path to speaker info CSV file (from ASD) |
| `--background-audio-input` | `None` | Path to background audio file (WAV or MP3) for mixing |
| `--output-mp4` | `outputs/lipsync_output.mp4` | Path for the output video file (`--output` is a deprecated alias) |

### LipSync Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--lipsync-input-audio-codec` | `MP3` | Audio codec for LipSync input (`WAV` or `MP3`) |
| `--lipsync-extend-audio` | `unspecified` | How to handle video longer than audio (`unspecified`, `silence`) |
| `--lipsync-extend-video` | `unspecified` | How to handle audio longer than video (`unspecified`, `forward`, `reverse`) |
| `--lipsync-output-bitrate-mbps` | `20` | Output video bitrate in Mbps |
| `--lipsync-output-idr-interval` | `8` | Output video IDR (keyframe) interval in frames |
| `--lipsync-head-movement-speed` | `None` | Head movement speed: 0 for static/slow, 1 for fast |
| `--lipsync-output-audio-codec` | `None` | Output audio codec (`WAV` or `MP3`) |
| `--lipsync-is-speaker-info-provided` | `False` | Flag indicating speaker bounding boxes are provided (from ASD) |
| `--lipsync-background-audio-volume` | `1.0` | Background audio volume (0.0 = muted, 1.0 = full volume) |
| `--lipsync-background-audio-codec` | `None` | Background audio codec (`WAV` or `MP3`). Auto-detected from file extension if omitted |

### Encoding Flags

These are included in the LipSync Configuration table above, but
are also available as standalone flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--lipsync-lossless` | `False` | Enable lossless video encoding (overrides bitrate/IDR settings) |
| `--lipsync-custom-encoding-params` | `None` | Custom encoding parameters in JSON format |

## Requirements

- Python 3.12+
- gRPC dependencies (installed via the project virtual environment)
- The LipSync NIM service must be running and accessible
- Input video must be MP4 format (streamable MP4 recommended for best performance)
- Input audio must be WAV or MP3 format

## Hosting the LipSync NIM

### Using Docker Compose

```bash
docker compose --profile controller-third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build
```

### Standalone NIM

Follow the instructions in the [LipSync NIM documentation](https://docs.nvidia.com/nim/maxine/lipsync/latest/getting-started.html).

## Troubleshooting

1. **Service not found**: Ensure the LipSync NIM is running on the specified port (default: 50054)
2. **Video format issues**: Only MP4 is supported. For streaming mode, convert to streamable MP4:
   ```bash
   ffmpeg -i input.mp4 -movflags +faststart output_streamable.mp4
   ```
3. **Audio format issues**: Only WAV and MP3 are supported
4. **Variable frame rate (VFR)**: VFR videos are not supported. Convert to constant frame rate before processing
5. **Permission errors**: Ensure you have write permissions for the output directory
6. **Memory issues with extend-video**: Video extension caches frames in memory, which can increase memory usage significantly for long videos

## Extending Audio and Video Duration

When audio and video have different durations:

- **`--lipsync-extend-audio silence`**: Pads audio with silence to match video length
- **`--lipsync-extend-video forward`**: Repeats the last 5 seconds of video frames forward to match audio length
- **`--lipsync-extend-video reverse`**: Plays the last 5 seconds of video frames in reverse to match audio length

By default (`unspecified`), no extension is performed and the output matches the shorter of the two inputs.

## Regenerating Proto Files

The proto definitions are in `protos/nvidia/ai4m/lipsync/v1/`. To regenerate the Python bindings:

```bash
bash protos/generate_protos.sh
```

## Example Workflow

```bash
# 1. Start the services
docker compose --profile controller-third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build

# 2. Run the LipSync client
python client/lipsync/app.py \
    --lipsync-server 127.0.0.1:50054 \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/sample_audio.wav \
    --output-mp4 outputs/lipsync_output.mp4

# 3. Check the output
ls -la outputs/lipsync_output.mp4
```

For more information, see the [LipSync NIM documentation](https://docs.nvidia.com/nim/maxine/lipsync/latest/index.html).
