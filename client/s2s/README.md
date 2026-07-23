# S2S (Speech-to-Speech) Client

This client provides a standalone interface for the Speech-to-Speech (S2S) service, which handles audio translation between languages.

## Overview

The S2S client processes audio input and sends it to the S2S service for translation. It supports:
- **Audio Translation**: Convert speech from one language to another
- **Streaming**: Process audio in chunks for low latency
- **Multiple Formats**: Support for WAV and MP3 audio formats
- **Latency Analysis**: Built-in performance monitoring and analysis

## Usage

### Basic Usage

```bash
python client/s2s/app.py
```

This will use the default settings:
- Input audio: `assets/sample_audio.wav`
- S2S server: `localhost:50050`
- Output audio: `outputs/sample_audio_output.mp3`
- Audio chunk size: 1 second

### Custom Parameters

```bash
python client/s2s/app.py \
    --s2s-server localhost:50050 \
    --input-audio assets/sample_audio.wav \
    --output-audio outputs/translated_audio.mp3 \
    --chunk-size-audio-secs 0.5
```

### CambAI Parameters

```bash
python client/s2s/app.py \
    --no-camb-ai-optimization \
    --camb-chosen-dictionaries 1,5,12
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--s2s-server` | `localhost:50050` | Address and port of the S2S gRPC service |
| `--input-audio` | `assets/sample_audio.wav` | Path to input audio file (WAV/MP3 format) |
| `--chunk-size-audio-secs` | `1` | Audio chunk size for streaming in seconds |
| `--output-audio` | `outputs/sample_audio_output.mp3` | Path to output audio file (WAV or MP3 format) |
| `--camb-ai-optimization` / `--no-camb-ai-optimization` | `True` | Enable CambAI AI optimization |
| `--camb-chosen-dictionaries` | `None` | Comma-separated CambAI dictionary IDs (e.g. `1,5,12`) |

## How It Works

1. **Audio Input**: The client reads audio files in configurable chunks
2. **Request Generation**: Creates `SpeechToSpeechRequest` objects containing audio data
3. **Service Communication**: Streams requests to the S2S service via gRPC
4. **Audio Translation**: The S2S service translates the audio to the target language
5. **Output Generation**: Receives translated audio chunks and writes them to the output file

## Troubleshooting

### Common Issues

1. **Service not found**: Ensure the S2S service is running on the specified port
2. **Audio format issues**: Convert your audio to WAV or MP3 format
3. **Permission errors**: Ensure you have read/write permissions for input/output files
4. **Memory issues**: Try reducing the chunk size if processing large files
5. **Network timeouts**: Check network connectivity and service availability

### Performance Issues

1. **High latency**: Reduce chunk size or check network connectivity
2. **Low throughput**: Increase chunk size or check service capacity
3. **Memory usage**: Monitor memory usage and adjust chunk sizes accordingly

## Example Workflow

### ElevenLabs

```bash
# 1. Start the S2S service with ElevenLabs backend S2S_SERVICE=EL_DUBBING
docker compose --profile third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build

# 2. Run the S2S client
python client/s2s/app.py \
    --input-audio assets/sample_audio.wav \
    --output-audio outputs/translated_audio.mp3 \
    --source-language en --target-language de
```

### CambAI

CambAI uses **integer language IDs** (e.g. `1` = English, `54` = Spanish). To get the full mapping, query the CambAI API or see the [source languages](https://docs.camb.ai/api-reference/endpoint/get-source-languages) and [target languages](https://docs.camb.ai/api-reference/endpoint/get-target-languages) docs.

```bash
# 1. Start the S2S service with CambAI backend S2S_SERVICE=CAMB_DUBBING
docker compose --profile third-party-s2s --env-file configs/camb.env --env-file .env up --build

# 2. Run the S2S client (CambAI uses numeric language IDs)
python client/s2s/app.py \
    --input-audio assets/sample_audio.wav \
    --output-audio outputs/translated_audio.mp3 \
    --source-language 1 --target-language 54 \
    --camb-ai-optimization
```

The S2S client provides a robust and efficient interface for audio translation, with comprehensive error handling.
