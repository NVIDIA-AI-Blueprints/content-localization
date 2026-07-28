# Content Localization

The Content Localization Blueprint is a comprehensive solution for translating and dubbing audio and video content using AI services. It is a microservices architecture that orchestrates three main AI services:

- **Speech-to-Speech (S2S)**: Translates audio from one language to another with voice preservation
- **Speaker Detection (ASD NIM)**: Identifies who needs to be dubbed in the video
- **LipSync**: Synchronizes lip movements with translated audio

The blueprint supports multiple client types to accommodate different deployment scenarios and use-case requirements.

Note: To get access to the LipSync feature of the Content localization Blueprint, please request to join our [NVIDIA AI for Media Private Access Program](https://developer.nvidia.com/ai-for-media/private-access-program)

## Overall Architecture

The diagram below shows the controller-centric end-to-end architecture used to orchestrate S2S, ASD, and LipSync workflows.

![Controller overall architecture](docs/Architecture%20Diagram.png)

## Notices: 

- You may not directly or indirectly use this Content Localization Blueprint to alter the name, likeness, image, or voice of any person in violation of applicable law or regulation or without the person’s express consent.
- The Content-Localization Blueprint is shared as reference and is provided "as is". The security in the production environment is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats; define the trust boundaries, implement logging and monitoring capabilities, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment up to date, ensure the containers/source code are secure and free of known vulnerabilities.
---

## License:

GOVERNING TERMS: The blueprint software is governed by the [Apache License 2.0](https://github.com/NVIDIA-AI-Blueprints/content-localization/LICENSE.md), and enables use of separate open source and proprietary software, models and services governed by their respective licenses, including those below.
  - [Active Speaker Detection NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/active-speaker-detection)
  - [LipSync NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/lipsync)
  - [Eleven Labs API service](https://elevenlabs.io/dubbing-studio)
  - [Camb.ai service](https://www.camb.ai/features/translation)

Sample Assets: Use of the assets is governed by the [NVIDIA Sample Data License](https://github.com/NVIDIA-AI-Blueprints/content-localization/blob/main/licenses/NVIDIA%20Maxine%20Sample%20Data%20License%20%282025.10.22%29.pdf).

Link to relevant licenses:

- [ThirdPartyLicenses.md](ThirdPartyLicenses.md) — Third-party open-source software licenses
- [NIMLICENSES.md](NIMLICENSES.md) — NVIDIA NIM container licenses (LipSync, ASD)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

- [Code of Conduct](CODE_OF_CONDUCT.md) — Standards for community participation
- [Security](SECURITY.md) — Reporting security vulnerabilities
---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Environment Setup](#development-environment-setup)
- [Running the Services](#running-the-services)
- [Client Applications](#client-applications)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Utility Scripts](#utility-scripts)
- [License](#license)
- [Additional Resources](#additional-resources)

---

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

### System Requirements

- **Operating System**: Linux (Ubuntu 22.04 or 24.04 recommended)
- **Python**: 3.12 or higher
- **Git**: With Git LFS enabled
- **NVIDIA GPU**: With CUDA-capable drivers installed
- **CUDA Toolkit**: CUDA 12.x
- **TensorRT**: Compatible with your CUDA version
- **Docker**: Docker Engine 24.x or higher with Docker Compose and NVIDIA Docker runtime.
- **Node.js/npm**: Node 24 (Optional, only for UI development and pre-commit hooks)

### Verify GPU and CUDA Installation

```bash
# Verify NVIDIA driver installation
nvidia-smi

# Check CUDA version
nvcc --version
```
**Note**: If `nvcc` is not found, ensure CUDA is properly installed and added to your PATH and LD_LIBRARY_PATH

### Install System Dependencies

**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

These packages are required:

```bash
sudo apt-get update
sudo apt-get install -y \
    curl \
    wget \
    git-lfs
```

### Required Credentials

You need the following credentials set as environment variables (or in a `.env` file).
Get your NGC API keys from: https://ngc.nvidia.com/setup/api-key

**NIM container keys** — `docker-compose.yml` maps these component-specific keys to
`NGC_API_KEY` inside each NIM container. Set the keys for the services you intend to run:

| Variable | Used by | When required |
|---|---|---|
| `LIPSYNC_API_KEY` | LipSync NIM container | When using Lipsync |
| `ASD_API_KEY` | ASD NIM container | When using Active Speaker Detection |

**Third-party API keys** — required only when using the corresponding S2S backend:

| Variable | When required |
|---|---|
| `ELEVENLABS_API_KEY` | When `S2S_SERVICE=EL_DUBBING` |
| `CAMB_API_KEY` | When `S2S_SERVICE=CAMB_DUBBING` (and for the CAMB.AI helper scripts) |
---

## Development Environment Setup

Follow these steps to set up your local development environment:

### 1. Create Environment File

Create a `.env` file in the project root with your credentials:

```bash
# .env file

# NIM container keys (mapped to NGC_API_KEY inside each container by docker-compose.yml)
LIPSYNC_API_KEY=your_ngc_api_key_here
ASD_API_KEY=your_ngc_api_key_here

# Third-party S2S backend keys
ELEVENLABS_API_KEY=your_11labs_api_key_here
# CAMB_API_KEY=your_camb_api_key_here

PYTHONPATH=path-to-root-of-repository
```

### 2. Install uv Package Manager

Install `uv`, the fast Python package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add `uv` to your PATH (or restart your shell):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Create Virtual Environment

Create and activate a Python 3.12 virtual environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 4. Install Python Dependencies

Install the project dependencies and commonly used extras:

```bash
# Install core dependencies with non-GPU extras (test, lint, docs)
uv sync --extra test --extra lint --extra docs
```

Install GPU extras only on hosts with CUDA Toolkit headers available (`cuda.h`):

```bash
# Optional: install GPU extras (requires CUDA Toolkit development headers)
uv sync --extra gpu
```

### 5. Generate Protocol Buffer Files

Generate Python code from the gRPC/protobuf definitions:

```bash
# Download gRPC health check proto
wget -O protos/health.proto https://raw.githubusercontent.com/grpc/grpc/master/src/proto/grpc/health/v1/health.proto

# Generate Python protobuf files
bash ./protos/generate_protos.sh
```

### 6. Install Development Tools

Install linting and pre-commit hooks:

```bash
# Install development tools
uv tool install pre-commit
uv tool install ruff

# Set up pre-commit hooks
pre-commit install
```

To update pre-commit hook versions (optional):

```bash
pre-commit autoupdate
```

To run pre-commit on all files manually:

```bash
pre-commit run --all-files
```

### 7. Set Python Path

Add the project root, `src`, `client`, and generated protobuf files to your `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"
```

Add this line to your shell profile (`.bashrc` or `.zshrc`) to make it permanent:

```bash
echo 'export PYTHONPATH="${PYTHONPATH}:'"${PWD}"':'"${PWD}"'/src:'"${PWD}"'/client:'"${PWD}"'/protos/generated"' >> ~/.bashrc
```

### 8. Create Required Directories

Create the directories for test outputs, builds, and the NIM model caches:

```bash
mkdir -p build outputs volumes/models/asd volumes/models/lipsync
```

The ASD and LipSync NIM services download their models into the bind-mounted
cache directories (`volumes/models/asd` and `volumes/models/lipsync`). On a
fresh clone these directories must already **exist** — they ship with a
`.gitkeep` placeholder — otherwise `docker compose up` fails with
`Permission denied (os error 13)` when Docker cannot resolve the mount path.

The NIM containers run as `root`, so they can write into these directories once
they exist. Ensure the `volumes`, `build`, and `outputs` trees are writable by
your host user and group (no world-writable `777` needed):

```bash
chmod -R u+rwX,g+rwX volumes build outputs
```

### 9. Deploy and Verify Services (First-Time Setup)

For first-time deployment, use the deploy scripts to verify each service individually. This approach downloads models and verifies that each service starts correctly before deploying the full stack.

#### Deploy ASD Service

Deploy the Active Speaker Detection service:

```bash
./scripts/nims/deploy_asd.sh
```

This will:
- Download the ASD model to `volumes/models/asd/`
- Start the ASD container on ports 8005 (HTTP) and 50055 (gRPC)
- Verify the service is running correctly

**Note**: Model download may take several minutes depending on your internet connection. Press `Ctrl+C` to stop the service once verified.

#### Deploy LipSync Service

Deploy the LipSync service:

```bash
./scripts/nims/deploy_lipsync.sh
```

This will:
- Download the LipSync models to `volumes/models/lipsync/`
- Start the LipSync container on ports 8004 (HTTP) and 50054 (gRPC)
- Verify the service is running correctly

**Note**: This requires the `LIPSYNC_API_KEY` (or `NGC_API_KEY`) environment variable. Press `Ctrl+C` to stop the service once verified.

#### Deployment Notes

- Each deploy script runs in interactive mode (`-it`) and will occupy your terminal
- Run each script in a separate terminal or stop (Ctrl+C) before running the next
- These scripts are for **verification only** - use docker compose for production deployments

### 10. Verify Setup

If all steps completed successfully, you're ready to run the full service stack!

To verify your setup:
1. All deploy scripts should have started successfully without errors
2. Model files should be present in `volumes/models/` subdirectories
3. TensorRT engines should be built in `volumes/models/asd/`

You can now proceed to [Running the Services](#running-the-services) to launch the full stack with docker compose.

---

## Running the Services

### Launch All Services

Start the full Content Localization stack:

```bash
# Full stack: S2S (ElevenLabs/CambAI) + ASD + LipSync + Controller + Demo App
docker compose --profile demo-app-third-party-s2s \
    --env-file configs/elevenlabs.env \
    --env-file .env \
    up --build
```

### Available Profiles

Different service combinations for various use cases. Use the `--profile` flag to select which services to run.
The canonical profile reference lives in [docs/source/deployment.rst](docs/source/deployment.rst).

| Profile | S2S | ASD | LipSync | Controller | Demo App | Description |
|---------|-----|-----|---------|------------|----------|-------------|
| `default` | ✓ | ✓ | ✓ | ✓ | ✓ | All services (for testing) |
| `third-party-s2s` | ✓ | - | - | - | - | S2S only with ElevenLabs/CambAI |
| `lipsync` | ✓ | - | ✓ | - | - | LipSync + S2S backend |
| `third-party-s2s-lipsync` | ✓ | - | ✓ | - | - | S2S (ElevenLabs/CambAI) + LipSync |
| `third-party-s2s-asd-lipsync` | ✓ | ✓ | ✓ | - | - | Full pipeline with ElevenLabs/CambAI |
| `asd` | - | ✓ | - | - | - | Active Speaker Detection only |
| `controller-third-party-s2s` | ✓ | ✓ | ✓ | ✓ | - | Orchestrated pipeline (ElevenLabs/CambAI) |
| `asd-lipsync` | - | ✓ | ✓ | ✓ | - | ASD + LipSync + Controller |
| `demo-app-third-party-s2s` | ✓ | ✓ | ✓ | ✓ | ✓ | Full stack with Web Demo (ElevenLabs/CambAI) |

#### Usage Examples

```bash
# ElevenLabs with full pipeline and demo app
docker compose --profile demo-app-third-party-s2s \
    --env-file configs/elevenlabs.env \
    --env-file .env \
    up --build

# CambAI S2S + ASD + LipSync (no controller, no demo)
docker compose --profile third-party-s2s-asd-lipsync \
    --env-file configs/camb.env \
    --env-file .env \
    up --build

# Controller orchestration with ElevenLabs (also similar for camb)
docker compose --profile controller-third-party-s2s \
    --env-file configs/elevenlabs.env \
    --env-file .env \
    up --build

# Basic S2S with ElevenLabs only (also similar for CambAI)
docker compose --profile third-party-s2s \
    --env-file configs/elevenlabs.env \
    --env-file .env \
    up --build
```

#### Profile Selection Guide

- **For Development/Testing**: Use `demo-app-third-party-s2s` for the full stack with web interface
- **For Production with ElevenLabs/CambAI**: Use `controller-third-party-s2s` for orchestrated processing
- **For Service Testing**: Use individual profiles like `third-party-s2s`, `lipsync`, or `asd`

### Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean state)
docker compose down -v
```

### View Logs

#### Real-Time Log Viewing

View logs in real-time as services are running:

```bash
# View logs from all services (follow mode)
docker compose logs -f

# View logs from specific service
docker compose logs -f speech-to-speech
docker compose logs -f asd
docker compose logs -f controller
docker compose logs -f lipsync

# View logs from multiple services
docker compose logs -f speech-to-speech controller

# View last 100 lines of logs
docker compose logs --tail=100
```

#### Copy Logs to Files

For debugging or sharing, use the log copy script to save logs to local files:

```bash
# Copy logs from all services to ./logs/ directory
./scripts/misc/copy_docker_logs.sh

# Copy logs from a specific service only
./scripts/misc/copy_docker_logs.sh s2s
./scripts/misc/copy_docker_logs.sh lipsync
./scripts/misc/copy_docker_logs.sh asd
./scripts/misc/copy_docker_logs.sh controller

# View help
./scripts/misc/copy_docker_logs.sh --help
```

This creates log files in `./logs/`:
- `./logs/s2s.log` - Speech-to-Speech service logs
- `./logs/lipsync.log` - LipSync service logs
- `./logs/asd.log` - Active Speaker Detection logs
- `./logs/controller.log` - Controller orchestration logs

**Benefits of copying logs**:
- Persist logs even after containers are stopped
- Easy to share with team members for debugging
- Can be archived or uploaded to issue trackers
- Includes line counts and helpful status messages

---

## Client Applications

This section describes the various client implementations for interacting with the Content-Localization services.
Each client is designed for specific use cases and can be used independently or together.

### Available Clients

| Client | File | Services | Use Case | Key Features | Usage |
|--------|------|----------|----------|--------------|-------|
| **Controller** | `client/controller/app.py` | Controller (S2S + ASD + LipSync orchestration) | Streamlined end-to-end content localization with single service communication | • Audio and video input processing<br>• Optional diarization input for speaker-aware dubbing<br>• Optional background audio pass-through to LipSync<br>• Pre-translated audio to bypass S2S (`--translated-audio`)<br>• Complete pipeline orchestration through Controller service<br>• Single gRPC "streaming" connection for entire workflow<br>• Simplified error handling and monitoring | `python client/controller/app.py [options]` |
| **Direct** | `client/direct/app.py` | S2S + ASD + LipSync (direct communication) | Full control over service pipeline with direct service coordination | • Direct communication with each service<br>• Custom pipeline orchestration (S2S → LipSync → ASD)<br>• Pre-translated audio to bypass S2S (`--translated-audio`)<br>• Complete control over service interactions, easy to test and debug various services individually and in combination | `python client/direct/app.py [options]` |
| **S2S** | `client/s2s/app.py` | S2S only | Audio translation between languages | • Audio input/output processing<br>• Real-time streaming with configurable chunks<br>• Built-in latency analysis and performance monitoring<br>• Support for WAV and MP3 formats. Ensure output file name is set to the desired format | `python client/s2s/app.py [options]` |
| **LipSync** | `client/lipsync/app.py` | LipSync only | Synchronize lip movements with audio | • Video and audio input processing<br>• Speaker info support from file<br>• Optional background audio mixing<br>• Multiple output formats and encoding options<br>• Streaming support with performance optimization | `python client/lipsync/app.py [options]` |
| **ASD** | `client/asd/app.py` | ASD only | Detect active speakers in video | • Video processing for speaker detection<br>• Multi-format diarization input (flat, ElevenLabs/CambAI)<br>• Speaker info generation in CSV format<br>• Real-time speaker detection<br>• Configurable chunk sizes for optimal performance | `python client/asd/app.py [options]` |
| **Demo Web App** | `client/demos` | Controller (full pipeline) | User-friendly interface for testing and demonstrations | • Web-based UI for file upload and processing<br>• Real-time progress monitoring<br>• Output preview and download<br>• Accessible via browser at `http://localhost:3000` | • Service is spun up from docker compose<br>• Open browser to `http://localhost:3000` for UI interface |

### Client Selection Guide

| Use Case | Recommended Client | Description |
|----------|-------------------|-------------|
| **Complete Pipeline (Simplified)** | `controller/` | Single service orchestration with minimal configuration. Ideal for production workflows. |
| **Complete Pipeline (Full Control)** | `direct/` | Direct service communication with complete pipeline control. Best for development and detailed monitoring. |
| **Audio Translation Only** | `s2s/` | Just speech-to-speech translation with performance analysis. |
| **Lip Sync Only** | `lipsync/` | Just lip synchronization with advanced encoding options |
| **Speaker Detection Only** | `asd/` | Just active speaker detection and speaker info generation |
| **Web Interface** | `demos/` | Browser-based interface for demonstrations |


### Shared Components

#### Source / Sink Helpers (`source_sink/`)

**Purpose**: Shared audio and video source and sink utilities for file I/O

- **Files**:
  - `base.py` - Abstract base class for file simulators (`BaseFileSimulator`)
  - `file.py` - Generic file source simulator for non-WAV formats (e.g., MP3)
  - `grpc/audio.py` - Audio source and sink simulators (gRPC-coupled)
  - `grpc/video.py` - Video source and sink simulators (gRPC-coupled)
- **Use Case**: Used by all clients for standardized file I/O operations
- **Features**:
  - WAV/MP3 audio processing
  - MP4 video processing with streaming support
  - Chunk-based processing for large files
  - Format validation and error handling
  - Latency analysis and performance monitoring

### Client Architecture Comparisons

The following diagrams illustrate the different client architectures.
Mermaid sources are available in `docs/source/uml_mermaid/`.

#### Direct Client Architecture

**Source**: [`docs/source/uml_mermaid/direct_client_architecture.mmd`](docs/source/uml_mermaid/direct_client_architecture.mmd)
**Benefits**:
- Full control over pipeline
- Custom orchestration
- Development flexibility
- Service-specific tuning

#### Individual Client Architectures

**S2S Client Architecture**

**Source**: [`docs/source/uml_mermaid/s2s_client_architecture.mmd`](docs/source/uml_mermaid/s2s_client_architecture.mmd)

**Benefits**:
- Audio translation only
- Real-time streaming with configurable chunks
- Built-in latency analysis and performance monitoring
- Support for WAV and MP3 formats

**LipSync Client Architecture**

**Source**: [`docs/source/uml_mermaid/lipsync_client_architecture.mmd`](docs/source/uml_mermaid/lipsync_client_architecture.mmd)

**Benefits**:
- Video and audio input processing
- Speaker info support from file or ASD NIM
- Multiple output formats and encoding options
- Streaming support with performance optimization

**ASD Client Architecture**

**Source**: [`docs/source/uml_mermaid/asd_client_architecture.mmd`](docs/source/uml_mermaid/asd_client_architecture.mmd)

**Benefits**:
- Video processing for speaker detection
- Speaker info generation in CSV format
- Real-time speaker detection
- Configurable chunk sizes for optimal performance

#### Client-Service Data Flow for Controller Client
**Source**: [`docs/source/uml_mermaid/client_service_flow.mmd`](docs/source/uml_mermaid/client_service_flow.mmd)
Detailed sequence diagram showing:
- Health checks for all services
- Controller service orchestration
- Audio processing pipeline (Controller → S2S → ElevenLabs/CambAI)
- Video processing pipeline (Controller → ASD → GPU/CPU fallback)
- LipSync processing coordination
- Error handling and fallback mechanisms

#### UI Demo App Client Architecture
**Source**: [`docs/source/uml_mermaid/ui_demo_app_architecture.mmd`](docs/source/uml_mermaid/ui_demo_app_architecture.mmd)

**Benefits**:
- User-friendly web interface for content localization workflows
- Upload and preview audio/video content with localization results
- Real-time output streaming to UI

**📖 Complete Documentation**: See [Demo Web Application Guide](docs/source/demo_app.rst) for detailed setup, configuration, and usage instructions.


### Client Prerequisites

Before running any client:

1. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Ensure Services are Running**:
   - **Controller Service** (default: `localhost:50056`) - for controller client
   - **S2S Service** (default: `localhost:50050`) - for direct client and S2S client
   - **LipSync Service** (default: `localhost:50054`) - for direct client and LipSync client
   - **ASD NIM Service** (default: `localhost:50055`) - for direct client and ASD client

3. **Required Input Files**:
   - Audio files (WAV/MP3 format)
   - Video files (MP4 format, streamable preferred)
   - Speaker info files (CSV format, optional for LipSync)

### Quick Start Examples

#### Start Services for Controller Client (Recommended)
```bash
docker compose --profile controller-third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build
```

#### Start Services for Direct Client
```bash
docker compose --profile third-party-s2s-asd-lipsync --env-file configs/elevenlabs.env --env-file .env up --build
```

#### Run Controller Client (Complete Pipeline)
```bash
python client/controller/app.py
```

#### Run Direct Client (Alternative Pipeline)
```bash
python client/direct/app.py
```

#### Run Individual Clients
```bash
# S2S only
python client/s2s/app.py

# LipSync only
python client/lipsync/app.py

# ASD only
python client/asd/app.py
```

### Client Configuration

Each client supports various command-line options for customization:

- **Server endpoints**: Configure service addresses and ports
- **Input/output paths**: Specify file locations
- **Chunk sizes**: Adjust streaming parameters for optimal performance
- **Audio/video formats**: Configure processing options
- **Performance settings**: Tune for latency vs. throughput

Run `python client/[client_name]/app.py --help` for detailed options.

### Client Output Files

Clients generate various output files:

- **Audio**: Translated audio (WAV format)
- **Video**: Lip-synchronized video (MP4 format)
- **Speaker Info**: Per-frame speaker metadata (CSV format)
- **Logs**: Processing logs and diagnostics
- **Performance**: Latency analysis plots and metrics

### Client Troubleshooting

#### Common Issues

1. **Module Import Errors**: Ensure virtual environment is activated
2. **Service Connection Errors**: Verify services are running and accessible
3. **File Format Issues**: Check input file formats and convert if necessary
4. **Memory Issues**: Reduce chunk sizes for large files
5. **Network Issues**: Check network connectivity and firewall settings

#### Client-Specific Issues

- **Controller Client**: Check Controller service health and configuration
- **Direct Client**: Verify all three services (S2S, LipSync, ASD) are running
- **S2S Client**: Check audio format and S2S service logs
- **LipSync Client**: Verify video format and LipSync service configuration
- **ASD Client**: Check video format and ASD NIM logs

### Advanced Client Usage

#### Batch Processing
```bash
# Process multiple files with controller client
for audio_file in audio/*.wav; do
    for video_file in video/*.mp4; do
        python client/controller/app.py \
            --input-audio "$audio_file" \
            --input-mp4 "$video_file" \
            --output-mp4 "outputs/$(basename "$audio_file" .wav)_$(basename "$video_file" .mp4)_output.mp4"
    done
done
```

#### Service Health Monitoring
```bash
# Check all services are running
python -c "
import grpc
services = [
    ('localhost:50050', 'S2S'),
    ('localhost:50054', 'LipSync'),
    ('localhost:50055', 'ASD'),
    ('localhost:50056', 'Controller')
]
for addr, name in services:
    try:
        channel = grpc.insecure_channel(addr)
        grpc.channel_ready_future(channel).result(timeout=1)
        print(f'✓ {name} service is running on {addr}')
    except:
        print(f'✗ {name} service is not accessible on {addr}')
"
```

---

## Configuration

> The settings below summarize the most common options. The complete, canonical
> configuration reference (all variables, types, and defaults) lives in
> [docs/source/configuration.rst](docs/source/configuration.rst).

### Environment Variables

#### Controller Service Configuration

The controller service supports various configuration options:

**Basic Configuration:**
- `CONTROLLER_GRPC_API_PORT`: gRPC service port (default: `50056`)
- `CONTROLLER_MAX_CONCURRENCY`: Maximum concurrent requests (default: `1`)
- `CONTROLLER_LOG_LEVEL`: Logging level (default: `INFO`)

**Service Endpoints:**
- `S2S_SERVER`: Speech-to-Speech service endpoint
- `ASD_SERVER`: Active Speaker Detection service endpoint (optional)
- `LIPSYNC_SERVER`: LipSync service endpoint

**Controller Processing:**
- ASD bypass a per-request option available via `bypass_asd=True` in `ContentLocalizationConfig` (LipSync uses internal face detection)

**Debug Configuration:**
- `CONTROLLER_DEBUG_PORT`: VS Code debug port (default: `5678`)
- `CONTROLLER_VS_CODE_DEBUG`: Enable VS Code debugging (default: `0`)

#### S2S Service Configuration

- `S2S_GRPC_API_PORT`: gRPC service port (default: `50050`)
- `S2S_LOG_LEVEL`: Logging level (default: `INFO`)

#### ASD NIM Configuration

- `ASD_GRPC_API_PORT`: gRPC service port (default: `50055`)
- `ASD_LOG_LEVEL`: Logging level (default: `INFO`)
- `ASD_MODEL_MOUNT_PATH`: Host path mounted as the ASD NIM model cache

#### Timeout and Polling Configuration

All timeout values are in seconds. See the full reference in the
[Sphinx Configuration docs](docs/source/configuration.rst).

**Shared (all services):**
- `HEALTH_CHECK_TIMEOUT`: HTTP and gRPC health-check timeout (default: `5.0`)
- `BUFFER_POLL_TIMEOUT`: Buffer iterator poll cadence (default: `0.1`)

**Controller:**
- `CONTROLLER_CONFIG_POLL_TIMEOUT`: Wait for per-request config messages (default: `5.0`)
- `CONTROLLER_CLEANUP_TIMEOUT`: Thread cleanup timeout (default: `10.0`)

**S2S:**
- `S2S_EL_DUBBING_POLL_INTERVAL`: ElevenLabs dubbing status poll interval (default: `10`)
- `S2S_EL_DUBBING_MAX_ATTEMPTS`: Max dubbing poll attempts (default: `120`)
- `S2S_EL_KEEPALIVE_INTERVAL`: Keepalive ping interval during dubbing (default: `1`)

### Configuration Files

Configuration files are located in the `configs/` directory:

- `configs/elevenlabs.env`: ElevenLabs S2S configuration
- `configs/camb.env`: CambAI S2S configuration
- `configs/debug.env`: Developer profile — same knobs as `configs/elevenlabs.env` with debug-oriented values (`LIPSYNC_DEBUG_MODE=1`)

---

## Development

### Project Structure

The tree below lists the top-level packages (not every file):

```
.
├── assets/             # Sample input media (audio/video) and diarization
├── client/             # Client applications
│   ├── asd/            # Active Speaker Detection client (app.py, args.py, config.py)
│   ├── batch_processing/  # Batch pipeline runner over a directory of videos
│   ├── common/         # Shared client helpers (timing, etc.)
│   ├── controller/     # Controller orchestration client (app.py, args.py, config.py)
│   ├── demos/          # Web demo application
│   ├── direct/         # Direct processing client (app.py, args.py, config.py, pipeline.py)
│   ├── lipsync/        # LipSync client (app.py, args.py, config.py)
│   └── s2s/            # Speech-to-Speech client (app.py, args.py, config.py)
├── configs/            # Service configuration files
├── dockerfiles/        # Dockerfiles for each service
├── docs/               # Sphinx documentation
├── functional_tests/   # End-to-end functional tests (require running services)
├── protos/             # gRPC/Protobuf definitions
├── scripts/            # Utility and standalone scripts
│   ├── camb/           # Camb AI scripts (s2s_infer.py, diarize.py, audio_isolation.py)
│   ├── docs/           # Documentation generators (CLI reference)
│   ├── elevenlabs/     # ElevenLabs scripts (s2s_infer.py, diarize.py, audio_isolation.py, stem_separation.py)
│   ├── functional_tests/  # Shell wrappers for functional test runs
│   ├── nims/           # deploy_*.sh for ASD/LipSync NIMs and parity_test.sh
│   ├── perf/           # Performance benchmarking scripts and reports
│   └── misc/           # Dev/media utilities (setup_env.sh, convert_to_streamable_mp4.sh, etc.)
├── src/                # Service implementation
│   ├── common/         # Shared utilities
│   ├── controller_service/  # Controller service code
│   ├── docker_entrypoints/  # Docker container entrypoints
│   └── s2s_service/    # S2S service code
├── tests/              # Unit and integration tests
└── volumes/            # Persistent data (models, cache, outputs)
```

### Regenerate Protocol Buffers

gRPC protobuf definitions are in the `protos/` folder. To regenerate:

```bash
cd protos
bash generate_protos.sh
cd ..
```

### Code Formatting and Linting

The project uses `ruff` for code formatting and linting:

```bash
# Check formatting
ruff format --check src/ tests/ client/

# Auto-format code
ruff format src/ tests/ client/

# Run linter
ruff check src/ tests/ client/

# Auto-fix linting issues
ruff check --fix src/ tests/ client/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit:

```bash
# Run manually on all files
pre-commit run --all-files

# Skip hooks (not recommended)
git commit --no-verify
```

---

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_s2s_service.py

# Run tests matching pattern
pytest -k "test_asd"
```

### Test Coverage

Coverage reports are generated in `build/coverage/`:

```bash
# Generate HTML coverage report
pytest --cov-report html:build/coverage

# Open coverage report
open build/coverage/index.html  # macOS
xdg-open build/coverage/index.html  # Linux
```

### Functional Tests

Comprehensive end-to-end functional tests are available to validate the complete pipeline with all clients and services. These tests run actual clients with sample inputs and verify outputs.

**Test Coverage:**
- Controller Client (orchestrated pipeline)
- Direct Client (direct service communication)
- S2S Client (audio translation with latency analysis)
- LipSync Client (lip synchronization)
- ASD Client (active speaker detection)

**Quick Start:**
```bash
# Run all functional tests
python -m pytest functional_tests/ -v

# Run specific client tests
python -m pytest functional_tests/test_controller_client.py -v
python -m pytest functional_tests/test_s2s_client.py -v
```

> **CambAI stacks:** the defaults use ElevenLabs-style language codes
> (`en`, `es`). CambAI expects integer language IDs, so pass them explicitly,
> otherwise the run fails with `Invalid CambAI source language ID: en`:
>
> ```bash
> source .venv/bin/activate && python -m pytest functional_tests/ -v --require-services \
>     --source-language 1 --target-language 54
> ```

**Prerequisites:**
- All services running (S2S, ASD, LipSync, Controller)
- Sample input files in `assets/`
- Python environment with dependencies

For detailed functional testing documentation, test configuration, and troubleshooting, see **[functional_tests/README.md](functional_tests/README.md)**

---

## Documentation

Comprehensive Sphinx-based documentation is available covering architecture, service modes, client types, and API references.

**Build Documentation:**

```bash
# Build HTML documentation
bash docs/build_docs.sh
# or
cd docs
make html
cd ..
```

**View Generated Documentation:**

```bash
open build/docs/html/index.html  # macOS
xdg-open build/docs/html/index.html  # Linux
```

**Output Locations:**
- HTML: `build/docs/html/index.html`

For documentation structure, maintenance guidelines see **[docs/README.md](docs/README.md)**

### Mermaid Diagram Sources

Mermaid diagrams for system and client architecture are available in:

`docs/source/uml_mermaid/`

This directory includes diagrams for:
- System architecture overview
- Client architectures (Controller, Direct, Individual)
- Service mode comparisons
- Client-service flow diagrams
- Demo app workflow

---

## Troubleshooting

### Common Issues

**Issue: `nvidia-smi` not found**
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Issue: Docker permission denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**Issue: TensorRT engine build fails**
- Verify CUDA and TensorRT versions match
- Ensure sufficient disk space (>10GB free)
- Check GPU compute capability compatibility

---

## Utility Scripts

The `scripts/` directory contains various utility scripts to help with development, deployment, and testing:

### Deployment Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy_lipsync.sh` | Deploy LipSync service | `./scripts/nims/deploy_lipsync.sh` |
| `deploy_asd.sh` | Deploy ASD NIM service | `./scripts/nims/deploy_asd.sh` |

These scripts download models and start individual services for verification before full deployment.

### Development Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_env.sh` | Setup complete development environment | `./scripts/misc/setup_env.sh` |
| `copy_docker_logs.sh` | Copy Docker container logs to files | `./scripts/misc/copy_docker_logs.sh [service]` |

### Media Processing Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `convert_to_streamable_mp4.sh` | Convert videos to streamable MP4 format | `./scripts/misc/convert_to_streamable_mp4.sh input.mp4` |
| `extract_audio_from_videos.sh` | Extract audio from video files | `./scripts/misc/extract_audio_from_videos.sh <input_dir> <output_dir> [sample_rate] [channels]` |

### Script Details

#### convert_to_streamable_mp4.sh

Converts video files to MP4 format suitable for streaming with the `faststart` flag.

**Usage:**
```bash
./scripts/misc/convert_to_streamable_mp4.sh input.mp4
# Output: input_streamable.mp4
```

**Features:**
- Automatically installs ffmpeg if not present
- Copies video/audio streams without re-encoding
- Adds `faststart` flag for progressive download
- Supports various input formats (avi, mkv, mp4, etc.)

#### extract_audio_from_videos.sh

Batch extract audio from all video files in a directory.

**Usage:**
```bash
# Basic usage with defaults (16kHz, mono, WAV)
./scripts/misc/extract_audio_from_videos.sh videos/ audio/

# Custom parameters (44.1kHz, stereo, MP3)
./scripts/misc/extract_audio_from_videos.sh videos/ audio/ 44100 2 mp3
```

**Arguments:**
- `input_dir` - Directory containing video files (required)
- `output_dir` - Directory to save audio files (required)
- `sample_rate` - Sample rate in Hz (default: 16000)
- `channels` - Audio channels: 1=mono, 2=stereo (default: 1)
- `format` - Output format: wav, mp3, flac (default: wav)

**Features:**
- Processes multiple video formats (mp4, avi, mkv, mov, webm)
- Configurable sample rate and channels
- Progress tracking and error reporting
- Creates output directory if needed

#### setup_env.sh

Complete automated setup of the development environment. This script:

1. Installs system packages (build tools, ffmpeg, etc.)
2. Installs Python 3.12
3. Installs `uv` package manager
4. Creates virtual environment and installs dependencies from `pyproject.toml`
5. Generates gRPC/protobuf Python code
6. Optionally installs dev tools (pre-commit, ruff) with `--dev`
7. Optionally installs Docker and NVIDIA GPU drivers

**Usage:**
```bash
./scripts/misc/setup_env.sh [--no-docker] [--no-gpu] [--dev] [--docs]
```

**Options:**
- `--no-docker` — Skip Docker and NVIDIA Container Toolkit installation
- `--no-gpu` — Skip NVIDIA GPU driver and CUDA toolkit installation
- `--dev` — Install development dependencies (lint, pre-commit)
- `--docs` — Install documentation build dependencies

**Requirements:**
- Internet connection for package downloads
- Ubuntu 22.04 or 24.04 recommended

#### copy_docker_logs.sh

Copy logs from Docker containers to local files for debugging and sharing.

**Usage:**
```bash
# Copy all service logs
./scripts/misc/copy_docker_logs.sh

# Copy specific service logs
./scripts/misc/copy_docker_logs.sh s2s
./scripts/misc/copy_docker_logs.sh controller
```

**Output:** Logs saved to `./logs/` directory with filenames like `s2s.log`, `controller.log`, etc.

### Diarization Scripts

These scripts generate diarization data (speaker segmentation) from audio files, producing JSON files that can be passed to the ASD client or Controller client via `--diarization-file`.

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/elevenlabs/diarize.py` | Generate diarization using ElevenLabs Scribe STT | `ELEVENLABS_API_KEY=<key> python scripts/elevenlabs/diarize.py --input-file audio.wav` |
| `scripts/camb/diarize.py` | Generate source-language diarization using Camb AI | `CAMB_API_KEY=<key> python scripts/camb/diarize.py --input-file audio.wav` |

#### scripts/elevenlabs/diarize.py

Generate diarization data using the ElevenLabs Speech-to-Text (Scribe) API. Outputs native ElevenLabs STT JSON format.

**Usage:**
```bash
ELEVENLABS_API_KEY=<key> python scripts/elevenlabs/diarize.py \
    --input-file audio.wav \
    --output-file diarization.json
```

**Arguments:**
- `--input-file` - Path to audio file (WAV, MP3, etc.) (required)
- `--output-file` - Path to output JSON file (default: `diarization.json`)
- `--language-code` - Language code (default: auto-detect)
- `--max-speakers` - Maximum number of speakers (default: model default)
- `--model-id` - Scribe model ID (default: `scribe_v2`)
- `--tag-audio-events` - Tag audio events such as (laughter) or (footsteps) in the transcript (flag, default off)

**Requirements:**
- `ELEVENLABS_API_KEY` environment variable

### Standalone Dubbing Scripts

These scripts perform end-to-end dubbing outside of the gRPC service pipeline, using cloud dubbing APIs directly.

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/elevenlabs/s2s_infer.py` | ElevenLabs end-to-end dubbing | `ELEVENLABS_API_KEY=<key> python scripts/elevenlabs/s2s_infer.py --input-file assets/sample_audio.wav --source-language-code en --target-language-code es -o output.wav` |
| `scripts/camb/s2s_infer.py` | CAMB end-to-end dubbing | `CAMB_API_KEY=<key> python scripts/camb/s2s_infer.py --input-file assets/sample_audio.wav --source-language 1 --target-language 54 -o output.mp3` |
| `scripts/elevenlabs/invoke_e2e.sh` | Wrapper for ElevenLabs E2E dubbing | `./scripts/elevenlabs/invoke_e2e.sh` |
| `scripts/camb/invoke_e2e.sh` | Wrapper for CAMB E2E dubbing | `./scripts/camb/invoke_e2e.sh` |

#### scripts/elevenlabs/s2s_infer.py

Invoke ElevenLabs end-to-end dubbing for local media files. Extracts audio from video, submits a dubbing request, and downloads the translated audio.

**Usage:**
```bash
ELEVENLABS_API_KEY=<key> python scripts/elevenlabs/s2s_infer.py \
    --input-file assets/sample_audio.wav \
    --source-language-code en \
    --target-language-code es \
    --output-file output.wav \
    --source-transcript-output-file source_transcript.json \
    --target-transcript-output-file target_transcript.json \
    --transcript-format json
```

Transcript outputs are optional. ElevenLabs supports `json`, `srt`, and `webvtt` formats;
`vtt` is accepted as an alias for `webvtt`. Use ElevenLabs Dubbing API JSON outputs
as ASD, Controller, or Direct diarization inputs with
`--diarization-format elevenlabs-dubbing-api`. See
[docs/source/diarization_formats.rst](docs/source/diarization_formats.rst) for the
full list of supported diarization formats and their schemas.

**Requirements:**
- `ELEVENLABS_API_KEY` environment variable
- `ffmpeg` installed (for video-to-audio extraction)

#### scripts/camb/s2s_infer.py

Invoke CAMB end-to-end dubbing for a local file or a public media URL. Submits a dubbing request, polls for completion, and downloads the translated audio. Pass exactly one of `--input-file` (uploaded via CAMB's `/files/upload-url` endpoint, requires the Files API to be enabled on your CAMB account) or `--input-url` (public media URL sent as `video_url`).

CAMB.AI uses **integer language IDs** (e.g. `1` = English, `54` = Spanish). To get the full mapping, query the CambAI API or see the [source languages](https://docs.camb.ai/api-reference/endpoint/get-source-languages) and [target languages](https://docs.camb.ai/api-reference/endpoint/get-target-languages) docs.

**Usage:**
```bash
CAMB_API_KEY=<key> python scripts/camb/s2s_infer.py \
    --input-file assets/sample_audio.wav \
    --source-language 1 \
    --target-language 54 \
    --output-file output.mp3 \
    --target-transcript-output-file target_transcript.json \
    --transcript-format json
```

Transcript outputs are optional and are always in the **target** language —
CAMB's dubbing API does not return a source-language transcript. JSON
writes the target-language diarized transcript from the `dub-result`
payload, and `txt`, `srt`, and `vtt` come from CAMB's formatted transcript
endpoint (also keyed on the target language). These can be passed to
clients with `--diarization-format camb` when target-language diarization
is acceptable. For source-language CAMB diarization (e.g. as ASD input),
use `scripts/camb/diarize.py` (Camb AI Transcription API).

**Requirements:**
- `CAMB_API_KEY` environment variable

#### deploy_asd.sh

Deploy the Active Speaker Detection (ASD) NIM container for standalone testing.

**Usage:**
```bash
./scripts/nims/deploy_asd.sh
```

**Features:**
- Deploys ASD NIM container with GPU support
- Configures ports: HTTP (`ASD_NIM_HTTP_API_PORT`, default 8005) and gRPC (`ASD_GRPC_API_PORT`, default 50055)
- Mounts model cache at `volumes/models/asd/`
- Requires `ASD_API_KEY` environment variable

---
