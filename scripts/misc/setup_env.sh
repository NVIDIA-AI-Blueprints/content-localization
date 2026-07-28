#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bootstrap script for setting up the Content Localization Blueprint
# development environment from a fresh Ubuntu installation.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh [--no-docker] [--no-gpu] [--dev] [--docs]
#
# Options:
#   --no-docker   Skip Docker and NVIDIA Container Toolkit installation
#   --no-gpu      Skip NVIDIA GPU driver and CUDA toolkit installation
#   --dev         Install development dependencies (lint, pre-commit)
#   --docs        Install documentation build dependencies

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────── #
#                              Parse Arguments                                 #
# ──────────────────────────────────────────────────────────────────────────── #

INSTALL_DOCKER=true
INSTALL_GPU=true
INSTALL_DEV=false
INSTALL_DOCS=false

for arg in "$@"; do
    case "$arg" in
        --no-docker) INSTALL_DOCKER=false ;;
        --no-gpu)    INSTALL_GPU=false ;;
        --dev)       INSTALL_DEV=true ;;
        --docs)      INSTALL_DOCS=true ;;
        --help|-h)
            sed -n '8,15p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_VERSION="3.12"

# ──────────────────────────────────────────────────────────────────────────── #
#                                Helpers                                       #
# ──────────────────────────────────────────────────────────────────────────── #

info()  { echo -e "\n\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

command_exists() { command -v "$1" &>/dev/null; }

# ──────────────────────────────────────────────────────────────────────────── #
#                        1. System Packages (apt)                              #
# ──────────────────────────────────────────────────────────────────────────── #

install_system_packages() {
    info "Installing system packages..."

    sudo apt-get update -qq

    # Core build tools and libraries needed by Python packages
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        git \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        pkg-config \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        tk-dev \
        libgdbm-dev \
        libnss3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        ffmpeg \
        unzip

    ok "System packages installed."
}

# ──────────────────────────────────────────────────────────────────────────── #
#                           2. Python 3.12                                     #
# ──────────────────────────────────────────────────────────────────────────── #

install_python() {
    if command_exists "python${PYTHON_VERSION}"; then
        ok "Python ${PYTHON_VERSION} already installed: $(python${PYTHON_VERSION} --version)"
        return
    fi

    info "Installing Python ${PYTHON_VERSION}..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-distutils"

    ok "Python ${PYTHON_VERSION} installed: $(python${PYTHON_VERSION} --version)"
}

# ──────────────────────────────────────────────────────────────────────────── #
#                              3. uv                                           #
# ──────────────────────────────────────────────────────────────────────────── #

install_uv() {
    if command_exists uv; then
        ok "uv already installed: $(uv --version)"
        return
    fi

    info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the env so uv is available in this session
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed: $(uv --version)"
}

# ──────────────────────────────────────────────────────────────────────────── #
#                     4. Virtual Environment & Dependencies                    #
# ──────────────────────────────────────────────────────────────────────────── #

setup_venv() {
    info "Creating Python virtual environment at ${REPO_ROOT}/.venv ..."
    cd "$REPO_ROOT"

    # Create venv with the correct Python version
    uv venv --python "python${PYTHON_VERSION}" .venv

    # Activate for the rest of this script
    # shellcheck disable=SC1091
    source .venv/bin/activate

    ok "Virtual environment created ($(python --version))."

    # Install project dependencies via uv sync (respects pyproject.toml + uv.lock)
    info "Installing project dependencies with uv sync..."

    local extras=""
    if [ "$INSTALL_DEV" = true ]; then
        extras="$extras --extra lint --extra test"
    else
        extras="$extras --extra test"
    fi
    if [ "$INSTALL_DOCS" = true ]; then
        extras="$extras --extra docs"
    fi

    # uv sync installs the project and its dependencies from uv.lock
    # shellcheck disable=SC2086
    uv sync $extras

    ok "Python dependencies installed."
}

# ──────────────────────────────────────────────────────────────────────────── #
#                        5. Generate Protobuf Code                             #
# ──────────────────────────────────────────────────────────────────────────── #

generate_protos() {
    info "Generating gRPC/protobuf Python code..."
    cd "$REPO_ROOT"
    source .venv/bin/activate
    bash protos/generate_protos.sh
    ok "Protobuf code generated in protos/generated/."
}

# ──────────────────────────────────────────────────────────────────────────── #
#                          6. Pre-commit Hooks                                 #
# ──────────────────────────────────────────────────────────────────────────── #

setup_precommit() {
    if [ "$INSTALL_DEV" != true ]; then
        return
    fi

    info "Installing pre-commit hooks..."
    cd "$REPO_ROOT"
    source .venv/bin/activate
    pre-commit install
    ok "Pre-commit hooks installed."
}

# ──────────────────────────────────────────────────────────────────────────── #
#                      7. Docker & Docker Compose                              #
# ──────────────────────────────────────────────────────────────────────────── #

install_docker() {
    if [ "$INSTALL_DOCKER" != true ]; then
        warn "Skipping Docker installation (--no-docker)."
        return
    fi

    if command_exists docker; then
        ok "Docker already installed: $(docker --version)"
    else
        info "Installing Docker Engine..."

        # Add Docker's official GPG key and repo
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg

        echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
            https://download.docker.com/linux/ubuntu \
            $(lsb_release -cs) stable" \
            | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        sudo apt-get update -qq
        sudo apt-get install -y \
            docker-ce \
            docker-ce-cli \
            containerd.io \
            docker-buildx-plugin \
            docker-compose-plugin

        # Allow current user to use Docker without sudo
        sudo usermod -aG docker "$USER"
        ok "Docker installed. Log out and back in for group changes to take effect."
    fi
}

# ──────────────────────────────────────────────────────────────────────────── #
#                  8. NVIDIA GPU Drivers & Container Toolkit                   #
# ──────────────────────────────────────────────────────────────────────────── #

install_nvidia_gpu() {
    if [ "$INSTALL_GPU" != true ]; then
        warn "Skipping NVIDIA GPU setup (--no-gpu)."
        return
    fi

    # NVIDIA drivers
    if command_exists nvidia-smi; then
        ok "NVIDIA driver already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    else
        info "Installing NVIDIA GPU drivers..."
        sudo apt-get install -y nvidia-driver-560
        warn "NVIDIA driver installed. A reboot is required before GPU is usable."
    fi

    # NVIDIA Container Toolkit (for Docker GPU passthrough)
    if [ "$INSTALL_DOCKER" = true ]; then
        if command_exists nvidia-ctk; then
            ok "NVIDIA Container Toolkit already installed."
        else
            info "Installing NVIDIA Container Toolkit..."

            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

            sudo apt-get update -qq
            sudo apt-get install -y nvidia-container-toolkit

            # Configure Docker to use the NVIDIA runtime
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker || true

            ok "NVIDIA Container Toolkit installed and configured."
        fi
    fi
}

# ──────────────────────────────────────────────────────────────────────────── #
#                           9. Environment File                                #
# ──────────────────────────────────────────────────────────────────────────── #

setup_env_file() {
    if [ -f "${REPO_ROOT}/.env" ]; then
        ok ".env file already exists — skipping."
        return
    fi

    info "Creating .env file from template..."
    cat > "${REPO_ROOT}/.env" <<'ENVEOF'
# API keys — fill in your actual keys before running services.
# This file is git-ignored and should never be committed.
NGC_API_KEY=
ELEVENLABS_API_KEY=
ENVEOF

    warn ".env created at ${REPO_ROOT}/.env — fill in your API keys before running services."
}

# ──────────────────────────────────────────────────────────────────────────── #
#                              10. Verify                                      #
# ──────────────────────────────────────────────────────────────────────────── #

verify_setup() {
    info "Verifying setup..."
    cd "$REPO_ROOT"
    source .venv/bin/activate

    # Quick import smoke test
    python -c "
import grpc
import cv2
import numpy
import scipy
import google.protobuf
print(f'grpc      {grpc.__version__}')
print(f'opencv    {cv2.__version__}')
print(f'numpy     {numpy.__version__}')
print(f'scipy     {scipy.__version__}')
print(f'protobuf  {google.protobuf.__version__}')
"

    # Verify proto generation
    python -c "
from nvidia.ai4m.controller.v1 import controller_pb2
from nvidia.ai4m.s2s.v1 import s2s_pb2
from nvidia.ai4m.lipsync.v1 import lipsync_pb2
from nvidia.ai4m.activespeakerdetection.v1 import activespeakerdetection_pb2
print('Proto imports OK')
" && ok "Proto imports verified." || warn "Proto imports failed — check protos/generate_protos.sh output."

    ok "Setup complete!"
}

# ──────────────────────────────────────────────────────────────────────────── #
#                                  Main                                        #
# ──────────────────────────────────────────────────────────────────────────── #

main() {
    echo "=============================================="
    echo " Content Localization Blueprint — Environment "
    echo "=============================================="
    echo ""
    echo "Options:"
    echo "  Docker:  $INSTALL_DOCKER"
    echo "  GPU:     $INSTALL_GPU"
    echo "  Dev:     $INSTALL_DEV"
    echo "  Docs:    $INSTALL_DOCS"
    echo ""

    install_system_packages
    install_python
    install_uv
    setup_venv
    generate_protos
    setup_precommit
    install_docker
    install_nvidia_gpu
    setup_env_file
    verify_setup

    echo ""
    echo "=============================================="
    echo " Next steps:                                  "
    echo "=============================================="
    echo ""
    echo "  1. Activate the environment:"
    echo "       source .venv/bin/activate"
    echo ""
    echo "  2. Set PYTHONPATH (for running outside Docker):"
    echo "       export PYTHONPATH=\"\${PYTHONPATH}:\${PWD}:\${PWD}/src:\${PWD}/client:\${PWD}/protos/generated\""
    echo ""
    echo "  3. Fill in API keys in .env:"
    echo "       NGC_API_KEY=<your-key>"
    echo "       ELEVENLABS_API_KEY=<your-key>"
    echo ""
    echo "  4. Run tests:"
    echo "       python -m pytest tests/"
    echo ""
    echo "  5. Run services with Docker:"
    echo "       docker compose --profile demo-app-third-party-s2s --env-file configs/elevenlabs.env --env-file .env up --build"
    echo ""
}

main "$@"
