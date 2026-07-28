# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build ffmpeg from source with LGPL-only, royalty-free codecs
# This ensures no GPL or patent-encumbered codecs are included

set -e

echo "Installing build dependencies..."
apk add --no-cache \
    build-base \
    coreutils \
    wget \
    tar \
    xz \
    pkgconf \
    yasm \
    nasm \
    libvpx-dev \
    opus-dev

echo "Downloading ffmpeg 8.1.2..."
cd /tmp
wget https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz
echo "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c  ffmpeg-8.1.2.tar.xz" | sha256sum -c -
tar xf ffmpeg-8.1.2.tar.xz
cd ffmpeg-8.1.2

echo "Configuring ffmpeg with LGPL-only, royalty-free codecs (VP9 + Opus only)..."
./configure \
    --prefix=/usr/local \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --disable-debug \
    --enable-small \
    --disable-gpl \
    --disable-nonfree \
    --enable-libvpx \
    --enable-libopus

echo "Building ffmpeg (this may take a while)..."
make -j$(nproc)

echo "Installing ffmpeg..."
make install

echo "Cleaning up..."
cd /
rm -rf /tmp/ffmpeg-8.1.2*

echo "ffmpeg installation complete!"
ffmpeg -version
