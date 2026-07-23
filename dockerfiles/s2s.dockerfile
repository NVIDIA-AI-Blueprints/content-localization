# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


ARG BASE_IMAGE="python:3.13.14-slim-bookworm"

# ---------------------------------------------------------------------------
# grpcurl builder — published grpcurl release binaries (through v1.9.3) are
# compiled with Go 1.21.1 and ship grpc-go 1.61.0 / x/net 0.22 / x/oauth2 0.14,
# which together carry ~28 High/Critical Go CVEs (stdlib, x/net, x/oauth2, and
# grpc-go). Rebuilding grpcurl from source with a current Go toolchain and
# patched modules clears them. Only the compiled static binary is copied into
# the final image; the Go toolchain never ships. grpcurl is used solely by the
# docker-compose healthcheck (grpc.health.v1.Health/Check on localhost), so its
# grpc-go version is independent of the Python grpcio the services speak.
# ---------------------------------------------------------------------------
FROM golang:1.26-bookworm AS grpcurl-builder
ARG GRPCURL_VERSION=v1.9.3
RUN set -eux; \
    git clone --depth 1 -b "${GRPCURL_VERSION}" https://github.com/fullstorydev/grpcurl /src; \
    cd /src; \
    go get golang.org/x/net@v0.57.0; \
    go get golang.org/x/oauth2@v0.36.0; \
    go get google.golang.org/grpc@v1.82.0; \
    go mod tidy; \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/grpcurl ./cmd/grpcurl

FROM ${BASE_IMAGE} AS final

ENV DEBIAN_FRONTEND=noninteractive 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 

# suppress warning about installing as root (pip 22+)
ENV PIP_ROOT_USER_ACTION=ignore 

# Keeps Python from buffering stdout and stderr to avoid situations where the application crashes
# without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Fail piped RUN commands (e.g. sha256sum verification) on the first error.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Security: install the patched OpenSSL/libssl security versions. Try the exact
# version from the CVE guidance, but fall back to the latest available build
# when that exact revision has been rotated out of the Debian mirror. A hard
# pin here breaks the build the moment Debian publishes a newer point release
# (see NVBug 6446229), while the latest available is always >= the pinned fix.
RUN set -eux; \
    apt-get update; \
    install_version() { pkg="$1"; ver="$2"; if apt-cache madison "$pkg" | awk '{print $3}' | grep -x "$ver" >/dev/null; then apt-get install -y --no-install-recommends --allow-downgrades "$pkg=$ver"; else apt-get install -y --no-install-recommends "$pkg"; fi }; \
    install_version libssl3 3.0.20-1~deb12u2; \
    install_version openssl 3.0.20-1~deb12u2; \
    rm -rf /var/lib/apt/lists/*

# The 3.13 slim base already provides /usr/local/bin/python(3) and pip, so no
# distro python3 is installed — that keeps the distro libpython3.11 (and its
# unfixed CVEs) out of the image. Only the base interpreter's pip and setuptools
# are upgraded here.
RUN python3 -m pip install --upgrade pip setuptools

# Mitigate CVE-2025-59375 and CVE-2026-45186: build libexpat >= 2.8.1 from source
# into the multiarch libdir so the patched library replaces the vulnerable distro
# one at runtime, then hold the distro package so later apt operations cannot
# reinstall the old shared object over ours. The equivs metadata package that
# makes CVE scanners report the patched version is registered in the final
# consolidated block near the end of this file (after all other apt work), since
# a content-less shim confuses apt dependency resolution for intervening installs.
# libexpat1 itself is installed explicitly here: git (installed later) depends on
# it, and the distro python3 that used to pull it in has been removed.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential ca-certificates wget xz-utils libexpat1; \
    wget -O /tmp/expat.tar.xz https://github.com/libexpat/libexpat/releases/download/R_2_8_1/expat-2.8.1.tar.xz; \
    echo "10b195ee78160a908388180a8fe3603d4e9a12f4755fbf5f3816b23a9d750da0  /tmp/expat.tar.xz" | sha256sum -c -; \
    tar -C /tmp -xf /tmp/expat.tar.xz; \
    cd /tmp/expat-2.8.1; \
    ./configure --prefix=/usr --libdir=/usr/lib/x86_64-linux-gnu; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    apt-mark hold libexpat1; \
    cd /; \
    rm -rf /tmp/expat-2.8.1 /tmp/expat.tar.xz; \
    apt-get purge -y build-essential; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/s2s/

# Install system dependencies
# Installing ffmpeg should be fine, since its OSS.
# Install grpcurl inside the container to run health checks.
RUN set -eux; \
    echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/backports.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends wget git git-man -t bookworm-backports; \
    rm -rf /var/lib/apt/lists/*
# grpcurl (compose healthcheck client) built from source in the grpcurl-builder
# stage above with a current Go toolchain and patched modules.
COPY --from=grpcurl-builder /out/grpcurl /usr/local/bin/grpcurl

# Install poetry, pinned to the version that generated poetry.lock so the
# locked install resolves identically across builds.
RUN pip install --no-cache-dir poetry==2.4.1

# Configure poetry to install packages in system Python
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy poetry configuration files (lock file included for reproducible installs)
COPY pyproject.toml poetry.lock README.md /opt/s2s/

# Install dependencies using poetry. Poetry 2.x fails when poetry.lock is out
# of sync with pyproject.toml, so this is a locked, reproducible install.
RUN poetry install --without dev --without gpu --no-root

# Copy proto files
COPY protos /opt/s2s/protos
RUN cd /opt/s2s/protos && chmod +x generate_protos.sh && ./generate_protos.sh

# Copy service code
COPY src/s2s_service/ /opt/s2s/s2s_service/
COPY src/common/ /opt/s2s/common/

# Set Python path to include service code and generated protos
ENV PYTHONPATH=/opt/s2s:/opt/s2s/protos/generated:${PYTHONPATH}

# Copy entrypoint script and make it executable
COPY src/docker_entrypoints/s2s/entrypoint.sh /opt/s2s/entrypoint.sh
RUN chmod +x /opt/s2s/entrypoint.sh

# Security: build libtiff 4.7.1 from source to fix CVE-2026-4775
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential cmake ca-certificates wget libjpeg-dev zlib1g-dev; \
    wget -O /tmp/tiff-4.7.1.tar.gz https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz; \
    echo "f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69  /tmp/tiff-4.7.1.tar.gz" | sha256sum -c -; \
    tar -C /tmp -xzf /tmp/tiff-4.7.1.tar.gz; \
    cd /tmp/tiff-4.7.1; \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build; \
    ldconfig; \
    apt-mark hold libtiff6; \
    cd /; \
    rm -rf /tmp/tiff-4.7.1 /tmp/tiff-4.7.1.tar.gz; \
    apt-get purge -y build-essential cmake; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Security: build libde265 1.1.0 from source to fix CVE-2026-33164,
# CVE-2026-49295, and CVE-2026-49346
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential cmake ca-certificates wget; \
    wget -O /tmp/libde265-1.1.0.tar.gz https://github.com/strukturag/libde265/releases/download/v1.1.0/libde265-1.1.0.tar.gz; \
    echo "afc19dd28e2fc523de5952bba5224ee1d28e286c72436d2843df126cca1181fd  /tmp/libde265-1.1.0.tar.gz" | sha256sum -c -; \
    tar -C /tmp -xzf /tmp/libde265-1.1.0.tar.gz; \
    cd /tmp/libde265-1.1.0; \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build; \
    ldconfig; \
    apt-mark hold libde265-0; \
    cd /; \
    rm -rf /tmp/libde265-*; \
    apt-get purge -y build-essential cmake; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Security: build libheif 1.22.0 from source to fix CVE-2026-41071,
# CVE-2026-32882, CVE-2026-32741, and CVE-2026-32740. HEIF decoding uses the
# source-built libde265 above; AV1/x265 encoders are disabled since they are
# not needed and would pull additional CVE-bearing codec dependencies.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential cmake ca-certificates wget; \
    wget -O /tmp/libheif-1.22.0.tar.gz https://github.com/strukturag/libheif/releases/download/v1.22.0/libheif-1.22.0.tar.gz; \
    echo "8bd20cfa3201997b8f63266cddfabea2e1481467d7f992e6a2595e0bec691fc2  /tmp/libheif-1.22.0.tar.gz" | sha256sum -c -; \
    tar -C /tmp -xzf /tmp/libheif-1.22.0.tar.gz; \
    cd /tmp/libheif-1.22.0; \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
        -DWITH_EXAMPLES=OFF -DWITH_AOM_ENCODER=OFF -DWITH_AOM_DECODER=OFF \
        -DWITH_X265=OFF; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build; \
    ldconfig; \
    apt-mark hold libheif1; \
    cd /; \
    rm -rf /tmp/libheif-*; \
    apt-get purge -y build-essential cmake; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Cleaning up git, wget, and unused packages with known CVEs. wget is only used
# at build time to fetch the source tarballs above; nothing at runtime needs it.
# linux-libc-dev/libc6-dev are kernel-header build cruft pulled in by the
# source-built codec libraries above; nothing at runtime needs kernel headers,
# and they carry a large volume of non-exploitable-in-container kernel CVEs, so
# they are purged here after all compilation is complete.
RUN set -eux; \
    apt-get purge -y git git-man wget; \
    apt-get remove -y libjs-underscore || true; \
    apt-get purge -y linux-libc-dev libc6-dev || true; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# EQUIVS METADATA PACKAGES — CVE-SCANNER VISIBILITY (must stay last)
# Register metadata-only dpkg entries so container CVE scanners report the
# locally-built, patched libexpat, libde265, libtiff, and libheif versions
# instead of the distro packages. Each shim is built from an explicit control
# file (equivs-control leaves Version/Architecture commented, so the old
# sed-based approach silently produced unusable version-1.0 packages). This
# block MUST be the final apt-affecting step: the content-less shims satisfy
# reverse-deps via Provides but confuse apt dependency resolution, so equivs is
# purged before the shims are registered and no apt install runs afterward. The
# pinned versions here MUST match the source tarball versions built above.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends equivs; \
    cd /tmp; \
    for spec in "libexpat1 2.8.1-99 libexpat" "libde265-0 1.1.0-99 libde265" "libtiff6 4.7.1-99 libtiff" "libheif1 1.22.0-99 libheif"; do \
        set -- $spec; pkg="$1"; ver="$2"; label="$3"; \
        printf 'Section: libs\nPriority: optional\nStandards-Version: 3.9.2\nPackage: %s\nVersion: %s\nArchitecture: amd64\nProvides: %s (= %s)\nReplaces: %s\nDescription: Metadata package marking %s %s present (built from source)\n Registers the source-built library so CVE scanners see the patched version.\n' \
            "$pkg" "$ver" "$pkg" "$ver" "$pkg" "$label" "${ver%-99}" > "/tmp/$pkg.ctl"; \
        equivs-build "/tmp/$pkg.ctl"; \
    done; \
    apt-get purge -y equivs; \
    apt-get autoremove -y; \
    for spec in "libexpat1 2.8.1-99" "libde265-0 1.1.0-99" "libtiff6 4.7.1-99" "libheif1 1.22.0-99"; do \
        set -- $spec; pkg="$1"; ver="$2"; \
        dpkg -i "/tmp/${pkg}_${ver}_amd64.deb"; \
        apt-mark hold "$pkg"; \
        rm -f "/tmp/$pkg.ctl" "/tmp/${pkg}_${ver}_amd64.deb"; \
    done; \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# PERL REMOVAL — clears CVE-2026-12087 (Critical), CVE-2026-48962, and
# CVE-2026-48959. Debian ships no fixed perl for bookworm, and nothing here
# needs it at runtime: the entrypoint is bash, the service is Python, grpcurl is
# a static Go binary, and dpkg/apt are C. perl-base is an Essential package, so
# it is purged directly via dpkg with the essential/depends force flags. This
# MUST run last — equivs-build above needs perl, and no apt/dpkg install runs
# afterward. The trailing check fails the build if a perl interpreter survives.
# ---------------------------------------------------------------------------
RUN set -eux; \
    pkgs=""; \
    for p in perl perl-modules-5.36 libperl5.36 perl-base; do \
        if dpkg -s "$p" >/dev/null 2>&1; then pkgs="$pkgs $p"; fi; \
    done; \
    if [ -n "$pkgs" ]; then \
        dpkg --purge --force-remove-essential --force-depends $pkgs; \
    fi; \
    ! command -v perl >/dev/null 2>&1; \
    rm -rf /var/lib/apt/lists/*

# Run the service
ENTRYPOINT ["/opt/s2s/entrypoint.sh"]
