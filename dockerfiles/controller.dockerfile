# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security fixes: try exact versions from guidance, else install latest available
RUN set -eux; \
    apt-get update; \
    install_version() { pkg="$1"; ver="$2"; if apt-cache madison "$pkg" | awk '{print $3}' | grep -x "$ver" >/dev/null; then apt-get install -y --no-install-recommends --allow-downgrades "$pkg=$ver"; else apt-get install -y --no-install-recommends "$pkg"; fi }; \
    install_version libc-bin 2.36-9+deb12u11; \
    install_version libc-dev-bin 2.36-9+deb12u11; \
    install_version libc6 2.36-9+deb12u11; \
    install_version libgnutls30 3.7.9-2+deb12u5; \
    apt-get install -y --no-install-recommends --allow-downgrades \
    libgssapi-krb5-2=1.20.1-2+deb12u3 \
    libk5crypto3=1.20.1-2+deb12u3 \
    libkrb5-3=1.20.1-2+deb12u3 \
    libkrb5support0=1.20.1-2+deb12u3 \
    || apt-get install -y --no-install-recommends \
    libgssapi-krb5-2 libk5crypto3 libkrb5-3 libkrb5support0; \
    install_version libsqlite3-0 3.40.1-2+deb12u2; \
    install_version libssl3 3.0.20-1~deb12u2; \
    install_version openssl 3.0.20-1~deb12u2; \
    install_version libsystemd0 252.23-1~deb12u1; \
    install_version libudev1 252.23-1~deb12u1; \
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
WORKDIR /opt/controller

# Install system dependencies
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
COPY pyproject.toml poetry.lock README.md /opt/controller/

# Install dependencies using poetry. Poetry 2.x fails when poetry.lock is out
# of sync with pyproject.toml, so this is a locked, reproducible install.
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --without gpu --no-root

# debugpy is intentionally NOT installed in the production image; the
# entrypoint installs it at container start when CONTROLLER_VS_CODE_DEBUG=1.

# Security: upgrade setuptools to fix CVE GHSA-5rjg-fvgr-3xxf
RUN pip install --no-cache-dir "setuptools>=78.1.1"



# Copy proto files and generate Python code
COPY protos/ /opt/controller/protos/
RUN cd /opt/controller/protos && chmod +x generate_protos.sh && ./generate_protos.sh

# Copy service code
COPY src/controller_service/ /opt/controller/controller_service/
COPY src/common/ /opt/controller/common/

# Set Python path to include service code and generated protos
ENV PYTHONPATH=/opt/controller:/opt/controller/protos/generated:${PYTHONPATH}

# Copy entrypoint script and make it executable
COPY src/docker_entrypoints/controller/entrypoint.sh /opt/controller/entrypoint.sh
RUN chmod +x /opt/controller/entrypoint.sh


# Cleaning up git, wget, and unused packages with known CVEs. wget is only used
# at build time to fetch the source tarballs above; nothing at runtime needs it.
RUN set -eux; \
    apt-get purge -y git git-man wget; \
    apt-get remove -y --allow-remove-essential libjs-underscore || true; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# EQUIVS METADATA PACKAGE — CVE-SCANNER VISIBILITY (must stay last)
# Register a metadata-only dpkg entry so container CVE scanners report the
# locally-built, patched libexpat version instead of the distro package. Built
# from an explicit control file (equivs-control leaves Version/Architecture
# commented, so the old sed-based approach produced unusable version-1.0
# packages). This MUST be the final apt-affecting step: the content-less shim
# satisfies reverse-deps via Provides but confuses apt dependency resolution for
# intervening installs, so equivs is purged before the shim is registered and no
# apt install runs afterward. The pinned version MUST match the source build above.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends equivs; \
    cd /tmp; \
    printf 'Section: libs\nPriority: optional\nStandards-Version: 3.9.2\nPackage: libexpat1\nVersion: 2.8.1-99\nArchitecture: amd64\nProvides: libexpat1 (= 2.8.1-99)\nReplaces: libexpat1\nDescription: Metadata package marking Expat 2.8.1 present (built from source)\n Registers the source-built libexpat so CVE scanners see the patched version.\n' > /tmp/libexpat1.ctl; \
    equivs-build /tmp/libexpat1.ctl; \
    apt-get purge -y equivs; \
    apt-get autoremove -y; \
    dpkg -i /tmp/libexpat1_2.8.1-99_amd64.deb; \
    apt-mark hold libexpat1; \
    rm -f /tmp/libexpat1.ctl /tmp/libexpat1_2.8.1-99_amd64.deb; \
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

# Set the entrypoint
ENTRYPOINT ["/opt/controller/entrypoint.sh"]
