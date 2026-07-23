# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM node:24-alpine AS base

# Step 1. Rebuild the source code only when needed
FROM base AS builder

# Build ffmpeg from source with LGPL-only, royalty-free codecs
COPY ./client/demos/build-ffmpeg.sh /build-ffmpeg.sh
RUN chmod +x /build-ffmpeg.sh && /build-ffmpeg.sh

RUN npm install -g npm@11.18.0

RUN apk add git

WORKDIR /app

# Install dependencies based on the preferred package manager
COPY ./client/demos/package.json ./client/demos/yarn.lock* ./client/demos/package-lock.json* ./client/demos/pnpm-lock.yaml* ./client/demos/.npmrc* ./

RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i; \
  # Allow install without lockfile, so example works even without Node.js installed locally
  else echo "Warning: Lockfile not found. It is recommended to commit lockfiles to version control." && npm install; \
  fi

COPY ./client/demos .

# Copy protos for build-time access if needed
COPY protos ./protos
COPY assets ./assets

# Environment variables must be present at build time
# https://github.com/vercel/next.js/discussions/14030
ARG ENV_VARIABLE
ENV ENV_VARIABLE=${ENV_VARIABLE}
ARG NEXT_PUBLIC_ENV_VARIABLE
ENV NEXT_PUBLIC_ENV_VARIABLE=${NEXT_PUBLIC_ENV_VARIABLE}
ARG NEXT_PUBLIC_INPUT_FILE_NAME
ENV NEXT_PUBLIC_INPUT_FILE_NAME=${NEXT_PUBLIC_INPUT_FILE_NAME}

# Next.js collects completely anonymous telemetry data about general usage. Learn more here: https://nextjs.org/telemetry
ENV NEXT_TELEMETRY_DISABLED 1

# Build Next.js based on the preferred package manager
RUN \
  if [ -f yarn.lock ]; then yarn build; \
  elif [ -f package-lock.json ]; then npm run build; \
  elif [ -f pnpm-lock.yaml ]; then pnpm build; \
  else npm run build; \
  fi

# Step 2. Production image, copy all the files and run next
FROM base AS runner

# Security: upgrade Alpine system packages to fix OpenSSL CVE-2026-2673
RUN apk upgrade --no-cache libssl3 libcrypto3

# Security: upgrade npm to 11.18.0 to fix bundled tar/minimatch/picomatch CVEs
RUN npm install -g npm@11.18.0

# Install runtime dependencies for ffmpeg (VP9 + Opus only)
RUN apk add --no-cache \
  libvpx \
  opus

WORKDIR /app

# Copy ffmpeg binaries and libraries from builder
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=builder /usr/local/lib/libav* /usr/local/lib/
COPY --from=builder /usr/local/lib/libsw* /usr/local/lib/
COPY --from=builder /usr/local/lib/libpostproc* /usr/local/lib/

# Alpine uses musl libc, which has no ldconfig. Tell the musl dynamic
# linker where ffmpeg's shared libraries live.
ENV LD_LIBRARY_PATH=/usr/local/lib

# Don't run production as root
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

RUN mkdir -p /app/outputs

# USER nextjs
COPY protos ./protos
COPY assets ./assets

# COPY --from=builder app/public ./public
COPY --from=builder app/package.json ./package.json
COPY --from=builder app/package-lock.json ./package-lock.json
COPY --from=builder app/dist ./dist
COPY --from=builder app/node_modules ./node_modules

# Copy ThirdPartyLicenses-UI.txt file
COPY --from=builder app/ThirdPartyLicenses-UI.txt ./ThirdPartyLicenses-UI.txt

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

CMD ["npm", "run", "start"]