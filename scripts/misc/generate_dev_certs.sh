#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate a throwaway development CA plus per-service TLS certificates for
# testing the TLS/mTLS surface of the content localization services.
#
# DEVELOPMENT AND TESTING ONLY. Production deployments must use certificates
# issued by the organization's certificate authority or service mesh; never
# ship or reuse the material produced by this script.
#
# Usage:
#   bash scripts/misc/generate_dev_certs.sh [output-dir]
#
#   output-dir  Where to write the PEM files (default: ./certs)
#   DAYS        Validity in days (env var, default: 365)
#
# Produces, matching the placeholder paths in configs/*.env:
#   root.pem                    CA root certificate  (CONTROLLER_NIM_SSL_ROOT_CERT)
#   client.key / client.pem     Client identity for mTLS
#                               (CONTROLLER_NIM_SSL_KEY / CONTROLLER_NIM_SSL_CERT)
#   <service>.key / <service>.pem
#                               Server key/cert per service (controller, s2s,
#                               asd, lipsync), SANs cover the compose hostname,
#                               localhost, and 127.0.0.1.
set -euo pipefail

OUT_DIR="${1:-certs}"
DAYS="${DAYS:-365}"
SERVICES=(controller s2s asd lipsync)

mkdir -p "$OUT_DIR"

echo "Generating development CA in $OUT_DIR (validity: $DAYS days)"
openssl req -x509 -newkey rsa:4096 -nodes -sha256 \
    -keyout "$OUT_DIR/ca.key" \
    -out "$OUT_DIR/root.pem" \
    -days "$DAYS" \
    -subj "/CN=h4m-ebu-dev-ca"

issue_cert() {
    local name="$1"
    local san="$2"
    openssl req -newkey rsa:4096 -nodes -sha256 \
        -keyout "$OUT_DIR/$name.key" \
        -out "$OUT_DIR/$name.csr" \
        -subj "/CN=$name"
    openssl x509 -req -sha256 \
        -in "$OUT_DIR/$name.csr" \
        -CA "$OUT_DIR/root.pem" \
        -CAkey "$OUT_DIR/ca.key" \
        -CAcreateserial \
        -out "$OUT_DIR/$name.pem" \
        -days "$DAYS" \
        -extfile <(printf "subjectAltName=%s" "$san")
    rm "$OUT_DIR/$name.csr"
}

for service in "${SERVICES[@]}"; do
    echo "Issuing server certificate: $service"
    issue_cert "$service" "DNS:$service,DNS:localhost,IP:127.0.0.1"
done

echo "Issuing client certificate for mTLS: client"
issue_cert "client" "DNS:localhost"

chmod 600 "$OUT_DIR"/*.key

echo
echo "Done. Development certificates written to $OUT_DIR/"
echo
echo "Controller -> NIM channel TLS (mount $OUT_DIR at /certs in compose):"
echo "  CONTROLLER_NIM_SSL_MODE=TLS"
echo "  CONTROLLER_NIM_SSL_ROOT_CERT=/certs/root.pem"
echo "  CONTROLLER_S2S_SSL_MODE=DISABLED   # the in-repo S2S serves plaintext only"
echo "For mTLS, additionally:"
echo "  CONTROLLER_NIM_SSL_MODE=MTLS"
echo "  CONTROLLER_NIM_SSL_KEY=/certs/client.key"
echo "  CONTROLLER_NIM_SSL_CERT=/certs/client.pem"
