# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real TLS/mTLS handshakes against a local gRPC server.

The other TLS tests mock ``grpc.ssl_channel_credentials`` and verify the
configuration wiring only; this module proves the cryptographic path:
ephemeral certificates are generated at test time and
``create_channel_credentials`` output must complete an actual handshake
with a TLS-terminated in-process gRPC server.
"""

import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from common.tls import create_channel_credentials

pytestmark = pytest.mark.integration

_HANDSHAKE_TIMEOUT_SECS = 5.0
_ONE_DAY = datetime.timedelta(days=1)


def _new_key() -> rsa.RSAPrivateKey:
    """Generate a throwaway RSA key (2048 bits keeps test runtime low)."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _key_pem(key: rsa.RSAPrivateKey) -> bytes:
    """Serialize a private key as unencrypted PEM bytes."""
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_builder(subject_cn: str, issuer_cn: str, public_key) -> x509.CertificateBuilder:
    """Build the shared skeleton of a short-lived test certificate."""
    now = datetime.datetime.now(datetime.UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_cn)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_cn)]))
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _ONE_DAY)
        .not_valid_after(now + _ONE_DAY)
    )


def _make_ca(common_name: str) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Create a self-signed test CA."""
    key = _new_key()
    cert = (
        _cert_builder(subject_cn=common_name, issuer_cn=common_name, public_key=key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    return cert, key


def _issue_cert(
    ca_cert: x509.Certificate,
    ca_key: rsa.RSAPrivateKey,
    common_name: str,
) -> tuple[bytes, bytes]:
    """Issue a leaf certificate for ``localhost`` signed by the test CA.

    Args:
        ca_cert (x509.Certificate): Issuing CA certificate.
        ca_key (rsa.RSAPrivateKey): Issuing CA private key.
        common_name (str): Subject common name for the leaf.

    Returns:
        tuple[bytes, bytes]: ``(cert_pem, key_pem)`` for the new identity.
    """
    key = _new_key()
    issuer_cn = ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    cert = (
        _cert_builder(subject_cn=common_name, issuer_cn=issuer_cn, public_key=key.public_key())
        .add_extension(
            # gRPC verifies the dialed host against the SAN, so every leaf
            # covers localhost — the only host these tests connect to.
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM), _key_pem(key)


class _TlsFixture:
    """PEM files for one CA plus a server and client identity it issued."""

    def __init__(self, directory: Path, ca_name: str) -> None:
        ca_cert, ca_key = _make_ca(common_name=ca_name)
        self.root_pem = directory / f"{ca_name}-root.pem"
        self.root_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

        server_cert, server_key = _issue_cert(
            ca_cert=ca_cert, ca_key=ca_key, common_name="localhost"
        )
        self.server_cert = server_cert
        self.server_key = server_key

        client_cert, client_key = _issue_cert(
            ca_cert=ca_cert, ca_key=ca_key, common_name="test-client"
        )
        self.client_cert_pem = directory / f"{ca_name}-client.pem"
        self.client_cert_pem.write_bytes(client_cert)
        self.client_key_pem = directory / f"{ca_name}-client.key"
        self.client_key_pem.write_bytes(client_key)


class TestRealTlsHandshake:
    """create_channel_credentials output must complete real handshakes."""

    @pytest.fixture()
    def tls(self, tmp_path: Path) -> _TlsFixture:
        """CA + identities the server trusts."""
        return _TlsFixture(directory=tmp_path, ca_name="trusted")

    @pytest.fixture()
    def other_ca(self, tmp_path: Path) -> _TlsFixture:
        """A second, unrelated CA for negative tests."""
        return _TlsFixture(directory=tmp_path, ca_name="untrusted")

    def _serve(self, tls: _TlsFixture, require_client_auth: bool) -> tuple[grpc.Server, int]:
        """Start a local TLS-terminated gRPC server on an ephemeral port."""
        server = grpc.server(ThreadPoolExecutor(max_workers=1))
        credentials = grpc.ssl_server_credentials(
            [(tls.server_key, tls.server_cert)],
            root_certificates=tls.root_pem.read_bytes(),
            require_client_auth=require_client_auth,
        )
        port = server.add_secure_port("localhost:0", credentials)
        server.start()
        return server, port

    @staticmethod
    def _client_args(tls: _TlsFixture, ssl_mode: str) -> argparse.Namespace:
        """Build the argparse surface create_channel_credentials expects."""
        return argparse.Namespace(
            ssl_mode=ssl_mode,
            ssl_root_cert=str(tls.root_pem),
            ssl_key=str(tls.client_key_pem),
            ssl_cert=str(tls.client_cert_pem),
        )

    def _assert_channel_ready(self, port: int, credentials: grpc.ChannelCredentials) -> None:
        """The channel must reach READY, which requires a completed handshake."""
        with grpc.secure_channel(f"localhost:{port}", credentials) as channel:
            grpc.channel_ready_future(channel).result(timeout=_HANDSHAKE_TIMEOUT_SECS)

    def test_tls_handshake_succeeds_with_trusted_root(self, tls: _TlsFixture) -> None:
        """TLS mode completes a handshake when the root matches the server."""
        server, port = self._serve(tls=tls, require_client_auth=False)
        try:
            credentials = create_channel_credentials(args=self._client_args(tls, "TLS"))
            self._assert_channel_ready(port=port, credentials=credentials)
        finally:
            server.stop(grace=None)

    def test_tls_handshake_fails_with_wrong_root(
        self, tls: _TlsFixture, other_ca: _TlsFixture
    ) -> None:
        """A root from a different CA must not verify the server."""
        server, port = self._serve(tls=tls, require_client_auth=False)
        try:
            credentials = create_channel_credentials(args=self._client_args(other_ca, "TLS"))
            with pytest.raises(grpc.FutureTimeoutError):
                self._assert_channel_ready(port=port, credentials=credentials)
        finally:
            server.stop(grace=None)

    def test_mtls_handshake_succeeds_with_client_identity(self, tls: _TlsFixture) -> None:
        """MTLS mode presents the client certificate the server requires."""
        server, port = self._serve(tls=tls, require_client_auth=True)
        try:
            credentials = create_channel_credentials(args=self._client_args(tls, "MTLS"))
            self._assert_channel_ready(port=port, credentials=credentials)
        finally:
            server.stop(grace=None)

    def test_mtls_server_rejects_client_without_certificate(self, tls: _TlsFixture) -> None:
        """A TLS-only client cannot reach READY against an mTLS server."""
        server, port = self._serve(tls=tls, require_client_auth=True)
        try:
            credentials = create_channel_credentials(args=self._client_args(tls, "TLS"))
            with pytest.raises(grpc.FutureTimeoutError):
                self._assert_channel_ready(port=port, credentials=credentials)
        finally:
            server.stop(grace=None)
