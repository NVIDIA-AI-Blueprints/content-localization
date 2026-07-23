# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for client utils module."""

import argparse
import csv
import io
import os
import tempfile
import wave
from unittest.mock import MagicMock
from unittest.mock import patch

import grpc
import pytest
from google.protobuf import any_pb2
from google.protobuf import wrappers_pb2
from grpc_health.v1 import health_pb2

from client.lipsync.request_generators import speaker_info_csv_reader
from common.audio_utils import create_wav_header
from common.health import check_service_health
from common.media import check_streamable
from common.media import is_file_available
from common.proto_utils import create_protobuf_any_value
from common.tls import create_channel_credentials
from common.tls import read_file_content

pytestmark = pytest.mark.unit


class TestCreateWaveHeader:
    """Test cases for create_wav_header function."""

    def test_create_wav_header_basic(self):
        """Test create_wav_header creates valid WAV header."""
        header = create_wav_header(n_channels=1, sample_width=2, frame_rate=16000, n_frames=1600)

        # Should return bytes
        assert isinstance(header, bytes)
        assert len(header) > 0

        # Should be valid WAV header
        assert header.startswith(b"RIFF")
        assert b"WAVE" in header
        assert b"fmt " in header

    def test_create_wav_header_stereo(self):
        """Test create_wav_header with stereo audio."""
        header = create_wav_header(n_channels=2, sample_width=2, frame_rate=44100, n_frames=44100)

        assert isinstance(header, bytes)
        assert len(header) > 0

    def test_create_wav_header_different_sample_widths(self):
        """Test create_wav_header with different sample widths."""
        # Test 16-bit
        header_16bit = create_wav_header(
            n_channels=1, sample_width=2, frame_rate=16000, n_frames=1600
        )
        assert isinstance(header_16bit, bytes)

        # Test 24-bit
        header_24bit = create_wav_header(
            n_channels=1, sample_width=3, frame_rate=16000, n_frames=1600
        )
        assert isinstance(header_24bit, bytes)

    def test_create_wav_header_zero_frames(self):
        """Test create_wav_header with zero frames."""
        header = create_wav_header(n_channels=1, sample_width=2, frame_rate=16000, n_frames=0)

        assert isinstance(header, bytes)
        assert len(header) > 0

    def test_create_wav_header_verification(self):
        """Test that created header can be read by wave module."""
        header = create_wav_header(n_channels=1, sample_width=2, frame_rate=16000, n_frames=1600)

        # Create a buffer with the header
        buffer = io.BytesIO(header)

        # Should be readable by wave module
        with wave.open(buffer, "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000

    def test_create_wav_header_different_parameters(self):
        """Test create_wav_header with different parameters."""
        n_channels = 2
        sample_width = 2
        frame_rate = 44100
        n_frames = 1000

        header = create_wav_header(n_channels, sample_width, frame_rate, n_frames)

        # Verify it's a valid WAV header
        assert isinstance(header, bytes)
        assert len(header) > 0

        # Test with different parameters
        header_mono = create_wav_header(1, 1, 22050, 500)
        assert isinstance(header_mono, bytes)
        assert header != header_mono


class TestCheckServiceHealth:
    """Test cases for check_service_health function."""

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_serving(self, mock_stub_class, mock_channel):
        """Test health check when service is serving."""
        # Mock the health check response
        mock_response = MagicMock()
        mock_response.status = 1  # SERVING status

        mock_stub = MagicMock()
        mock_stub.Check.return_value = mock_response
        mock_stub_class.return_value = mock_stub

        # Test the function
        result = check_service_health("localhost:50050")
        assert result is True

        # Verify the calls
        mock_channel.assert_called_once_with("localhost:50050")
        mock_stub_class.assert_called_once()
        mock_stub.Check.assert_called_once()

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_not_serving(self, mock_stub_class, mock_channel):
        """Test health check when service is not serving."""
        # Mock the health check response
        mock_response = MagicMock()
        mock_response.status = 2  # NOT_SERVING status

        mock_stub = MagicMock()
        mock_stub.Check.return_value = mock_response
        mock_stub_class.return_value = mock_stub

        # Test the function should raise ConnectionError
        with pytest.raises(ConnectionError, match="not healthy: status=2"):
            check_service_health("localhost:50050")

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_unknown_status(self, mock_stub_class, mock_channel):
        """Test health check with unknown status."""
        # Mock the health check response
        mock_response = MagicMock()
        mock_response.status = 999  # Unknown status

        mock_stub = MagicMock()
        mock_stub.Check.return_value = mock_response
        mock_stub_class.return_value = mock_stub

        # Test the function should raise ConnectionError
        with pytest.raises(ConnectionError, match="not healthy: status=999"):
            check_service_health("localhost:50050")

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_grpc_error(self, mock_stub_class, mock_channel):
        """Test health check when gRPC call fails."""
        mock_stub = MagicMock()
        mock_stub.Check.side_effect = grpc.RpcError("gRPC error")
        mock_stub_class.return_value = mock_stub

        # Test the function should raise ConnectionError
        with pytest.raises(
            ConnectionError, match="localhost:50050 health check failed: gRPC error"
        ):
            check_service_health("localhost:50050")

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_rpc_error(self, mock_stub_class, mock_channel):
        """Test health check when RPC error occurs."""
        from common.handles import grpc

        mock_stub = MagicMock()
        mock_stub.Check.side_effect = grpc.RpcError("RPC failed")
        mock_stub_class.return_value = mock_stub

        # Test the function should raise ConnectionError
        with pytest.raises(
            ConnectionError, match="localhost:50050 health check failed: RPC failed"
        ):
            check_service_health("localhost:50050")

    @patch("common.handles.grpc.insecure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_different_server_address(self, mock_stub_class, mock_channel):
        """Test health check with different server address."""
        # Mock the health check response
        mock_response = MagicMock()
        mock_response.status = 1  # SERVING status

        mock_stub = MagicMock()
        mock_stub.Check.return_value = mock_response
        mock_stub_class.return_value = mock_stub

        # Test with different address
        result = check_service_health("192.168.1.100:8080")
        assert result is True

        # Verify the calls
        mock_channel.assert_called_once_with("192.168.1.100:8080")

    @patch("common.handles.grpc.secure_channel")
    @patch("common.handles.health_pb2_grpc.HealthStub")
    def test_health_with_channel_credentials(self, mock_stub_class, mock_secure_channel):
        """Test health check probes over a secure channel when credentials are given."""
        mock_response = MagicMock()
        mock_response.status = 1  # SERVING status

        mock_stub = MagicMock()
        mock_stub.Check.return_value = mock_response
        mock_stub_class.return_value = mock_stub

        credentials = MagicMock()
        result = check_service_health("localhost:50050", channel_credentials=credentials)
        assert result is True

        # The credentials must be honored by the probe channel
        mock_secure_channel.assert_called_once_with("localhost:50050", credentials)

    @patch("grpc.insecure_channel")
    def test_check_service_health_success(self, mock_channel):
        """Test check_service_health with successful response."""
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.status = health_pb2.HealthCheckResponse.SERVING
        mock_stub.Check.return_value = mock_response
        mock_channel.return_value = MagicMock()

        with patch("common.handles.health_pb2_grpc.HealthStub", return_value=mock_stub):
            result = check_service_health("localhost:50051")
            assert result is True
            mock_stub.Check.assert_called_once()

    @patch("grpc.insecure_channel")
    def test_check_service_health_not_serving(self, mock_channel):
        """Test check_service_health when service is not serving."""
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.status = health_pb2.HealthCheckResponse.NOT_SERVING
        mock_stub.Check.return_value = mock_response
        mock_channel.return_value = MagicMock()

        with patch("common.handles.health_pb2_grpc.HealthStub", return_value=mock_stub):
            with pytest.raises(ConnectionError):
                check_service_health("localhost:50051")

    @patch("grpc.insecure_channel")
    def test_check_service_health_grpc_error(self, mock_channel):
        """Test check_service_health with gRPC error."""
        mock_stub = MagicMock()
        mock_stub.Check.side_effect = grpc.RpcError("Connection failed")
        mock_channel.return_value = MagicMock()

        with patch("common.handles.health_pb2_grpc.HealthStub", return_value=mock_stub):
            with pytest.raises(ConnectionError):
                check_service_health("localhost:50051")


class TestFileOperations:
    """Test cases for file operation functions."""

    def test_is_file_available_success(self):
        """Test is_file_available with existing file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file_path = temp_file.name

        try:
            result = is_file_available(temp_file_path, ["txt"])
            assert result is True
        finally:
            os.unlink(temp_file_path)

    def test_is_file_available_file_not_found(self):
        """Test is_file_available with non-existent file."""
        with pytest.raises(FileNotFoundError):
            is_file_available("nonexistent.txt", ["txt"])

    def test_is_file_available_wrong_extension(self):
        """Test is_file_available with wrong file extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file_path = temp_file.name

        try:
            result = is_file_available(temp_file_path, ["pdf"])
            assert result is False
        finally:
            os.unlink(temp_file_path)

    def test_read_file_content(self):
        """Test read_file_content function."""
        test_content = b"test file content"
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            result = read_file_content(temp_file_path)
            assert result == test_content
        finally:
            os.unlink(temp_file_path)

    def test_speaker_info_csv_reader(self):
        """Test speaker_info_csv_reader function."""
        csv_data = "col1,col2\n1,2\n3,4\n5,6\n7,8\n"
        csv_file = io.StringIO(csv_data)
        reader = csv.reader(csv_file)

        # Test reading with row_count=2
        batches = list(speaker_info_csv_reader(reader, 2))
        assert len(batches) == 3  # Fixed: should be 3 batches for 4 data rows
        assert batches[0] == [["col1", "col2"], ["1", "2"]]
        assert batches[1] == [["3", "4"], ["5", "6"]]
        assert batches[2] == [["7", "8"]]  # Last batch has only one row


class TestCheckStreamable:
    """Test cases for check_streamable function."""

    def _check(self, content: bytes) -> bool:
        """Write content to a temporary MP4 file and run check_streamable."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        try:
            return check_streamable(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_check_streamable_true(self):
        """Test check_streamable with streamable MP4."""
        # [size=32]["ftyp"][24 bytes of brands] followed directly by moov
        content = (
            b"\x00\x00\x00\x20"  # ftyp size (32 bytes)
            b"ftyp"  # ftyp atom type
            b"isomiso2mp41"  # major brand + compatible brands (12 bytes)
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # padding to 32 bytes
            b"\x00\x00\x00\x10"  # moov size (16 bytes)
            b"moov"  # moov atom type
            b"\x00\x00\x00\x08"  # moov data
            b"\x00\x00\x00\x00"  # moov data
        )

        assert self._check(content) is True

    def test_check_streamable_false(self):
        """Test check_streamable with non-streamable MP4 (mdat between ftyp and moov)."""
        content = (
            b"\x00\x00\x00\x20"  # ftyp size (32 bytes)
            b"ftyp"  # ftyp atom type
            b"isomiso2mp41"  # major brand + compatible brands (12 bytes)
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # padding to 32 bytes
            b"\x00\x00\x00\x10"  # mdat size (16 bytes)
            b"mdat"  # mdat atom type
            b"\x00\x00\x00\x08"  # mdat data
            b"\x00\x00\x00\x00"  # mdat data
        )

        assert self._check(content) is False

    def test_check_streamable_large_ftyp(self):
        """Regression test: an oversized ftyp atom must not hide a following moov atom."""
        # A 48-byte ftyp (many compatible brands) followed directly by moov.
        ftyp_body = b"isom" + b"iso2avc1mp41" * 3  # 40 bytes of brands
        content = (
            b"\x00\x00\x00\x30"  # ftyp size (48 bytes)
            b"ftyp"  # ftyp atom type
            + ftyp_body
            + b"\x00\x00\x00\x10"  # moov size (16 bytes)
            + b"moov"  # moov atom type at offset 52
            + b"\x00" * 8  # moov data
        )

        assert self._check(content) is True

    def test_check_streamable_huge_declared_ftyp_size(self):
        """Test check_streamable with a declared ftyp size far beyond the file length.

        The declared atom size is untrusted file data; a malformed size close
        to the 32-bit maximum must return ``False`` without attempting to read
        the declared number of bytes into memory.
        """
        content = (
            b"\xff\xff\xff\xff"  # ftyp size (~4 GiB, far beyond file length)
            b"ftyp"
            b"isomiso2mp41"
        )

        assert self._check(content) is False

    def test_check_streamable_invalid_ftyp(self):
        """Test check_streamable with invalid first atom type."""
        content = (
            b"\x00\x00\x00\x20"  # size
            b"free"  # not ftyp
            b"isomiso2mp41"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x10"
            b"moov"
            b"\x00" * 8
        )

        assert self._check(content) is False

    def test_check_streamable_file_too_small(self):
        """Test check_streamable with file too small for an atom header."""
        assert self._check(b"\x00\x00\x00\x10") is False

    def test_check_streamable_truncated_after_ftyp(self):
        """Test check_streamable when no atom header follows the ftyp atom."""
        content = (
            b"\x00\x00\x00\x20"  # ftyp size (32 bytes)
            b"ftyp"
            b"isomiso2mp41"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # file ends with ftyp
        )

        assert self._check(content) is False


class TestChannelCredentials:
    """Test cases for create_channel_credentials function."""

    @patch("common.tls.read_file_content")
    def test_create_channel_credentials_mtls(self, mock_read_file):
        """Test create_channel_credentials with MTLS mode."""
        mock_read_file.return_value = b"certificate_data"

        args = argparse.Namespace()
        args.ssl_mode = "MTLS"
        args.ssl_key = "key.pem"
        args.ssl_cert = "cert.pem"
        args.ssl_root_cert = "root.pem"

        with patch("grpc.ssl_channel_credentials") as mock_ssl_creds:
            mock_ssl_creds.return_value = MagicMock()
            result = create_channel_credentials(args)

            mock_read_file.assert_called()
            mock_ssl_creds.assert_called_once()

    @patch("common.tls.read_file_content")
    def test_create_channel_credentials_tls(self, mock_read_file):
        """Test create_channel_credentials with TLS mode."""
        mock_read_file.return_value = b"certificate_data"

        args = argparse.Namespace()
        args.ssl_mode = "TLS"
        args.ssl_root_cert = "root.pem"

        with patch("grpc.ssl_channel_credentials") as mock_ssl_creds:
            mock_ssl_creds.return_value = MagicMock()
            result = create_channel_credentials(args)

            mock_read_file.assert_called_once_with("root.pem")
            mock_ssl_creds.assert_called_once()

    def test_create_channel_credentials_mtls_missing_files(self):
        """Test create_channel_credentials with MTLS mode but missing files."""
        args = argparse.Namespace()
        args.ssl_mode = "MTLS"
        args.ssl_key = None
        args.ssl_cert = None
        args.ssl_root_cert = None

        with pytest.raises(RuntimeError):
            create_channel_credentials(args)

    def test_create_channel_credentials_tls_missing_root_cert(self):
        """Test create_channel_credentials with TLS mode but missing root cert."""
        args = argparse.Namespace()
        args.ssl_mode = "TLS"
        args.ssl_root_cert = None

        with pytest.raises(RuntimeError):
            create_channel_credentials(args)

    @patch("common.tls.read_file_content")
    def test_create_channel_credentials_ssl_mode_override(self, mock_read_file):
        """The ssl_mode keyword overrides args.ssl_mode for one connection."""
        mock_read_file.return_value = b"certificate_data"

        args = argparse.Namespace()
        # args say MTLS, but the per-connection override asks for TLS: the
        # TLS branch must run (root cert only, no client key material).
        args.ssl_mode = "MTLS"
        args.ssl_key = "key.pem"
        args.ssl_cert = "cert.pem"
        args.ssl_root_cert = "root.pem"

        with patch("grpc.ssl_channel_credentials") as mock_ssl_creds:
            mock_ssl_creds.return_value = MagicMock()
            create_channel_credentials(args=args, ssl_mode="TLS")

            mock_read_file.assert_called_once_with("root.pem")
            mock_ssl_creds.assert_called_once_with(root_certificates=b"certificate_data")


class TestProtobufAnyValue:
    """Test cases for create_protobuf_any_value function."""

    def test_create_protobuf_any_value_bool(self):
        """Test create_protobuf_any_value with boolean value."""
        result = create_protobuf_any_value(True)
        assert isinstance(result, any_pb2.Any)

        # Unpack and verify
        wrapper = wrappers_pb2.BoolValue()
        result.Unpack(wrapper)
        assert wrapper.value is True

    def test_create_protobuf_any_value_int32(self):
        """Test create_protobuf_any_value with int32 value."""
        result = create_protobuf_any_value(42)
        assert isinstance(result, any_pb2.Any)

        # Unpack and verify
        wrapper = wrappers_pb2.Int32Value()
        result.Unpack(wrapper)
        assert wrapper.value == 42

    def test_create_protobuf_any_value_int64(self):
        """Test create_protobuf_any_value with int64 value."""
        large_int = 2147483648  # Larger than int32 max
        result = create_protobuf_any_value(large_int)
        assert isinstance(result, any_pb2.Any)

        # Unpack and verify
        wrapper = wrappers_pb2.Int64Value()
        result.Unpack(wrapper)
        assert wrapper.value == large_int

    def test_create_protobuf_any_value_float(self):
        """Test create_protobuf_any_value packs floats as DoubleValue."""
        # A value that is not exactly representable in 32-bit floats, so
        # a FloatValue round-trip would lose precision.
        float_value = 3.141592653589793
        result = create_protobuf_any_value(float_value)
        assert isinstance(result, any_pb2.Any)
        assert result.Is(wrappers_pb2.DoubleValue.DESCRIPTOR)

        # Unpack and verify: DoubleValue preserves Python float precision
        wrapper = wrappers_pb2.DoubleValue()
        result.Unpack(wrapper)
        assert wrapper.value == float_value

    def test_create_protobuf_any_value_string(self):
        """Test create_protobuf_any_value with string value."""
        result = create_protobuf_any_value("test string")
        assert isinstance(result, any_pb2.Any)

        # Unpack and verify
        wrapper = wrappers_pb2.StringValue()
        result.Unpack(wrapper)
        assert wrapper.value == "test string"

    def test_create_protobuf_any_value_unsupported_type(self):
        """Test create_protobuf_any_value with unsupported type."""
        with pytest.raises(TypeError):
            create_protobuf_any_value([1, 2, 3])


if __name__ == "__main__":
    pytest.main([__file__])
