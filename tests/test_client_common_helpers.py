# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared client helpers in ``client.common``."""

import shutil
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path

import pytest
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV

from client.common.audio import create_audio_source
from client.common.audio import detect_audio_codec
from client.common.bypass import resolve_bypass_asd
from client.common.paths import ensure_parent_dir
from common.source_sink.file import FileSourceSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator

pytestmark = pytest.mark.unit


def _write_wav(path: Path) -> None:
    """Write a minimal valid WAV file.

    Args:
        path (Path): Destination file path.
    """
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)


class _TempDirTestCase(unittest.TestCase):
    """Base class providing a per-test temporary directory."""

    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp(prefix="client-common-helpers-"))
        self.addCleanup(shutil.rmtree, self.tmp_path, True)


class TestCreateAudioSource(_TempDirTestCase):
    """create_audio_source selects the simulator by file content."""

    def _assert_source_type(self, file_path: str, expected_type: type) -> None:
        """Create a source, check its type, and always close it."""
        source = create_audio_source(file_path=file_path)
        try:
            self.assertIsInstance(source, expected_type)
        finally:
            if source.is_open():
                source.close()

    def test_wav_content_uses_audio_source_simulator(self) -> None:
        """Genuine WAV content gets the WAV-aware simulator."""
        wav_path = self.tmp_path / "audio.wav"
        _write_wav(wav_path)
        self._assert_source_type(file_path=str(wav_path), expected_type=AudioSourceSimulator)

    def test_mp3_content_in_wav_filename_uses_file_source(self) -> None:
        """MP3 bytes inside a .wav filename stream as raw bytes."""
        fake_wav = self.tmp_path / "translated.wav"
        fake_wav.write_bytes(b"ID3\x04\x00" + b"\x00" * 64)
        self._assert_source_type(file_path=str(fake_wav), expected_type=FileSourceSimulator)

    def test_mp3_file_uses_file_source(self) -> None:
        """Plain MP3 files stream as raw bytes."""
        mp3_path = self.tmp_path / "audio.mp3"
        mp3_path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)
        self._assert_source_type(file_path=str(mp3_path), expected_type=FileSourceSimulator)


class TestDetectAudioCodec(_TempDirTestCase):
    """detect_audio_codec sniffs content, matching create_audio_source."""

    def test_wav_content_detected_as_wav(self) -> None:
        """Genuine WAV content reports AUDIO_CODEC_WAV."""
        wav_path = self.tmp_path / "audio.wav"
        _write_wav(wav_path)
        self.assertEqual(detect_audio_codec(file_path=str(wav_path)), AUDIO_CODEC_WAV)

    def test_mp3_content_detected_as_mp3_despite_wav_name(self) -> None:
        """MP3 bytes inside a .wav filename report AUDIO_CODEC_MP3."""
        fake_wav = self.tmp_path / "translated.wav"
        fake_wav.write_bytes(b"ID3\x04\x00" + b"\x00" * 64)
        self.assertEqual(detect_audio_codec(file_path=str(fake_wav)), AUDIO_CODEC_MP3)

    def test_mp3_file_detected_as_mp3(self) -> None:
        """Plain MP3 files report AUDIO_CODEC_MP3."""
        mp3_path = self.tmp_path / "audio.mp3"
        mp3_path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)
        self.assertEqual(detect_audio_codec(file_path=str(mp3_path)), AUDIO_CODEC_MP3)


class TestEnsureParentDir(_TempDirTestCase):
    """ensure_parent_dir handles bare filenames and nested paths."""

    def test_bare_filename_is_a_noop(self) -> None:
        """A path without a directory component must not raise."""
        ensure_parent_dir(path="output.mp4")

    def test_creates_missing_parent_directories(self) -> None:
        """Nested parent directories are created on demand."""
        target = self.tmp_path / "a" / "b" / "output.mp4"
        ensure_parent_dir(path=str(target))
        self.assertTrue(target.parent.is_dir())

    def test_existing_parent_directory_is_accepted(self) -> None:
        """An already-existing parent directory must not raise."""
        target = self.tmp_path / "output.mp4"
        ensure_parent_dir(path=str(target))
        self.assertTrue(self.tmp_path.is_dir())


class TestResolveBypassAsd(unittest.TestCase):
    """resolve_bypass_asd combines the explicit flag with auto-detection."""

    def test_explicit_flag_wins(self) -> None:
        """--bypass-asd forces bypass even with a diarization file."""
        args = Namespace(bypass_asd=True, diarization_file="diar.json")
        self.assertTrue(resolve_bypass_asd(args=args))

    def test_auto_bypass_without_diarization(self) -> None:
        """Missing diarization file auto-bypasses ASD by default."""
        args = Namespace(bypass_asd=False, diarization_file=None)
        self.assertTrue(resolve_bypass_asd(args=args))

    def test_no_bypass_with_diarization(self) -> None:
        """A diarization file keeps ASD enabled."""
        args = Namespace(bypass_asd=False, diarization_file="diar.json")
        self.assertFalse(resolve_bypass_asd(args=args))

    def test_auto_bypass_opt_out(self) -> None:
        """auto_bypass_asd=False keeps ASD enabled without diarization."""
        args = Namespace(bypass_asd=False, diarization_file=None)
        self.assertFalse(resolve_bypass_asd(args=args, auto_bypass_asd=False))

    def test_explicit_flag_wins_with_opt_out(self) -> None:
        """--bypass-asd is honored even when auto-detection is disabled."""
        args = Namespace(bypass_asd=True, diarization_file=None)
        self.assertTrue(resolve_bypass_asd(args=args, auto_bypass_asd=False))

    def test_missing_attributes_default_to_auto_bypass(self) -> None:
        """Namespaces without the attributes behave like empty values."""
        self.assertTrue(resolve_bypass_asd(args=Namespace()))
        self.assertFalse(resolve_bypass_asd(args=Namespace(), auto_bypass_asd=False))


if __name__ == "__main__":
    unittest.main()
