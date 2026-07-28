# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for batch-processing bypass-S2S and diarization-granularity wiring."""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from client.batch_processing.app import _resolve_translated_audio
from client.batch_processing.app import _set_lipsync_codec_for_translated_audio
from client.common.audio import AUDIO_CODEC_CONFIGS
from client.controller.config import ControllerConfig

_ASSERTIONS = unittest.TestCase()


@pytest.mark.unit
class TestResolveTranslatedAudio:
    """Test cases for ``_resolve_translated_audio``."""

    def test_falls_back_to_source_when_no_dir(self):
        """No directory -> the extracted source WAV is used as a stand-in."""
        resolved = _resolve_translated_audio(
            translated_audio_dir=None,
            stem="clip",
            fallback_wav_path="prep/clip.wav",
        )
        assert resolved == "prep/clip.wav"

    def test_uses_dir_file_when_present(self, tmp_path):
        """A matching ``{stem}.wav`` in the directory is preferred."""
        translated = tmp_path / "clip.wav"
        translated.write_bytes(b"RIFF")
        resolved = _resolve_translated_audio(
            translated_audio_dir=str(tmp_path),
            stem="clip",
            fallback_wav_path="prep/clip.wav",
        )
        assert resolved == str(translated)

    def test_uses_mp3_when_wav_is_absent(self, tmp_path):
        """A matching ``{stem}.mp3`` is accepted for LipSync MP3 bypass."""
        translated = tmp_path / "clip.mp3"
        translated.write_bytes(b"ID3")
        resolved = _resolve_translated_audio(
            translated_audio_dir=str(tmp_path),
            stem="clip",
            fallback_wav_path="prep/clip.wav",
        )
        assert resolved == str(translated)

    def test_raises_when_dir_file_missing(self, tmp_path):
        """A directory without the expected file is a hard error."""
        with pytest.raises(FileNotFoundError):
            _resolve_translated_audio(
                translated_audio_dir=str(tmp_path),
                stem="missing",
                fallback_wav_path="prep/missing.wav",
            )


@pytest.mark.unit
class TestRunSingleVideoBypass:
    """Test cases for ``run_single_video`` translated-audio forwarding."""

    @pytest.fixture(autouse=True)
    def _stub_detect_audio_codec(self):
        """Stub codec sniffing — these tests use fake paths that don't exist."""
        with patch(
            "client.batch_processing.runner.detect_audio_codec",
            return_value=AUDIO_CODEC_CONFIGS["wav"],
        ):
            yield

    @patch("client.batch_processing.runner.write_output_from_response")
    @patch("client.batch_processing.runner.ContentLocalizationControllerStub")
    @patch("client.batch_processing.runner.grpc")
    @patch("client.batch_processing.runner.VideoSourceSimulator")
    @patch("client.batch_processing.runner.create_audio_source")
    @patch("client.batch_processing.runner.AudioSourceSimulator")
    @patch("client.batch_processing.runner.create_controller_request_generator")
    def test_forwards_translated_audio_source(
        self,
        mock_gen,
        mock_audio_cls,
        mock_create_audio_source,
        mock_video_cls,
        mock_grpc,
        mock_stub,
        mock_writer,
    ):
        """A translated_audio_path produces a translated_audio_source arg."""
        from client.batch_processing.runner import run_single_video

        original_audio = MagicMock()
        translated_audio = MagicMock()
        original_audio.is_open.return_value = False
        translated_audio.is_open.return_value = False
        mock_audio_cls.return_value = original_audio
        mock_create_audio_source.return_value = translated_audio
        mock_video_cls.return_value.is_open.return_value = False

        config = MagicMock()
        config.bypass_asd = False
        config.diarization_rows_per_chunk = 10
        run_single_video(
            audio_path="a.wav",
            video_path="v.mp4",
            output_path="o.mp4",
            config=config,
            diarization_info=None,
            translated_audio_path="translated.wav",
        )

        # The translated source is selected by content sniffing, so MP3
        # data inside a .wav filename cannot crash the WAV parser.
        mock_create_audio_source.assert_called_once_with(file_path="translated.wav")
        _, kwargs = mock_gen.call_args
        assert kwargs["translated_audio_source"] is translated_audio
        assert kwargs["bypass_asd"] is False
        assert kwargs["diarization_rows_per_chunk"] == 10

    @patch("client.batch_processing.runner.write_output_from_response")
    @patch("client.batch_processing.runner.ContentLocalizationControllerStub")
    @patch("client.batch_processing.runner.grpc")
    @patch("client.batch_processing.runner.VideoSourceSimulator")
    @patch("client.batch_processing.runner.create_audio_source")
    @patch("client.batch_processing.runner.AudioSourceSimulator")
    @patch("client.batch_processing.runner.create_controller_request_generator")
    def test_no_translated_audio_source_in_normal_mode(
        self,
        mock_gen,
        mock_audio_cls,
        mock_create_audio_source,
        mock_video_cls,
        mock_grpc,
        mock_stub,
        mock_writer,
    ):
        """Without translated audio, translated_audio_source stays None."""
        from client.batch_processing.runner import run_single_video

        mock_audio_cls.return_value.is_open.return_value = False
        mock_video_cls.return_value.is_open.return_value = False
        config = MagicMock()
        config.bypass_asd = True
        config.diarization_rows_per_chunk = None

        run_single_video(
            audio_path="a.wav",
            video_path="v.mp4",
            output_path="o.mp4",
            config=config,
        )

        _, kwargs = mock_gen.call_args
        assert kwargs["translated_audio_source"] is None
        assert kwargs["bypass_asd"] is True
        assert kwargs["diarization_rows_per_chunk"] is None
        # Only the original audio source is created.
        assert mock_audio_cls.call_count == 1
        mock_create_audio_source.assert_not_called()

    @patch("client.batch_processing.runner.write_output_from_response")
    @patch("client.batch_processing.runner.ContentLocalizationControllerStub")
    @patch("client.batch_processing.runner.grpc")
    @patch("client.batch_processing.runner.VideoSourceSimulator")
    @patch("client.batch_processing.runner.create_audio_source")
    @patch("client.batch_processing.runner.AudioSourceSimulator")
    @patch("client.batch_processing.runner.create_controller_request_generator")
    def test_mp3_translated_audio_uses_content_sniffing_factory(
        self,
        mock_gen,
        mock_audio_cls,
        mock_create_audio_source,
        mock_video_cls,
        mock_grpc,
        mock_stub,
        mock_writer,
    ):
        """MP3 translated audio goes through the content-sniffing factory."""
        from client.batch_processing.runner import run_single_video

        original_audio = MagicMock()
        translated_audio = MagicMock()
        original_audio.is_open.return_value = False
        translated_audio.is_open.return_value = False
        mock_audio_cls.return_value = original_audio
        mock_create_audio_source.return_value = translated_audio
        mock_video_cls.return_value.is_open.return_value = False

        config = MagicMock()
        config.bypass_asd = False
        config.diarization_rows_per_chunk = 10
        run_single_video(
            audio_path="a.wav",
            video_path="v.mp4",
            output_path="o.mp4",
            config=config,
            diarization_info=None,
            translated_audio_path="translated.mp3",
        )

        mock_create_audio_source.assert_called_once_with(file_path="translated.mp3")
        _, kwargs = mock_gen.call_args
        assert kwargs["translated_audio_source"] is translated_audio


@pytest.mark.unit
class TestBatchConfig:
    """Test cases for batch-specific controller config adjustments."""

    @patch("client.controller.config.lipsync_config_from_args")
    @patch("client.controller.config.asd_config_from_args")
    @patch("client.controller.config.s2s_config_from_args")
    def test_batch_config_keeps_asd_enabled_without_cli_diarization(
        self,
        _mock_s2s_config,
        mock_asd_config,
        _mock_lipsync_config,
    ):
        """Batch opts out of auto-bypass because diarization is discovered per video."""
        asd_config = MagicMock()
        mock_asd_config.return_value = asd_config
        args = MagicMock()
        args.bypass_asd = False
        args.diarization_file = None
        args.translated_audio = None
        args.lipsync_input_audio_codec = None
        args.controller_server = "localhost:50056"
        args.chunk_size_audio_secs = 1.0
        args.chunk_size_video_bytes = 1048576

        config = ControllerConfig.from_args(args=args, auto_bypass_asd=False)

        _ASSERTIONS.assertFalse(config.bypass_asd)
        _ASSERTIONS.assertIs(config.asd_config, asd_config)
        mock_asd_config.assert_called_once_with(args)

    @patch("client.controller.config.lipsync_config_from_args")
    @patch("client.controller.config.asd_config_from_args")
    @patch("client.controller.config.s2s_config_from_args")
    def test_batch_config_honors_explicit_bypass_asd(
        self,
        _mock_s2s_config,
        mock_asd_config,
        _mock_lipsync_config,
    ):
        """An explicit --bypass-asd flag wins even when auto-bypass is off."""
        args = MagicMock()
        args.bypass_asd = True
        args.diarization_file = None
        args.translated_audio = None
        args.lipsync_input_audio_codec = None
        args.controller_server = "localhost:50056"
        args.chunk_size_audio_secs = 1.0
        args.chunk_size_video_bytes = 1048576

        config = ControllerConfig.from_args(args=args, auto_bypass_asd=False)

        _ASSERTIONS.assertTrue(config.bypass_asd)
        _ASSERTIONS.assertIsNone(config.asd_config)
        mock_asd_config.assert_not_called()

    @patch("client.batch_processing.app.is_wav_file", return_value=True)
    def test_translated_audio_codec_uses_detected_wav(self, mock_is_wav_file):
        """Detected WAV content sets LipSync input codec to WAV."""
        config = MagicMock()

        _set_lipsync_codec_for_translated_audio(
            config=config,
            translated_audio_path="translated.wav",
        )

        _ASSERTIONS.assertEqual(
            config.lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["wav"],
        )
        mock_is_wav_file.assert_called_once_with("translated.wav")

    @patch("client.batch_processing.app.is_wav_file", return_value=False)
    def test_translated_audio_codec_uses_detected_mp3(self, mock_is_wav_file):
        """Detected non-WAV content sets LipSync input codec to MP3."""
        config = MagicMock()

        _set_lipsync_codec_for_translated_audio(
            config=config,
            translated_audio_path="translated.mp3",
        )

        _ASSERTIONS.assertEqual(
            config.lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["mp3"],
        )
        mock_is_wav_file.assert_called_once_with("translated.mp3")

    @patch("client.batch_processing.app.is_wav_file")
    def test_translated_audio_codec_keeps_explicit_customer_value(self, mock_is_wav_file):
        """Explicit customer codec skips detection and is forwarded."""
        config = MagicMock()

        _set_lipsync_codec_for_translated_audio(
            config=config,
            translated_audio_path="translated.wav",
            explicit_input_audio_codec="MP3",
        )

        _ASSERTIONS.assertEqual(
            config.lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["mp3"],
        )
        mock_is_wav_file.assert_not_called()


@pytest.mark.unit
class TestProcessSingleVideoGranularity:
    """Test cases for diarization-granularity + bypass threading in app."""

    @patch("client.batch_processing.app.run_single_video")
    @patch("client.batch_processing.app.load_diarization_info")
    @patch("client.batch_processing.app.ensure_diarization")
    @patch("client.batch_processing.app.preprocess_video")
    @patch("client.batch_processing.app.is_wav_file", return_value=True)
    @patch("client.batch_processing.app.os.path.getsize", return_value=1234)
    def test_combine_flag_and_bypass_threaded(
        self,
        mock_size,
        mock_is_wav_file,
        mock_preprocess,
        mock_ensure,
        mock_load,
        mock_run,
    ):
        """combine flag flows to load_diarization_info; bypass resolves audio."""
        from client.batch_processing.app import _process_single_video

        mock_preprocess.return_value = ("prep/v.wav", 17.0, 1920, 1080, 510)
        mock_ensure.return_value = "diar/v.json"
        mock_load.return_value = MagicMock(segments=[1, 2])

        result = _process_single_video(
            video_path="v.mp4",
            output_dir="out",
            target_language="de",
            config=MagicMock(explicit_lipsync_input_audio_codec=None),
            bypass_s2s=True,
            translated_audio_dir=None,
            combine_chunks_per_speaker=False,
        )

        assert result.success is True
        assert result.video_duration_secs == 17.0
        _ASSERTIONS.assertEqual(mock_is_wav_file.call_args.args[0], "prep/v.wav")
        # Granularity flag forwarded verbatim.
        assert mock_load.call_args.kwargs["combine_chunks_per_speaker"] is False
        # Bypass with no dir falls back to the extracted source WAV.
        assert mock_run.call_args.kwargs["translated_audio_path"] == "prep/v.wav"
        _ASSERTIONS.assertEqual(
            mock_run.call_args.kwargs["config"].lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["wav"],
        )

    @patch("client.batch_processing.app.run_single_video")
    @patch("client.batch_processing.app.load_diarization_info")
    @patch("client.batch_processing.app.ensure_diarization")
    @patch("client.batch_processing.app.preprocess_video")
    @patch("client.batch_processing.app.is_wav_file")
    @patch("client.batch_processing.app.os.path.getsize", return_value=1234)
    def test_bypass_uses_explicit_lipsync_codec_without_detection(
        self,
        mock_size,
        mock_is_wav_file,
        mock_preprocess,
        mock_ensure,
        mock_load,
        mock_run,
    ):
        """Batch bypass honors explicit LipSync input codec over detection."""
        from client.batch_processing.app import _process_single_video

        mock_preprocess.return_value = ("prep/v.wav", 17.0, 1920, 1080, 510)
        mock_ensure.return_value = "diar/v.json"
        mock_load.return_value = MagicMock(segments=[1, 2])

        _process_single_video(
            video_path="v.mp4",
            output_dir="out",
            target_language="de",
            config=MagicMock(explicit_lipsync_input_audio_codec="MP3"),
            bypass_s2s=True,
            translated_audio_dir=None,
            combine_chunks_per_speaker=False,
        )

        mock_is_wav_file.assert_not_called()
        _ASSERTIONS.assertEqual(
            mock_run.call_args.kwargs["config"].lipsync_config.input_audio_codec,
            AUDIO_CODEC_CONFIGS["mp3"],
        )

    @patch("client.batch_processing.app.run_single_video")
    @patch("client.batch_processing.app.load_diarization_info")
    @patch("client.batch_processing.app.ensure_diarization")
    @patch("client.batch_processing.app.preprocess_video")
    @patch("client.batch_processing.app.os.path.getsize", return_value=1234)
    def test_normal_mode_has_no_translated_audio(
        self,
        mock_size,
        mock_preprocess,
        mock_ensure,
        mock_load,
        mock_run,
    ):
        """Without bypass, no translated audio path is passed to the runner."""
        from client.batch_processing.app import _process_single_video

        mock_preprocess.return_value = ("prep/v.wav", 20.0, 1280, 720, 600)
        mock_ensure.return_value = "diar/v.json"
        mock_load.return_value = MagicMock(segments=[1])

        _process_single_video(
            video_path="v.mp4",
            output_dir="out",
            target_language="de",
            config=MagicMock(),
            bypass_s2s=False,
        )

        assert mock_run.call_args.kwargs["translated_audio_path"] is None
        assert mock_load.call_args.kwargs["combine_chunks_per_speaker"] is True


if __name__ == "__main__":
    pytest.main([__file__])
