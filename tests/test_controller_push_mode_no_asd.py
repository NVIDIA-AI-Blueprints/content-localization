# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bypass-ASD controller flow."""

import os
import threading
import unittest
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionResult,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.errors import PipelineInputError
from common.errors import PipelinePreconditionError
from controller_service.service import ControllerService


class _FakeS2SClient:
    def __init__(self, handle: object) -> None:
        self.handle = handle

    def __call__(
        self,
        request_iterator: Iterator[object],
        output_buffer: Any,
        context: object,
        request_id: str,
    ) -> None:
        _ = context
        _ = request_id
        for _, _req in zip(range(2), request_iterator, strict=False):
            pass
        output_buffer.put(SpeechToSpeechResponse(audio_data=b"s2s-audio", audio_format="mp3"))
        output_buffer.done = True


class _FakeLipsyncClient:
    def __init__(self, handle: object) -> None:
        self.handle = handle

    def __call__(
        self,
        request_iterator: Iterator[object],
        output_buffer: Any,
        context: object,
        request_id: str,
    ) -> None:
        _ = context
        _ = request_id
        for _, _req in zip(range(4), request_iterator, strict=False):
            pass
        output_buffer.put(LipsyncResponse(video_file_data=b"lipsync-video"))
        output_buffer.done = True


class _FakeAsdClient:
    def __init__(self, handle: object) -> None:
        self.handle = handle

    def __call__(
        self,
        request_iterator: Iterator[object],
        output_buffer: Any,
        context: object,
        request_id: str,
    ) -> None:
        _ = context
        _ = request_id
        for _, _req in zip(range(4), request_iterator, strict=False):
            pass
        output_buffer.put(
            DetectActiveSpeakerResponse(
                active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
            )
        )
        output_buffer.done = True


def _make_request_stream_with_configs(
    asd_codec: int = AUDIO_CODEC_WAV,
    lipsync_codec: int = AUDIO_CODEC_MP3,
    bypass_asd: bool = False,
    include_asd_config: bool = True,
    include_translated_audio: bool = False,
) -> Iterator[ContentLocalizationRequest]:
    """Build a request stream with configs first, then data (matching new protocol)."""
    msgs: list[ContentLocalizationRequest] = [
        # controller_config must be sent so _extract_config doesn't
        # block for the full 5-second timeout in tests
        ContentLocalizationRequest(
            controller_config=ContentLocalizationConfig(
                bypass_s2s=False,
                bypass_asd=bypass_asd,
            ),
        ),
        ContentLocalizationRequest(
            s2s_config=SpeechToSpeechConfig(target_language="es"),
        ),
    ]
    if include_asd_config:
        msgs.append(
            ContentLocalizationRequest(
                asd_config=ActiveSpeakerDetectionConfig(
                    input_audio_config=AudioConfig(encoding=asd_codec),
                ),
            )
        )
    msgs.extend(
        [
            ContentLocalizationRequest(
                lipsync_config=LipsyncConfig(input_audio_codec=lipsync_codec),
            ),
            ContentLocalizationRequest(audio_data=b"client-audio"),
            ContentLocalizationRequest(video_file_data=b"client-video"),
        ]
    )
    if include_translated_audio:
        msgs.append(ContentLocalizationRequest(translated_audio_data=b"unexpected-translated"))
    return iter(msgs)


@pytest.mark.integration
@patch.dict(os.environ, {"S2S_SERVICE": "EL_DUBBING"})
class TestControllerPushModeBypassAsd(unittest.TestCase):
    """End-to-end ``infer()`` tests for bypass-ASD controller behavior.

    These tests drive the full push-mode pipeline, spinning real
    deserializer/client threads, so they carry the integration marker.
    """

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.pipeline.ActiveSpeakerDetectionClient")
    def test_controller_impl_bypass_asd_skips_asd_client(
        self,
        mock_asd_client: MagicMock,
    ) -> None:
        """Bypass-ASD push path produces output without creating ASD client."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-no-asd",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_asd_client.assert_not_called()

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.pipeline.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    @patch("controller_service.pipeline.asd_request_generator")
    def test_controller_impl_asd_passes_client_config(
        self,
        mock_asd_request_generator: MagicMock,
    ) -> None:
        """ASD-enabled push path passes client-provided asd_config to generator."""
        mock_asd_request_generator.return_value = iter([])
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(asd_codec=AUDIO_CODEC_MP3)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-asd",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        self.assertTrue(mock_asd_request_generator.called)
        passed_config = mock_asd_request_generator.call_args.kwargs["asd_config"]
        self.assertEqual(passed_config.input_audio_config.encoding, AUDIO_CODEC_MP3)

    @patch.dict(os.environ, {"S2S_SERVICE": "CAMB_DUBBING"})
    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_camb_mp3_output_overrides_lipsync_codec(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """CAMB_DUBBING overrides mismatched lipsync codec to MP3."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(
            asd_codec=AUDIO_CODEC_WAV,
            lipsync_codec=AUDIO_CODEC_WAV,
            bypass_asd=True,
        )

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-camb-mirror-codec",
            )
        )

        self.assertEqual(len(responses), 1)
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("(MP3)", warning_msg)

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_missing_input_audio_config_warns_and_defaults_to_wav(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Without input_audio_config or asd_config the server warns about WAV."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        # bypass_asd + no asd_config: no codec source remains, so the
        # controller must fall back to WAV and say so.
        requests = _make_request_stream_with_configs(bypass_asd=True, include_asd_config=False)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-default-wav",
            )
        )

        self.assertEqual(len(responses), 1)
        warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("assuming WAV input audio" in msg for msg in warnings))

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_server_overrides_is_speaker_info_bypass_asd(self) -> None:
        """Server sets is_speaker_info_provided=False when ASD is bypassed."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=True)

        with patch(
            "controller_service.pipeline.lipsync_request_generator",
            return_value=iter([]),
        ) as mock_generator:
            responses = list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-speaker-no-asd",
                )
            )

        self.assertEqual(len(responses), 1)
        lipsync_config = mock_generator.call_args.kwargs["lipsync_config"]
        self.assertFalse(lipsync_config.is_speaker_info_provided)

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.pipeline.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    def test_server_overrides_is_speaker_info_with_asd(self) -> None:
        """Server sets is_speaker_info_provided=True when ASD is enabled."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs()

        with patch(
            "controller_service.pipeline.lipsync_request_generator",
            return_value=iter([]),
        ) as mock_generator:
            responses = list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-speaker-with-asd",
                )
            )

        self.assertEqual(len(responses), 1)
        lipsync_config = mock_generator.call_args.kwargs["lipsync_config"]
        self.assertTrue(lipsync_config.is_speaker_info_provided)

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_lipsync_codec_mismatch_warns_and_overrides(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Mismatched lipsync input codec triggers warning and gets overridden."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(lipsync_codec=AUDIO_CODEC_WAV, bypass_asd=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-mismatch",
            )
        )

        self.assertEqual(len(responses), 1)
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("does not match S2S output format", warning_msg)

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_lipsync_codec_match_no_warning(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Matching lipsync input codec does not trigger a warning."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(lipsync_codec=AUDIO_CODEC_MP3, bypass_asd=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-match",
            )
        )

        self.assertEqual(len(responses), 1)
        for call_args in mock_logger.warning.call_args_list:
            self.assertNotIn("does not match S2S output format", call_args[0][0])

    # -- bypass S2S tests --------------------------------------------------

    @patch("controller_service.pipeline.SpeechToSpeechClient")
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_bypass_s2s_skips_s2s_client(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """When bypass_s2s=True, S2S client is never instantiated."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream()

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-s2s",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()

    @patch("controller_service.pipeline.SpeechToSpeechClient")
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_bypass_s2s_no_codec_override_warning(
        self,
        mock_logger: MagicMock,
        _mock_s2s_client: MagicMock,
    ) -> None:
        """In bypass mode, lipsync codec mismatch does NOT trigger a warning."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        # Client sets WAV codec — normally this would trigger override
        # warning against the MP3 S2S output, but bypass mode should skip
        requests = _make_bypass_s2s_request_stream(
            lipsync_codec=AUDIO_CODEC_WAV,
        )

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-no-warn",
            )
        )

        self.assertEqual(len(responses), 1)
        for call_args in mock_logger.warning.call_args_list:
            self.assertNotIn("does not match S2S output format", call_args[0][0])

    @patch("controller_service.pipeline.SpeechToSpeechClient")
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.pipeline.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    def test_bypass_s2s_with_asd_enabled(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """Bypass S2S + ASD enabled: S2S skipped but ASD still runs."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream(include_asd_config=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-with-asd",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()

    @patch("controller_service.pipeline.SpeechToSpeechClient")
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_no_s2s_server_without_bypass_aborts(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """s2s_server=None + bypass_s2s=False raises PipelinePreconditionError."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=None,
        )
        context = MagicMock()
        # Explicitly send bypass_s2s=False so _extract_config
        # returns immediately instead of blocking for 5s
        requests = iter(
            [
                ContentLocalizationRequest(
                    controller_config=ContentLocalizationConfig(
                        bypass_s2s=False,
                        bypass_asd=True,
                    ),
                ),
                ContentLocalizationRequest(
                    lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
                ),
                ContentLocalizationRequest(audio_data=b"audio"),
                ContentLocalizationRequest(video_file_data=b"video"),
            ]
        )

        with self.assertRaises(PipelinePreconditionError):
            list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-no-s2s-no-bypass",
                )
            )
        mock_s2s_client.assert_not_called()

    @patch("controller_service.pipeline.SpeechToSpeechClient")
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_no_s2s_server_with_bypass_succeeds(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """When s2s_server is None but bypass_s2s=True, pipeline succeeds."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream()

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-no-s2s-with-bypass",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()
        context.abort.assert_not_called()

    # -- bypass_asd tests -------------------------------------------------

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.pipeline.ActiveSpeakerDetectionClient")
    def test_bypass_asd_with_server_configured(
        self,
        mock_asd_client: MagicMock,
    ) -> None:
        """ASD server present + bypass_asd=True -> ASD skipped."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-asd-with-server",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_asd_client.assert_not_called()

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_no_asd_server_without_bypass_aborts(self) -> None:
        """ASD server=None + bypass_asd=False raises PipelinePreconditionError."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=False)

        with self.assertRaises(PipelinePreconditionError):
            list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-no-asd-no-bypass",
                )
            )

    @patch("controller_service.pipeline.SpeechToSpeechClient")
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_combined_bypass_s2s_and_bypass_asd(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """Both bypass_s2s=True and bypass_asd=True -> only LipSync runs."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream(bypass_asd=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-both-bypass",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()
        context.abort.assert_not_called()


@pytest.mark.unit
class TestControllerBackendSelection(unittest.TestCase):
    """Construction-time S2S backend selection checks (no pipeline threads)."""

    @patch.dict(os.environ, {"S2S_SERVICE": "UNKNOWN_BACKEND"})
    def test_init_rejects_unknown_s2s_backend(self) -> None:
        """Controller init raises ValueError for unknown S2S_SERVICE values."""
        with self.assertRaises(ValueError):
            ControllerService(
                lipsync_server=MagicMock(),
                s2s_server=MagicMock(),
                asd_server=None,
            )

    @patch.dict(os.environ, {"S2S_SERVICE": "EL_DUBBING"})
    def test_s2s_output_audio_format_el_dubbing(self) -> None:
        """EL_DUBBING backend sets s2s_output_audio_format to MP3."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        self.assertEqual(controller.s2s_output_audio_format, "MP3")

    @patch.dict(os.environ, {"S2S_SERVICE": "CAMB_DUBBING"})
    def test_s2s_output_audio_format_camb_is_mp3(self) -> None:
        """CAMB_DUBBING backend outputs MP3 via CambAI alt-format API."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        self.assertEqual(controller.s2s_output_audio_format, "MP3")


def _make_bypass_s2s_request_stream(
    lipsync_codec: int = AUDIO_CODEC_MP3,
    include_asd_config: bool = True,
    bypass_asd: bool = False,
    include_translated_audio: bool = True,
) -> Iterator[ContentLocalizationRequest]:
    """Build a request stream for bypass-S2S mode.

    Sends controller_config with bypass_s2s=True, no s2s_config,
    then translated audio alongside original audio/video. The ASD config is
    included by default because ASD-active requests require one.
    """
    msgs: list[ContentLocalizationRequest] = [
        ContentLocalizationRequest(
            controller_config=ContentLocalizationConfig(
                bypass_s2s=True,
                bypass_asd=bypass_asd,
            ),
        ),
        ContentLocalizationRequest(
            lipsync_config=LipsyncConfig(input_audio_codec=lipsync_codec),
        ),
    ]
    if include_asd_config:
        msgs.append(
            ContentLocalizationRequest(
                asd_config=ActiveSpeakerDetectionConfig(
                    input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV),
                ),
            )
        )
    msgs.extend(
        [
            ContentLocalizationRequest(audio_data=b"original-audio"),
            ContentLocalizationRequest(video_file_data=b"client-video"),
        ]
    )
    if include_translated_audio:
        msgs.append(ContentLocalizationRequest(translated_audio_data=b"translated-mp3"))
    return iter(msgs)


@pytest.mark.integration
@patch.dict(os.environ, {"S2S_SERVICE": "EL_DUBBING"})
class TestControllerRequestValidation(unittest.TestCase):
    """Request-stream contract validation at the controller.

    These tests drive the full push-mode pipeline through ``infer()``,
    spinning real threads, so they carry the integration marker.
    """

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.pipeline.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    @patch("controller_service.helpers.CONFIG_POLL_TIMEOUT", 0.1)
    def test_asd_active_without_config_raises_invalid_argument(self) -> None:
        """ASD active with no asd_config in the stream raises PipelineInputError."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(include_asd_config=False)

        with self.assertRaises(PipelineInputError):
            list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-no-asd-config",
                )
            )

    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_bypass_s2s_without_translated_audio_raises(self) -> None:
        """bypass_s2s=True with no translated_audio_data raises PipelineInputError."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream(
            bypass_asd=True,
            include_translated_audio=False,
        )

        with self.assertRaises(PipelineInputError):
            list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-bypass-no-translated",
                )
            )

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_translated_audio_without_bypass_raises(self) -> None:
        """translated_audio_data with bypass_s2s=False raises PipelineInputError."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(
            bypass_asd=True,
            include_translated_audio=True,
        )

        with self.assertRaises(PipelineInputError):
            list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-unexpected-translated",
                )
            )

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    def test_translated_audio_without_bypass_emits_no_output_before_error(self) -> None:
        """An invalid translated-audio request produces no responses at all."""

        class _DrainingLipsyncClient:
            """Consumes its whole input before responding, so the deserializer
            has routed the forbidden chunk by the time a response exists."""

            def __init__(self, handle: object) -> None:
                self.handle = handle

            def __call__(
                self,
                request_iterator: Iterator[object],
                output_buffer: Any,
                context: object,
                request_id: str,
            ) -> None:
                _ = context
                _ = request_id
                for _req in request_iterator:
                    pass
                output_buffer.put(LipsyncResponse(video_file_data=b"lipsync-video"))
                output_buffer.done = True

        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(
            bypass_asd=True,
            include_translated_audio=True,
        )

        responses = []
        with (
            patch("controller_service.pipeline.LipsyncClient", new=_DrainingLipsyncClient),
            self.assertRaises(PipelineInputError),
        ):
            for response in controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-no-partial-output",
            ):
                responses.append(response)

        self.assertEqual(responses, [])

    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_s2s_receives_audio_before_inbound_stream_completes(self) -> None:
        """S2S consumes audio while the inbound stream is still open."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        s2s_got_audio = threading.Event()
        observed = {"audio_before_stream_end": False}

        class _RecordingS2SClient:
            def __init__(self, handle: object) -> None:
                self.handle = handle

            def __call__(
                self,
                request_iterator: Iterator[object],
                output_buffer: Any,
                context: object,
                request_id: str,
            ) -> None:
                _ = context
                _ = request_id
                for req in request_iterator:
                    if req.HasField("audio_data"):
                        s2s_got_audio.set()
                        break
                for _req in request_iterator:
                    pass
                output_buffer.put(
                    SpeechToSpeechResponse(audio_data=b"s2s-audio", audio_format="mp3")
                )
                output_buffer.done = True

        def requests() -> Iterator[ContentLocalizationRequest]:
            yield ContentLocalizationRequest(
                controller_config=ContentLocalizationConfig(bypass_s2s=False, bypass_asd=True),
            )
            yield ContentLocalizationRequest(
                s2s_config=SpeechToSpeechConfig(target_language="es"),
            )
            yield ContentLocalizationRequest(
                lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
            )
            yield ContentLocalizationRequest(audio_data=b"chunk-1")
            # Holding the stream open here proves S2S consumed the audio
            # without waiting for the inbound stream to complete.
            observed["audio_before_stream_end"] = s2s_got_audio.wait(timeout=10.0)
            yield ContentLocalizationRequest(video_file_data=b"client-video")

        with patch(
            "controller_service.pipeline.SpeechToSpeechClient",
            new=_RecordingS2SClient,
        ):
            responses = list(
                controller.infer(
                    request_iterator=requests(),
                    context=context,
                    request_id="req-ungated-s2s",
                )
            )

        self.assertTrue(observed["audio_before_stream_end"])
        self.assertEqual(len(responses), 1)

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_client_request_id_is_echoed_in_responses(self) -> None:
        """Responses carry the client-supplied request_id when one is sent."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()

        def requests() -> Iterator[ContentLocalizationRequest]:
            for message in _make_request_stream_with_configs(bypass_asd=True):
                message.request_id = "client-correlation-7"
                yield message

        responses = list(
            controller.infer(
                request_iterator=requests(),
                context=context,
                request_id="server-generated-id",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].request_id, "client-correlation-7")

    @patch("controller_service.pipeline.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.pipeline.LipsyncClient", new=_FakeLipsyncClient)
    def test_input_audio_config_on_controller_config_sets_s2s_format(self) -> None:
        """The controller config's input_audio_config declares the S2S input codec."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = iter(
            [
                ContentLocalizationRequest(
                    controller_config=ContentLocalizationConfig(
                        bypass_s2s=False,
                        bypass_asd=True,
                        input_audio_config=AudioConfig(encoding=AUDIO_CODEC_MP3),
                    ),
                ),
                ContentLocalizationRequest(
                    s2s_config=SpeechToSpeechConfig(target_language="es"),
                ),
                ContentLocalizationRequest(
                    lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
                ),
                ContentLocalizationRequest(audio_data=b"client-audio"),
                ContentLocalizationRequest(video_file_data=b"client-video"),
            ]
        )

        with patch("controller_service.pipeline.to_s2s_request") as mock_to_s2s:
            mock_to_s2s.side_effect = lambda req, input_audio_format: SpeechToSpeechRequest(
                audio_data=req.audio_data,
                audio_format=input_audio_format,
            )
            responses = list(
                controller.infer(
                    request_iterator=requests,
                    context=context,
                    request_id="req-input-codec",
                )
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(mock_to_s2s.call_args.kwargs["input_audio_format"], "MP3")

    @patch("controller_service.pipeline.LipsyncClient")
    def test_lipsync_keepalives_are_forwarded_to_the_client(
        self,
        mock_lipsync_client: MagicMock,
    ) -> None:
        """LipSync keepalives surface as controller keepalive responses."""

        class _KeepaliveThenVideoLipsyncClient:
            def __init__(self, handle: object) -> None:
                self.handle = handle

            def __call__(
                self,
                request_iterator: Iterator[object],
                output_buffer: Any,
                context: object,
                request_id: str,
            ) -> None:
                _ = context
                _ = request_id
                for _, _req in zip(range(4), request_iterator, strict=False):
                    pass
                keepalive = LipsyncResponse()
                keepalive.keepalive.SetInParent()
                output_buffer.put(keepalive)
                output_buffer.put(LipsyncResponse(video_file_data=b"lipsync-video"))
                output_buffer.done = True

        mock_lipsync_client.side_effect = _KeepaliveThenVideoLipsyncClient
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream(bypass_asd=True)

        responses = list(
            controller.infer(
                request_iterator=requests,
                context=context,
                request_id="req-keepalive",
            )
        )

        self.assertEqual(len(responses), 2)
        self.assertTrue(responses[0].HasField("keepalive"))
        self.assertEqual(responses[0].request_id, "req-keepalive")
        self.assertEqual(responses[1].video_file_data, b"lipsync-video")
        self.assertEqual(responses[1].request_id, "req-keepalive")
