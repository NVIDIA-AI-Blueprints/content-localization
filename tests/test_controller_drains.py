# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for unused-input drains and setup-failure cleanup coverage."""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import BackgroundAudioConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig

from common.errors import PipelineInputError
from controller_service import pipeline
from controller_service.config import _PipelineConfig
from controller_service.config import _RequestServices
from controller_service.deserializer import ContentLocalizationDeserializer
from controller_service.service import ControllerService

pytestmark = pytest.mark.unit


def _make_service() -> ControllerService:
    return ControllerService(
        lipsync_server=MagicMock(),
        s2s_server=MagicMock(),
        asd_server=None,
    )


def _make_request_services() -> _RequestServices:
    return _RequestServices(
        lipsync_server=MagicMock(),
        s2s_server=MagicMock(),
        asd_server=None,
    )


def _make_pipeline_config(
    bypass_s2s: bool,
    lipsync_config: LipsyncConfig,
) -> _PipelineConfig:
    return _PipelineConfig(
        bypass_s2s=bypass_s2s,
        bypass_asd=True,
        asd_active=False,
        asd_config=None,
        lipsync_config=lipsync_config,
        input_audio_format="wav",
        s2s_output_format=None if bypass_s2s else "MP3",
    )


class TestForbiddenTranslatedAudioCheck(unittest.TestCase):
    """Eager rejection of translated audio while S2S is active."""

    def test_raises_with_chunk_count_when_not_bypassing(self) -> None:
        """Routed translated audio with bypass_s2s=False raises immediately."""
        deserializer = ContentLocalizationDeserializer(iter([]))
        for _ in range(2):
            deserializer.translated_audio_buffer.put(
                ContentLocalizationRequest(translated_audio_data=b"chunk")
            )

        with self.assertRaises(PipelineInputError) as raised:
            pipeline._raise_if_forbidden_translated_audio(
                deserializer=deserializer,
                bypass_s2s=False,
            )
        self.assertIn("2 translated_audio_data", str(raised.exception))

    def test_no_translated_audio_passes(self) -> None:
        """No routed translated audio means no violation."""
        deserializer = ContentLocalizationDeserializer(iter([]))

        pipeline._raise_if_forbidden_translated_audio(
            deserializer=deserializer,
            bypass_s2s=False,
        )

    def test_bypass_mode_allows_translated_audio(self) -> None:
        """bypass_s2s=True legitimately streams translated audio."""
        deserializer = ContentLocalizationDeserializer(iter([]))
        deserializer.translated_audio_buffer.put(
            ContentLocalizationRequest(translated_audio_data=b"chunk")
        )

        pipeline._raise_if_forbidden_translated_audio(
            deserializer=deserializer,
            bypass_s2s=True,
        )


class TestUnusedInputDrains(unittest.TestCase):
    """Unused translated/background audio buffers must be drained, not leaked."""

    def test_wrong_mode_buffers_are_drained(self) -> None:
        """translated audio (non-bypass) and background audio (undeclared) drain."""
        deserializer = ContentLocalizationDeserializer(iter([]))
        for _ in range(3):
            deserializer.translated_audio_buffer.put(
                ContentLocalizationRequest(translated_audio_data=b"chunk")
            )
            deserializer.background_audio_buffer.put(
                ContentLocalizationRequest(background_audio_data=b"chunk")
            )
        deserializer.translated_audio_buffer.done = True
        deserializer.background_audio_buffer.done = True

        threads = pipeline._start_unused_input_drains(
            deserializer=deserializer,
            pipeline_config=_make_pipeline_config(
                bypass_s2s=False,
                lipsync_config=LipsyncConfig(),
            ),
            request_id="r-drain",
        )

        self.assertEqual(len(threads), 1)
        for thread in threads:
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
        self.assertEqual(deserializer.translated_audio_buffer.qsize(0), 0)
        self.assertEqual(deserializer.background_audio_buffer.qsize(0), 0)

    def test_no_drains_when_buffers_are_consumed_by_pipeline(self) -> None:
        """No drain threads when bypass-S2S consumes translated audio and
        background audio is declared (consumed by the LipSync leg)."""
        deserializer = ContentLocalizationDeserializer(iter([]))
        lipsync_config = LipsyncConfig(
            background_audio_config=BackgroundAudioConfig(is_background_audio_provided=True),
        )

        threads = pipeline._start_unused_input_drains(
            deserializer=deserializer,
            pipeline_config=_make_pipeline_config(
                bypass_s2s=True,
                lipsync_config=lipsync_config,
            ),
            request_id="r-nodrain",
        )

        self.assertEqual(threads, [])


class TestSetupFailureCleanup(unittest.TestCase):
    """_controller_impl must clean up when setup raises before the yield loop."""

    def test_config_extraction_failure_still_runs_cleanup(self) -> None:
        """An exception in config extraction stops the deserializer via cleanup."""
        service = _make_service()
        context = MagicMock()

        with (
            patch.object(
                service,
                "_extract_and_apply_configs",
                side_effect=RuntimeError("config extraction failed"),
            ),
            patch("controller_service.pipeline._cleanup_threads") as mock_cleanup,
            self.assertRaises(RuntimeError),
        ):
            list(
                service._controller_impl(
                    request_iterator=iter([]),
                    context=context,
                    request_id="r-setupfail",
                    services=_make_request_services(),
                )
            )

        mock_cleanup.assert_called_once()
        # No pipeline threads were launched before the failure.
        self.assertEqual(mock_cleanup.call_args.kwargs["threads"], [])

    def test_partial_thread_launch_failure_cleans_started_threads(self) -> None:
        """A failure launching a later thread still joins the earlier ones."""
        service = _make_service()
        context = MagicMock()
        s2s_thread = MagicMock(name="s2s-thread")

        with (
            patch.object(service, "_extract_and_apply_configs") as mock_cfg,
            patch.object(service, "_check_services_health", return_value=True),
            patch(
                "controller_service.pipeline._start_s2s_thread",
                return_value=s2s_thread,
            ),
            patch(
                "controller_service.pipeline._start_asd_thread",
                side_effect=RuntimeError("asd launch failed"),
            ),
            patch("controller_service.pipeline._cleanup_threads") as mock_cleanup,
        ):
            mock_cfg.return_value = _make_pipeline_config(
                bypass_s2s=False,
                lipsync_config=LipsyncConfig(),
            )
            with self.assertRaises(RuntimeError):
                list(
                    service._controller_impl(
                        request_iterator=iter([]),
                        context=context,
                        request_id="r-partial",
                        services=_make_request_services(),
                    )
                )

        mock_cleanup.assert_called_once()
        self.assertEqual(mock_cleanup.call_args.kwargs["threads"], [s2s_thread])


class TestUnconsumedItemsWarning(unittest.TestCase):
    """Buffers ending a request partially consumed are reported in the logs."""

    def test_warns_for_partially_consumed_buffer(self) -> None:
        deserializer = ContentLocalizationDeserializer(iter([]))
        deserializer.audio_buffer.put(ContentLocalizationRequest(audio_data=b"chunk"))
        # Consume only one of the two fan-out queues.
        _ = deserializer.audio_buffer.get(0)

        with patch("controller_service.pipeline.logger") as mock_logger:
            pipeline._warn_on_unconsumed_items(deserializer=deserializer)

        warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
        self.assertEqual(len(warnings), 1)
        self.assertIn("audio_buffer", warnings[0])
        self.assertIn("1 item(s) put", warnings[0])

    def test_silent_when_all_buffers_fully_consumed(self) -> None:
        deserializer = ContentLocalizationDeserializer(iter([]))
        deserializer.video_buffer.put(ContentLocalizationRequest(video_file_data=b"v"))
        _ = deserializer.video_buffer.get(0)
        _ = deserializer.video_buffer.get(1)

        with patch("controller_service.pipeline.logger") as mock_logger:
            pipeline._warn_on_unconsumed_items(deserializer=deserializer)

        mock_logger.warning.assert_not_called()
