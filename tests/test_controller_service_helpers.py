# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import grpc
import pytest
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse

from common.errors import PipelineInputError
from common.errors import PipelinePreconditionError
from controller_service.conversions import create_wav_header
from controller_service.service import ControllerService
from controller_service.servicer import ControllerServiceServicer

pytestmark = pytest.mark.unit


def test_create_wav_header_basic():
    header = create_wav_header(n_channels=1, sample_width=2, frame_rate=16000, n_frames=0)
    assert header.startswith(b"RIFF")
    assert b"WAVE" in header
    assert len(header) == 44


def test_check_services_health_calls_servers():
    controller = ControllerService(
        lipsync_server=MagicMock(),
        s2s_server=MagicMock(),
        asd_server=MagicMock(),
    )
    services = controller._create_request_services()
    controller._check_services_health(services=services)
    services.lipsync_server.is_healthy.assert_called_once()
    services.s2s_server.is_healthy.assert_called_once()
    services.asd_server.is_healthy.assert_called_once()


def test_create_request_services_opens_and_closes_per_request_channels():
    """Each request gets its own cloned handles; close touches only those."""
    lipsync_server = MagicMock()
    s2s_server = MagicMock()
    controller = ControllerService(
        lipsync_server=lipsync_server,
        s2s_server=s2s_server,
        asd_server=None,
    )
    services = controller._create_request_services()

    lipsync_server.clone.assert_called_once()
    s2s_server.clone.assert_called_once()
    assert services.lipsync_server is lipsync_server.clone.return_value
    assert services.asd_server is None
    services.lipsync_server.connect.assert_called_once()
    services.s2s_server.connect.assert_called_once()
    # Channel options carry the configured message size on every hop.
    options = services.lipsync_server.connect.call_args.kwargs["channel_options"]
    assert ("grpc.max_receive_message_length", controller.message_size) in options
    assert ("grpc.max_send_message_length", controller.message_size) in options

    services.close()
    services.lipsync_server.close.assert_called_once()
    services.s2s_server.close.assert_called_once()
    # The shared configured handles are never opened or closed per request.
    lipsync_server.connect.assert_not_called()
    lipsync_server.close.assert_not_called()


def test_create_request_services_closes_partial_channels_on_connect_failure():
    """A failed later connect() closes channels already opened for the request."""
    lipsync_server = MagicMock()
    s2s_server = MagicMock()
    asd_server = MagicMock()
    controller = ControllerService(
        lipsync_server=lipsync_server,
        s2s_server=s2s_server,
        asd_server=asd_server,
    )
    s2s_server.clone.return_value.connect.side_effect = ConnectionError("s2s unreachable")

    with pytest.raises(ConnectionError, match="s2s unreachable"):
        controller._create_request_services()

    # The already-connected LipSync channel is released; the never-connected
    # ASD clone is closed too, which handle.close() treats as a no-op.
    lipsync_server.clone.return_value.close.assert_called_once()
    s2s_server.clone.return_value.close.assert_called_once()
    asd_server.clone.return_value.close.assert_called_once()
    asd_server.clone.return_value.connect.assert_not_called()


def test_servicer_streams_responses():
    response_iter = iter([ContentLocalizationResponse(), ContentLocalizationResponse()])
    service = MagicMock()
    service.infer.return_value = response_iter
    service.intermediate_audio_format = "MP3"
    servicer = ControllerServiceServicer(service)
    context = MagicMock()
    context.peer.return_value = "peer"

    results = list(
        servicer.StreamContentLocalization(
            request_iterator=iter([]),
            context=context,
        )
    )
    assert len(results) == 2
    service.infer.assert_called_once()


def test_servicer_maps_pipeline_errors_to_specific_status_codes():
    """The servicer's single abort point preserves specific status codes."""
    cases = [
        (PipelineInputError("bad request stream"), grpc.StatusCode.INVALID_ARGUMENT),
        (PipelinePreconditionError("service missing"), grpc.StatusCode.FAILED_PRECONDITION),
        (ConnectionError("NIM unreachable"), grpc.StatusCode.UNAVAILABLE),
        (RuntimeError("unexpected"), grpc.StatusCode.INTERNAL),
    ]
    for error, expected_code in cases:
        service = MagicMock()
        service.infer.side_effect = error
        servicer = ControllerServiceServicer(service)
        context = MagicMock()
        context.peer.return_value = "peer"
        context.abort.side_effect = RuntimeError("aborted")

        with pytest.raises(RuntimeError, match="aborted"):
            list(
                servicer.StreamContentLocalization(
                    request_iterator=iter([]),
                    context=context,
                )
            )
        code, details = context.abort.call_args.args
        assert code == expected_code
        assert str(error) in details


def test_servicer_aborts_on_infer_error():
    service = MagicMock()
    service.infer.side_effect = RuntimeError("boom")
    service.intermediate_audio_format = "MP3"
    servicer = ControllerServiceServicer(service)
    context = MagicMock()
    context.peer.return_value = "peer"
    context.abort.side_effect = RuntimeError("aborted")

    with pytest.raises(RuntimeError, match="aborted"):
        list(
            servicer.StreamContentLocalization(
                request_iterator=iter([]),
                context=context,
            )
        )
    context.abort.assert_called_once()
