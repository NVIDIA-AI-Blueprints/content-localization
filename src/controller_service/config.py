# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request configuration and service-handle containers for the Controller.

``_PipelineConfig`` bundles the effective pipeline configuration extracted
from the client's config buffers for a single request, and
``_RequestServices`` holds that request's channels to the downstream S2S,
ASD, and LipSync services.
"""

import dataclasses

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig

from common.nims import ActiveSpeakerDetectionHandle
from common.nims import LipsyncHandle
from common.nims import SpeechToSpeechHandle


@dataclasses.dataclass
class _PipelineConfig:
    """Bundled configuration extracted per-request from client buffers.

    Groups the bypass flags, NIM configs, and derived audio format
    so they can be threaded through the pipeline helpers.

    Attributes:
        bypass_s2s: Skip S2S, use translated audio for LipSync.
        bypass_asd: Skip ASD, LipSync uses internal face detection.
        asd_active: Single predicate for "ASD runs in this request":
            ASD is configured and not bypassed. Used at every ASD gate
            so the gates cannot diverge.
        asd_config: ASD protobuf config, or ``None`` when bypassed.
        lipsync_config: LipSync protobuf config (with server
            overrides applied).
        input_audio_format: Derived from ASD config audio encoding.
        s2s_output_format: Resolved S2S output format for this
            request. ``None`` when S2S is bypassed.
    """

    bypass_s2s: bool
    bypass_asd: bool
    asd_active: bool
    asd_config: ActiveSpeakerDetectionConfig | None
    lipsync_config: LipsyncConfig
    input_audio_format: str
    s2s_output_format: str | None


@dataclasses.dataclass
class _RequestServices:
    """Per-request handles to the downstream services.

    Each request opens its own channels so that closing them during request
    cleanup cannot cancel streams belonging to other in-flight requests.

    Attributes:
        lipsync_server: LipSync handle (always present).
        s2s_server: S2S handle, or ``None`` when S2S is not configured.
        asd_server: ASD handle, or ``None`` when ASD is not configured.
    """

    lipsync_server: LipsyncHandle
    s2s_server: SpeechToSpeechHandle | None
    asd_server: ActiveSpeakerDetectionHandle | None

    def close(self) -> None:
        """Close every open channel held by this request.

        Explicitly terminating each NIM channel ends the connection
        immediately so a single-concurrency NIM (e.g. LipSync) is freed for
        the next request's health check.

        Returns:
            None.

        Examples:
            >>> services.close()  # doctest: +SKIP
        """
        for handle in (self.lipsync_server, self.s2s_server, self.asd_server):
            if handle is not None:
                handle.close()
