# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video encoding helpers for the LipSync client."""

from nvidia.ai4m.video.v1.video_pb2 import CustomEncodingParams
from nvidia.ai4m.video.v1.video_pb2 import LossyEncoding
from nvidia.ai4m.video.v1.video_pb2 import VideoEncoding

from client.lipsync.config import LipSyncConfig
from common.proto_utils import create_protobuf_any_value


def create_custom_encoding_params(params: dict) -> CustomEncodingParams:
    """Create CustomEncodingParams from a dictionary.

    Args:
        params (dict): Dictionary with string keys and values of type
            bool, int, float, or str.

    Returns:
        CustomEncodingParams: Protobuf message from video.v1.

    Examples:
        >>> msg = create_custom_encoding_params(
        ...     params={"crf": 23},
        ... )  # doctest: +SKIP
    """
    if not isinstance(params, dict):
        raise ValueError("Custom encoding parameters must be a JSON object (e.g. '{\"crf\": 23}')")

    # Protobuf map keys are strings; validate early to surface a
    # user-friendly CLI error instead of a lower-level protobuf failure.
    non_string_keys = [key for key in params if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError("Custom encoding parameter keys must all be strings")

    custom_params = CustomEncodingParams()
    for key, value in params.items():
        custom_params.custom[key].CopyFrom(create_protobuf_any_value(value))
    return custom_params


def create_output_video_encoding(config: LipSyncConfig) -> VideoEncoding:
    """Create VideoEncoding based on configuration.

    Args:
        config (LipSyncConfig): LipSync configuration object.

    Returns:
        VideoEncoding: Protobuf message from video.v1.

    Examples:
        >>> enc = create_output_video_encoding(config=cfg)  # doctest: +SKIP
    """
    if config.lossless:
        return VideoEncoding(lossless=True)

    if config.custom_encoding_params:
        return VideoEncoding(
            custom_encoding=create_custom_encoding_params(
                params=config.custom_encoding_params,
            )
        )

    return VideoEncoding(
        lossy=LossyEncoding(
            bitrate_mbps=config.bitrate_mbps,
            idr_interval=config.idr_interval,
        )
    )
