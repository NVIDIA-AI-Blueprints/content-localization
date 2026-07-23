# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller service helper functions and codec/format mappings."""

import os
import queue as queue_module

from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV

from common.base_utils import logger
from common.buffers import Buffer

_CODEC_TO_FORMAT = {AUDIO_CODEC_WAV: "WAV", AUDIO_CODEC_MP3: "MP3"}
_FORMAT_TO_CODEC = {"WAV": AUDIO_CODEC_WAV, "MP3": AUDIO_CODEC_MP3}

_S2S_OUTPUT_FORMAT: dict[str, str] = {
    "EL_DUBBING": "MP3",
    "CAMB_DUBBING": "MP3",
}

# Seconds to wait for each per-request config message (controller_config,
# asd_config, lipsync_config) before treating it as absent. Configs are
# the first messages the client sends, so they arrive quickly. A long
# timeout only matters when an older client omits a config field.
CONFIG_POLL_TIMEOUT: float = float(os.environ.get("CONTROLLER_CONFIG_POLL_TIMEOUT", "5.0"))

# Seconds to wait for the deserializer thread and client threads to
# finish during the finally-block cleanup of each request.
CONTROLLER_CLEANUP_TIMEOUT: float = float(os.environ.get("CONTROLLER_CLEANUP_TIMEOUT", "10.0"))


def _audio_codec_to_format_string(codec: int) -> str:
    """Map an ``AudioCodec`` enum value to its string representation.

    Args:
        codec (int): Protobuf ``AudioCodec`` enum value.

    Returns:
        str: Uppercase format string (e.g. ``"WAV"``, ``"MP3"``).
            Falls back to ``"WAV"`` for unknown codecs.

    Examples:
        >>> from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
        >>> _audio_codec_to_format_string(codec=AUDIO_CODEC_MP3)
        'MP3'
    """
    return _CODEC_TO_FORMAT.get(codec, "WAV")


def _extract_config(
    buffer: Buffer,
    field_name: str,
    timeout: float | None = None,
) -> object | None:
    """Read a single config message from a 1-queue buffer.

    Blocks for up to *timeout* seconds waiting for the config to arrive.
    Returns ``None`` if no config message is received.

    Args:
        buffer (Buffer): Single-queue buffer expected to hold at most
            one config.
        field_name (str): Protobuf field name to extract
            (e.g. ``"asd_config"``).
        timeout (float | None): Maximum seconds to wait. ``None`` (default)
            uses the current :data:`CONFIG_POLL_TIMEOUT`
            (env ``CONTROLLER_CONFIG_POLL_TIMEOUT``, default ``5.0``).

    Returns:
        object | None: The extracted config protobuf message, or
            ``None`` if nothing arrived within *timeout*.

    Examples:
        >>> from common.buffers import Buffer
        >>> buf = Buffer(num_queues=1)
        >>> _extract_config(
        ...     buffer=buf,
        ...     field_name="asd_config",
        ...     timeout=1.0,
        ... )  # returns None (empty)
    """
    if timeout is None:
        timeout = CONFIG_POLL_TIMEOUT
    try:
        req = buffer.get(consumer_id=0, timeout=timeout)
        if req.HasField(field_name):
            return getattr(req, field_name)
    except queue_module.Empty:
        logger.debug(f"No {field_name} received from client within {timeout}s timeout")
    return None
