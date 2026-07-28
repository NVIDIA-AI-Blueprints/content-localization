# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared service-bypass resolution for the client applications."""

import argparse

from common.base_utils import logger


def resolve_bypass_asd(
    args: argparse.Namespace,
    auto_bypass_asd: bool = True,
) -> bool:
    """Determine whether the ASD service should be bypassed.

    An explicit ``--bypass-asd`` flag always wins. Otherwise, when
    *auto_bypass_asd* is ``True`` and no diarization file was provided,
    ASD is bypassed automatically because it cannot produce useful
    speaker segments without diarization input. Callers that discover
    diarization later (e.g. the batch processing client, which resolves
    a diarization file per video) should pass ``auto_bypass_asd=False``
    to keep ASD enabled.

    Args:
        args (argparse.Namespace): Parsed CLI arguments. Reads the
            optional ``bypass_asd`` and ``diarization_file`` attributes.
        auto_bypass_asd (bool): When ``True`` (default), bypass ASD
            automatically if no ``--diarization-file`` was provided.

    Returns:
        bool: ``True`` if ASD should be bypassed.

    Examples:
        >>> import argparse
        >>> args = argparse.Namespace(bypass_asd=False, diarization_file=None)
        >>> resolve_bypass_asd(args=args)
        True
        >>> resolve_bypass_asd(args=args, auto_bypass_asd=False)
        False
    """
    if getattr(args, "bypass_asd", False):
        return True
    if auto_bypass_asd and getattr(args, "diarization_file", None) is None:
        logger.info(
            "ASD bypassed: no diarization file provided — LipSync will use internal face detection"
        )
        return True
    return False
