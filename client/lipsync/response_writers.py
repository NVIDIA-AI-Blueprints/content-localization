# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LipSync response writers (video output from inference responses)."""

import os
import time
from collections.abc import Iterator

from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.lipsync.config import LipSyncConfig
from common.base_utils import logger


def write_output_file_from_response(
    response_iter: Iterator[lipsync_pb2.LipsyncResponse],
    output_filepath: os.PathLike,
) -> int:
    """Write the video data from LipsyncResponse messages to an output file.

    Responses without ``video_file_data`` (e.g. acknowledgments) are
    skipped by field inspection, so the very first video chunk is never
    dropped.

    Args:
        response_iter (Iterator[lipsync_pb2.LipsyncResponse]): Iterator of
            LipsyncResponse messages from the LipSync service.
        output_filepath (os.PathLike): Path where the output video will be saved.

    Returns:
        int: Number of video chunks written.

    Raises:
        RuntimeError: If there are errors writing the output file.

    Examples:
        >>> chunks = write_output_file_from_response(
        ...     response_iter=responses,
        ...     output_filepath="/tmp/output.mp4",
        ... )  # doctest: +SKIP
    """
    try:
        chunk_number = 0
        with open(output_filepath, "wb") as fd:
            for response in response_iter:
                if response.HasField("video_file_data"):
                    if chunk_number == 0:
                        logger.info(f"Writing output file {output_filepath}")
                    chunk_number += 1
                    fd.write(response.video_file_data)
        logger.info(f"Output file written successfully: {output_filepath} ({chunk_number} chunks)")
        return chunk_number
    except OSError as e:
        raise RuntimeError(f"Error writing output file: {e}") from e


def process_response_iter(
    response_iter: Iterator[lipsync_pb2.LipsyncResponse],
    lipsync_config: LipSyncConfig,
) -> None:
    """Process gRPC response iterator and write output.

    Args:
        response_iter (Iterator[lipsync_pb2.LipsyncResponse]): Iterator
            of LipsyncResponse messages.
        lipsync_config (LipSyncConfig): Configuration for the LipSync service.

    Raises:
        RuntimeError: If no video data is received from the service.
        Exception: If any errors occur during processing.

    Examples:
        >>> process_response_iter(
        ...     response_iter=responses,
        ...     lipsync_config=cfg,
        ... )  # doctest: +SKIP
    """
    try:
        start_time = time.time()

        chunk_count = write_output_file_from_response(
            response_iter=response_iter,
            output_filepath=lipsync_config.output_filepath,
        )
        if chunk_count == 0:
            raise RuntimeError("No video data received from LipSync service")

        end_time = time.time()
        logger.info(f"Function invocation completed in {end_time - start_time:.2f}s")

    except Exception:
        logger.exception("An error occurred while processing LipSync responses")
        raise
