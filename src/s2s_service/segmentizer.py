# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segmentizer utilities. - Splits text into chunks using different strategies."""

import time
import typing

from common.base_utils import logger


def sentence_segmentizer(chunks: typing.Iterator[str]) -> typing.Iterator[str]:
    """Segment input text chunks into sentences, yielding each as a chunk.

    Args:
        chunks (Iterator[str]): The input text chunks.

    Returns:
        Iterator[str]: The segmented text chunks, each ending with a space if not already present.
    """
    # These separators split into sentences
    splitters = (".", "?", "!")

    buffer = ""
    for text in chunks:
        if buffer.endswith(splitters):
            yield buffer if buffer.endswith(" ") else buffer + " "
            buffer = text
        elif text.startswith(splitters):
            output = buffer + text[0]
            yield output if output.endswith(" ") else output + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer:
        yield buffer + " "


def length_segmentizer(chunks: typing.Iterator[str], chunk_size: int = 200) -> typing.Iterator[str]:
    """Segment input text chunks to a maximum length of characters per chunk.

    Args:
        chunks (Iterator[str]): The input text chunks.
        chunk_size (int): The max length of the chunks.

    Returns:
        Iterator[str]: The segmented text chunks.
    """
    buffer = ""
    start_time_for_segment = time.time()
    count = 0
    for text in chunks:
        buffer += text
        while len(buffer) >= chunk_size:
            logger.debug(f"Time taken for segment {count}: {time.time() - start_time_for_segment}")
            start_time_for_segment = time.time()
            yield buffer[:chunk_size]
            buffer = buffer[chunk_size:]

    if buffer:
        logger.debug(f"Time taken for segment {count}: {time.time() - start_time_for_segment}")
        start_time_for_segment = time.time()
        yield buffer + " "
