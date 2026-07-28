# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LipSync request stream generators for the standalone LipSync client."""

import csv
import itertools
from collections.abc import Iterator

from nvidia.ai4m.common.v1.common_pb2 import BoundingBox
from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.common.audio import DATA_CHUNK_SIZE
from client.lipsync.config import LipSyncConfig
from client.lipsync.constants import SPEAKER_INFO_FRAME_COUNT
from common.base_utils import logger
from common.feeder_stream import FeederSource
from common.feeder_stream import FeederStream


def speaker_info_csv_reader(
    reader: Iterator[list[str]],
    row_count: int,
) -> Iterator[list[list[str]]]:
    """Read CSV data in batches of multiple rows.

    Args:
        reader (Iterator[list[str]]): CSV reader object to read from.
        row_count (int): Number of rows to include in each batch.

    Yields:
        list[list[str]]: CSV rows in batches of the specified row count.

    Examples:
        >>> import csv
        >>> import io
        >>> rows = csv.reader(io.StringIO("a,b\\nc,d\\ne,f\\n"))
        >>> next(speaker_info_csv_reader(reader=rows, row_count=2))
        [['a', 'b'], ['c', 'd']]
    """
    while True:
        rows = list(itertools.islice(reader, row_count))
        if not rows:
            break
        yield rows


def _speaker_info_from_row(row: list) -> tuple[int, lipsync_pb2.SpeakerInfo]:
    """Parse a single CSV row into a frame ID and SpeakerInfo protobuf.

    Args:
        row (list): List containing speaker info columns in one of these formats:
            - [frame_id, x, y, width, height]
            - [frame_id, x, y, width, height, diarized_speaker_id, face_id,
              is_speaking, ...]
            frame_id is used as the frame identifier.
            x, y, width, height define the speaker bounding box.
            face_id and is_speaking are consumed when provided.

    Returns:
        tuple[int, lipsync_pb2.SpeakerInfo]: The frame ID and a SpeakerInfo
            protobuf message for one detected face.

    Examples:
        >>> fid, info = _speaker_info_from_row(
        ...     row=["0", "10", "20", "30", "40"],
        ... )  # doctest: +SKIP
    """
    frame_id, x, y, width, height = row[0], *map(float, row[1:5])
    speaker_info = lipsync_pb2.SpeakerInfo(
        speaker_bbox=BoundingBox(
            x=float(x),
            y=float(y),
            width=float(width),
            height=float(height),
        )
    )

    # Optional metadata written by ASD CSV output:
    # [frame_id,x,y,w,h,diarized_speaker_id,face_id,is_speaking,confidence]
    if len(row) >= 7 and row[6] != "":
        speaker_info.speaker_id = int(row[6])

    if len(row) >= 8 and row[7] != "":
        speaker_info.is_speaking = row[7].strip().lower() in {
            "1",
            "true",
            "t",
            "yes",
            "y",
        }

    return int(frame_id), speaker_info


def group_rows_into_per_frame_infos(
    rows: list[list],
) -> list[lipsync_pb2.SpeakerInfoPerFrame]:
    """Group CSV rows by frame ID into SpeakerInfoPerFrame messages.

    Multiple speakers in the same frame are grouped into a single
    ``SpeakerInfoPerFrame`` with all their ``SpeakerInfo`` entries,
    matching how the direct client streams ASD results to LipSync.

    Args:
        rows (list[list]): Batch of CSV rows from the ASD output.

    Returns:
        list[lipsync_pb2.SpeakerInfoPerFrame]: One message per unique
            frame ID, each containing all speakers detected in that frame.

    Examples:
        >>> infos = group_rows_into_per_frame_infos(
        ...     rows=[["0", "10", "20", "30", "40"]],
        ... )  # doctest: +SKIP
    """
    # Preserve frame ordering while grouping speakers per frame
    frames: dict[int, list[lipsync_pb2.SpeakerInfo]] = {}
    for row in rows:
        frame_id, speaker_info = _speaker_info_from_row(row)
        frames.setdefault(frame_id, []).append(speaker_info)

    return [
        lipsync_pb2.SpeakerInfoPerFrame(
            frame_id=frame_id,
            speaker_infos=speaker_infos,
        )
        for frame_id, speaker_infos in frames.items()
    ]


def _file_chunk_iterator(file_path: str) -> Iterator[bytes]:
    """Yield fixed-size byte chunks from a file.

    The file handle is opened lazily and closed when the iterator is
    exhausted or garbage collected, so it is safe to hand this iterator
    to a feeder thread.

    Args:
        file_path (str): Path to the file to stream.

    Yields:
        bytes: Chunks of up to ``DATA_CHUNK_SIZE`` bytes.

    Raises:
        RuntimeError: If the file cannot be read.

    Examples:
        >>> chunks = _file_chunk_iterator(file_path="video.mp4")  # doctest: +SKIP
    """
    try:
        with open(file_path, "rb") as f:
            while True:
                data = f.read(DATA_CHUNK_SIZE)
                if not data:
                    break
                yield data
    except OSError as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}") from e


def _speaker_info_batch_iterator(
    file_path: str,
) -> Iterator[list[lipsync_pb2.SpeakerInfoPerFrame]]:
    """Yield per-frame speaker-info batches from an ASD CSV file.

    Args:
        file_path (str): Path to the speaker-info CSV (with header row).

    Yields:
        list[lipsync_pb2.SpeakerInfoPerFrame]: Batches of grouped
            per-frame speaker infos.

    Raises:
        RuntimeError: If the CSV cannot be read or parsed.

    Examples:
        >>> batches = _speaker_info_batch_iterator(
        ...     file_path="speaker_info.csv",
        ... )  # doctest: +SKIP
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip the header row; an empty file simply yields no batches
            # (a bare next() would surface as "generator raised StopIteration").
            if next(reader, None) is None:
                return
            for rows in speaker_info_csv_reader(
                reader=reader,
                row_count=SPEAKER_INFO_FRAME_COUNT,
            ):
                yield group_rows_into_per_frame_infos(rows)
    except OSError as e:
        raise RuntimeError(f"Failed to read speaker info file {file_path}: {e}") from e
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to process speaker info data: {e}") from e


def _build_feeder_sources(
    lipsync_config: LipSyncConfig,
    audio_iterator: Iterator[bytes],
) -> list[FeederSource]:
    """Create one feeder source per input stream for the LipSync request.

    Args:
        lipsync_config (LipSyncConfig): Validated configuration with the
            input file paths.
        audio_iterator (Iterator[bytes]): Chunk iterator for the input
            audio file. Passed in (rather than built here) so the caller
            can send a priming chunk before the feeder threads start.

    Returns:
        list[FeederSource]: Video and audio sources, plus speaker-info
            and background-audio sources when the corresponding files
            are configured.

    Examples:
        >>> sources = _build_feeder_sources(
        ...     lipsync_config=cfg,
        ...     audio_iterator=iter([b"chunk"]),
        ... )  # doctest: +SKIP
    """
    sources: list[FeederSource] = [
        FeederSource(
            name="video",
            iterator=_file_chunk_iterator(file_path=lipsync_config.video_filepath),
            transform=lambda c: lipsync_pb2.LipsyncRequest(
                input=lipsync_pb2.LipsyncInputData(video_file_data=c),
            ),
        ),
        FeederSource(
            name="audio",
            iterator=audio_iterator,
            transform=lambda c: lipsync_pb2.LipsyncRequest(
                input=lipsync_pb2.LipsyncInputData(audio_file_data=c),
            ),
        ),
    ]
    if lipsync_config.speaker_info_filepath:
        sources.append(
            FeederSource(
                name="speaker_info",
                iterator=_speaker_info_batch_iterator(
                    file_path=lipsync_config.speaker_info_filepath,
                ),
                transform=lambda batch: lipsync_pb2.LipsyncRequest(
                    input=lipsync_pb2.LipsyncInputData(per_frame_speaker_infos=batch),
                ),
            )
        )
    if lipsync_config.background_audio_filepath:
        sources.append(
            FeederSource(
                name="background_audio",
                iterator=_file_chunk_iterator(
                    file_path=lipsync_config.background_audio_filepath,
                ),
                transform=lambda c: lipsync_pb2.LipsyncRequest(
                    input=lipsync_pb2.LipsyncInputData(background_audio_file_data=c),
                ),
            )
        )
    return sources


def generate_request_for_inference(
    lipsync_config: LipSyncConfig,
    config_proto: lipsync_pb2.LipsyncConfig,
) -> Iterator[lipsync_pb2.LipsyncRequest]:
    """Generate a stream of LipsyncRequest messages for the LipSync service.

    Sends the configuration message first, then a single audio priming
    chunk so the server initializes its sample rate/resampler before any
    video or speaker-info arrives, then concurrently merges the video,
    audio, optional speaker-info, and optional background-audio streams
    via :class:`~common.feeder_stream.FeederStream`, matching the
    streaming behavior of the ASD, direct, and controller clients.

    Args:
        lipsync_config (LipSyncConfig): Validated configuration with the
            input file paths.
        config_proto (lipsync_pb2.LipsyncConfig): Protobuf configuration
            message, built via ``lipsync_config_from_args`` and updated
            with the values resolved during validation.

    Yields:
        lipsync_pb2.LipsyncRequest: Messages containing either
            configuration or chunks of input data.

    Raises:
        RuntimeError: If an input file cannot be read.

    Examples:
        >>> gen = generate_request_for_inference(
        ...     lipsync_config=cfg,
        ...     config_proto=proto,
        ... )  # doctest: +SKIP
    """
    logger.debug("Generating request for inference")

    # Send configuration first so the service can initialize decoders
    yield lipsync_pb2.LipsyncRequest(config=config_proto)

    logger.debug("Sending data for inference")

    # Prime audio before the feeder threads start: the feeder merge gives
    # no cross-source ordering, but the server needs the first audio chunk
    # to initialize its sample rate/resampler before video/speaker-info
    # arrives (mirrors the direct client's lipsync adapter).
    audio_iterator = _file_chunk_iterator(file_path=lipsync_config.audio_filepath)
    primed_audio_chunks = 0
    try:
        first_audio_chunk = next(audio_iterator)
    except StopIteration:
        logger.info("LipSync standalone | audio stream empty, no priming chunk")
    else:
        primed_audio_chunks = 1
        yield lipsync_pb2.LipsyncRequest(
            input=lipsync_pb2.LipsyncInputData(audio_file_data=first_audio_chunk),
        )
        logger.info("LipSync standalone | audio priming chunk sent")

    stream: FeederStream[lipsync_pb2.LipsyncRequest] = FeederStream(
        sources=_build_feeder_sources(
            lipsync_config=lipsync_config,
            audio_iterator=audio_iterator,
        ),
    )
    stream.start(request_id="lipsync-standalone")
    try:
        yield from stream
    finally:
        stream.stop()

    counts = stream.chunk_counts
    logger.info(
        f"Data sending completed - Video: {counts.get('video', 0)}, "
        f"Audio: {counts.get('audio', 0) + primed_audio_chunks}, "
        f"Speaker info: {counts.get('speaker_info', 0)}, "
        f"Background audio: {counts.get('background_audio', 0)} chunks"
    )
    stream.raise_on_error()
