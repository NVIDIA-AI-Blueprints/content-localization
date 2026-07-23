# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASD response writers (CSV output for speaker detection data)."""

import csv
from collections.abc import Iterator
from pathlib import Path

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)

from common.base_utils import logger

# Column order for the ASD speaker-info CSV. Exposed as a module-level constant
# so documentation and downstream readers (e.g. the LipSync client) can derive
# the schema from code instead of hand-copying the header.
SPEAKER_INFO_CSV_FIELDNAMES = [
    "frame_id",
    "x",
    "y",
    "width",
    "height",
    "diarized_speaker_id",
    "face_id",
    "is_speaking",
    "face_detection_confidence",
]


def write_asd_outputs_from_response(
    response_iter: Iterator[DetectActiveSpeakerResponse],
    output_csv_path: str,
) -> None:
    """Write Active Speaker Detection data from response iterator to CSV.

    Parses ActiveSpeakerDetectionResult messages and writes per-speaker
    bounding box and detection metadata to CSV.

    Args:
        response_iter (Iterator[DetectActiveSpeakerResponse]): Iterator
            of DetectActiveSpeakerResponse objects.
        output_csv_path (str): Path to the output CSV file.

    Examples:
        >>> write_asd_outputs_from_response(
        ...     response_iter=responses,
        ...     output_csv_path="/tmp/asd_output.csv",
        ... )  # doctest: +SKIP
    """
    response_count = 0
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SPEAKER_INFO_CSV_FIELDNAMES)
        writer.writeheader()

        for response in response_iter:
            response_count += 1
            if response_count % 100 == 0:
                logger.debug(f"Processing ASD Response: {response_count}")

            # Extract the active speaker detection result
            if response.HasField("active_speaker_detection_result"):
                result = response.active_speaker_detection_result
                if result.speaker_data:
                    for speaker in result.speaker_data:
                        bbox = speaker.speaker_bbox
                        writer.writerow(
                            {
                                "frame_id": result.frame_id,
                                "x": bbox.x,
                                "y": bbox.y,
                                "width": bbox.width,
                                "height": bbox.height,
                                "diarized_speaker_id": speaker.diarized_speaker_id,
                                "face_id": speaker.face_id,
                                "is_speaking": speaker.is_speaking,
                                "face_detection_confidence": speaker.face_detection_confidence,
                            }
                        )
                else:
                    # No speakers detected for this frame
                    writer.writerow(
                        {
                            "frame_id": result.frame_id,
                            "x": 0,
                            "y": 0,
                            "width": 0,
                            "height": 0,
                            "diarized_speaker_id": 0,
                            "face_id": 0,
                            "is_speaking": False,
                            "face_detection_confidence": 0.0,
                        }
                    )

    logger.info(f"ASD data written to {output_csv_path} ({response_count} responses processed).")
