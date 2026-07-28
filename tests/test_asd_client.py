# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ASD client functionality."""

from unittest.mock import MagicMock

import pytest

from client.direct.stream_adapters import speaker_info_from_asd_response
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionHandle


@pytest.mark.unit
class TestASDClient:
    """Test cases for ASD client functionality."""

    def test_speaker_info_from_asd_response_with_speaker(self):
        """Test speaker_info_from_asd_response when ASD response contains speaker data."""
        # Create mock ASD DetectActiveSpeakerResponse with speaker data
        mock_response = MagicMock()
        mock_speaker = MagicMock()
        mock_speaker.speaker_bbox.x = 100
        mock_speaker.speaker_bbox.y = 200
        mock_speaker.speaker_bbox.width = 300
        mock_speaker.speaker_bbox.height = 400
        mock_speaker.face_id = 1
        mock_speaker.is_speaking = True
        mock_result = MagicMock()
        mock_result.speaker_data = [mock_speaker]
        mock_result.frame_id = 0
        mock_response.active_speaker_detection_result = mock_result

        # Create response iterator
        response_iterator = iter([mock_response])

        # Call the function
        speaker_info_iter = speaker_info_from_asd_response(response_iterator)

        # Get the first LipsyncInputData
        lipsync_input = next(speaker_info_iter)

        # Verify the speaker info data
        assert len(lipsync_input.per_frame_speaker_infos) == 1
        frame_info = lipsync_input.per_frame_speaker_infos[0]
        assert frame_info.frame_id == 0
        assert len(frame_info.speaker_infos) == 1
        speaker = frame_info.speaker_infos[0]
        assert speaker.speaker_bbox.x == 100
        assert speaker.speaker_bbox.y == 200
        assert speaker.speaker_bbox.width == 300
        assert speaker.speaker_bbox.height == 400
        assert speaker.speaker_id == 1
        assert speaker.is_speaking is True

    def test_speaker_info_from_asd_response_without_speaker(self):
        """Test speaker_info_from_asd_response when ASD response has no speaker data."""
        # Create mock ASD DetectActiveSpeakerResponse without speaker data
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_result.speaker_data = []
        mock_result.frame_id = 0
        mock_response.active_speaker_detection_result = mock_result

        # Create response iterator
        response_iterator = iter([mock_response])

        # Call the function
        speaker_info_iter = speaker_info_from_asd_response(response_iterator)

        # Get the first LipsyncInputData
        lipsync_input = next(speaker_info_iter)

        # Verify the speaker info data (should be default/empty speaker info)
        assert len(lipsync_input.per_frame_speaker_infos) == 1
        frame_info = lipsync_input.per_frame_speaker_infos[0]
        assert len(frame_info.speaker_infos) == 0

    def test_speaker_info_from_asd_response_multiple_frames(self):
        """Test speaker_info_from_asd_response with multiple frames."""
        # Create mock ASD responses
        mock_response1 = MagicMock()
        mock_speaker1 = MagicMock()
        mock_speaker1.speaker_bbox.x = 100
        mock_speaker1.speaker_bbox.y = 200
        mock_speaker1.speaker_bbox.width = 300
        mock_speaker1.speaker_bbox.height = 400
        mock_speaker1.face_id = 1
        mock_speaker1.is_speaking = True
        mock_result1 = MagicMock()
        mock_result1.speaker_data = [mock_speaker1]
        mock_result1.frame_id = 0
        mock_response1.active_speaker_detection_result = mock_result1

        mock_response2 = MagicMock()
        mock_result2 = MagicMock()
        mock_result2.speaker_data = []  # No speaker in second frame
        mock_result2.frame_id = 1
        mock_response2.active_speaker_detection_result = mock_result2

        # Create response iterator
        response_iterator = iter([mock_response1, mock_response2])

        # Call the function
        speaker_info_iter = speaker_info_from_asd_response(response_iterator)

        # Get the first LipsyncInputData
        lipsync_input1 = next(speaker_info_iter)
        assert len(lipsync_input1.per_frame_speaker_infos) == 1
        frame_info1 = lipsync_input1.per_frame_speaker_infos[0]
        assert len(frame_info1.speaker_infos) == 1
        speaker1 = frame_info1.speaker_infos[0]
        assert speaker1.speaker_bbox.x == 100
        assert speaker1.speaker_bbox.y == 200
        assert speaker1.speaker_bbox.width == 300
        assert speaker1.speaker_bbox.height == 400

        # Get the second LipsyncInputData
        lipsync_input2 = next(speaker_info_iter)
        assert len(lipsync_input2.per_frame_speaker_infos) == 1
        frame_info2 = lipsync_input2.per_frame_speaker_infos[0]
        assert len(frame_info2.speaker_infos) == 0

    def test_asd_client_connection(self):
        """Test ASD client connection setup."""
        handle = ActiveSpeakerDetectionHandle(host="localhost", port=50051)
        client = ActiveSpeakerDetectionClient(handle=handle)

        assert client.handle == handle


if __name__ == "__main__":
    pytest.main([__file__])
