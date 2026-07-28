# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia.ai4m.lipsync.v1 import lipsync_pb2

# Constants for data handling
SPEAKER_INFO_FRAME_COUNT = 2048  # Number of speaker info entries to read at once
DEFAULT_BITRATE_MBPS = 20  # Mbps
DEFAULT_IDR_INTERVAL = 8  # frames
DEFAULT_VIDEO_PATH = "assets/sample_video_streamable.mp4"
DEFAULT_AUDIO_PATH = "assets/sample_audio.wav"


# Configuration mappings for different options
EXTEND_AUDIO_CONFIGS = {
    "unspecified": lipsync_pb2.ExtendAudio.EXTEND_AUDIO_UNSPECIFIED,
    "silence": lipsync_pb2.ExtendAudio.EXTEND_AUDIO_SILENCE,
}

# Configuration constants for extend video options
EXTEND_VIDEO_CONFIGS = {
    "unspecified": lipsync_pb2.ExtendVideo.EXTEND_VIDEO_UNSPECIFIED,
    "forward": lipsync_pb2.ExtendVideo.EXTEND_VIDEO_FORWARD,
    "reverse": lipsync_pb2.ExtendVideo.EXTEND_VIDEO_REVERSE,
}
