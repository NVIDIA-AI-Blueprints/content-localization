# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a single video through the controller pipeline."""

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerStub

from client.common.audio import create_audio_source
from client.common.audio import detect_audio_codec
from client.controller.config import ControllerConfig
from client.controller.request_generators import create_controller_request_generator
from client.controller.response_writers import write_output_from_response
from common.source_sink.base import BaseFileSimulator
from common.source_sink.grpc.audio import AudioSourceSimulator
from common.source_sink.grpc.video import VideoSourceSimulator


def run_single_video(
    audio_path: str,
    video_path: str,
    output_path: str,
    config: ControllerConfig,
    diarization_info: AudioDiarizationInfo | None = None,
    translated_audio_path: str | None = None,
) -> None:
    """Run one video through the controller content-localization pipeline.

    Creates input sources from the given audio and video files, streams
    them to the controller service, and writes the output video.

    Args:
        audio_path (str): Path to the extracted WAV audio file.
        video_path (str): Path to the input MP4 video file.
        output_path (str): Path for the output MP4 video file.
        config (ControllerConfig): Pipeline configuration bundle.
        diarization_info (AudioDiarizationInfo | None): Optional
            diarization metadata for ASD.
        translated_audio_path (str | None): Path to pre-translated WAV or MP3
            audio. When provided, S2S is bypassed and this audio is
            streamed directly to LipSync (ASD still runs on the original
            audio).

    Raises:
        grpc.RpcError: If the controller service returns an error.
        RuntimeError: If input files are invalid or output fails.

    Examples:
        >>> run_single_video(
        ...     audio_path="audio.wav",
        ...     video_path="video.mp4",
        ...     output_path="output.mp4",
        ...     config=cfg,
        ... )  # doctest: +SKIP
    """
    # Initialize before the try so the finally block can close whatever was
    # successfully opened, even if a later constructor raises.
    input_audio: AudioSourceSimulator | None = None
    input_video: VideoSourceSimulator | None = None
    translated_audio_source: BaseFileSimulator | None = None
    channel = grpc.insecure_channel(config.controller_server)

    try:
        input_audio = AudioSourceSimulator(file_path=audio_path)
        input_video = VideoSourceSimulator(file_path=video_path)
        # Pre-translated audio bypasses S2S; ASD still consumes the original
        # audio. The source is selected by file content, not extension, so
        # MP3 data inside a .wav filename streams as raw bytes instead of
        # failing WAV parsing.
        if translated_audio_path:
            translated_audio_source = create_audio_source(file_path=translated_audio_path)

        stub = ContentLocalizationControllerStub(channel)
        request_generator = create_controller_request_generator(
            audio_source=input_audio,
            video_source=input_video,
            chunk_size_audio_secs=config.chunk_size_audio_secs,
            chunk_size_video_bytes=config.chunk_size_video_bytes,
            s2s_config=config.s2s_config,
            asd_config=config.asd_config,
            lipsync_config=config.lipsync_config,
            diarization_info=diarization_info,
            translated_audio_source=translated_audio_source,
            bypass_asd=config.bypass_asd,
            diarization_rows_per_chunk=config.diarization_rows_per_chunk,
            input_audio_codec=detect_audio_codec(file_path=audio_path),
        )

        response_iter = stub.StreamContentLocalization(request_generator)

        write_output_from_response(
            response_iter=response_iter,
            output_mp4_path=output_path,
            chunk_size_video_bytes=config.chunk_size_video_bytes,
        )
    finally:
        if input_audio is not None and input_audio.is_open():
            input_audio.close()
        if input_video is not None and input_video.is_open():
            input_video.close()
        if translated_audio_source is not None and translated_audio_source.is_open():
            translated_audio_source.close()
        channel.close()
