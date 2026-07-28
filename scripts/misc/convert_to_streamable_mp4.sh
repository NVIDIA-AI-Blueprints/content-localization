#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to convert a regular video file to a mp4 file suitable for streaming.

# Usage:
# ./scripts/misc/convert_to_streamable_mp4.sh <input_video_file.ext>
# where ext can be avi, mkv, mp4, etc.
# output file will be <input_video_file>_streamable.ext

# Example:
# ./scripts/misc/convert_to_streamable_mp4.sh input.mp4
# output file will be input_streamable.mp4

# check if ffmpeg is installed
# Use POSIX redirection (>/dev/null 2>&1) instead of bash-only `&>` so the check
# works even when this script is sourced/run under dash (/bin/sh on Ubuntu),
# where `&>` backgrounds the command and misfires into the install branch.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFMPEG could not be found, installing FFMPEG package."
    sudo apt-get install -y ffmpeg
fi

# Check if input file exists
if [ ! -f "$1" ]; then
    echo "Input file does not exist."
    exit 1
fi

filename=$1
filename_no_ext="${filename%.*}"
output_streamable_file="${filename_no_ext}_streamable.mp4"
ffmpeg -y -loglevel quiet -nostats -hide_banner -i "${filename}" -c copy -movflags faststart "${output_streamable_file}"

echo "${filename} converted successfully to ${output_streamable_file}"
