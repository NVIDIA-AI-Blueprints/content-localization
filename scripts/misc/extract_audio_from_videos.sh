#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Extract audio from video files (single file or directory)
#
# Usage:
#   ./scripts/misc/extract_audio_from_videos.sh <input_path> <output_path> [sample_rate] [channels] [format]
#
# Arguments:
#   input_path   - Video file path OR directory containing video files
#   output_path  - Output audio file path OR directory to save extracted audio files
#   sample_rate  - Target sample rate in Hz (default: 16000)
#   channels     - Number of channels: 1 for mono, 2 for stereo (default: 1)
#   format       - Output format: wav, mp3, flac (default: wav)
#
# Examples:
#   ./scripts/misc/extract_audio_from_videos.sh video.mp4 audio.wav
#   ./scripts/misc/extract_audio_from_videos.sh videos/ audio/
#   ./scripts/misc/extract_audio_from_videos.sh videos/ audio/ 44100 2 mp3

set -e  # Exit on error

# Check if ffmpeg is installed
if ! command -v ffmpeg >/dev/null 2>&1 && ! type ffmpeg >/dev/null 2>&1 && [ ! -f /usr/bin/ffmpeg ]; then
    echo "Error: ffmpeg is not installed"
    echo "Install with: sudo apt install ffmpeg"
    exit 1
fi

# Ensure ffmpeg is in PATH or use full path
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_CMD="ffmpeg"
elif [ -f /usr/bin/ffmpeg ]; then
    FFMPEG_CMD="/usr/bin/ffmpeg"
else
    FFMPEG_CMD="ffmpeg"  # Fallback, will fail if not found
fi

# Parse arguments
INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
SAMPLE_RATE="${3:-16000}"
CHANNELS="${4:-1}"
FORMAT="${5:-wav}"

# Validate arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_path> <output_path> [sample_rate] [channels] [format]"
    echo ""
    echo "Arguments:"
    echo "  input_path   - Video file path OR directory containing video files (required)"
    echo "  output_path  - Output audio file path OR directory to save extracted audio files (required)"
    echo "  sample_rate  - Target sample rate in Hz (default: 16000)"
    echo "  channels     - Number of channels: 1 for mono, 2 for stereo (default: 1)"
    echo "  format       - Output format: wav, mp3, flac (default: wav)"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4 audio.wav"
    echo "  $0 videos/ audio/"
    echo "  $0 videos/ audio/ 44100 2 mp3"
    exit 1
fi

# Determine if input is a file or directory
INPUT_IS_FILE=false
if [ -f "$INPUT_DIR" ]; then
    INPUT_IS_FILE=true
elif [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input path '$INPUT_DIR' does not exist (not a file or directory)"
    exit 1
fi

# Create output directory if it doesn't exist (or parent directory if output is a file)
if [ "$INPUT_IS_FILE" = true ]; then
    # If input is a file, output should be a file - create parent directory
    OUTPUT_PARENT=$(dirname "$OUTPUT_DIR")
    if [ "$OUTPUT_PARENT" != "." ] && [ "$OUTPUT_PARENT" != "" ]; then
        mkdir -p "$OUTPUT_PARENT"
    fi
else
    # If input is a directory, output should be a directory
    mkdir -p "$OUTPUT_DIR"
fi

# Video extensions to process (POSIX-compatible space-separated list)
VIDEO_EXTENSIONS="mp4 avi mkv mov webm MP4 AVI MKV MOV WEBM"

# Count files
total_count=0
success_count=0
fail_count=0

echo "=========================================="
echo "Audio Extraction Script"
echo "=========================================="
if [ "$INPUT_IS_FILE" = true ]; then
    echo "Input file:       $INPUT_DIR"
    echo "Output file:      $OUTPUT_DIR"
else
    echo "Input directory:  $INPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
fi
echo "Sample rate:      $SAMPLE_RATE Hz"
echo "Channels:         $CHANNELS ($([ $CHANNELS -eq 1 ] && echo 'mono' || echo 'stereo'))"
echo "Format:           $FORMAT"
echo "=========================================="
echo ""

# Process single file or directory
if [ "$INPUT_IS_FILE" = true ]; then
    # Process single file
    video_file="$INPUT_DIR"
    filename=$(basename "$video_file")
    
    # Determine output file path
    if [ -d "$OUTPUT_DIR" ]; then
        # If output is a directory, create filename based on input
        basename="${filename%.*}"
        output_file="$OUTPUT_DIR/${basename}.${FORMAT}"
    else
        # If output is a file path, use it directly
        output_file="$OUTPUT_DIR"
    fi
    
    total_count=1
    echo "Processing: $filename"
    
    if "$FFMPEG_CMD" -i "$video_file" \
        -vn \
        -acodec pcm_s16le \
        -ar "$SAMPLE_RATE" \
        -ac "$CHANNELS" \
        -y \
        "$output_file" \
        -loglevel error 2>&1; then
        echo "  ✓ Extracted: $(basename "$output_file")"
        success_count=$((success_count + 1))
    else
        echo "  ✗ Failed: $filename"
        fail_count=$((fail_count + 1))
    fi
    echo ""
else
    # Find and process all video files in directory
    # Use a temporary file to avoid subshell issues with piped while loop
    TMPFILE=$(mktemp)
    trap "rm -f $TMPFILE" EXIT
    
    for ext in $VIDEO_EXTENSIONS; do
        find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.${ext}" >> "$TMPFILE"
    done
    
    while IFS= read -r video_file; do
        total_count=$((total_count + 1))
        
        # Get filename without path and extension
        filename=$(basename "$video_file")
        basename="${filename%.*}"
        
        # Create output path
        output_file="$OUTPUT_DIR/${basename}.${FORMAT}"
        
        # Extract audio using ffmpeg
        echo "Processing: $filename"
        
        if "$FFMPEG_CMD" -i "$video_file" \
            -vn \
            -acodec pcm_s16le \
            -ar "$SAMPLE_RATE" \
            -ac "$CHANNELS" \
            -y \
            "$output_file" \
            -loglevel error 2>&1; then
            echo "  ✓ Extracted: ${basename}.${FORMAT}"
            success_count=$((success_count + 1))
        else
            echo "  ✗ Failed: $filename"
            fail_count=$((fail_count + 1))
        fi
        echo ""
        
    done < "$TMPFILE"
    rm -f "$TMPFILE"
fi

# Print summary
echo "=========================================="
echo "Processing Complete"
echo "=========================================="
echo "Total files:      $total_count"
echo "✓ Successful:     $success_count"
echo "✗ Failed:         $fail_count"
echo "=========================================="

# Exit with error code if any files failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi

exit 0

