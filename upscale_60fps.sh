#!/bin/bash
# High-Quality Video Upscaler with 60 FPS
# Created by Md. Mahir Labib
# Copyright © 2026 Md. Mahir Labib. All rights reserved.
# Uses Real-ESRGAN (4x upscale) + FFmpeg minterpolate (60 FPS)

set -e

INPUT_VIDEO="${1:-video.mp4}"
OUTPUT_VIDEO="${2:-video-upscaled.mp4}"
TARGET_FPS="${3:-60}"
SCALE="${4:-4}"

echo "============================================================"
echo "🎥 HIGH-QUALITY VIDEO UPSCALER"
echo "   Created by Md. Mahir Labib"
echo "   Real-ESRGAN ${SCALE}x + ${TARGET_FPS} FPS Motion Interpolation"
echo "============================================================"
echo ""

# Check if input exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "❌ Input file not found: $INPUT_VIDEO"
    exit 1
fi

# Make Real-ESRGAN executable
chmod +x ./realesrgan-ncnn-vulkan

# Create temp directories
TEMP_DIR="temp_upscale_$$"
LR_FRAMES="$TEMP_DIR/lr_frames"
HR_FRAMES="$TEMP_DIR/hr_frames"
mkdir -p "$LR_FRAMES" "$HR_FRAMES"

echo "📁 Using temp directory: $TEMP_DIR"

# Get original FPS
ORIGINAL_FPS=$(ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "$INPUT_VIDEO" | head -1)
echo "📽️ Original video FPS: $ORIGINAL_FPS"

# Extract frames
echo ""
echo "📽️ Extracting frames from video..."
ffmpeg -y -i "$INPUT_VIDEO" -qscale:v 2 "$LR_FRAMES/frame_%07d.png" 2>/dev/null
FRAME_COUNT=$(ls -1 "$LR_FRAMES"/*.png | wc -l | tr -d ' ')
echo "✅ Extracted $FRAME_COUNT frames"

# Upscale with Real-ESRGAN
echo ""
echo "🔍 Upscaling frames with Real-ESRGAN (${SCALE}x)..."
echo "   This may take a while depending on frame count and resolution..."
./realesrgan-ncnn-vulkan -i "$LR_FRAMES" -o "$HR_FRAMES" -n realesrgan-x4plus -s $SCALE -f png

echo "✅ All frames upscaled!"

# Create video with 60 FPS interpolation
echo ""
echo "🎬 Creating smooth ${TARGET_FPS} FPS video with motion interpolation..."

# Parse original FPS for framerate calculation
if [[ "$ORIGINAL_FPS" == *"/"* ]]; then
    FPS_NUM=$(echo "$ORIGINAL_FPS" | cut -d'/' -f1)
    FPS_DEN=$(echo "$ORIGINAL_FPS" | cut -d'/' -f2)
    ORIG_FPS_FLOAT=$(echo "scale=2; $FPS_NUM / $FPS_DEN" | bc)
else
    ORIG_FPS_FLOAT="$ORIGINAL_FPS"
fi

echo "   Interpolating from $ORIG_FPS_FLOAT FPS to $TARGET_FPS FPS..."

# Create upscaled video with frame interpolation to 60 FPS
ffmpeg -y \
    -framerate "$ORIGINAL_FPS" \
    -i "$HR_FRAMES/frame_%07d.png" \
    -vf "minterpolate=fps=$TARGET_FPS:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" \
    -c:v libx264 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p \
    -an \
    "$TEMP_DIR/temp_no_audio.mp4" 2>/dev/null

echo "✅ Video created with $TARGET_FPS FPS"

# Add audio from original
echo ""
echo "🔊 Adding audio from original video..."
ffmpeg -y \
    -i "$TEMP_DIR/temp_no_audio.mp4" \
    -i "$INPUT_VIDEO" \
    -c:v copy \
    -c:a aac \
    -map 0:v:0 \
    -map 1:a:0? \
    -shortest \
    "$OUTPUT_VIDEO" 2>/dev/null || cp "$TEMP_DIR/temp_no_audio.mp4" "$OUTPUT_VIDEO"

echo "✅ Audio added!"

# Cleanup
echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Show final info
echo ""
echo "============================================================"
echo "✅ SUCCESS! Output saved to: $OUTPUT_VIDEO"
echo "============================================================"

# Get output video info
OUTPUT_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$OUTPUT_VIDEO")
WIDTH=$(echo "$OUTPUT_INFO" | cut -d',' -f1)
HEIGHT=$(echo "$OUTPUT_INFO" | cut -d',' -f2)
OUT_FPS=$(echo "$OUTPUT_INFO" | cut -d',' -f3)

echo "   Resolution: ${WIDTH}x${HEIGHT}"
echo "   Frame Rate: $OUT_FPS"
echo "   File Size:  $(ls -lh "$OUTPUT_VIDEO" | awk '{print $5}')"
echo ""
