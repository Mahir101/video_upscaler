#!/usr/bin/env bash
# Photo Enhancement Shell Wrapper
# Created by Md. Mahir Labib
# Copyright © 2026 Md. Mahir Labib. All rights reserved.
#
# Uses the NCNN binary (realesrgan-ncnn-vulkan) — no Python required.
# For face restoration + batch processing use photo_enhance.py instead.
#
# Usage:
#   ./upscale_photo.sh photo.jpg                      # 4x, default model
#   ./upscale_photo.sh photo.jpg -o out.png           # explicit output
#   ./upscale_photo.sh photo.jpg -m realesrgan-x4plus-anime
#   ./upscale_photo.sh photos/                        # batch folder
#   ./upscale_photo.sh photos/ -o enhanced/

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INPUT=""
OUTPUT=""
MODEL="realesrgan-x4plus"
SCALE=4
TILE=1024

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
RESET='\033[0m'

print_header() {
    echo ""
    echo "================================================================"
    echo "  PHOTO ENHANCEMENT — NCNN Binary Pipeline"
    echo "  Created by Md. Mahir Labib"
    echo "  Real-ESRGAN NCNN Vulkan"
    echo "================================================================"
    echo ""
}

usage() {
    echo "Usage: $0 <input-image-or-folder> [options]"
    echo ""
    echo "Options:"
    echo "  -o, --output <path>   Output file or folder (default: auto)"
    echo "  -m, --model  <name>   NCNN model (default: realesrgan-x4plus)"
    echo "                        Available: realesrgan-x4plus"
    echo "                                   realesrgan-x4plus-anime"
    echo "                                   realesrnet-x4plus"
    echo "  -s, --scale  <n>      Scale factor (default: 4)"
    echo "  -t, --tile   <n>      Tile size for VRAM (default: 1024)"
    echo "  -h, --help            Show this message"
    echo ""
    echo "Examples:"
    echo "  $0 photo.jpg"
    echo "  $0 photo.jpg -o photo_4k.png -m realesrgan-x4plus"
    echo "  $0 photos/  -o photos/enhanced/"
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then usage; fi
INPUT="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        -m|--model)   MODEL="$2";  shift 2 ;;
        -s|--scale)   SCALE="$2";  shift 2 ;;
        -t|--tile)    TILE="$2";   shift 2 ;;
        -h|--help)    usage ;;
        *) echo -e "${RED}Unknown option: $1${RESET}"; usage ;;
    esac
done

print_header

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -e "$INPUT" ]]; then
    echo -e "${RED}ERROR: Input not found: $INPUT${RESET}"
    exit 1
fi

if [[ ! -x "./realesrgan-ncnn-vulkan" ]]; then
    echo -e "${RED}ERROR: realesrgan-ncnn-vulkan binary not found.${RESET}"
    echo "Run:  ./scripts/fetch_realesrgan_ncnn.sh"
    exit 1
fi

# Validate model token (only allow safe characters)
if [[ ! "$MODEL" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo -e "${RED}ERROR: Invalid model name: $MODEL${RESET}"
    exit 1
fi

# Validate numeric args
if [[ ! "$SCALE" =~ ^[0-9]+$ ]] || [[ ! "$TILE" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}ERROR: scale and tile must be positive integers${RESET}"
    exit 1
fi

echo -e "  Input   : ${CYAN}$INPUT${RESET}"
echo -e "  Model   : ${CYAN}$MODEL${RESET}"
echo -e "  Scale   : ${CYAN}${SCALE}x${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Single image mode
# ---------------------------------------------------------------------------
process_single() {
    local src="$1"
    local dest="$2"

    # derive output name if not set
    if [[ -z "$dest" ]]; then
        local stem="${src%.*}"
        dest="${stem}_enhanced.png"
    fi

    echo -e "  ${YELLOW}Enhancing:${RESET} $src  →  $dest"
    ./realesrgan-ncnn-vulkan \
        -i "$src" \
        -o "$dest" \
        -n "$MODEL" \
        -s "$SCALE" \
        -t "$TILE" \
        -f png

    echo -e "  ${GREEN}Saved: $dest${RESET}"
}

# ---------------------------------------------------------------------------
# Batch folder mode
# ---------------------------------------------------------------------------
process_folder() {
    local src_dir="$1"
    local out_dir="${2:-${src_dir}/enhanced}"

    mkdir -p "$out_dir"
    local count=0

    for img in "$src_dir"/*.{jpg,jpeg,png,bmp,webp,JPG,JPEG,PNG,BMP,WEBP}; do
        [[ -f "$img" ]] || continue
        local stem
        stem="$(basename "${img%.*}")"
        process_single "$img" "$out_dir/${stem}_enhanced.png"
        (( count++ )) || true
    done

    if [[ $count -eq 0 ]]; then
        echo -e "${YELLOW}No images found in: $src_dir${RESET}"
    else
        echo ""
        echo -e "${GREEN}Done: $count image(s) enhanced → $out_dir${RESET}"
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
START=$(date +%s)

if [[ -d "$INPUT" ]]; then
    process_folder "$INPUT" "$OUTPUT"
else
    process_single "$INPUT" "$OUTPUT"
fi

END=$(date +%s)
echo ""
echo -e "  Total time: $((END - START))s"
echo "================================================================"
