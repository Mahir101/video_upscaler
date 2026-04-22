#!/usr/bin/env bash
# Download all model weights used by this project.
# Run once after cloning — models are never committed to git.
#
# Models fetched:
#   Real-ESRGAN Python weights  → ~/.cache/realesrgan/
#   GFPGAN face-restoration     → ~/.cache/gfpgan/
#   NCNN binary (macOS)         → ./  (realesrgan-ncnn-vulkan + model files)
#
# Usage:
#   ./scripts/fetch_models.sh            # fetch all
#   ./scripts/fetch_models.sh --esrgan   # Python ESRGAN weights only
#   ./scripts/fetch_models.sh --gfpgan   # GFPGAN weights only
#   ./scripts/fetch_models.sh --ncnn     # NCNN binary only

set -euo pipefail

GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

ESRGAN_CACHE="${HOME}/.cache/realesrgan"
GFPGAN_CACHE="${HOME}/.cache/gfpgan"

FETCH_ESRGAN=true
FETCH_GFPGAN=true
FETCH_NCNN=true

for arg in "$@"; do
    case "$arg" in
        --esrgan) FETCH_GFPGAN=false; FETCH_NCNN=false ;;
        --gfpgan) FETCH_ESRGAN=false; FETCH_NCNN=false ;;
        --ncnn)   FETCH_ESRGAN=false; FETCH_GFPGAN=false ;;
    esac
done

# ---------------------------------------------------------------------------
safe_download() {
    local url="$1"
    local dest="$2"
    local label="${3:-$(basename "$dest")}"

    if [[ -f "$dest" ]]; then
        echo -e "  ${GREEN}Already cached:${RESET} $label"
        return
    fi

    echo -e "  ${YELLOW}Downloading:${RESET} $label"
    mkdir -p "$(dirname "$dest")"
    curl -L --fail --progress-bar -o "$dest" "$url"
    echo -e "  ${GREEN}Saved:${RESET} $dest"
}

echo ""
echo "================================================================"
echo "  Model Downloader"
echo "  Created by Md. Mahir Labib"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Real-ESRGAN Python weights
# ---------------------------------------------------------------------------
if [[ "$FETCH_ESRGAN" == "true" ]]; then
    echo -e "${CYAN}[1/3] Real-ESRGAN Python weights → $ESRGAN_CACHE${RESET}"

    safe_download \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        "$ESRGAN_CACHE/RealESRGAN_x4plus.pth" \
        "RealESRGAN_x4plus.pth (general 4x)"

    safe_download \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
        "$ESRGAN_CACHE/RealESRGAN_x2plus.pth" \
        "RealESRGAN_x2plus.pth (light 2x)"

    safe_download \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" \
        "$ESRGAN_CACHE/RealESRGAN_x4plus_anime_6B.pth" \
        "RealESRGAN_x4plus_anime_6B.pth (anime 4x)"

    safe_download \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrnet_x4plus.pth" \
        "$ESRGAN_CACHE/realesrnet_x4plus.pth" \
        "realesrnet_x4plus.pth (natural scenes 4x)"

    echo ""
fi

# ---------------------------------------------------------------------------
# 2. GFPGAN face restoration weights
# ---------------------------------------------------------------------------
if [[ "$FETCH_GFPGAN" == "true" ]]; then
    echo -e "${CYAN}[2/3] GFPGAN face restoration weights → $GFPGAN_CACHE${RESET}"

    safe_download \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth" \
        "$GFPGAN_CACHE/GFPGANv1.3.pth" \
        "GFPGANv1.3.pth"

    safe_download \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
        "$GFPGAN_CACHE/GFPGANv1.4.pth" \
        "GFPGANv1.4.pth (recommended)"

    # Detection model used internally by GFPGAN
    safe_download \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
        "$GFPGAN_CACHE/detection_Resnet50_Final.pth" \
        "detection_Resnet50_Final.pth (face detection)"

    safe_download \
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
        "$GFPGAN_CACHE/parsing_parsenet.pth" \
        "parsing_parsenet.pth (face parsing)"

    echo ""
fi

# ---------------------------------------------------------------------------
# 3. Real-ESRGAN NCNN Vulkan binary (macOS)
# ---------------------------------------------------------------------------
if [[ "$FETCH_NCNN" == "true" ]]; then
    echo -e "${CYAN}[3/3] Real-ESRGAN NCNN Vulkan binary${RESET}"

    if [[ -x "./realesrgan-ncnn-vulkan" ]]; then
        echo -e "  ${GREEN}Already present:${RESET} ./realesrgan-ncnn-vulkan"
    else
        VERSION="${REALESRGAN_NCNN_VERSION:-20220424}"
        ARCHIVE="realesrgan-ncnn-vulkan-${VERSION}-macos.zip"
        URL="https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v${VERSION}/${ARCHIVE}"

        echo "  Downloading NCNN binary archive ..."
        curl -L --fail --progress-bar -o "$ARCHIVE" "$URL"
        unzip -o "$ARCHIVE" -d "realesrgan-ncnn-extracted"

        # Binary may be at root or in a subdirectory
        BIN=$(find realesrgan-ncnn-extracted -name "realesrgan-ncnn-vulkan" -type f | head -1)
        if [[ -n "$BIN" ]]; then
            cp "$BIN" ./realesrgan-ncnn-vulkan
            chmod +x ./realesrgan-ncnn-vulkan

            # Copy model param/bin files if present
            MODEL_DIR=$(find realesrgan-ncnn-extracted -name "*.param" | head -1 | xargs dirname 2>/dev/null || true)
            if [[ -n "$MODEL_DIR" && -d "$MODEL_DIR" ]]; then
                cp "$MODEL_DIR"/*.param ./ 2>/dev/null || true
                cp "$MODEL_DIR"/*.bin   ./ 2>/dev/null || true
                echo -e "  ${GREEN}NCNN model files copied.${RESET}"
            fi

            rm -rf realesrgan-ncnn-extracted "$ARCHIVE"
            echo -e "  ${GREEN}Ready: ./realesrgan-ncnn-vulkan${RESET}"
        else
            echo "  WARNING: Could not find binary inside archive."
        fi
    fi

    echo ""
fi

echo "================================================================"
echo -e "  ${GREEN}All models ready.${RESET}"
echo ""
echo "  Python pipeline:   python photo_enhance.py --list-models"
echo "  NCNN shell:        ./upscale_photo.sh photo.jpg"
echo "  C++ binary:        ./upscaler_ult --input photo.jpg"
echo "================================================================"
echo ""
