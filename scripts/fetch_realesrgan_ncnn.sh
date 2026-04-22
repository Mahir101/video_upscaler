#!/usr/bin/env bash
set -euo pipefail

VERSION="${REALESRGAN_NCNN_VERSION:-20220424}"
ARCHIVE="realesrgan-ncnn-vulkan-${VERSION}-macos.zip"
URL="https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v${VERSION}/${ARCHIVE}"

echo "Downloading ${URL}"
curl -L --fail --output "${ARCHIVE}" "${URL}"
unzip -o "${ARCHIVE}"

if [ -f "realesrgan-ncnn-vulkan-${VERSION}-macos/realesrgan-ncnn-vulkan" ]; then
  cp "realesrgan-ncnn-vulkan-${VERSION}-macos/realesrgan-ncnn-vulkan" ./realesrgan-ncnn-vulkan
fi

chmod +x ./realesrgan-ncnn-vulkan
echo "Ready: ./realesrgan-ncnn-vulkan"
