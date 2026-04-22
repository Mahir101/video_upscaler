<h1>Ultimate Video Upscaler</h1>
<p><strong>Created by Md. Mahir Labib</strong><br>
Copyright © 2026 Md. Mahir Labib. All rights reserved.</p>

<p>A professional-grade media engine for video upscaling and frame interpolation, optimized for <strong>Apple Silicon M4</strong>. This version features hardware-accelerated HEVC and ProRes encoding, Real-ESRGAN upscaling, optional RIFE integration when the RIFE binary is installed, FFmpeg interpolation fallback, and progress monitoring.</p>

<div align="center">
  <img src="https://img.shields.io/badge/Codec-HEVC%20%2F%20ProRes-blue?style=for-the-badge" alt="Codecs">
  <img src="https://img.shields.io/badge/Hardware-Apple%20M4-orange?style=for-the-badge&logo=apple" alt="Hardware">
  <img src="https://img.shields.io/badge/Interpolation-RIFE%20AI-green?style=for-the-badge" alt="AI">
</div>

## Key Features

- 🏎️ **Apple Silicon Suite**: Native support for `hevc_videotoolbox` and `prores_videotoolbox`.
- ✨ **Interpolation Options**: RIFE support when `rife-ncnn-vulkan` is installed; otherwise FFmpeg `minterpolate` is used.
- 📂 **Managed Temp Workspaces**: Temporary frame folders are cleaned on success, failure, or interrupt.
- 🎥 **ProRes 422 HQ**: Generate masterfiles ready for professional editing in Final Cut or Premiere.
- 🛡️ **Signal Resilience**: Graceful interrupt handling with auto-cleanup of temporary data.

## Installation

Install system tools:

```bash
brew install ffmpeg
```

Build the C++ CLI:

```bash
g++ -O3 -o upscaler_ult main.cpp -lpthread -std=c++17
```

Fetch the Real-ESRGAN NCNN Vulkan binary instead of committing binaries into Git:

```bash
./scripts/fetch_realesrgan_ncnn.sh
```

Optional Python pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 🚀 High Efficiency (HEVC)
Best for sharing online or keeping high-quality backups.
```bash
./upscaler_ult --input my_movie.mp4 --hevc --fps 60
```

### 🎬 Professional Workflow (ProRes)
Best for masterfiles and further editing.
```bash
./upscaler_ult --input raw_clip.mov --prores
```

### ✨ AI Fluid Motion (RIFE)
Requires `rife-ncnn-vulkan` binary in the project root. Without it, the C++ tool falls back to FFmpeg interpolation.
```bash
./upscaler_ult --input low_fps.mp4 --rife --fps 120
```

### 🧪 Quick Test
Limit processing to the first few frames:

```bash
./upscaler_ult --input sample.mp4 --frames 20 --fps 60
python3 upscale_60fps.py sample.mp4 --frames 20
```

## CLI Reference

| Flag | Description |
|---|---|
| `--hevc` | Enables H.265 Hardware Encoder (VideoToolbox). |
| `--prores` | Enables Apple ProRes 422 HQ Encoder. |
| `--rife` | Activates RIFE AI Frame Interpolation. |
| `--input / -i` | Path to the source file. |
| `--scale / -s` | Upscale factor (default: 4). |
| `--fps / -f` | Target frames per second (default: 60). |
| `--output / -o` | Custom output path. |
| `--frames / -n` | Limit extracted frames for testing. |

## Repository Policy

Generated media, model weights, zip archives, and platform-specific binaries should not be committed to Git. Use `scripts/fetch_realesrgan_ncnn.sh`, GitHub Releases, or Git LFS for large runtime assets.

## Compatibility

| Workflow | Target |
|---|---|
| `main.cpp` / `upscaler_ult` | macOS Apple Silicon with FFmpeg and NCNN Vulkan binary |
| `upscale_60fps.py` | Python environment with PyTorch, Real-ESRGAN, BasicSR, and FFmpeg |
| notebooks | Colab/CUDA-style experimentation |

## Why this is "Ultimate"?
Unlike basic upscalers, this tool handles the entire **Media Lifecycle**:
1. **Extraction**: High-bitrate intermediate processing.
2. **AI Logic**: Leveraging NCNN Vulkan for GPU-based super-resolution.
3. **Interpolation**: Intelligent motion estimation.
4. **Mastering**: Encoding directly to hardware-supported codecs for maximum speed and quality.
