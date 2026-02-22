<h1>Ultimate Video Upscaler</h1>
<p><strong>Created by Md. Mahir Labib</strong><br>
Copyright © 2026 Md. Mahir Labib. All rights reserved.</p>

<p>The ultimate professional-grade media engine for video upscaling and frame interpolation, fully optimized for <strong>Apple Silicon M4</strong>. This version features hardware-accelerated HEVC and ProRes encoding, AI-driven motion interpolation, and real-time performance monitoring.</p>

<div align="center">
  <img src="https://img.shields.io/badge/Codec-HEVC%20%2F%20ProRes-blue?style=for-the-badge" alt="Codecs">
  <img src="https://img.shields.io/badge/Hardware-Apple%20M4-orange?style=for-the-badge&logo=apple" alt="Hardware">
  <img src="https://img.shields.io/badge/Interpolation-RIFE%20AI-green?style=for-the-badge" alt="AI">
</div>

## Key Features

- 🏎️ **Apple Silicon Suite**: Native support for `hevc_videotoolbox` and `prores_videotoolbox`.
- ✨ **RIFE AI Integration**: Support for RIFE (Real-Time Intermediate Flow Estimation) for perfectly smooth 60+ FPS motion.
- 📂 **Zero-Disk Strategy**: Optimized internal pipelining and temporary file management to minimize SSD wear.
- 🎥 **ProRes 422 HQ**: Generate masterfiles ready for professional editing in Final Cut or Premiere.
- 🛡️ **Signal Resilience**: Graceful interrupt handling with auto-cleanup of temporary data.

## Installation

```bash
# Build the Ultimate tool
g++ -O3 -o upscaler_ult main.cpp -lpthread -std=c++17
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
Requires `rife-ncnn-vulkan` binary in the project root.
```bash
./upscaler_ult --input low_fps.mp4 --rife --fps 120
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

## Why this is "Ultimate"?
Unlike basic upscalers, this tool handles the entire **Media Lifecycle**:
1. **Extraction**: High-bitrate intermediate processing.
2. **AI Logic**: Leveraging NCNN Vulkan for GPU-based super-resolution.
3. **Interpolation**: Intelligent motion estimation.
4. **Mastering**: Encoding directly to hardware-supported codecs for maximum speed and quality.
