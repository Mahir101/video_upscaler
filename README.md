<h1>Ultimate Video &amp; Photo Upscaler</h1>
<p><strong>Created by Md. Mahir Labib</strong><br>
Copyright © 2026 Md. Mahir Labib. All rights reserved.</p>

<p>A professional-grade media engine for <strong>video upscaling</strong>, <strong>photo enhancement</strong>, and <strong>frame interpolation</strong>, optimised for <strong>Apple Silicon M4</strong>. Supports Real-ESRGAN super-resolution, GFPGAN face restoration, HEVC/ProRes encoding, and optional RIFE AI motion interpolation.</p>

<div align="center">
  <img src="https://img.shields.io/badge/Codec-HEVC%20%2F%20ProRes-blue?style=for-the-badge" alt="Codecs">
  <img src="https://img.shields.io/badge/Hardware-Apple%20M4-orange?style=for-the-badge&logo=apple" alt="Hardware">
  <img src="https://img.shields.io/badge/AI-Real--ESRGAN%20%2B%20GFPGAN-green?style=for-the-badge" alt="AI">
  <img src="https://img.shields.io/badge/Interpolation-RIFE%20AI-purple?style=for-the-badge" alt="RIFE">
</div>

---

## Key Features

- **Real-ESRGAN** super-resolution — 2x, 4x, or chained 8x for video and photos
- **GFPGAN face restoration** — repair faces in old, blurry, or low-res photos
- **Multi-model support** — x4plus, x2plus, anime-6B, and RealESRNet v3
- **Three pipelines** — C++ binary (fastest), Python (GPU batching), Shell (no Python needed)
- **Apple Silicon Suite** — `hevc_videotoolbox` and `prores_videotoolbox` hardware encoders
- **RIFE AI interpolation** — fluid 60/120 FPS when `rife-ncnn-vulkan` is installed
- **Comparison output** — side-by-side before/after images saved automatically
- **Signal resilience** — SIGINT-safe temp-dir cleanup

---

## Installation

### 1 — System tools

```bash
brew install ffmpeg
```

### 2 — Fetch NCNN binary and model weights

```bash
# Fetch the Real-ESRGAN NCNN Vulkan binary (macOS)
./scripts/fetch_realesrgan_ncnn.sh

# Download all Python model weights (ESRGAN variants + GFPGAN)
./scripts/fetch_models.sh
```

Individual fetches are also available:

```bash
./scripts/fetch_models.sh --esrgan   # Real-ESRGAN Python weights only
./scripts/fetch_models.sh --gfpgan   # GFPGAN weights only
./scripts/fetch_models.sh --ncnn     # NCNN binary only
```

### 3 — Build the C++ binary

```bash
g++ -O3 -o upscaler_ult main.cpp -lpthread -std=c++17
```

### 4 — Python environment (for photo_enhance.py and upscale_60fps.py)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Photo Enhancement

### Python pipeline — `photo_enhance.py`

Full-featured: multi-model, GFPGAN face restoration, batch processing, comparison images.

```bash
# Enhance a single photo (4x, face restoration on by default)
python photo_enhance.py photo.jpg

# Explicit output path
python photo_enhance.py photo.jpg -o photo_4k.png

# Batch-process a folder → enhanced/ subfolder
python photo_enhance.py photos/ -o enhanced/

# Anime or illustration
python photo_enhance.py art.png --model x4plus-anime --no-face

# Light 2x enhancement (faster)
python photo_enhance.py photo.jpg --model x2plus

# Two-pass 8x super-resolution
python photo_enhance.py photo.jpg --scale 8

# Stronger face restoration (blend weight 0.0–1.0)
python photo_enhance.py portrait.jpg --denoise 0.8

# List all available models
python photo_enhance.py --list-models
```

#### Photo CLI Reference

| Flag | Description |
|---|---|
| `--model` | Upscaling model: `x4plus` (default), `x2plus`, `x4plus-anime`, `x4-v3` |
| `--scale` | Output scale — 2, 4 (default), 8 (two-pass 4→4) |
| `--face` / `--no-face` | Enable/disable GFPGAN face restoration (default: on) |
| `--face-version` | GFPGAN version: `v1.3` or `v1.4` (default) |
| `--denoise` | Face restoration blend weight 0.0–1.0 (default: 0.5) |
| `--tile` | Tile size for VRAM management (default: 256) |
| `--no-comparison` | Skip side-by-side comparison image |
| `--png-compression` | PNG compression 0–9 (default: 1 = fast) |
| `--list-models` | Print available models and exit |

### Shell pipeline — `upscale_photo.sh`

No Python required — uses the NCNN binary directly.

```bash
# Single photo
./upscale_photo.sh photo.jpg

# Explicit output + model
./upscale_photo.sh photo.jpg -o photo_hd.png -m realesrgan-x4plus

# Anime mode
./upscale_photo.sh art.png -m realesrgan-x4plus-anime

# Batch folder
./upscale_photo.sh photos/ -o photos/enhanced/
```

### C++ binary — `upscaler_ult`

Detects `.jpg`/`.jpeg`/`.png`/`.bmp`/`.webp` inputs automatically and runs the fast NCNN pipeline.

```bash
./upscaler_ult --input photo.jpg
./upscaler_ult --input photo.jpg --scale 4 --model realesrgan-x4plus
./upscaler_ult --input photo.jpg --output photo_4k.png --model realesrgan-x4plus-anime
```

---

## Video Upscaling

### Quick Starts

```bash
# Default H.264, 60 FPS
./upscaler_ult --input video.mp4

# H.265 hardware encode, 60 FPS
./upscaler_ult --input my_movie.mp4 --hevc --fps 60

# Apple ProRes 422 HQ masterfile
./upscaler_ult --input raw_clip.mov --prores

# RIFE AI fluid motion (requires rife-ncnn-vulkan binary)
./upscaler_ult --input low_fps.mp4 --rife --fps 120

# Test with first 20 frames only
./upscaler_ult --input sample.mp4 --frames 20 --fps 60

# Use anime model for animated content
./upscaler_ult --input anime.mp4 --model realesrgan-x4plus-anime --hevc
```

### Python pipeline — `upscale_60fps.py`

```bash
python upscale_60fps.py video.mp4
python upscale_60fps.py video.mp4 -o output.mp4 --fps 60 --frames 20
```

### Shell pipeline — `upscale_60fps.sh`

```bash
./upscale_60fps.sh input.mp4 output.mp4 60 4
```

### Video CLI Reference (`upscaler_ult`)

| Flag | Description |
|---|---|
| `--input / -i` | Source file (image or video) |
| `--output / -o` | Custom output path |
| `--model / -m` | NCNN model name (default: `realesrgan-x4plus`) |
| `--scale / -s` | Upscale factor (default: 4) |
| `--fps / -f` | Target frames per second (default: 60) |
| `--frames / -n` | Limit extracted frames for testing |
| `--hevc` | H.265 hardware encoder (VideoToolbox) |
| `--prores` | Apple ProRes 422 HQ encoder |
| `--rife` | RIFE AI frame interpolation |

---

## Available Models

| Key | Scale | Best For |
|---|---|---|
| `realesrgan-x4plus` | 4x | General real-world photos and video (default) |
| `realesrgan-x2plus` | 2x | Fast/subtle enhancement |
| `realesrgan-x4plus-anime` | 4x | Anime, illustrations, and hand-drawn art |
| `realesrnet-x4plus` | 4x | Natural scenes, less aggressive enhancement |

---

## How HD Upscaling Works

```
Source 480p (854×480)  ──┐
                          │  Real-ESRGAN 4x
                          ▼
Intermediate 4x (3416×1920)
                          │  Optional: RIFE / minterpolate
                          ▼
Final 4K-class output + 60 FPS
```

1. **Frame extraction** — FFmpeg pulls lossless PNG frames at maximum quality
2. **Super-resolution** — Real-ESRGAN neural network infers missing high-frequency details tile-by-tile
3. **Frame interpolation** — RIFE AI (or FFmpeg minterpolate as fallback) synthesises smooth in-between frames
4. **Hardware encode** — Apple VideoToolbox HEVC/ProRes for maximum speed and quality

---

## Compatibility

| Workflow | Target |
|---|---|
| `upscaler_ult` / `upscale_photo.sh` | macOS Apple Silicon, NCNN Vulkan binary |
| `photo_enhance.py` | PyTorch + Real-ESRGAN + GFPGAN (CUDA / MPS / CPU) |
| `upscale_60fps.py` | PyTorch + Real-ESRGAN + FFmpeg (CUDA / MPS / CPU) |
| Notebooks | Colab / CUDA-style experimentation |

---

## Repository Policy

Generated media, model weights (`.pth`, `.bin`, `.param`), zip archives, and platform binaries must not be committed to git. Use `scripts/fetch_models.sh`, GitHub Releases, or Git LFS for large runtime assets.
