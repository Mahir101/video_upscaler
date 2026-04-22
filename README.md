---
title: Video Upscaler
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.46.1
app_file: app.py
pinned: false
---

# Video Upscaler

A Gradio app for upscaling short videos with StablePy upscaler models.

## What It Does

- Upscales videos frame by frame instead of loading the whole output into memory.
- Preserves audio for MP4 output when `ffmpeg` is available.
- Writes each run to a unique file in `outputs/` to avoid accidental overwrites.
- Uses CUDA when available, then MPS on Apple Silicon, then CPU.
- Downloads external upscaler weights into `upscalers/` using atomic writes, size checks, and cached SHA-256 sidecars.
- Lets users set output codec, MP4 quality, audio/metadata preservation, and safety limits for frame count and source dimensions.
- Shows model guidance in the UI so users can choose realistic, anime, sharp, or smoother upscalers more intentionally.
- Includes optional temporal smoothing to reduce frame-to-frame shimmer from image upscalers.
- Adds an optional temporal Video SR mode through OpenMMLab MMagic for BasicVSR, BasicVSR++, and RealBasicVSR.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `ffmpeg` if you want MP4 audio preservation and H.264 output:

```bash
brew install ffmpeg
```

## Optional Video SR Backend

The default mode uses StablePy image upscalers frame by frame. For true temporal video super-resolution, install the optional OpenMMLab stack and choose `Video SR` in the UI.

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -r requirements-vsr.txt
```

Video SR models are much heavier than frame upscalers. Start with short clips and use `Video SR Max Sequence Length` if you run out of VRAM.

## Run

```bash
python3 app.py
```

## Test

```bash
python3 -m unittest discover -s tests
```

## Notes

- GPU acceleration is strongly recommended. CPU fallback can be very slow.
- GIF output does not preserve audio.
- Hugging Face ZeroGPU runs are limited to short videos by the app.
- To enforce upstream checksum validation for downloaded model files, add known SHA-256 hashes to `UPSCALER_SHA256` in `app.py`.
- `Frame Upscaler` mode applies image upscalers to individual frames. Temporal smoothing can reduce shimmer.
- `Video SR` mode uses temporal models through MMagic, so it can use neighboring frames for more consistent detail.
