#!/usr/bin/env python3
"""
Photo Enhancement Studio
Created by Md. Mahir Labib
Copyright © 2026 Md. Mahir Labib. All rights reserved.

Super-resolution via Real-ESRGAN + optional GFPGAN face restoration.
Supports single images, batch folders, multiple models, and 2-pass 8x upscaling.
"""

import os
import sys
import argparse
import glob
import time
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ESRGAN_MODELS = {
    "x4plus": {
        "desc": "General 4x — best for real-world photos",
        "scale": 4,
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "num_block": 23,
    },
    "x2plus": {
        "desc": "Light 2x — fast, subtle enhancement",
        "scale": 2,
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "num_block": 23,
    },
    "x4plus-anime": {
        "desc": "Anime / illustration 4x",
        "scale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
    },
    "x4-v3": {
        "desc": "RealESRNet v3 — natural scenes, less aggressive",
        "scale": 4,
        "filename": "realesrnet_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrnet_x4plus.pth",
        "num_block": 23,
    },
}

GFPGAN_MODELS = {
    "v1.3": {
        "filename": "GFPGANv1.3.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    },
    "v1.4": {
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    },
}

ESRGAN_CACHE = os.path.expanduser("~/.cache/realesrgan")
GFPGAN_CACHE = os.path.expanduser("~/.cache/gfpgan")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependencies() -> dict:
    deps = {}
    for name, import_expr in [
        ("torch",      "import torch"),
        ("realesrgan", "from realesrgan import RealESRGANer"),
        ("basicsr",    "from basicsr.archs.rrdbnet_arch import RRDBNet"),
        ("gfpgan",     "from gfpgan import GFPGANer"),
        ("cv2",        "import cv2"),
        ("PIL",        "from PIL import Image"),
    ]:
        try:
            exec(import_expr)
            deps[name] = True
        except Exception:
            deps[name] = False

    missing_critical = [k for k in ("torch", "realesrgan", "basicsr", "cv2") if not deps[k]]
    if missing_critical:
        print("\n  Missing critical dependencies:")
        for k, v in deps.items():
            print(f"    {'OK' if v else 'MISSING':8s}  {k}")
        print("\n  Run: pip install -r requirements.txt")
        sys.exit(1)

    return deps


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> Tuple[str, str]:
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS)"
    return "cpu", "CPU (no GPU found)"


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _hook(count, block, total):
        if total > 0:
            pct = min(int(count * block * 100 / total), 100)
            bar = "█" * (pct // 3) + "░" * (34 - pct // 3)
            print(f"\r    [{bar}] {pct}%", end="", flush=True)

    print(f"  Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest, _hook)
    print()


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def build_upsampler(model_key: str, tile: int):
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    cfg = ESRGAN_MODELS[model_key]
    path = os.path.join(ESRGAN_CACHE, cfg["filename"])
    if not os.path.exists(path):
        _download(cfg["url"], path)

    device, _ = get_device()
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=cfg["num_block"], num_grow_ch=32,
        scale=cfg["scale"],
    )
    return RealESRGANer(
        scale=cfg["scale"],
        model_path=path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=(device != "cpu"),
        device=device,
    )


def build_face_restorer(version: str):
    try:
        from gfpgan import GFPGANer
    except ImportError:
        return None

    cfg = GFPGAN_MODELS[version]
    path = os.path.join(GFPGAN_CACHE, cfg["filename"])
    if not os.path.exists(path):
        _download(cfg["url"], path)

    return GFPGANer(
        model_path=path,
        upscale=1,        # upscaling is done separately by Real-ESRGAN
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


# ---------------------------------------------------------------------------
# Core enhancement
# ---------------------------------------------------------------------------

def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def enhance_image(
    img_path: str,
    upsampler,
    face_restorer,
    scale: int,
    face_blend: float,
    model_scale: int,
) -> np.ndarray:
    img = _read_bgr(img_path)
    passes = scale // model_scale
    out = img

    for _ in range(max(passes, 1)):
        out, _ = upsampler.enhance(out, outscale=model_scale)

    if face_restorer is not None:
        _, _, out = face_restorer.enhance(
            out,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=face_blend,
        )

    return out


# ---------------------------------------------------------------------------
# Comparison image
# ---------------------------------------------------------------------------

def make_comparison(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    h_e, w_e = enhanced.shape[:2]
    h_o, w_o = original.shape[:2]
    orig_r = cv2.resize(original, (int(w_o * h_e / h_o), h_e), interpolation=cv2.INTER_LANCZOS4)

    comp = np.hstack([orig_r, enhanced])
    font, lw = cv2.FONT_HERSHEY_SIMPLEX, 2
    for text, x, color in [
        ("ORIGINAL", 20, (255, 255, 255)),
        ("ENHANCED", orig_r.shape[1] + 20, (60, 240, 120)),
    ]:
        cv2.putText(comp, text, (x, 60), font, 1.6, (0, 0, 0), lw + 4)
        cv2.putText(comp, text, (x, 60), font, 1.6, color, lw)

    mid = orig_r.shape[1]
    cv2.line(comp, (mid, 0), (mid, h_e), (0, 200, 255), 4)
    return comp


# ---------------------------------------------------------------------------
# Single / batch processing
# ---------------------------------------------------------------------------

def _output_path(src: str, out_dir: str, suffix: str, ext: str) -> str:
    stem = Path(src).stem
    return os.path.join(out_dir, f"{stem}{suffix}{ext}")


def process_one(
    src: str,
    out_dir: str,
    upsampler,
    face_restorer,
    scale: int,
    face_blend: float,
    model_scale: int,
    save_comparison: bool,
    png_compression: int,
) -> bool:
    t0 = time.time()
    original = _read_bgr(src)
    h, w = original.shape[:2]

    try:
        enhanced = enhance_image(src, upsampler, face_restorer, scale, face_blend, model_scale)
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    enh_h, enh_w = enhanced.shape[:2]
    out = _output_path(src, out_dir, "_enhanced", ".png")
    cv2.imwrite(out, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

    if save_comparison:
        comp_path = _output_path(src, out_dir, "_comparison", ".jpg")
        cv2.imwrite(comp_path, make_comparison(original, enhanced), [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"    {w}x{h} → {enh_w}x{enh_h}  ({time.time()-t0:.1f}s)  →  {os.path.basename(out)}")
    return True


def process_batch(
    files: List[str],
    out_dir: str,
    upsampler,
    face_restorer,
    scale: int,
    face_blend: float,
    model_scale: int,
    save_comparison: bool,
    png_compression: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ok, fail = 0, 0
    t_start = time.time()

    for i, src in enumerate(files, 1):
        print(f"\n  [{i}/{len(files)}]  {os.path.basename(src)}")
        if process_one(src, out_dir, upsampler, face_restorer, scale, face_blend,
                       model_scale, save_comparison, png_compression):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t_start
    print(f"\n  Done: {ok} OK, {fail} failed  ({elapsed:.0f}s total)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print("=" * 64)
    print("  PHOTO ENHANCEMENT STUDIO")
    print("  Created by Md. Mahir Labib")
    print("  Real-ESRGAN super-resolution + GFPGAN face restoration")
    print("=" * 64)


def _collect_images(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        found = []
        for ext in IMAGE_EXTS:
            found += glob.glob(os.path.join(path, f"*{ext}"))
            found += glob.glob(os.path.join(path, f"*{ext.upper()}"))
        return sorted(set(found))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Photo Enhancement: Real-ESRGAN upscaling + GFPGAN face restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python photo_enhance.py photo.jpg
  python photo_enhance.py photos/ -o enhanced/
  python photo_enhance.py art.png --model x4plus-anime --no-face
  python photo_enhance.py photo.jpg --scale 8          # two-pass 8x
  python photo_enhance.py photo.jpg --model x2plus --no-comparison
  python photo_enhance.py --list-models
        """,
    )

    parser.add_argument("input", nargs="?", help="Input image or folder")
    parser.add_argument("-o", "--output", default=None, help="Output file or folder")
    parser.add_argument(
        "--model", default="x4plus", choices=list(ESRGAN_MODELS),
        help="Upscaling model (default: x4plus)",
    )
    parser.add_argument("--scale", type=int, default=None, help="Override output scale (e.g. 8 for two-pass 4→4)")
    parser.add_argument("--face", dest="face", action="store_true", default=True, help="Enable GFPGAN face restoration (default: on)")
    parser.add_argument("--no-face", dest="face", action="store_false", help="Disable face restoration")
    parser.add_argument("--face-version", default="v1.4", choices=list(GFPGAN_MODELS), help="GFPGAN version (default: v1.4)")
    parser.add_argument("--denoise", type=float, default=0.5, metavar="0-1", help="Face restoration blend weight (default: 0.5)")
    parser.add_argument("--tile", type=int, default=256, help="Tile size — reduce if OOM (default: 256)")
    parser.add_argument("--no-comparison", dest="comparison", action="store_false", default=True, help="Skip side-by-side comparison image")
    parser.add_argument("--png-compression", type=int, default=1, choices=range(0, 10), metavar="0-9", help="PNG compression level (default: 1 = fast + small)")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    _print_header()

    if args.list_models:
        print("\n  Available models:")
        for key, cfg in ESRGAN_MODELS.items():
            print(f"    {key:<20}  {cfg['scale']}x  —  {cfg['desc']}")
        print("\n  GFPGAN face restoration versions: " + ", ".join(GFPGAN_MODELS))
        print()
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    # --- collect input files ---
    files = _collect_images(args.input)
    if not files:
        print(f"\n  ERROR: No images found at: {args.input}")
        sys.exit(1)

    is_batch = os.path.isdir(args.input) or len(files) > 1
    model_cfg = ESRGAN_MODELS[args.model]
    model_scale = model_cfg["scale"]
    scale = args.scale if args.scale else model_scale

    if scale % model_scale != 0 or scale < model_scale:
        print(f"  WARNING: scale {scale} is not a multiple of model scale {model_scale}; clamping to {model_scale}")
        scale = model_scale

    # --- determine output location ---
    if is_batch:
        out_dir = args.output or os.path.join(args.input, "enhanced")
    else:
        if args.output and not os.path.isdir(args.output):
            out_dir = str(Path(args.output).parent) or "."
        else:
            out_dir = args.output or str(Path(files[0]).parent)

    # --- print config ---
    print()
    print(f"  Input      {len(files)} image(s)")
    print(f"  Model      {args.model} — {model_cfg['desc']}")
    print(f"  Scale      {scale}x" + (" (2-pass)" if scale > model_scale else ""))
    print(f"  Output     {out_dir}")

    # --- check deps ---
    print("\n  Checking dependencies ...")
    deps = check_dependencies()
    device, device_name = get_device()
    print(f"  Device     {device_name}")

    # --- load models ---
    print("\n  Loading Real-ESRGAN ...")
    upsampler = build_upsampler(args.model, args.tile)

    face_restorer = None
    if args.face:
        if deps.get("gfpgan"):
            print(f"  Loading GFPGAN {args.face_version} ...")
            face_restorer = build_face_restorer(args.face_version)
            print(f"  Face blend {args.denoise:.1f} (0 = subtle, 1 = full restore)")
        else:
            print("  WARNING: gfpgan not installed — face restoration disabled")

    # --- process ---
    print(f"\n  Processing {len(files)} image(s) ...\n")
    os.makedirs(out_dir, exist_ok=True)

    if is_batch:
        process_batch(
            files, out_dir, upsampler, face_restorer,
            scale, args.denoise, model_scale,
            args.comparison, args.png_compression,
        )
    else:
        src = files[0]
        dest = args.output if (args.output and not os.path.isdir(args.output)) else None
        if dest:
            # explicit output path — write directly
            original = _read_bgr(src)
            enhanced = enhance_image(src, upsampler, face_restorer, scale, args.denoise, model_scale)
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            cv2.imwrite(dest, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
            if args.comparison:
                comp = _output_path(dest, str(Path(dest).parent), "_comparison", ".jpg")
                cv2.imwrite(comp, make_comparison(original, enhanced), [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"  Comparison  {comp}")
            h, w = original.shape[:2]
            enh_h, enh_w = enhanced.shape[:2]
            print(f"\n  {w}x{h} → {enh_w}x{enh_h}   Saved: {dest}")
        else:
            process_one(
                src, out_dir, upsampler, face_restorer,
                scale, args.denoise, model_scale,
                args.comparison, args.png_compression,
            )

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
