#!/usr/bin/env python3
# High-Quality Video Upscaler with 60 FPS Frame Interpolation
# Created by Md. Mahir Labib
# Copyright © 2026 Md. Mahir Labib. All rights reserved.
# Uses Real-ESRGAN for upscaling + FFmpeg minterpolate for smooth 60 FPS

import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def check_dependencies():
    """Check if all required dependencies are available"""
    # Check FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg first.")
        sys.exit(1)
    
    # Check Real-ESRGAN
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip3 install realesrgan basicsr")
        sys.exit(1)

def setup_realesrgan(scale=4):
    """Initialize Real-ESRGAN upscaler"""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import torch
    
    # Use RealESRGAN-x4plus model (best quality)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Model path - will download automatically
    model_path = os.path.expanduser('~/.cache/realesrgan/RealESRGAN_x4plus.pth')
    
    # Check if we need to download the model
    if not os.path.exists(model_path):
        print("📥 Downloading Real-ESRGAN model (first time only)...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import urllib.request
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        urllib.request.urlretrieve(url, model_path)
        print("✅ Model downloaded!")
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        print("🚀 Using CUDA GPU acceleration")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Using Apple Metal GPU acceleration")
    else:
        device = 'cpu'
        print("⚠️ Using CPU (this will be slower)")
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,  # Process in tiles to save memory
        tile_pad=10,
        pre_pad=0,
        half=False if device == 'cpu' else True,
        device=device
    )
    
    return upsampler

def extract_frames(video_path, output_dir, fps=None):
    """Extract frames from video using FFmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original FPS if not specified
    if fps is None:
        probe_cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
                     '-show_entries', 'stream=r_frame_rate', video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
    
    print(f"📽️ Extracting frames at {fps:.2f} FPS...")
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-qscale:v', '2',
        '-start_number', '0',
        f'{output_dir}/frame_%07d.png'
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    
    frames = sorted(glob.glob(f'{output_dir}/frame_*.png'))
    print(f"✅ Extracted {len(frames)} frames")
    
    return frames, fps

def upscale_frames(upsampler, input_dir, output_dir, frames):
    """Upscale all frames using Real-ESRGAN"""
    os.makedirs(output_dir, exist_ok=True)
    
    total = len(frames)
    print(f"🔍 Upscaling {total} frames with Real-ESRGAN...")
    
    for i, frame_path in enumerate(frames):
        # Read image
        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        
        # Upscale
        output, _ = upsampler.enhance(img, outscale=4)
        
        # Save
        output_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(output_path, output)
        
        # Progress
        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"   Progress: {i + 1}/{total} frames ({100*(i+1)/total:.1f}%)")
    
    print("✅ All frames upscaled!")

def create_video_with_interpolation(frames_dir, output_path, source_fps, target_fps=60):
    """
    Create video from frames with frame interpolation for smooth 60 FPS
    Uses FFmpeg's minterpolate filter for motion-compensated interpolation
    """
    print(f"🎬 Creating smooth {target_fps} FPS video with motion interpolation...")
    
    # Calculate interpolation factor
    fps_multiplier = target_fps / source_fps
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(source_fps),
        '-i', f'{frames_dir}/frame_%07d.png',
        '-vf', f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    print(f"   Using motion-compensated frame interpolation ({source_fps} FPS → {target_fps} FPS)")
    subprocess.run(cmd, check=True)
    print(f"✅ Video created: {output_path}")

def add_audio(video_no_audio, original_video, output_path):
    """Copy audio from original video to upscaled video"""
    print("🔊 Adding audio from original video...")
    
    # Check if original has audio
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a',
                 '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', original_video]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    
    if 'audio' not in result.stdout:
        print("   No audio in original video, skipping...")
        shutil.copy(video_no_audio, output_path)
        return
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_no_audio,
        '-i', original_video,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print("✅ Audio added!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Upscale video with Real-ESRGAN + 60 FPS interpolation')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', default='video-upscaled.mp4', help='Output video file')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS (default: 60)')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print("🎥 HIGH-QUALITY VIDEO UPSCALER")
    print("   Created by Md. Mahir Labib")
    print("   Real-ESRGAN 4x + 60 FPS Motion Interpolation")
    print("=" * 60)
    
    # Check dependencies
    check_dependencies()
    
    # Setup directories
    temp_dir = Path('temp_upscale')
    lr_frames_dir = temp_dir / 'lr_frames'
    hr_frames_dir = temp_dir / 'hr_frames'
    temp_video = temp_dir / 'temp_no_audio.mp4'
    
    # Clean previous temp if exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    try:
        # Step 1: Extract frames
        frames, original_fps = extract_frames(args.input, str(lr_frames_dir))
        
        # Step 2: Setup Real-ESRGAN
        print("\n🔧 Initializing Real-ESRGAN...")
        upsampler = setup_realesrgan()
        
        # Step 3: Upscale frames
        print()
        upscale_frames(upsampler, str(lr_frames_dir), str(hr_frames_dir), frames)
        
        # Step 4: Create video with 60 FPS interpolation
        print()
        hr_frames = sorted(glob.glob(f'{hr_frames_dir}/frame_*.png'))
        create_video_with_interpolation(str(hr_frames_dir), str(temp_video), original_fps, args.fps)
        
        # Step 5: Add audio
        print()
        add_audio(str(temp_video), args.input, args.output)
        
        print()
        print("=" * 60)
        print(f"✅ SUCCESS! Output saved to: {args.output}")
        print("=" * 60)
        
        # Show file info
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height,r_frame_rate',
                     '-of', 'csv=p=0', args.output]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        info = result.stdout.strip().split(',')
        if len(info) >= 3:
            print(f"   Resolution: {info[0]}x{info[1]}")
            print(f"   Frame Rate: {info[2]}")
        
    finally:
        if not args.keep_temp and temp_dir.exists():
            print("\n🧹 Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
