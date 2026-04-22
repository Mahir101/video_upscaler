import os
import shutil
import subprocess
import tempfile

import spaces
import cv2
import imageio
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
import gradio as gr
from stablepy import load_upscaler_model, ALL_BUILTIN_UPSCALERS
import gc
import torch

from video_utils import (
    OUTPUT_DIRECTORY,
    build_mux_command,
    cached_checksum,
    checksum_sidecar_path,
    make_output_path,
    remove_if_exists,
    sha256_file,
    upscaler_filename_from_url,
    validate_video_limits,
    write_checksum_sidecar,
)

DIRECTORY_UPSCALERS = "upscalers"
IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
ZERO_GPU_MAX_FRAMES = 90
ZERO_GPU_MAX_WIDTH = 1024
ZERO_GPU_MAX_HEIGHT = 1024
DEFAULT_MAX_FRAMES = 0
DEFAULT_MAX_WIDTH = 0
DEFAULT_MAX_HEIGHT = 0
ALL_BUILTIN_UPSCALERS = ALL_BUILTIN_UPSCALERS[8:]
VALID_UPSCALERS = {bu: bu for bu in ALL_BUILTIN_UPSCALERS if "ESRGAN" in bu or "ScuNET" in bu} if IS_ZERO_GPU else {bu: bu for bu in ALL_BUILTIN_UPSCALERS}
UPSCALER_DICT_GUI = {
    **VALID_UPSCALERS,
    # "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    # "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    # "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    # "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    # "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    # "realesr-general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "4x-UltraSharp": "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "Real-ESRGAN-Anime-finetuning": "https://huggingface.co/danhtran2mind/Real-ESRGAN-Anime-finetuning/resolve/main/Real-ESRGAN-Anime-finetuning.pth",
    "4x_foolhardy_Remacri": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "Remacri4xExtraSmoother": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    "AnimeSharp4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "lollypop": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    "RealisticRescaler4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    "NickelbackFS4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth"
}
UPSCALER_KEYS = list(UPSCALER_DICT_GUI.keys())
UPSCALER_SHA256 = {}
MODEL_GUIDANCE = {
    "RealESRNet_x4plus": "General realistic footage. More conservative than very sharp ESRGAN models.",
    "4x-UltraSharp": "High sharpness/detail. Good for crisp source material, but can exaggerate artifacts.",
    "Real-ESRGAN-Anime-finetuning": "Anime and illustration content with Real-ESRGAN style restoration.",
    "4x_foolhardy_Remacri": "Detailed ESRGAN look for art and textured footage.",
    "Remacri4xExtraSmoother": "Smoother Remacri variant. Useful when sharp models look noisy.",
    "AnimeSharp4x": "Anime, cartoons, line art, and cel-shaded clips.",
    "lollypop": "Stylized ESRGAN option; useful to compare against sharper models.",
    "RealisticRescaler4x": "Realistic/detail-oriented ESRGAN option.",
    "NickelbackFS4x": "Strong detail reconstruction; can be aggressive on noisy footage.",
}
CUSTOM_CSS = """
h1 {
    color: #333;
    text-align: center;
}
.gradio-container {
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    padding: 20px;
}
"""
DEMO_GPU = (
    "\n\nThis demo only works with short videos—up to 90 frames (about 3 seconds for 30 fps videos) "
    "and a maximum resolution of 1024×1024 pixels."
)
DESCRIPTION = (
    "# Video Upscaler\n\n"
    "Upscale your videos using powerful upscaler models."
    f"{DEMO_GPU if IS_ZERO_GPU else ''}"
)


def free_memory():
    # CPU
    gc.collect()

    # GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_compute_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_guidance_text(upscaler_name):
    return MODEL_GUIDANCE.get(
        upscaler_name,
        "Built-in StablePy upscaler. Test on a short clip first; frame-by-frame models can flicker.",
    )


def download_upscaler(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    expected_hash = UPSCALER_SHA256.get(url)
    temp_path = f"{local_path}.download"

    print(f"Downloading upscaler from {url} ...")
    with requests.get(url, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        expected_size = int(r.headers.get("content-length") or 0)
        bytes_written = 0
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

    if bytes_written == 0:
        os.remove(temp_path)
        raise RuntimeError(f"Downloaded model is empty: {url}")

    if expected_size and bytes_written != expected_size:
        os.remove(temp_path)
        raise RuntimeError(
            f"Incomplete model download for {url}. Expected {expected_size} bytes, got {bytes_written}."
        )

    actual_hash = sha256_file(temp_path)
    if expected_hash:
        if actual_hash.lower() != expected_hash.lower():
            os.remove(temp_path)
            raise RuntimeError(
                f"Checksum mismatch for {url}. Expected {expected_hash}, got {actual_hash}."
            )

    os.replace(temp_path, local_path)
    write_checksum_sidecar(local_path, actual_hash)
    print(f"Downloaded upscaler to {local_path}")


def resolve_upscaler_model(upscaler_name):
    if upscaler_name not in UPSCALER_DICT_GUI:
        raise ValueError(f"Unknown upscaler: {upscaler_name}")

    model = UPSCALER_DICT_GUI[upscaler_name]
    if "https://" not in str(model):
        return model

    filename = upscaler_filename_from_url(model)
    local_path = os.path.join(DIRECTORY_UPSCALERS, filename)
    if not os.path.exists(local_path):
        download_upscaler(model, local_path)
    else:
        actual_hash = sha256_file(local_path)
        expected_hash = UPSCALER_SHA256.get(model) or cached_checksum(local_path)
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            os.remove(local_path)
            sidecar_path = checksum_sidecar_path(local_path)
            if os.path.exists(sidecar_path):
                os.remove(sidecar_path)
            download_upscaler(model, local_path)
        elif not expected_hash:
            write_checksum_sidecar(local_path, actual_hash)

    return local_path


def read_video_metadata(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError("Could not open the input video.")
    metadata = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def upscale_frame(frame, scaler_beta, upscaler_factor):
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pil_frame.mode != "RGB":
        pil_frame = pil_frame.convert("RGB")

    processed_pil = scaler_beta.upscale(pil_frame, upscaler_factor, True)
    if not isinstance(processed_pil, Image.Image):
        raise TypeError("Upscaler must return a PIL.Image")

    processed_np = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
    del pil_frame, processed_pil
    return processed_np


def apply_temporal_smoothing(current_frame, previous_frame, amount):
    if amount <= 0 or previous_frame is None or previous_frame.shape != current_frame.shape:
        return current_frame
    return cv2.addWeighted(current_frame, 1.0 - amount, previous_frame, amount, 0)


def write_upscaled_video(video_file, output_path, scaler_beta, upscaler_factor, speed_factor, temporal_smoothing=0.0, progress=None):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError("Could not open the input video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    new_fps = fps * speed_factor if speed_factor > 0 else fps
    writer = None
    count = 0
    new_width = 0
    new_height = 0
    previous_frame = None

    try:
        with tqdm(total=total_frames or None, desc="Upscaling video", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_np = upscale_frame(frame, scaler_beta, upscaler_factor)
                processed_np = apply_temporal_smoothing(processed_np, previous_frame, temporal_smoothing)

                if writer is None:
                    new_height, new_width = processed_np.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, new_fps, (new_width, new_height))
                    if not writer.isOpened():
                        raise RuntimeError("Could not create output video writer.")

                writer.write(processed_np)
                previous_frame = processed_np.copy() if temporal_smoothing > 0 else None
                count += 1
                pbar.update(1)
                pbar.set_description(f"Frames {count}/{total_frames or '?'}")
                if progress and total_frames:
                    progress(count / total_frames, desc=f"Upscaled {count}/{total_frames} frames")

                del frame, processed_np
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if count == 0:
        raise RuntimeError("No frames were processed.")

    return {"frames": count, "width": new_width, "height": new_height, "fps": new_fps}


def write_upscaled_gif(video_file, output_path, scaler_beta, upscaler_factor, speed_factor, temporal_smoothing=0.0, progress=None):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError("Could not open the input video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    new_fps = fps * speed_factor if speed_factor > 0 else fps
    duration = 1.0 / new_fps
    count = 0
    new_width = 0
    new_height = 0
    previous_frame = None

    try:
        with imageio.get_writer(output_path, mode="I", duration=duration) as gif_writer:
            with tqdm(total=total_frames or None, desc="Upscaling GIF", unit="frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_np = upscale_frame(frame, scaler_beta, upscaler_factor)
                    processed_np = apply_temporal_smoothing(processed_np, previous_frame, temporal_smoothing)
                    new_height, new_width = processed_np.shape[:2]
                    gif_writer.append_data(cv2.cvtColor(processed_np, cv2.COLOR_BGR2RGB))
                    previous_frame = processed_np.copy() if temporal_smoothing > 0 else None

                    count += 1
                    pbar.update(1)
                    pbar.set_description(f"Frames {count}/{total_frames or '?'}")
                    if progress and total_frames:
                        progress(count / total_frames, desc=f"Upscaled {count}/{total_frames} frames")

                    del frame, processed_np
    finally:
        cap.release()

    if count == 0:
        raise RuntimeError("No frames were processed.")

    return {"frames": count, "width": new_width, "height": new_height, "fps": new_fps}


def mux_media(input_video_path, silent_video_path, output_path, keep_audio=True, keep_metadata=True, codec="libx264", crf=18):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        os.replace(silent_video_path, output_path)
        print("ffmpeg was not found; returning OpenCV-encoded video.")
        return False

    command = build_mux_command(
        ffmpeg,
        input_video_path,
        silent_video_path,
        output_path,
        codec=codec,
        crf=crf,
        keep_audio=keep_audio,
        keep_metadata=keep_metadata,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        os.replace(silent_video_path, output_path)
        print("ffmpeg mux failed; returning OpenCV-encoded video.")
        print(result.stderr)
        return False

    os.remove(silent_video_path)
    return True


@spaces.GPU(duration=120)
def zero_comp(
    video_file,
    output_path,
    as_gif,
    upscaler_factor,
    speed_factor,
    progress,
    upscaler_params,
    keep_audio,
    keep_metadata,
    codec,
    crf,
    temporal_smoothing,
):
    scaler_beta = load_upscaler_model(**upscaler_params)
    silent_path = None
    try:
        if as_gif:
            stats = write_upscaled_gif(
                video_file,
                output_path,
                scaler_beta,
                upscaler_factor,
                speed_factor,
                temporal_smoothing,
                progress,
            )
        else:
            silent_fd, silent_path = tempfile.mkstemp(suffix=".mp4", prefix="silent_", dir=OUTPUT_DIRECTORY)
            os.close(silent_fd)
            stats = write_upscaled_video(
                video_file,
                silent_path,
                scaler_beta,
                upscaler_factor,
                speed_factor,
                temporal_smoothing,
                progress,
            )
            stats["muxed_with_ffmpeg"] = mux_media(
                video_file,
                silent_path,
                output_path,
                keep_audio=keep_audio,
                keep_metadata=keep_metadata,
                codec=codec,
                crf=crf,
            )
            silent_path = None
    finally:
        remove_if_exists(silent_path)
        scaler_beta = None
        del scaler_beta
        free_memory()

    return stats


def start_upscaler(
    video_file,
    upscaler_name="R-ESRGAN 4x+ Anime6B",
    as_gif=False,
    speed_factor=1.0,
    half_precision=True,
    tile=192,
    tile_overlap=8,
    upscaler_factor=1.5,
    max_frames=DEFAULT_MAX_FRAMES,
    max_width=DEFAULT_MAX_WIDTH,
    max_height=DEFAULT_MAX_HEIGHT,
    keep_audio=True,
    keep_metadata=True,
    codec="libx264",
    crf=18,
    temporal_smoothing=0.0,
    progress=gr.Progress(track_tqdm=True),
):
    if video_file is None:
        raise ValueError("Error: No video file provided.")
    if not isinstance(video_file, str):
        video_file = video_file.name
    ext = "gif" if as_gif else "mp4"
    output_path = make_output_path(video_file, ext)

    metadata = read_video_metadata(video_file)
    total_frames = metadata["total_frames"]
    width = metadata["width"]
    height = metadata["height"]

    if IS_ZERO_GPU:
        max_frames = ZERO_GPU_MAX_FRAMES
        max_width = ZERO_GPU_MAX_WIDTH
        max_height = ZERO_GPU_MAX_HEIGHT
    validate_video_limits(
        metadata,
        max_frames=int(max_frames or 0),
        max_width=int(max_width or 0),
        max_height=int(max_height or 0),
    )

    cl_name_upscaler = resolve_upscaler_model(upscaler_name)
    device = get_compute_device()
    use_half_precision = half_precision and device == "cuda"

    upscaler_params = dict(
        model=cl_name_upscaler,
        tile=tile,
        tile_overlap=tile_overlap,
        device=device,
        half=use_half_precision,
    )
    try:
        stats = zero_comp(
            video_file,
            output_path,
            as_gif,
            upscaler_factor,
            speed_factor,
            progress,
            upscaler_params,
            keep_audio,
            keep_metadata,
            codec,
            int(crf),
            float(temporal_smoothing or 0.0),
        )
        print(
            f"Saved {output_path} ({stats['frames']} frames, "
            f"{stats['width']}x{stats['height']}, {stats['fps']:.2f} FPS)"
        )
    except Exception:
        remove_if_exists(output_path)
        raise

    return output_path


with gr.Blocks(css=CUSTOM_CSS, title="Video Upscaler") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # inp_video = gr.Video(label="Input Video", format="mp4")
        inp_video = gr.File(label="Input Video", file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
        upscaler_choice = gr.Dropdown(
            choices=UPSCALER_KEYS,
            label="Upscaler",
            value=(UPSCALER_KEYS[0] if UPSCALER_KEYS else None),
            info="Select the upscaler model to use.",
        )
        model_guidance = gr.Textbox(
            label="Model guidance",
            value=model_guidance_text(UPSCALER_KEYS[0] if UPSCALER_KEYS else ""),
            interactive=False,
            lines=2,
        )
    with gr.Row():
        upscaler_factor_slider = gr.Slider(
            minimum=1.1, maximum=4.0, step=0.1,
            label="Upscaler Factor", value=1.5,
            info="Set how much to upscale the video. For example, 2.0 doubles the resolution, 3.0 triples it, etc.",
        )

    with gr.Accordion("Settings", open=False):

        with gr.Row():
            gif_checkbox = gr.Checkbox(
                label="Output as GIF", value=False,
                info="If checked, the output will be a GIF file instead of MP4.",
            )
            speed_slider = gr.Slider(
                minimum=0.1, maximum=2.0, step=0.1,
                label="Speed Factor", value=1.0,
                info="Adjust the speed of the output video. Values >1.0 speed up the video, <1.0 slow it down.",
            )
            keep_audio_checkbox = gr.Checkbox(
                label="Keep Audio", value=True,
                info="Preserve audio in MP4 output when ffmpeg is available.",
            )
            keep_metadata_checkbox = gr.Checkbox(
                label="Keep Metadata", value=True,
                info="Preserve source metadata tags when ffmpeg is available.",
            )

        with gr.Row():
            half_check = gr.Checkbox(
                label="Half-Precision", value=True,
                info="Use half-precision (FP16) for upscaling. This reduces VRAM usage and may speed up processing on compatible GPUs.",
                interactive=(get_compute_device() == "cuda" and not IS_ZERO_GPU),
            )
            tile_slider = gr.Slider(
                minimum=0, maximum=512, step=16,
                label="Tile Size", value=(0 if IS_ZERO_GPU else 192),
                interactive=(not IS_ZERO_GPU),
                info="0 means no tiling. Larger tiles may improve quality but use more VRAM.",
            )
            overlap_slider = gr.Slider(
                minimum=0, maximum=48, step=8,
                label="Tile Overlap", value=8,
                info="Higher values can reduce seams but increase VRAM usage.",
            )

        with gr.Row():
            codec_choice = gr.Dropdown(
                choices=["libx264", "libx265"],
                label="MP4 Codec",
                value="libx264",
                info="H.264 is most compatible; H.265 is smaller but slower and less universally supported.",
            )
            crf_slider = gr.Slider(
                minimum=12,
                maximum=30,
                step=1,
                label="MP4 Quality (CRF)",
                value=18,
                info="Lower means higher quality and larger files. 18 is visually high quality.",
            )
            temporal_smoothing_slider = gr.Slider(
                minimum=0.0,
                maximum=0.4,
                step=0.05,
                label="Temporal Smoothing",
                value=0.0,
                info="Blends a little of the previous upscaled frame to reduce shimmer. Higher values can cause ghosting.",
            )

        with gr.Row():
            max_frames_number = gr.Number(
                label="Max Frames",
                value=(ZERO_GPU_MAX_FRAMES if IS_ZERO_GPU else DEFAULT_MAX_FRAMES),
                precision=0,
                minimum=0,
                info="0 means no custom limit. Useful for preventing accidental huge jobs.",
                interactive=(not IS_ZERO_GPU),
            )
            max_width_number = gr.Number(
                label="Max Width",
                value=(ZERO_GPU_MAX_WIDTH if IS_ZERO_GPU else DEFAULT_MAX_WIDTH),
                precision=0,
                minimum=0,
                info="0 means no custom width limit.",
                interactive=(not IS_ZERO_GPU),
            )
            max_height_number = gr.Number(
                label="Max Height",
                value=(ZERO_GPU_MAX_HEIGHT if IS_ZERO_GPU else DEFAULT_MAX_HEIGHT),
                precision=0,
                minimum=0,
                info="0 means no custom height limit.",
                interactive=(not IS_ZERO_GPU),
            )
    upscale_button = gr.Button("Upscale Video", variant="primary")
    output_text = gr.File(label="Upscaled video")

    upscaler_choice.change(
        fn=model_guidance_text,
        inputs=upscaler_choice,
        outputs=model_guidance,
    )

    upscale_button.click(
        fn=start_upscaler,
        inputs=[
            inp_video,
            upscaler_choice,
            gif_checkbox,
            speed_slider,
            half_check,
            tile_slider,
            overlap_slider,
            upscaler_factor_slider,
            max_frames_number,
            max_width_number,
            max_height_number,
            keep_audio_checkbox,
            keep_metadata_checkbox,
            codec_choice,
            crf_slider,
            temporal_smoothing_slider,
        ],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(
        debug=True,
        show_error=True,
        quiet=False,
    )
