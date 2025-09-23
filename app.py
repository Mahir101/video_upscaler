import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

DIRECTORY_UPSCALERS = "upscalers"
IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
MULTI_PROCESSING = True
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


# @spaces.GPU()
def process_frames(input_path, cap, total_frames, scaler_beta, upscaler_factor, progress=None):

    frames = []
    count = 0
    with tqdm(total=total_frames, desc="Processing video", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV (BGR numpy) -> PIL (RGB)
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if pil_frame.mode != "RGB":
                pil_frame = pil_frame.convert("RGB")

            processed_pil = scaler_beta.upscale(pil_frame, upscaler_factor, True)

            # Ensure it comes back as PIL
            if not isinstance(processed_pil, Image.Image):
                raise TypeError("demo() must return a PIL.Image")

            # Convert PIL (RGB) -> numpy BGR for video
            processed_np = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)

            frames.append(processed_np)

            count += 1
            pbar.update(1)
            pbar.set_description(f"Frames_{count}/{total_frames}")

    cap.release()

    return frames


# @spaces.GPU()
def process_frames_multithreaded(input_path, raw_frames, total_frames, scaler_beta, upscaler_factor, workers=None):
    if workers is None:
        workers = 16 if IS_ZERO_GPU else min(os.cpu_count(), 64)

    print(f"Upscaling {len(raw_frames)} frames with {workers} workers...")

    def upscale_frame(idx, frame):
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pil_frame.mode != "RGB":
            pil_frame = pil_frame.convert("RGB")
        processed_pil = scaler_beta.upscale(pil_frame, upscaler_factor, True)

        # Free GPU memory immediately after conversion
        result = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
        del pil_frame, processed_pil
        free_memory()
        return idx, result

    frames = [None] * len(raw_frames)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upscale_frame, i, f): i for i, f in enumerate(raw_frames)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Upscaling frames", unit="frames"):
            idx, result = future.result()
            frames[idx] = result
            # Release reference to the finished future
            del future

    return frames


@spaces.GPU(duration=120)
def zero_comp(video_file, upscaler_factor, progress, cap, total_frames, upscaler_params, workers):
    scaler_beta = load_upscaler_model(**upscaler_params)
    if cap is None:
        cap = cv2.VideoCapture(video_file)

    if MULTI_PROCESSING:
        frames = process_frames_multithreaded(video_file, cap, total_frames, scaler_beta, upscaler_factor, workers)
    else:
        frames = process_frames(video_file, cap, total_frames, scaler_beta, upscaler_factor, progress)

    scaler_beta = None
    del scaler_beta
    free_memory()

    return frames


def read_video_frames(cap):
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    return raw_frames


def start_upscaler(video_file, upscaler_name="R-ESRGAN 4x+ Anime6B", as_gif=False, speed_factor=1.0, half_precision=True, tile=192, tile_overlap=8, upscaler_factor=1.5, workers=4, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise ValueError("Error: No video file provided.")
    if not isinstance(video_file, str):
        video_file = video_file.name
    ext = "gif" if as_gif else "mp4"
    base_filename = os.path.splitext(os.path.basename(video_file))[0]
    output_path = f"{base_filename}_upscaled.{ext}"

    cap = cv2.VideoCapture(video_file)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if IS_ZERO_GPU:
        if total_frames > 90:
            raise ValueError(f"Video has too many frames ({total_frames}). Maximum allowed is 90.")
        if width > 1024 or height > 1024:
            raise ValueError(f"Video resolution too high ({width}x{height}). Maximum allowed is 1024x1024.")

    cl_name_upscaler = UPSCALER_DICT_GUI[upscaler_name]
    if "https://" in str(cl_name_upscaler):
        local_path = f"./{DIRECTORY_UPSCALERS}/{cl_name_upscaler.split('/')[-1]}"

        if not os.path.exists(local_path):
            print(f"⬇️ Downloading upscaler from {cl_name_upscaler} ...")
            os.makedirs(DIRECTORY_UPSCALERS, exist_ok=True)

            # Stream download to avoid memory issues
            with requests.get(cl_name_upscaler, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"✅ Downloaded upscaler to {local_path}")

        cl_name_upscaler = local_path

    if MULTI_PROCESSING:
        cap = read_video_frames(cap)
    elif IS_ZERO_GPU and not MULTI_PROCESSING:
        cap.release()
        cap = None

    upscaler_params = dict(
        model=cl_name_upscaler,
        tile=tile,
        tile_overlap=tile_overlap,
        device="cuda",
        half=half_precision,
    )
    frames = zero_comp(video_file, upscaler_factor, progress, cap, total_frames, upscaler_params, workers)

    if not frames:
        msg = "Error: No frames were processed."
        print(msg)
        raise RuntimeError(msg)

    # Detect NEW size after upscaling
    new_height, new_width = frames[0].shape[:2]
    new_fps = fps * speed_factor if speed_factor > 0 else fps

    if as_gif:
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        duration = 1.0 / new_fps
        imageio.mimsave(output_path, frames_rgb, duration=duration)
        print(f"✅ GIF saved: {output_path} ({len(frames)} frames, {new_width}x{new_height})")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, new_fps, (new_width, new_height))
        for f in frames:
            out.write(f)
        out.release()
        print(f"✅ MP4 saved: {output_path} ({len(frames)} frames, {new_width}x{new_height}, {new_fps:.2f} FPS)")

    # free_memory()

    return output_path


with gr.Blocks(css=CUSTOM_CSS, title="Video Upscaler") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # inp_video = gr.Video(label="Input Video", format="mp4")
        inp_video = gr.File(label="Input Video (mp4)", file_types=[".mp4", ".avi"])
        upscaler_choice = gr.Dropdown(
            choices=UPSCALER_KEYS,
            label="Upscaler",
            value="R-ESRGAN AnimeVideo",
            info="Select the upscaler model to use.",
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

        with gr.Row():
            half_check = gr.Checkbox(
                label="Half-Precision", value=True,
                info="Use half-precision (FP16) for upscaling. This reduces VRAM usage and may speed up processing on compatible GPUs.",
                interactive=(not IS_ZERO_GPU),
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
        num_workers = gr.Number(
            label="Workers",
            value=(42 if IS_ZERO_GPU else 8),
            precision=0,
            minimum=2,
            maximum=64,
            info="Number of worker threads for multi-threaded upscaling. Higher values may speed up processing but use more VRAM.",
            interactive=(not IS_ZERO_GPU),
        )

    upscale_button = gr.Button("Upscale Video", variant="primary")
    output_text = gr.File(label="Upscaled video")

    upscale_button.click(
        fn=start_upscaler,
        inputs=[inp_video, upscaler_choice, gif_checkbox, speed_slider, half_check, tile_slider, overlap_slider, upscaler_factor_slider, num_workers],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(debug=True)
