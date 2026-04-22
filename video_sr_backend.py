import os


SUPPORTED_VIDEO_SR_MODELS = {
    "BasicVSR": {
        "mmagic_name": "basicvsr",
        "description": "Temporal x4 video super-resolution baseline. Best for cleaner synthetic/downsampled clips.",
    },
    "BasicVSR++": {
        "mmagic_name": "basicvsr_pp",
        "description": "Stronger temporal propagation and alignment. Heavier than BasicVSR.",
    },
    "RealBasicVSR": {
        "mmagic_name": "real_basicvsr",
        "description": "Designed for real-world degraded videos. Usually the best fit for practical footage.",
    },
}


def video_sr_guidance_text(model_name):
    if model_name not in SUPPORTED_VIDEO_SR_MODELS:
        return "Select a temporal video super-resolution model."
    return SUPPORTED_VIDEO_SR_MODELS[model_name]["description"]


def run_mmagic_video_sr(
    input_video,
    output_frames_dir,
    model_name,
    device,
    max_seq_len=0,
    window_size=0,
):
    try:
        from mmagic.apis import MMagicInferencer
    except ImportError as exc:
        raise RuntimeError(
            "Video SR mode requires MMagic/OpenMMLab dependencies. Install the optional video-SR stack first; "
            "see README.md for the setup notes."
        ) from exc

    if model_name not in SUPPORTED_VIDEO_SR_MODELS:
        raise ValueError(f"Unsupported video SR model: {model_name}")

    os.makedirs(output_frames_dir, exist_ok=True)
    extra_parameters = {}
    if max_seq_len:
        extra_parameters["max_seq_len"] = int(max_seq_len)
    if window_size:
        extra_parameters["window_size"] = int(window_size)

    editor = MMagicInferencer(
        SUPPORTED_VIDEO_SR_MODELS[model_name]["mmagic_name"],
        device=device,
        extra_parameters=extra_parameters or None,
    )
    editor.infer(video=input_video, result_out_dir=output_frames_dir)
    return output_frames_dir
