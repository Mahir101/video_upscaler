import hashlib
import os
import uuid
from urllib.parse import unquote, urlparse


OUTPUT_DIRECTORY = "outputs"


def make_output_path(video_file, ext, output_directory=OUTPUT_DIRECTORY):
    os.makedirs(output_directory, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(video_file))[0]
    run_id = uuid.uuid4().hex[:8]
    return os.path.join(output_directory, f"{base_filename}_upscaled_{run_id}.{ext}")


def upscaler_filename_from_url(url):
    parsed = urlparse(url)
    filename = unquote(os.path.basename(parsed.path))
    if not filename:
        raise ValueError(f"Could not determine model filename from URL: {url}")
    return filename


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_sidecar_path(path):
    return f"{path}.sha256"


def write_checksum_sidecar(path, digest):
    with open(checksum_sidecar_path(path), "w", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")


def cached_checksum(path):
    sidecar_path = checksum_sidecar_path(path)
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path, "r", encoding="utf-8") as f:
        contents = f.read().split()
    return contents[0] if contents else None


def remove_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def validate_video_limits(metadata, max_frames=0, max_width=0, max_height=0):
    total_frames = metadata["total_frames"]
    width = metadata["width"]
    height = metadata["height"]

    if max_frames and total_frames > max_frames:
        raise ValueError(f"Video has too many frames ({total_frames}). Maximum allowed is {max_frames}.")
    if max_width and width > max_width:
        raise ValueError(f"Video width is too high ({width}). Maximum allowed is {max_width}.")
    if max_height and height > max_height:
        raise ValueError(f"Video height is too high ({height}). Maximum allowed is {max_height}.")


def build_mux_command(
    ffmpeg,
    input_video_path,
    silent_video_path,
    output_path,
    codec="libx264",
    crf=18,
    keep_audio=True,
    keep_metadata=True,
):
    command = [
        ffmpeg,
        "-y",
        "-i",
        silent_video_path,
        "-i",
        input_video_path,
        "-map",
        "0:v:0",
    ]

    if keep_audio:
        command.extend(["-map", "1:a?"])

    command.extend([
        "-c:v",
        codec,
        "-preset",
        "medium",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
    ])

    if keep_audio:
        command.extend(["-c:a", "aac"])

    if keep_metadata:
        command.extend(["-map_metadata", "1", "-movflags", "+use_metadata_tags"])

    command.extend(["-shortest", output_path])
    return command
