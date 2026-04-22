import os
import tempfile
import unittest

from video_utils import (
    build_frame_encode_command,
    build_mux_command,
    cached_checksum,
    make_output_path,
    sha256_file,
    upscaler_filename_from_url,
    validate_video_limits,
    write_checksum_sidecar,
)


class VideoUtilsTests(unittest.TestCase):
    def test_make_output_path_is_unique_and_uses_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = make_output_path("/tmp/source clip.mp4", "mp4", output_directory=temp_dir)
            second = make_output_path("/tmp/source clip.mp4", "mp4", output_directory=temp_dir)

            self.assertNotEqual(first, second)
            self.assertEqual(os.path.dirname(first), temp_dir)
            self.assertTrue(os.path.basename(first).startswith("source clip_upscaled_"))
            self.assertTrue(first.endswith(".mp4"))

    def test_upscaler_filename_from_url_decodes_escaped_names(self):
        url = "https://example.com/models/Remacri%204x%20ExtraSmoother.pth"

        self.assertEqual(upscaler_filename_from_url(url), "Remacri 4x ExtraSmoother.pth")

    def test_checksum_sidecar_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = os.path.join(temp_dir, "model.pth")
            with open(model_file, "wb") as f:
                f.write(b"model bytes")

            digest = sha256_file(model_file)
            write_checksum_sidecar(model_file, digest)

            self.assertEqual(cached_checksum(model_file), digest)

    def test_validate_video_limits_allows_zero_limits(self):
        validate_video_limits({"total_frames": 10, "width": 1920, "height": 1080})

    def test_validate_video_limits_rejects_too_many_frames(self):
        with self.assertRaisesRegex(ValueError, "too many frames"):
            validate_video_limits({"total_frames": 101, "width": 1280, "height": 720}, max_frames=100)

    def test_validate_video_limits_rejects_large_dimensions(self):
        with self.assertRaisesRegex(ValueError, "width"):
            validate_video_limits({"total_frames": 10, "width": 1921, "height": 720}, max_width=1920)

        with self.assertRaisesRegex(ValueError, "height"):
            validate_video_limits({"total_frames": 10, "width": 1280, "height": 1081}, max_height=1080)

    def test_build_mux_command_can_disable_audio_and_metadata(self):
        command = build_mux_command(
            "ffmpeg",
            "input.mp4",
            "silent.mp4",
            "output.mp4",
            codec="libx265",
            crf=22,
            keep_audio=False,
            keep_metadata=False,
        )

        self.assertIn("-map", command)
        self.assertNotIn("1:a?", command)
        self.assertNotIn("-c:a", command)
        self.assertNotIn("-map_metadata", command)
        self.assertIn("libx265", command)
        self.assertIn("22", command)

    def test_build_mux_command_preserves_audio_and_metadata_by_default(self):
        command = build_mux_command("ffmpeg", "input.mp4", "silent.mp4", "output.mp4")

        self.assertIn("1:a?", command)
        self.assertIn("-c:a", command)
        self.assertIn("-map_metadata", command)

    def test_build_frame_encode_command_uses_source_fps_and_pattern(self):
        command = build_frame_encode_command(
            "ffmpeg",
            "frames/%08d.png",
            29.97003,
            "encoded.mp4",
            codec="libx264",
            crf=19,
        )

        self.assertIn("-framerate", command)
        self.assertIn("29.970030", command)
        self.assertIn("frames/%08d.png", command)
        self.assertIn("encoded.mp4", command)


if __name__ == "__main__":
    unittest.main()
