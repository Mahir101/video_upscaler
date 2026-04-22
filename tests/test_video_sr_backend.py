import unittest

from video_sr_backend import SUPPORTED_VIDEO_SR_MODELS, video_sr_guidance_text


class VideoSRBackendTests(unittest.TestCase):
    def test_expected_temporal_models_are_registered(self):
        self.assertIn("BasicVSR", SUPPORTED_VIDEO_SR_MODELS)
        self.assertIn("BasicVSR++", SUPPORTED_VIDEO_SR_MODELS)
        self.assertIn("RealBasicVSR", SUPPORTED_VIDEO_SR_MODELS)

    def test_video_sr_guidance_mentions_selected_model_family(self):
        self.assertIn("real-world", video_sr_guidance_text("RealBasicVSR"))

    def test_video_sr_guidance_handles_unknown_model(self):
        self.assertIn("temporal", video_sr_guidance_text("unknown"))


if __name__ == "__main__":
    unittest.main()
