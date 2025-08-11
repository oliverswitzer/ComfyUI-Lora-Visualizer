"""
Unit tests for the LoRA metadata utilities.

These tests verify that the shared metadata loading and parsing functionality
works correctly with mock data and different metadata file formats.
"""

import os
import unittest
from unittest.mock import patch, mock_open

from nodes.lora_metadata_utils import (
    LoRAMetadataLoader,
    get_metadata_loader,
    load_lora_metadata,
    get_lora_trigger_words,
    is_video_lora,
)

os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")


class TestLoRAMetadataLoader(unittest.TestCase):
    """Unit tests for LoRAMetadataLoader class."""

    def setUp(self):
        self.loader = LoRAMetadataLoader()

    def test_extract_trigger_words_from_civitai(self):
        """extract_trigger_words should extract from civitai.trainedWords."""
        metadata = {
            "civitai": {"trainedWords": ["beautiful", "anime-style", "portrait"]}
        }

        trigger_words = self.loader.extract_trigger_words(metadata)

        self.assertEqual(len(trigger_words), 3)
        self.assertIn("beautiful", trigger_words)
        self.assertIn("anime-style", trigger_words)
        self.assertIn("portrait", trigger_words)

    def test_extract_trigger_words_empty_metadata(self):
        """extract_trigger_words should return empty list for None or missing data."""
        self.assertEqual(self.loader.extract_trigger_words(None), [])
        self.assertEqual(self.loader.extract_trigger_words({}), [])
        self.assertEqual(self.loader.extract_trigger_words({"civitai": {}}), [])

    def test_extract_trigger_words_filters_empty_strings(self):
        """extract_trigger_words should filter out empty or whitespace-only strings."""
        metadata = {
            "civitai": {"trainedWords": ["beautiful", "", "  ", "anime-style", None]}
        }

        trigger_words = self.loader.extract_trigger_words(metadata)

        self.assertEqual(len(trigger_words), 2)
        self.assertIn("beautiful", trigger_words)
        self.assertIn("anime-style", trigger_words)

    def test_is_video_lora_base_model(self):
        """is_video_lora should detect video LoRAs from base_model field."""
        video_metadata = {"base_model": "Wan Video 14B i2v 480p"}
        image_metadata = {"base_model": "Stable Diffusion XL"}

        self.assertTrue(self.loader.is_video_lora(video_metadata))
        self.assertFalse(self.loader.is_video_lora(image_metadata))

    def test_is_video_lora_civitai_base_model(self):
        """is_video_lora should detect video LoRAs from civitai.baseModel field."""
        video_metadata = {"civitai": {"baseModel": "Wan Video 14B i2v 480p"}}
        image_metadata = {"civitai": {"baseModel": "SDXL 1.0"}}

        self.assertTrue(self.loader.is_video_lora(video_metadata))
        self.assertFalse(self.loader.is_video_lora(image_metadata))

    def test_is_video_lora_case_insensitive(self):
        """is_video_lora should be case insensitive."""
        metadata = {"base_model": "WAN VIDEO 14B I2V 480P"}
        self.assertTrue(self.loader.is_video_lora(metadata))

    def test_is_video_lora_empty_metadata(self):
        """is_video_lora should return False for None or empty metadata."""
        self.assertFalse(self.loader.is_video_lora(None))
        self.assertFalse(self.loader.is_video_lora({}))

    @patch("nodes.lora_metadata_utils.folder_paths")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_load_metadata_success(self, mock_exists, mock_file, mock_folder_paths):
        """load_metadata should successfully load JSON from metadata file."""
        # Mock folder_paths
        mock_folder_paths.get_folder_paths.return_value = ["/fake/loras"]

        # Mock file existence
        mock_exists.return_value = True

        # Mock file content
        mock_metadata = {
            "model_name": "Test LoRA",
            "civitai": {"trainedWords": ["test"]},
        }
        mock_file.return_value.read.return_value = (
            '{"model_name": "Test LoRA", "civitai": {"trainedWords": ["test"]}}'
        )

        # Create new loader to pick up mocked folder_paths
        loader = LoRAMetadataLoader()
        result = loader.load_metadata("test-lora")

        self.assertIsNotNone(result)
        self.assertEqual(result["model_name"], "Test LoRA")

    @patch("nodes.lora_metadata_utils.folder_paths", None)
    def test_load_metadata_no_folder_paths(self):
        """load_metadata should handle missing folder_paths gracefully."""
        loader = LoRAMetadataLoader()
        result = loader.load_metadata("test-lora")

        self.assertIsNone(result)

    def test_get_lora_info(self):
        """get_lora_info should combine trigger words and video detection."""
        mock_metadata = {
            "base_model": "Wan Video 14B i2v",
            "civitai": {"trainedWords": ["motion-blur", "cinematic"]},
        }

        with patch.object(self.loader, "load_metadata", return_value=mock_metadata):
            info = self.loader.get_lora_info("test-lora")

            self.assertEqual(info["trigger_words"], ["motion-blur", "cinematic"])
            self.assertTrue(info["is_video_lora"])
            self.assertEqual(info["metadata"], mock_metadata)


class TestConvenienceFunctions(unittest.TestCase):
    """Test the convenience functions that use the global loader."""

    @patch("nodes.lora_metadata_utils.get_metadata_loader")
    def test_load_lora_metadata(self, mock_get_loader):
        """load_lora_metadata should delegate to the global loader."""
        mock_loader = mock_get_loader.return_value
        mock_loader.load_metadata.return_value = {"test": "data"}

        result = load_lora_metadata("test-lora")

        mock_loader.load_metadata.assert_called_once_with("test-lora")
        self.assertEqual(result, {"test": "data"})

    @patch("nodes.lora_metadata_utils.get_metadata_loader")
    def test_get_lora_trigger_words(self, mock_get_loader):
        """get_lora_trigger_words should return trigger words for a LoRA."""
        mock_loader = mock_get_loader.return_value
        mock_loader.load_metadata.return_value = {"civitai": {"trainedWords": ["test"]}}
        mock_loader.extract_trigger_words.return_value = ["test"]

        result = get_lora_trigger_words("test-lora")

        self.assertEqual(result, ["test"])

    @patch("nodes.lora_metadata_utils.get_metadata_loader")
    def test_is_video_lora_convenience(self, mock_get_loader):
        """is_video_lora convenience function should work correctly."""
        mock_loader = mock_get_loader.return_value
        mock_loader.load_metadata.return_value = {"base_model": "Wan Video"}
        mock_loader.is_video_lora.return_value = True

        result = is_video_lora("test-lora")

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
