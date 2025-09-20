"""
Unit tests for the LoRA metadata utilities.

These tests verify that the shared metadata loading and parsing functionality
works correctly with mock data and different metadata file formats.
"""

import os
import unittest
from unittest.mock import Mock, mock_open, patch

from nodes.lora_metadata_utils import (
    LoRAMetadataLoader,
    classify_lora_type,
    extract_embeddable_content,
    extract_example_prompts,
    extract_recommended_weight,
    find_lora_absolute_path,
    get_lora_trigger_words,
    is_video_lora,
    is_wan_2_2_lora,
    load_lora_metadata,
    parse_lora_tag,
)

os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")


class TestLoRAMetadataLoader(unittest.TestCase):
    """Unit tests for LoRAMetadataLoader class."""

    def setUp(self):
        self.loader = LoRAMetadataLoader()

    def test_extract_trigger_words_from_civitai(self):
        """extract_trigger_words should extract from civitai.trainedWords."""
        metadata = {"civitai": {"trainedWords": ["beautiful", "anime-style", "portrait"]}}

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
        metadata = {"civitai": {"trainedWords": ["beautiful", "", "  ", "anime-style", None]}}

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
    @patch("nodes.lora_metadata_utils.glob.glob")
    def test_load_metadata_success(self, mock_glob, mock_file, mock_folder_paths):
        """load_metadata should successfully load JSON from metadata file."""
        # Mock folder_paths
        mock_folder_paths.get_folder_paths.return_value = ["/fake/loras"]

        # Mock glob.glob to return a file path
        mock_glob.return_value = ["/fake/loras/subfolder/test-lora.metadata.json"]

        # Mock file content
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


class TestNewMetadataFunctions(unittest.TestCase):
    """Tests for the new metadata utility functions."""

    def test_extract_embeddable_content_basic(self):
        """extract_embeddable_content should combine various metadata fields."""
        metadata = {
            "model_name": "Test LoRA",
            "modelDescription": "<p>A great <strong>style</strong> LoRA</p>",
            "tags": ["anime", "portrait"],
            "civitai": {
                "model": {
                    "name": "Civitai Model",
                    "description": "<h1>Detailed</h1> description",
                    "tags": ["character", "detailed"],
                },
                "trainedWords": ["trigger1", "trigger2"],
            },
        }

        content = extract_embeddable_content(metadata)

        self.assertIn("Test LoRA", content)
        self.assertIn("A great style LoRA", content)  # HTML stripped
        self.assertIn("Civitai Model", content)
        self.assertIn("Detailed description", content)  # HTML stripped
        self.assertIn("anime", content)
        self.assertIn("portrait", content)
        self.assertIn("character", content)
        self.assertIn("detailed", content)
        self.assertIn("trigger1", content)
        self.assertIn("trigger2", content)

    def test_extract_embeddable_content_empty(self):
        """extract_embeddable_content should handle empty metadata gracefully."""
        self.assertEqual(extract_embeddable_content({}), "")
        self.assertEqual(extract_embeddable_content({"civitai": {}}), "")

    def test_extract_example_prompts(self):
        """extract_example_prompts should extract prompts from civitai images."""
        metadata = {
            "civitai": {
                "images": [
                    {"meta": {"prompt": "beautiful anime girl, detailed"}},
                    {"meta": {"prompt": "cyberpunk scene, neon"}},
                    {"meta": {"prompt": ""}},  # Empty prompt should be skipped
                    {"meta": {}},  # No prompt field
                    {"meta": {"prompt": "landscape, mountains"}},
                ]
            }
        }

        prompts = extract_example_prompts(metadata, limit=3)

        self.assertEqual(len(prompts), 3)
        self.assertIn("beautiful anime girl, detailed", prompts)
        self.assertIn("cyberpunk scene, neon", prompts)
        self.assertIn("landscape, mountains", prompts)

    def test_extract_example_prompts_empty(self):
        """extract_example_prompts should handle empty metadata gracefully."""
        self.assertEqual(extract_example_prompts({}), [])
        self.assertEqual(extract_example_prompts({"civitai": {}}), [])
        self.assertEqual(extract_example_prompts({"civitai": {"images": []}}), [])

    def test_classify_lora_type_video(self):
        """classify_lora_type should identify video LoRAs correctly."""
        metadata_variants = [
            {"base_model": "Wan Video 14B i2v 480p"},
            {"base_model": "Video Generation Model"},
            {"civitai": {"baseModel": "Wan Video"}},
            {"civitai": {"baseModel": "I2V Model"}},
        ]

        for metadata in metadata_variants:
            with self.subTest(metadata=metadata):
                self.assertEqual(classify_lora_type(metadata), "video")

    def test_classify_lora_type_image(self):
        """classify_lora_type should identify image LoRAs correctly."""
        metadata_variants = [
            {"base_model": "SDXL 1.0"},
            {"base_model": "Illustrious"},
            {"base_model": "NoobAI"},
            {"civitai": {"baseModel": "FLUX.1"}},
            {"civitai": {"baseModel": "SD1.5"}},
        ]

        for metadata in metadata_variants:
            with self.subTest(metadata=metadata):
                self.assertEqual(classify_lora_type(metadata), "image")

    def test_classify_lora_type_unknown(self):
        """classify_lora_type should return unknown for unrecognized models."""
        self.assertEqual(classify_lora_type({}), "unknown")
        self.assertEqual(classify_lora_type({"base_model": "Unknown Model"}), "unknown")

    def test_extract_recommended_weight_from_description(self):
        """extract_recommended_weight should parse weights from descriptions."""
        test_cases = [
            ("Best result with weight between : 0.3-0.7", 0.3),
            ("Recommended strength: 0.8", 0.8),
            ("Use 0.6 strength for best results", 0.6),
            ("Weight 1.2 works well", 1.2),
            ("Best at 0.9", 0.9),
        ]

        for description, expected_weight in test_cases:
            with self.subTest(description=description):
                metadata = {"modelDescription": description}
                weight = extract_recommended_weight(metadata)
                self.assertEqual(weight, expected_weight)

    def test_extract_recommended_weight_defaults(self):
        """extract_recommended_weight should use appropriate defaults."""
        # Video LoRA default
        video_metadata = {"base_model": "Wan Video"}
        self.assertEqual(extract_recommended_weight(video_metadata), 0.6)

        # Image LoRA default
        image_metadata = {"base_model": "SDXL"}
        self.assertEqual(extract_recommended_weight(image_metadata), 0.8)

        # Unknown LoRA default
        unknown_metadata = {}
        self.assertEqual(extract_recommended_weight(unknown_metadata), 0.8)

    def test_extract_recommended_weight_bounds_checking(self):
        """extract_recommended_weight should reject unreasonable weights."""
        # Too high
        metadata = {"modelDescription": "Use weight 5.0"}
        self.assertEqual(extract_recommended_weight(metadata), 0.8)  # Should use default

        # Too low
        metadata = {"modelDescription": "Use weight 0.05"}
        self.assertEqual(extract_recommended_weight(metadata), 0.8)  # Should use default

    def test_is_wan_2_2_lora_base_model(self):
        """is_wan_2_2_lora should detect WAN 2.2 from base_model field."""
        # Test various WAN 2.2 indicators in base_model
        metadata = {"base_model": "WAN2.2"}
        self.assertTrue(is_wan_2_2_lora(metadata))

        metadata = {"base_model": "wan 2.2 video model"}
        self.assertTrue(is_wan_2_2_lora(metadata))

        metadata = {"base_model": "WAN v2.2"}
        self.assertTrue(is_wan_2_2_lora(metadata))

        # Test non-WAN 2.2 models
        metadata = {"base_model": "WAN2.1"}
        self.assertFalse(is_wan_2_2_lora(metadata))

        metadata = {"base_model": "SDXL"}
        self.assertFalse(is_wan_2_2_lora(metadata))

    def test_is_wan_2_2_lora_civitai_model(self):
        """is_wan_2_2_lora should detect WAN 2.2 from civitai fields."""
        # Test civitai baseModel
        metadata = {"civitai": {"baseModel": "WAN2.2"}}
        self.assertTrue(is_wan_2_2_lora(metadata))

        # Test civitai model name
        metadata = {"civitai": {"model": {"name": "Character WAN2.2 LoRA"}}}
        self.assertTrue(is_wan_2_2_lora(metadata))

        # Test non-WAN 2.2
        metadata = {"civitai": {"baseModel": "WAN"}}
        self.assertFalse(is_wan_2_2_lora(metadata))

    def test_is_wan_2_2_lora_model_name(self):
        """is_wan_2_2_lora should detect WAN 2.2 from model_name field."""
        metadata = {"model_name": "Dancing Girl WAN2.2"}
        self.assertTrue(is_wan_2_2_lora(metadata))

        metadata = {"model_name": "Regular LoRA"}
        self.assertFalse(is_wan_2_2_lora(metadata))

    def test_is_wan_2_2_lora_empty_metadata(self):
        """is_wan_2_2_lora should handle empty or None metadata."""
        self.assertFalse(is_wan_2_2_lora(None))
        self.assertFalse(is_wan_2_2_lora({}))
        self.assertFalse(is_wan_2_2_lora({"unrelated_field": "value"}))


class TestLoRATagParsing(unittest.TestCase):
    """Tests for LoRA tag parsing functions."""

    def test_parse_lora_tag_basic(self):
        """parse_lora_tag should parse basic LoRA tags correctly."""
        tag = "<lora:style_HIGH:0.8>"
        result = parse_lora_tag(tag)

        expected = {"name": "style_HIGH", "strength": "0.8", "tag": "<lora:style_HIGH:0.8>"}
        self.assertEqual(result, expected)

    def test_parse_lora_tag_default_strength(self):
        """parse_lora_tag should default to 1.0 strength when not provided."""
        tag = "<lora:character>"
        result = parse_lora_tag(tag)

        self.assertEqual(result["name"], "character")
        self.assertEqual(result["strength"], "1.0")
        self.assertEqual(result["tag"], "<lora:character>")

    def test_parse_lora_tag_with_extra_params(self):
        """parse_lora_tag should handle tags with extra parameters."""
        tag = "<lora:model_name:0.6:extra_param>"
        result = parse_lora_tag(tag)

        self.assertEqual(result["name"], "model_name")
        self.assertEqual(result["strength"], "0.6")
        self.assertEqual(result["tag"], "<lora:model_name:0.6:extra_param>")

    def test_parse_lora_tag_malformed(self):
        """parse_lora_tag should handle malformed tags gracefully."""
        tag = "not_a_lora_tag"
        result = parse_lora_tag(tag)

        self.assertEqual(result["name"], "not_a_lora_tag")
        self.assertEqual(result["strength"], "1.0")
        self.assertEqual(result["tag"], "not_a_lora_tag")

    def test_parse_lora_tag_empty_parts(self):
        """parse_lora_tag should handle empty tag parts."""
        tag = "<lora::0.5>"
        result = parse_lora_tag(tag)

        self.assertEqual(result["name"], "")
        self.assertEqual(result["strength"], "0.5")

    @patch("nodes.lora_metadata_utils.folder_paths")
    def test_find_lora_absolute_path_no_folder_paths(self, mock_folder_paths):
        """find_lora_absolute_path should handle missing folder_paths module."""
        # Temporarily patch the module-level folder_paths
        with patch("nodes.lora_metadata_utils.folder_paths", None):
            result = find_lora_absolute_path("test_lora")
            self.assertEqual(result, "[ComfyUI not available] test_lora.safetensors")


class TestFindLoraRelativePath(unittest.TestCase):
    """Tests for find_lora_relative_path function."""

    @patch("nodes.lora_metadata_utils.folder_paths")
    @patch("nodes.lora_metadata_utils.Path")
    def test_find_lora_relative_path_success(self, mock_path_class, mock_folder_paths):
        """find_lora_relative_path should return Linux-style relative path."""
        from nodes.lora_metadata_utils import find_lora_relative_path

        # Mock folder paths
        mock_folder_paths.get_folder_paths.return_value = ["C:\\ComfyUI\\models\\loras"]

        # Mock Path objects and file system
        mock_lora_dir = Mock()
        mock_file_path = Mock()

        # Mock the file path relative_to method to return a Windows-style path
        mock_relative_path = Mock()
        mock_relative_path.__str__ = Mock(return_value=r"wan\wan2.2\style_lora.safetensors")
        mock_file_path.relative_to.return_value = mock_relative_path
        mock_file_path.is_file.return_value = True

        mock_lora_dir.rglob.return_value = [mock_file_path]
        mock_path_class.return_value = mock_lora_dir

        result = find_lora_relative_path("style_lora")

        # Should return Windows-style path with backslashes
        self.assertEqual(result, r"wan\wan2.2\style_lora.safetensors")
        mock_file_path.relative_to.assert_called_once_with(mock_lora_dir)

    @patch("nodes.lora_metadata_utils.folder_paths")
    def test_find_lora_relative_path_no_folder_paths(self, mock_folder_paths):
        """find_lora_relative_path should handle missing folder_paths module."""
        from nodes.lora_metadata_utils import find_lora_relative_path

        # Temporarily patch the module-level folder_paths
        with patch("nodes.lora_metadata_utils.folder_paths", None):
            result = find_lora_relative_path("test_lora")
            self.assertEqual(result, "[ComfyUI not available] test_lora.safetensors")

    @patch("nodes.lora_metadata_utils.folder_paths")
    def test_find_lora_relative_path_no_lora_paths(self, mock_folder_paths):
        """find_lora_relative_path should handle empty lora paths."""
        from nodes.lora_metadata_utils import find_lora_relative_path

        mock_folder_paths.get_folder_paths.return_value = []

        result = find_lora_relative_path("test_lora")
        self.assertEqual(result, "[No LoRA folders configured] test_lora.safetensors")

    @patch("nodes.lora_metadata_utils.folder_paths")
    @patch("nodes.lora_metadata_utils.Path")
    def test_find_lora_relative_path_not_found(self, mock_path_class, mock_folder_paths):
        """find_lora_relative_path should handle file not found."""
        from nodes.lora_metadata_utils import find_lora_relative_path

        mock_folder_paths.get_folder_paths.return_value = ["C:\\ComfyUI\\models\\loras"]

        # Mock empty search results
        mock_lora_dir = Mock()
        mock_lora_dir.rglob.return_value = []
        mock_path_class.return_value = mock_lora_dir

        result = find_lora_relative_path("nonexistent_lora")
        self.assertEqual(result, "nonexistent_lora.safetensors")

    @patch("nodes.lora_metadata_utils.folder_paths")
    def test_find_lora_absolute_path_no_lora_folders(self, mock_folder_paths):
        """find_lora_absolute_path should handle missing LoRA folders."""
        mock_folder_paths.get_folder_paths.return_value = []

        result = find_lora_absolute_path("test_lora")
        self.assertEqual(result, "[No LoRA folders configured] test_lora.safetensors")

    @patch("nodes.lora_metadata_utils.folder_paths")
    @patch("nodes.lora_metadata_utils.Path")
    def test_find_lora_absolute_path_found(self, mock_path_class, mock_folder_paths):
        """find_lora_absolute_path should return absolute path when file is found."""
        # Mock folder_paths
        mock_folder_paths.get_folder_paths.return_value = ["/fake/loras"]

        # Mock Path and rglob
        mock_lora_dir = mock_path_class.return_value
        mock_file_path = mock_path_class.return_value
        mock_file_path.is_file.return_value = True
        mock_file_path.absolute.return_value = "/absolute/path/test_lora.safetensors"
        mock_lora_dir.rglob.return_value = [mock_file_path]

        # Mock Path constructor to return our mocked directory
        mock_path_class.side_effect = (
            lambda x: mock_lora_dir if x == "/fake/loras" else mock_file_path
        )

        result = find_lora_absolute_path("test_lora")
        self.assertEqual(result, "/absolute/path/test_lora.safetensors")

    @patch("nodes.lora_metadata_utils.folder_paths")
    @patch("nodes.lora_metadata_utils.Path")
    def test_find_lora_absolute_path_not_found(self, mock_path_class, mock_folder_paths):
        """find_lora_absolute_path should return expected path when file is not found."""
        # Mock folder_paths
        mock_folder_paths.get_folder_paths.return_value = ["/fake/loras"]

        # Mock Path and empty rglob results
        mock_lora_dir = mock_path_class.return_value
        mock_lora_dir.rglob.return_value = []  # No files found

        # Mock expected path construction
        expected_path = mock_path_class.return_value
        expected_path.__truediv__ = lambda self, other: f"/fake/loras/{other}"

        # Mock Path constructor behavior
        def path_constructor(path_str):
            if path_str == "/fake/loras":
                return mock_lora_dir
            else:
                return expected_path

        mock_path_class.side_effect = path_constructor

        result = find_lora_absolute_path("test_lora")
        # Should return the expected path when file is not found
        self.assertIn("/fake/loras", result)
        self.assertIn("test_lora.safetensors", result)

    @patch("nodes.lora_metadata_utils.folder_paths")
    def test_find_lora_absolute_path_exception(self, mock_folder_paths):
        """find_lora_absolute_path should handle exceptions gracefully."""
        mock_folder_paths.get_folder_paths.side_effect = Exception("Test error")

        result = find_lora_absolute_path("test_lora")
        self.assertIn("[Error finding path: Test error]", result)
        self.assertIn("test_lora.safetensors", result)


if __name__ == "__main__":
    unittest.main()
