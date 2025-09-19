"""
Unit tests for the LoRA metadata utilities.

These tests verify that the shared metadata loading and parsing functionality
works correctly with mock data and different metadata file formats.
"""

import os
import unittest
from unittest.mock import mock_open, patch

from nodes.lora_metadata_utils import (
    LoRAMetadataLoader,
    classify_lora_pairs_with_ollama,
    classify_lora_type,
    extract_embeddable_content,
    extract_example_prompts,
    extract_recommended_weight,
    find_lora_pair_fuzzy,
    get_lora_trigger_words,
    is_video_lora,
    is_wan_2_2_lora,
    load_lora_metadata,
    parse_ollama_classification_response,
    split_prompt_by_lora_high_low_with_ollama,
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


class TestHighLowSplittingFunctions(unittest.TestCase):
    """Tests for the HIGH/LOW LoRA splitting utility functions."""

    def test_find_lora_pair_fuzzy_success(self):
        """find_lora_pair_fuzzy should find matching pairs based on similarity."""
        lora_names = ["Model-22-H-e8", "Model-22-L-e8", "character_high", "character_low"]

        # Should find pairs based on similarity
        pair1 = find_lora_pair_fuzzy("Model-22-H-e8", lora_names)
        self.assertEqual(pair1, "Model-22-L-e8")

        pair2 = find_lora_pair_fuzzy("character_high", lora_names)
        self.assertEqual(pair2, "character_low")

    def test_find_lora_pair_fuzzy_no_match(self):
        """find_lora_pair_fuzzy should return None when no suitable match is found."""
        lora_names = ["Model-22-H-e8", "Model-22-L-e8", "unrelated_lora"]

        no_pair = find_lora_pair_fuzzy("unrelated_lora", lora_names)
        self.assertIsNone(no_pair)

    def test_find_lora_pair_fuzzy_empty_candidates(self):
        """find_lora_pair_fuzzy should handle empty candidate lists."""
        result = find_lora_pair_fuzzy("test_lora", [])
        self.assertIsNone(result)

        result = find_lora_pair_fuzzy("test_lora", ["test_lora"])  # Only self
        self.assertIsNone(result)

    def test_split_prompt_by_lora_high_low_with_ollama_simple_mode(self):
        """split_prompt_by_lora_high_low_with_ollama should work in simple mode (no Ollama)."""
        prompt = "test scene <lora:character_high:0.8> <lora:character_low:0.6>"

        # Test simple mode (no Ollama, treats all LoRAs as singles)
        high_prompt, low_prompt = split_prompt_by_lora_high_low_with_ollama(
            prompt, use_ollama=False
        )

        # Both outputs should contain the base prompt
        self.assertIn("test scene", high_prompt)
        self.assertIn("test scene", low_prompt)

        # In simple mode without rapidfuzz, LoRAs are treated as singles (included in both outputs)
        # With rapidfuzz available, similar LoRAs might be paired via fuzzy matching
        # Both HIGH and LOW should contain the base prompt
        self.assertEqual(high_prompt.count("test scene"), 1)
        self.assertEqual(low_prompt.count("test scene"), 1)

        # Both LoRAs should appear in the output (either paired or as singles)
        self.assertIn("<lora:character_high:0.8>", high_prompt)
        self.assertIn("<lora:character_low:0.6>", low_prompt)

    @patch("nodes.ollama_utils.ensure_model_available")
    @patch("nodes.ollama_utils.call_ollama_chat")
    def test_classify_lora_pairs_with_ollama_success(self, mock_chat, mock_ensure):
        """classify_lora_pairs_with_ollama should successfully classify pairs."""
        candidate_pairs = [("Model-22-H-e8", "Model-22-L-e8")]

        # Mock successful Ollama response
        mock_response = """{
            "classifications": [
                {
                    "pair_index": 1,
                    "high_lora": "Model-22-H-e8",
                    "low_lora": "Model-22-L-e8",
                    "reasoning": "H indicator suggests high noise"
                }
            ]
        }"""
        mock_chat.return_value = mock_response

        classifications = classify_lora_pairs_with_ollama(candidate_pairs)

        expected_key = ("Model-22-H-e8", "Model-22-L-e8")
        self.assertIn(expected_key, classifications)
        self.assertEqual(classifications[expected_key]["high_lora"], "Model-22-H-e8")
        self.assertEqual(classifications[expected_key]["low_lora"], "Model-22-L-e8")
        self.assertEqual(
            classifications[expected_key]["reasoning"], "H indicator suggests high noise"
        )

    @patch("nodes.ollama_utils.ensure_model_available")
    def test_classify_lora_pairs_with_ollama_model_unavailable(self, mock_ensure):
        """classify_lora_pairs_with_ollama should raise exception when model is unavailable."""
        candidate_pairs = [("test_high", "test_low")]

        # Mock model unavailable
        mock_ensure.side_effect = Exception("Model not found")

        with self.assertRaises(Exception) as context:
            classify_lora_pairs_with_ollama(candidate_pairs)

        self.assertIn("qwen-coder:7b model is not available", str(context.exception))
        self.assertIn("ollama pull qwen-coder:7b", str(context.exception))

    def test_parse_ollama_classification_response_success(self):
        """parse_ollama_classification_response should parse valid JSON responses."""
        response = """{
            "classifications": [
                {
                    "pair_index": 1,
                    "high_lora": "test_high",
                    "low_lora": "test_low",
                    "reasoning": "test reasoning"
                }
            ]
        }"""
        candidate_pairs = [("test_high", "test_low")]

        classifications = parse_ollama_classification_response(response, candidate_pairs)

        expected_key = ("test_high", "test_low")
        self.assertIn(expected_key, classifications)
        self.assertEqual(classifications[expected_key]["high_lora"], "test_high")
        self.assertEqual(classifications[expected_key]["low_lora"], "test_low")

    def test_parse_ollama_classification_response_invalid_json(self):
        """parse_ollama_classification_response should handle invalid JSON gracefully."""
        response = "This is not valid JSON"
        candidate_pairs = [("test_high", "test_low")]

        classifications = parse_ollama_classification_response(response, candidate_pairs)

        self.assertEqual(classifications, {})

    def test_parse_ollama_classification_response_with_code_blocks(self):
        """parse_ollama_classification_response should handle markdown code blocks."""
        response = """```json
        {
            "classifications": [
                {
                    "pair_index": 1,
                    "high_lora": "test_high",
                    "low_lora": "test_low",
                    "reasoning": "test reasoning"
                }
            ]
        }
        ```"""
        candidate_pairs = [("test_high", "test_low")]

        classifications = parse_ollama_classification_response(response, candidate_pairs)

        expected_key = ("test_high", "test_low")
        self.assertIn(expected_key, classifications)

    def test_split_prompt_by_lora_high_low_with_ollama_no_pairs(self):
        """split_prompt_by_lora_high_low_with_ollama should handle prompts with no pairs."""
        prompt = "simple scene <lora:style1:0.8> <lora:character:0.6>"

        high_prompt, low_prompt = split_prompt_by_lora_high_low_with_ollama(
            prompt, use_ollama=False
        )

        # Both outputs should be identical (base + all single lora tags)
        self.assertEqual(high_prompt, low_prompt)
        self.assertIn("simple scene", high_prompt)
        self.assertIn("<lora:style1:0.8>", high_prompt)
        self.assertIn("<lora:character:0.6>", high_prompt)

    @patch("nodes.ollama_utils.ensure_model_available")
    @patch("nodes.ollama_utils.call_ollama_chat")
    def test_split_prompt_by_lora_high_low_with_ollama_advanced_mode(self, mock_chat, mock_ensure):
        """split_prompt_by_lora_high_low_with_ollama should work with Ollama classification."""
        prompt = "dancing robot <lora:character_high:0.8> <lora:character_low:0.6>"

        # Mock successful Ollama response
        mock_response = """{
            "classifications": [
                {
                    "pair_index": 1,
                    "high_lora": "character_high",
                    "low_lora": "character_low",
                    "reasoning": "high vs low naming pattern"
                }
            ]
        }"""
        mock_chat.return_value = mock_response

        high_prompt, low_prompt = split_prompt_by_lora_high_low_with_ollama(prompt, use_ollama=True)

        # HIGH prompt should have base + HIGH lora tag from the pair
        self.assertIn("dancing robot", high_prompt)
        self.assertIn("<lora:character_high:0.8>", high_prompt)
        self.assertNotIn("<lora:character_low:0.6>", high_prompt)

        # LOW prompt should have base + LOW lora tag from the pair
        self.assertIn("dancing robot", low_prompt)
        self.assertIn("<lora:character_low:0.6>", low_prompt)
        self.assertNotIn("<lora:character_high:0.8>", low_prompt)


if __name__ == "__main__":
    unittest.main()
