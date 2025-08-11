"""
Unit tests for the LoRA Prompt Composer Node.

These tests verify the prompt composition functionality while mocking
external dependencies like sentence-transformers.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Set environment variable to skip initialization for testing
os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")

from nodes.prompt_composer_node import PromptComposerNode


class TestPromptComposerNode(unittest.TestCase):
    """Unit tests for PromptComposerNode class."""

    def setUp(self):
        """Set up test fixtures."""
        self.node = PromptComposerNode()

    def test_input_types_structure(self):
        """INPUT_TYPES should return the correct structure."""
        input_types = PromptComposerNode.INPUT_TYPES()

        # Check required inputs
        self.assertIn("required", input_types)
        required = input_types["required"]

        self.assertIn("scene_description", required)
        self.assertIn("max_image_loras", required)
        self.assertIn("max_video_loras", required)

        # Check optional inputs
        self.assertIn("optional", input_types)
        optional = input_types["optional"]

        self.assertIn("content_boost", optional)
        self.assertIn("style_preference", optional)

    def test_return_types(self):
        """Return types should be properly defined."""
        self.assertEqual(
            PromptComposerNode.RETURN_TYPES, ("STRING", "STRING", "STRING")
        )
        self.assertEqual(
            PromptComposerNode.RETURN_NAMES,
            ("composed_prompt", "lora_analysis", "metadata_summary"),
        )

    def test_compose_prompt_empty_description(self):
        """compose_prompt should handle empty scene descriptions gracefully."""
        result = self.node.compose_prompt("")

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "No scene description provided.")

    def test_compose_prompt_whitespace_only(self):
        """compose_prompt should handle whitespace-only descriptions."""
        result = self.node.compose_prompt("   \n\t   ")

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "No scene description provided.")

    @patch("nodes.prompt_composer_node.discover_all_loras")
    def test_embeddings_initialization_failure(self, mock_discover):
        """Should handle embeddings initialization failure gracefully."""
        mock_discover.return_value = {}

        # Mock the import to fail
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = self.node.compose_prompt("test scene")

        self.assertEqual(len(result), 3)
        self.assertIn("Error: Could not initialize embeddings system", result[0])

    def test_is_content_lora_with_character_tags(self):
        """_is_content_lora should identify character LoRAs correctly."""
        metadata = {"civitai": {"model": {"tags": ["character", "anime"]}}}

        result = self.node._is_content_lora(metadata)
        self.assertTrue(result)

    def test_is_content_lora_with_trigger_words(self):
        """_is_content_lora should identify LoRAs with trigger words."""
        metadata = {"civitai": {"trainedWords": ["woman877", "character_name"]}}

        result = self.node._is_content_lora(metadata)
        self.assertTrue(result)

    def test_is_content_lora_style_only(self):
        """_is_content_lora should not identify style-only LoRAs as content."""
        metadata = {
            "civitai": {"model": {"tags": ["style", "artistic"]}, "trainedWords": []}
        }

        result = self.node._is_content_lora(metadata)
        self.assertFalse(result)

    def test_analyze_prompt_style_empty(self):
        """_analyze_prompt_style should handle empty prompt lists."""
        result = self.node._analyze_prompt_style([])

        self.assertEqual(result["patterns"], [])
        self.assertEqual(result["common_terms"], [])
        self.assertEqual(result["structure"], "simple")

    def test_analyze_prompt_style_with_content(self):
        """_analyze_prompt_style should extract style patterns."""
        prompts = [
            "beautiful anime girl, masterpiece, detailed",
            "stunning portrait, high quality, cinematic lighting",
        ]

        result = self.node._analyze_prompt_style(prompts)

        self.assertIn("beautiful", result["patterns"])
        self.assertIn("masterpiece", result["patterns"])
        self.assertIn("detailed", result["patterns"])

    def test_compose_final_prompt_structure(self):
        """_compose_final_prompt should create properly structured prompts."""
        scene_description = "cyberpunk woman in alley"

        image_loras = [
            {
                "name": "character_lora",
                "recommended_weight": 0.8,
                "trigger_words": ["woman877"],
            }
        ]

        video_loras = [
            {
                "name": "video_lora",
                "recommended_weight": 0.6,
                "trigger_words": ["motion"],
            }
        ]

        # Mock metadata for style analysis
        for lora in image_loras + video_loras:
            lora["metadata"] = {"civitai": {"images": []}}

        result = self.node._compose_final_prompt(
            scene_description, image_loras, video_loras, "natural"
        )

        # Should contain LoRA tags
        self.assertIn("<lora:character_lora:0.8>", result)
        self.assertIn("<wanlora:video_lora:0.6>", result)

        # Should contain trigger words
        self.assertIn("woman877", result)
        self.assertIn("motion", result)

        # Should contain scene description
        self.assertIn("cyberpunk woman in alley", result)

    def test_compose_final_prompt_no_duplicates(self):
        """_compose_final_prompt should remove duplicate trigger words."""
        scene_description = "test scene"

        image_loras = [
            {
                "name": "lora1",
                "recommended_weight": 0.8,
                "trigger_words": ["common_word", "unique1"],
                "metadata": {"civitai": {"images": []}},
            }
        ]

        video_loras = [
            {
                "name": "lora2",
                "recommended_weight": 0.6,
                "trigger_words": ["common_word", "unique2"],  # Duplicate "common_word"
                "metadata": {"civitai": {"images": []}},
            }
        ]

        result = self.node._compose_final_prompt(
            scene_description, image_loras, video_loras, "natural"
        )

        # Should only contain "common_word" once
        word_count = result.count("common_word")
        self.assertEqual(word_count, 1)

    @patch("nodes.prompt_composer_node.discover_all_loras")
    def test_successful_embeddings_initialization(self, mock_discover):
        """Should successfully initialize embeddings with mocked dependencies."""
        # Mock LoRA discovery
        mock_discover.return_value = {
            "test_lora": {
                "metadata": {"model_name": "Test LoRA"},
                "trigger_words": ["test"],
            }
        }

        # Mock sentence-transformers import and functionality
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]  # Mock embedding

        with patch.dict(
            "sys.modules",
            {
                "sentence_transformers": MagicMock(
                    SentenceTransformer=MagicMock(return_value=mock_model)
                ),
                "numpy": MagicMock(),
                "sklearn.metrics.pairwise": MagicMock(),
            },
        ):
            with patch(
                "nodes.prompt_composer_node.extract_embeddable_content"
            ) as mock_extract:
                mock_extract.return_value = "test lora description"

                result = self.node._initialize_embeddings()

        self.assertTrue(result)
        self.assertTrue(self.node._embeddings_initialized)


if __name__ == "__main__":
    unittest.main()
