"""
Unit tests for the WANLoRAHighLowSplitterNode.

Tests the LoRA tag classification and prompt reconstruction logic.
"""

import json
import os
import unittest
from unittest.mock import patch

from nodes.wan_lora_high_low_splitter import WANLoRAHighLowSplitterNode

os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")


class TestWANLoRAHighLowSplitterNode(unittest.TestCase):
    """Unit tests for the WAN LoRA High/Low splitter node."""

    def setUp(self):
        self.node = WANLoRAHighLowSplitterNode()

    def test_extract_lora_tags(self):
        """Test extraction of LoRA tags from prompt."""
        prompt = "woman dancing <lora:style_HIGH:0.8> beautiful <lora:style_LOW:0.6> detailed"
        tags = self.node._extract_lora_tags(prompt)
        expected = ["<lora:style_HIGH:0.8>", "<lora:style_LOW:0.6>"]
        self.assertEqual(tags, expected)

    def test_extract_lora_tags_no_tags(self):
        """Test extraction with no LoRA tags."""
        prompt = "woman dancing beautiful detailed"
        tags = self.node._extract_lora_tags(prompt)
        self.assertEqual(tags, [])

    @patch("nodes.wan_lora_high_low_splitter._shared_call_ollama_chat")
    def test_classify_lora_tags_with_llm_success(self, mock_ollama):
        """Test successful LLM classification."""
        mock_ollama.return_value = json.dumps(
            {
                "high_tags": ["<lora:style_HIGH:0.8>"],
                "low_tags": ["<lora:style_LOW:0.6>"],
            }
        )

        tags = ["<lora:style_HIGH:0.8>", "<lora:style_LOW:0.6>"]
        result = self.node._classify_lora_tags_with_llm(tags)

        expected = {
            "high_tags": ["<lora:style_HIGH:0.8>"],
            "low_tags": ["<lora:style_LOW:0.6>"],
        }
        self.assertEqual(result, expected)

    @patch("nodes.wan_lora_high_low_splitter._shared_ensure_model_available")
    @patch("nodes.wan_lora_high_low_splitter._shared_call_ollama_chat")
    def test_split_wan_prompt_basic(self, mock_ollama, mock_ensure):
        """Test basic prompt splitting functionality."""
        # Mock the LLM response
        mock_ollama.return_value = json.dumps(
            {
                "high_tags": ["<lora:style_HIGH:0.8>"],
                "low_tags": ["<lora:style_LOW:0.6>"],
            }
        )

        prompt = "woman dancing <lora:style_HIGH:0.8> beautiful <lora:style_LOW:0.6>"
        high_prompt, low_prompt, analysis = self.node.split_wan_prompt(prompt)

        # Check that base content is preserved and correct tags are added
        self.assertIn("woman dancing beautiful", high_prompt)
        self.assertIn("<lora:style_HIGH:0.8>", high_prompt)
        self.assertNotIn("<lora:style_LOW:0.6>", high_prompt)

        self.assertIn("woman dancing beautiful", low_prompt)
        self.assertIn("<lora:style_LOW:0.6>", low_prompt)
        self.assertNotIn("<lora:style_HIGH:0.8>", low_prompt)

    @patch("nodes.wan_lora_high_low_splitter._shared_ensure_model_available")
    @patch("nodes.wan_lora_high_low_splitter._shared_call_ollama_chat")
    def test_enhanced_analysis_output(self, mock_ollama, mock_ensure):
        """Test the enhanced analysis output with detailed LoRA information."""
        # Mock the LLM response
        mock_ollama.return_value = json.dumps(
            {
                "high_tags": ["<lora:style_HIGH:0.8>", "<lora:quality_HIGH:1.2>"],
                "low_tags": ["<lora:motion_LOW:0.6>"],
            }
        )

        prompt = "woman <lora:style_HIGH:0.8> dancing <lora:motion_LOW:0.6> <lora:quality_HIGH:1.2>"
        high_prompt, low_prompt, analysis_json = self.node.split_wan_prompt(prompt)

        # Parse analysis
        analysis = json.loads(analysis_json)

        # Test structure
        self.assertIn("prompt_no_lora_tags", analysis)
        self.assertIn("high_lora_1", analysis)
        self.assertIn("high_lora_2", analysis)
        self.assertIn("low_lora_1", analysis)

        # Test prompt without LoRA tags
        self.assertEqual(analysis["prompt_no_lora_tags"], "woman dancing")

        # Test high_lora_1
        self.assertEqual(analysis["high_lora_1"]["tag"], "<lora:style_HIGH:0.8>")
        self.assertEqual(analysis["high_lora_1"]["strength"], "0.8")
        self.assertIn("rel_path", analysis["high_lora_1"])

        # Test high_lora_2
        self.assertEqual(analysis["high_lora_2"]["tag"], "<lora:quality_HIGH:1.2>")
        self.assertEqual(analysis["high_lora_2"]["strength"], "1.2")
        self.assertIn("rel_path", analysis["high_lora_2"])

        # Test low_lora_1
        self.assertEqual(analysis["low_lora_1"]["tag"], "<lora:motion_LOW:0.6>")
        self.assertEqual(analysis["low_lora_1"]["strength"], "0.6")
        self.assertIn("rel_path", analysis["low_lora_1"])

        # Should not have high_lora_3 or low_lora_2
        self.assertNotIn("high_lora_3", analysis)
        self.assertNotIn("low_lora_2", analysis)

    @patch("nodes.wan_lora_high_low_splitter._shared_ensure_model_available")
    @patch("nodes.wan_lora_high_low_splitter._shared_call_ollama_chat")
    def test_analysis_with_no_lora_tags(self, mock_ollama, mock_ensure):
        """Test analysis output when there are no HIGH/LOW LoRA tags."""
        # Mock the LLM response with empty tags
        mock_ollama.return_value = json.dumps(
            {
                "high_tags": [],
                "low_tags": [],
            }
        )

        prompt = "beautiful woman dancing"
        high_prompt, low_prompt, analysis_json = self.node.split_wan_prompt(prompt)

        # Parse analysis
        analysis = json.loads(analysis_json)

        # Should only have prompt_no_lora_tags
        self.assertEqual(len(analysis), 1)
        self.assertIn("prompt_no_lora_tags", analysis)
        self.assertEqual(analysis["prompt_no_lora_tags"], "beautiful woman dancing")
        self.assertNotIn("high_lora_1", analysis)
        self.assertNotIn("low_lora_1", analysis)

    def test_parse_lora_tag_integration(self):
        """Test that parse_lora_tag function works correctly when called from node."""
        from nodes.lora_metadata_utils import parse_lora_tag

        # Test various LoRA tag formats
        test_cases = [
            ("<lora:style_HIGH:0.8>", "style_HIGH", "0.8"),
            ("<lora:motion_LOW:0.6>", "motion_LOW", "0.6"),
            ("<lora:quality_HIGH:1.2>", "quality_HIGH", "1.2"),
            ("<lora:simple>", "simple", "1.0"),  # Default strength
        ]

        for tag, expected_name, expected_strength in test_cases:
            with self.subTest(tag=tag):
                result = parse_lora_tag(tag)
                self.assertEqual(result["name"], expected_name)
                self.assertEqual(result["strength"], expected_strength)
                self.assertEqual(result["tag"], tag)


if __name__ == "__main__":
    unittest.main()
