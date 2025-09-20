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
        mock_ollama.return_value = json.dumps({
            "high_tags": ["<lora:style_HIGH:0.8>"],
            "low_tags": ["<lora:style_LOW:0.6>"],
        })

        prompt = "woman dancing <lora:style_HIGH:0.8> beautiful <lora:style_LOW:0.6>"
        high_prompt, low_prompt, analysis = self.node.split_wan_prompt(prompt)

        # Check that base content is preserved and correct tags are added
        self.assertIn("woman dancing beautiful", high_prompt)
        self.assertIn("<lora:style_HIGH:0.8>", high_prompt)
        self.assertNotIn("<lora:style_LOW:0.6>", high_prompt)

        self.assertIn("woman dancing beautiful", low_prompt)
        self.assertIn("<lora:style_LOW:0.6>", low_prompt)
        self.assertNotIn("<lora:style_HIGH:0.8>", low_prompt)


if __name__ == "__main__":
    unittest.main()
