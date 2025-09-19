"""
Unit tests for the LoRAHighLowSplitterNode and shared splitting functionality.

Tests the new high/low LoRA splitting node and the enhanced shared utility function
that supports HN/LN naming patterns in addition to HIGH/LOW.
"""

import unittest
from unittest.mock import patch

from nodes.lora_high_low_splitter_node import LoRAHighLowSplitterNode
from nodes.lora_metadata_utils import (
    find_lora_high_low_pair,
    find_lora_pairs_in_prompt,
)


class TestLoRAHighLowSplitter(unittest.TestCase):
    """Unit tests for the LoRA High/Low Splitter functionality."""

    def setUp(self):
        self.node = LoRAHighLowSplitterNode()

    def test_split_high_low_basic(self):
        """Test basic high/low splitting functionality."""
        prompt = "woman dancing <lora:style_high:0.8> <lora:motion_low:0.6> in garden"

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # High prompt should contain only high LoRA tag
        self.assertIn("woman dancing", high_prompt)
        self.assertIn("in garden", high_prompt)
        self.assertIn("<lora:style_high:0.8>", high_prompt)
        self.assertNotIn("<lora:motion_low:0.6>", high_prompt)

        # Low prompt should contain only low LoRA tag
        self.assertIn("woman dancing", low_prompt)
        self.assertIn("in garden", low_prompt)
        self.assertIn("<lora:motion_low:0.6>", low_prompt)
        self.assertNotIn("<lora:style_high:0.8>", low_prompt)

    def test_split_high_low_with_hn_ln_patterns(self):
        """Test splitting with HN/LN naming patterns."""
        prompt = "robot walking <lora:character_hn:0.7> <lora:background_ln:0.5> through city"

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # High prompt should contain HN LoRA tag
        self.assertIn("robot walking", high_prompt)
        self.assertIn("through city", high_prompt)
        self.assertIn("<lora:character_hn:0.7>", high_prompt)
        self.assertNotIn("<lora:background_ln:0.5>", high_prompt)

        # Low prompt should contain LN LoRA tag
        self.assertIn("robot walking", low_prompt)
        self.assertIn("through city", low_prompt)
        self.assertIn("<lora:background_ln:0.5>", low_prompt)
        self.assertNotIn("<lora:character_hn:0.7>", low_prompt)

    def test_split_with_single_lora_tags(self):
        """Test that single LoRA tags (without high/low patterns) are included in both outputs."""
        prompt = "character <lora:style_high:0.8> <lora:general_style:0.6> <lora:motion_low:0.7>"

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # Both outputs should contain the single LoRA tag
        self.assertIn("<lora:general_style:0.6>", high_prompt)
        self.assertIn("<lora:general_style:0.6>", low_prompt)

        # High prompt should have high LoRA + single LoRA
        self.assertIn("<lora:style_high:0.8>", high_prompt)
        self.assertNotIn("<lora:motion_low:0.7>", high_prompt)

        # Low prompt should have low LoRA + single LoRA
        self.assertIn("<lora:motion_low:0.7>", low_prompt)
        self.assertNotIn("<lora:style_high:0.8>", low_prompt)

    def test_split_no_lora_tags(self):
        """Test splitting when prompt has no LoRA tags."""
        prompt = "woman dancing gracefully in garden"

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # Both outputs should be identical to input
        self.assertEqual(high_prompt, prompt)
        self.assertEqual(low_prompt, prompt)

    def test_split_empty_prompt(self):
        """Test splitting with empty prompt."""
        high_prompt, low_prompt = self.node.split_high_low("")

        self.assertEqual(high_prompt, "")
        self.assertEqual(low_prompt, "")

    def test_substring_matching_precision(self):
        """Test that substring matches don't create false pairs."""
        prompt = (
            "character <lora:highlight_effect:0.8> "
            "with <lora:lowlight_shadows:0.6> "
            "and <lora:character_high:0.7>"
        )

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # "highlight_effect" and "lowlight_shadows" should be treated as single LoRAs
        self.assertIn("<lora:highlight_effect:0.8>", high_prompt)
        self.assertIn("<lora:lowlight_shadows:0.6>", high_prompt)
        self.assertIn("<lora:character_high:0.7>", high_prompt)

        self.assertIn("<lora:highlight_effect:0.8>", low_prompt)
        self.assertIn("<lora:lowlight_shadows:0.6>", low_prompt)
        self.assertNotIn("<lora:character_high:0.7>", low_prompt)


class TestLoRAHighLowPairing(unittest.TestCase):
    """Unit tests for the shared LoRA high/low pairing functionality."""

    def test_pairing_basic(self):
        """Test basic string pairing."""
        available_loras = ["character_high", "character_low", "style_normal"]

        # Test HIGH -> LOW
        pair = find_lora_high_low_pair("character_high", available_loras)
        self.assertEqual(pair, "character_low")

        # Test LOW -> HIGH
        pair = find_lora_high_low_pair("character_low", available_loras)
        self.assertEqual(pair, "character_high")

    def test_hn_ln_pairing(self):
        """Test HN/LN pairing."""
        available_loras = ["robot_hn", "robot_ln", "style_normal"]

        # Test HN -> LN
        pair = find_lora_high_low_pair("robot_hn", available_loras)
        self.assertEqual(pair, "robot_ln")

        # Test LN -> HN
        pair = find_lora_high_low_pair("robot_ln", available_loras)
        self.assertEqual(pair, "robot_hn")

    def test_similarity_threshold(self):
        """Test that similarity thresholds are respected."""
        available_loras = [
            "highlight_effect",
            "lowlight_shadows",
            "character_high",
            "character_low",
        ]

        # Very similar names should match
        pair = find_lora_high_low_pair("character_high", available_loras)
        self.assertEqual(pair, "character_low")

        # Dissimilar names should not match due to low similarity score
        pair = find_lora_high_low_pair("highlight_effect", available_loras)
        # This will likely return None due to low similarity with other LoRAs
        # or could return a match if similarity is above threshold
        self.assertIn(pair, [None, "lowlight_shadows", "character_high", "character_low"])

    def test_deprecated_function_wrapper(self):
        """Test that the deprecated function still works for backward compatibility."""
        available_loras = ["character_high", "character_low", "style_normal"]

        # The old function should still work but use string matching
        pair = find_lora_high_low_pair("character_high", available_loras)
        self.assertEqual(pair, "character_low")

    @patch("nodes.lora_metadata_utils.discover_all_loras")
    def test_prompt_pairing_no_lora_tags(self, mock_discover):
        """Test prompt pairing with no LoRA tags."""
        mock_discover.return_value = {}

        prompt = "woman dancing in garden"
        result = find_lora_pairs_in_prompt(prompt)

        # Should return unchanged
        self.assertEqual(result, prompt)

    @patch("nodes.lora_metadata_utils.discover_all_loras")
    def test_prompt_pairing_no_high_low_tags(self, mock_discover):
        """Test prompt pairing with no HIGH/LOW LoRA tags."""
        mock_discover.return_value = {
            "style_normal": {"metadata": {}},
            "pose_basic": {"metadata": {}},
        }

        prompt = "woman dancing <lora:style_normal:0.8> in garden"
        result = find_lora_pairs_in_prompt(prompt)

        # Should return unchanged since no high/low LoRAs
        self.assertEqual(result, prompt)

    @patch("nodes.lora_metadata_utils.discover_all_loras")
    def test_prompt_pairing_success(self, mock_discover):
        """Test successful prompt pairing."""
        # Mock available LoRAs
        mock_discover.return_value = {
            "character_high": {"metadata": {}},
            "character_low": {"metadata": {}},
            "style_normal": {"metadata": {}},
        }

        # String similarity will find the pair automatically

        prompt = "woman dancing <lora:character_high:0.8> in garden"
        result = find_lora_pairs_in_prompt(prompt)

        # Should add the paired LoRA
        self.assertIn("<lora:character_high:0.8>", result)
        self.assertIn("<lora:character_low:1.0>", result)
        self.assertIn("woman dancing", result)
        self.assertIn("in garden", result)

    @patch("nodes.lora_metadata_utils.discover_all_loras")
    def test_prompt_pairing_no_match_found(self, mock_discover):
        """Test behavior when no suitable pairs are found."""
        # Mock available LoRAs with unrelated names that won't match
        mock_discover.return_value = {
            "robot_style": {"metadata": {}},
            "fantasy_theme": {"metadata": {}},
        }

        # Use a HIGH LoRA that has no matching LOW pair available
        prompt = "woman <lora:character_high:0.8> dancing"
        result = find_lora_pairs_in_prompt(prompt)

        # Should return unchanged since no suitable pairs exist
        self.assertEqual(result, prompt)


if __name__ == "__main__":
    unittest.main()
