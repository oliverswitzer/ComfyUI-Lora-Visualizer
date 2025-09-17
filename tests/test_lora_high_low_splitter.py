"""
Unit tests for the LoRAHighLowSplitterNode and shared splitting functionality.

Tests the new high/low LoRA splitting node and the enhanced shared utility function
that supports HN/LN naming patterns in addition to HIGH/LOW.
"""

import unittest
from nodes.lora_high_low_splitter_node import LoRAHighLowSplitterNode
from nodes.lora_metadata_utils import split_prompt_by_lora_high_low


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

    def test_split_mixed_case_patterns(self):
        """Test splitting with mixed case HIGH/LOW/HN/LN patterns."""
        prompt = "scene <lora:Style_HIGH:0.8> <lora:Motion_low:0.6> <lora:Effect_HN:0.7> <lora:Bg_ln:0.5>"

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # High prompt should have HIGH and HN tags
        self.assertIn("<lora:Style_HIGH:0.8>", high_prompt)
        self.assertIn("<lora:Effect_HN:0.7>", high_prompt)
        self.assertNotIn("<lora:Motion_low:0.6>", high_prompt)
        self.assertNotIn("<lora:Bg_ln:0.5>", high_prompt)

        # Low prompt should have low and ln tags
        self.assertIn("<lora:Motion_low:0.6>", low_prompt)
        self.assertIn("<lora:Bg_ln:0.5>", low_prompt)
        self.assertNotIn("<lora:Style_HIGH:0.8>", low_prompt)
        self.assertNotIn("<lora:Effect_HN:0.7>", low_prompt)

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

    def test_split_only_single_lora_tags(self):
        """Test splitting when all LoRA tags are single (no high/low patterns)."""
        prompt = "character <lora:style:0.8> <lora:lighting:0.6> <lora:pose:0.7>"

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # Both outputs should be identical
        self.assertEqual(high_prompt, low_prompt)
        self.assertIn("<lora:style:0.8>", high_prompt)
        self.assertIn("<lora:lighting:0.6>", high_prompt)
        self.assertIn("<lora:pose:0.7>", high_prompt)

    def test_shared_utility_function(self):
        """Test the shared utility function directly."""
        prompt = "test <lora:char_high:0.8> <lora:bg_low:0.6> scene"

        high_prompt, low_prompt = split_prompt_by_lora_high_low(prompt)

        self.assertIn("test", high_prompt)
        self.assertIn("scene", high_prompt)
        self.assertIn("<lora:char_high:0.8>", high_prompt)
        self.assertNotIn("<lora:bg_low:0.6>", high_prompt)

        self.assertIn("test", low_prompt)
        self.assertIn("scene", low_prompt)
        self.assertIn("<lora:bg_low:0.6>", low_prompt)
        self.assertNotIn("<lora:char_high:0.8>", low_prompt)

    def test_complex_wan_22_example(self):
        """Test with a realistic WAN 2.2 example using complex LoRA names."""
        prompt = (
            "futuristic cityscape <lora:Wan22-I2V-HIGH-Cyberpunk:0.7> "
            "with neon lights <lora:Wan22-I2V-LOW-Cyberpunk:0.7> "
            "and flying cars <lora:General-SciFi:0.5>"
        )

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # High prompt should have HIGH LoRA + general LoRA
        self.assertIn("futuristic cityscape", high_prompt)
        self.assertIn("with neon lights", high_prompt)
        self.assertIn("and flying cars", high_prompt)
        self.assertIn("<lora:Wan22-I2V-HIGH-Cyberpunk:0.7>", high_prompt)
        self.assertIn("<lora:General-SciFi:0.5>", high_prompt)
        self.assertNotIn("<lora:Wan22-I2V-LOW-Cyberpunk:0.7>", high_prompt)

        # Low prompt should have LOW LoRA + general LoRA
        self.assertIn("futuristic cityscape", low_prompt)
        self.assertIn("with neon lights", low_prompt)
        self.assertIn("and flying cars", low_prompt)
        self.assertIn("<lora:Wan22-I2V-LOW-Cyberpunk:0.7>", low_prompt)
        self.assertIn("<lora:General-SciFi:0.5>", low_prompt)
        self.assertNotIn("<lora:Wan22-I2V-HIGH-Cyberpunk:0.7>", low_prompt)

    def test_substring_matching_precision(self):
        """Test that substring matching works correctly and doesn't create false positives."""
        # Test case where LoRA names contain high/low as substrings but not as intended patterns
        prompt = (
            "character <lora:highlight_effect:0.8> "
            "with <lora:lowlight_shadows:0.6> "
            "and <lora:character_high:0.7> "
            "plus <lora:bg_low:0.5>"
        )

        high_prompt, low_prompt = self.node.split_high_low(prompt)

        # "highlight_effect" and "lowlight_shadows" should be treated as single LoRAs (in both)
        # Only "character_high" and "bg_low" should be split

        # High prompt should have character_high + single LoRAs
        self.assertIn("<lora:character_high:0.7>", high_prompt)
        self.assertIn("<lora:highlight_effect:0.8>", high_prompt)
        self.assertIn("<lora:lowlight_shadows:0.6>", high_prompt)
        self.assertNotIn("<lora:bg_low:0.5>", high_prompt)

        # Low prompt should have bg_low + single LoRAs
        self.assertIn("<lora:bg_low:0.5>", low_prompt)
        self.assertIn("<lora:highlight_effect:0.8>", low_prompt)
        self.assertIn("<lora:lowlight_shadows:0.6>", low_prompt)
        self.assertNotIn("<lora:character_high:0.7>", low_prompt)


if __name__ == "__main__":
    unittest.main()
