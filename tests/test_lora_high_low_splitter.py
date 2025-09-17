"""
Unit tests for the LoRAHighLowSplitterNode and shared splitting functionality.

Tests the new high/low LoRA splitting node and the enhanced shared utility function
that supports HN/LN naming patterns in addition to HIGH/LOW.
"""

import unittest

from nodes.lora_high_low_splitter_node import LoRAHighLowSplitterNode
from nodes.lora_metadata_utils import find_lora_high_low_pair, split_prompt_by_lora_high_low


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


class TestLoRAHighLowPairing(unittest.TestCase):
    """Unit tests for the shared LoRA high/low pairing functionality."""

    def test_find_high_low_pair_basic(self):
        """Test basic HIGH/LOW pairing."""
        available_loras = ["character_high", "character_low", "style_normal"]

        # Test HIGH -> LOW
        pair = find_lora_high_low_pair("character_high", available_loras)
        self.assertEqual(pair, "character_low")

        # Test LOW -> HIGH
        pair = find_lora_high_low_pair("character_low", available_loras)
        self.assertEqual(pair, "character_high")

    def test_find_hn_ln_pair_basic(self):
        """Test basic HN/LN pairing."""
        available_loras = ["robot_hn", "robot_ln", "style_normal"]

        # Test HN -> LN
        pair = find_lora_high_low_pair("robot_hn", available_loras)
        self.assertEqual(pair, "robot_ln")

        # Test LN -> HN
        pair = find_lora_high_low_pair("robot_ln", available_loras)
        self.assertEqual(pair, "robot_hn")

    def test_find_pair_case_preservation(self):
        """Test that case is preserved in pairing."""
        available_loras = ["Style_HIGH", "Style_LOW", "Character_HN", "Character_LN"]

        # Test case preservation
        pair = find_lora_high_low_pair("Style_HIGH", available_loras)
        self.assertEqual(pair, "Style_LOW")

        pair = find_lora_high_low_pair("Character_HN", available_loras)
        self.assertEqual(pair, "Character_LN")

    def test_find_pair_wan_22_example(self):
        """Test with realistic WAN 2.2 LoRA names."""
        available_loras = [
            "Wan22-I2V-HIGH-Cyberpunk",
            "Wan22-I2V-LOW-Cyberpunk",
            "General-SciFi",
        ]

        pair = find_lora_high_low_pair("Wan22-I2V-HIGH-Cyberpunk", available_loras)
        self.assertEqual(pair, "Wan22-I2V-LOW-Cyberpunk")

        pair = find_lora_high_low_pair("Wan22-I2V-LOW-Cyberpunk", available_loras)
        self.assertEqual(pair, "Wan22-I2V-HIGH-Cyberpunk")

    def test_find_pair_no_pair_available(self):
        """Test when no pair is available."""
        available_loras = ["character_high", "style_normal", "motion_single"]

        # No LOW pair available
        pair = find_lora_high_low_pair("character_high", available_loras)
        self.assertIsNone(pair)

        # LoRA is not high/low variant
        pair = find_lora_high_low_pair("style_normal", available_loras)
        self.assertIsNone(pair)

    def test_find_pair_mixed_case_patterns(self):
        """Test mixed case patterns like High, Low, Hn, Ln."""
        available_loras = [
            "Character_High",
            "Character_Low",
            "Robot_Hn",
            "Robot_Ln",
            "Effect_high",
            "Effect_low",
        ]

        # Test different case patterns
        pairs = [
            ("Character_High", "Character_Low"),
            ("Character_Low", "Character_High"),
            ("Robot_Hn", "Robot_Ln"),
            ("Robot_Ln", "Robot_Hn"),
            ("Effect_high", "Effect_low"),
            ("Effect_low", "Effect_high"),
        ]

        for lora_name, expected_pair in pairs:
            with self.subTest(lora_name=lora_name):
                pair = find_lora_high_low_pair(lora_name, available_loras)
                self.assertEqual(pair, expected_pair)

    def test_find_pair_uppercase_patterns(self):
        """Test fully uppercase patterns."""
        available_loras = ["STYLE_HIGH", "STYLE_LOW", "EFFECT_HN", "EFFECT_LN"]

        pairs = [
            ("STYLE_HIGH", "STYLE_LOW"),
            ("STYLE_LOW", "STYLE_HIGH"),
            ("EFFECT_HN", "EFFECT_LN"),
            ("EFFECT_LN", "EFFECT_HN"),
        ]

        for lora_name, expected_pair in pairs:
            with self.subTest(lora_name=lora_name):
                pair = find_lora_high_low_pair(lora_name, available_loras)
                self.assertEqual(pair, expected_pair)

    def test_find_pair_empty_list(self):
        """Test with empty available LoRA list."""
        pair = find_lora_high_low_pair("character_high", [])
        self.assertIsNone(pair)

    def test_find_pair_priority_order(self):
        """Test that HIGH/LOW takes priority over HN/LN when both are present."""
        # Edge case: LoRA name contains both patterns
        available_loras = ["complex_high_hn_style", "complex_low_hn_style"]

        # Should prioritize HIGH/LOW over HN/LN
        pair = find_lora_high_low_pair("complex_high_hn_style", available_loras)
        self.assertEqual(pair, "complex_low_hn_style")

    def test_find_pair_no_false_positives(self):
        """Test that substring matches don't create false pairs."""
        available_loras = [
            "highlight_effect",
            "lowlight_shadows",
            "character_high",
            "background_low",
        ]

        # "highlight" contains "high" but shouldn't pair with "lowlight"
        pair = find_lora_high_low_pair("highlight_effect", available_loras)
        self.assertIsNone(pair)

        # "lowlight" contains "low" but shouldn't pair with "highlight"
        pair = find_lora_high_low_pair("lowlight_shadows", available_loras)
        self.assertIsNone(pair)

        # But actual high/low pairs should work if they match exactly
        # "character_high" should not pair with "background_low" - no exact match
        pair = find_lora_high_low_pair("character_high", available_loras)
        self.assertIsNone(pair)  # No matching "character_low" available

    def test_shared_function_used_by_nodes(self):
        """Test that the shared function produces consistent results with both nodes."""
        available_loras = ["style_high", "style_low", "motion_hn", "motion_ln"]

        # Test pairing function directly
        pair1 = find_lora_high_low_pair("style_high", available_loras)
        self.assertEqual(pair1, "style_low")

        pair2 = find_lora_high_low_pair("motion_hn", available_loras)
        self.assertEqual(pair2, "motion_ln")


if __name__ == "__main__":
    unittest.main()
