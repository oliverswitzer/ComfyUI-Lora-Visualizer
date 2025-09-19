"""
Unit tests for the PromptSplitterNode using unittest.

These tests use the built-in ``unittest`` framework and the ``unittest.mock``
module to replace network calls and model checks.  The goal is to
exercise the logic of the ``PromptSplitterNode`` without requiring a
running Ollama server.  For these tests to run, the repository must be
on the Python path (pytest and additional third-party libraries are
not required).
"""

import os
import unittest
from unittest.mock import patch

from nodes.prompt_splitter_node import PromptSplitterNode

os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")


class TestPromptSplitterNode(unittest.TestCase):
    """Unit tests for the prompt splitting node."""

    def setUp(self):
        self.node = PromptSplitterNode()

    def test_split_prompt_returns_ollama_response(self):
        """split_prompt should return whatever _call_ollama returns if non-empty."""
        with patch.object(self.node, "_call_ollama", return_value=("image", "video")) as mock_call:
            with patch.object(self.node, "_ensure_model_available") as mock_ensure:
                image, wan, wan_high, wan_low, analysis = self.node.split_prompt("A test prompt")
        self.assertEqual(image, "image")
        self.assertEqual(wan, "video")
        mock_call.assert_called_once()
        mock_ensure.assert_called_once()

    def test_split_prompt_raises_exception_when_ollama_empty(self):
        """When _call_ollama returns empty, split_prompt should raise an exception."""
        with patch.object(self.node, "_call_ollama", return_value=("", "")):
            with patch.object(self.node, "_ensure_model_available"):
                with self.assertRaises(Exception) as context:
                    self.node.split_prompt("A test prompt")
                self.assertIn("AI model returned empty response", str(context.exception))

    def test_default_model_is_used_when_none(self):
        """If model_name is None, the default model should be used."""
        used = {}

        def fake_call(prompt, model_name, api_url, system_prompt):  # pylint: disable=unused-argument
            used["model"] = model_name
            return ("x", "y")

        with patch.object(self.node, "_call_ollama", side_effect=fake_call):
            with patch.object(self.node, "_ensure_model_available"):
                _, _, _, _, _ = self.node.split_prompt("Prompt without model", model_name=None)
        # Expect the default model name defined in the node to be used
        self.assertEqual(used["model"], "nollama/mythomax-l2-13b:Q4_K_M")

    def test_ensure_model_called_once(self):
        """_ensure_model_available should be called exactly once per split call."""
        with patch.object(self.node, "_call_ollama", return_value=("a", "b")):
            with patch.object(self.node, "_ensure_model_available") as mock_ensure:
                _, _, _, _, _ = self.node.split_prompt("Another test prompt")
        mock_ensure.assert_called_once()

    def test_override_model_name(self):
        """Providing a model_name should override the default."""
        used = {}

        def fake_call(prompt, model_name, api_url, system_prompt):  # pylint: disable=unused-argument
            used["model"] = model_name
            return ("image", "wan")

        with patch.object(self.node, "_call_ollama", side_effect=fake_call):
            with patch.object(self.node, "_ensure_model_available"):
                _, _, _, _, _ = self.node.split_prompt("Test override", model_name="custom-model")
        self.assertEqual(used["model"], "custom-model")

    def test_ensure_model_download_called_when_missing(self):
        """_ensure_model_available should delegate to the shared utility."""
        node = PromptSplitterNode()

        # Patch the shared utility function directly
        with patch("nodes.prompt_splitter_node._shared_ensure_model_available") as mock_ensure:
            node._ensure_model_available("test-model", "http://localhost:11434")

        # Verify that the shared utility was called with correct parameters
        # Note: requests_module should be the actual requests module now that it's installed
        import requests

        mock_ensure.assert_called_once_with(
            "test-model",
            "http://localhost:11434",
            requests_module=requests,
            status_channel="prompt_splitter_status",
        )

    def test_ensure_model_no_download_when_present(self):
        """_ensure_model_available should delegate to shared utility regardless of availability."""
        node = PromptSplitterNode()

        # Patch the shared utility function directly
        with patch("nodes.prompt_splitter_node._shared_ensure_model_available") as mock_ensure:
            node._ensure_model_available("installed-model", "http://localhost:11434")

        # Verify that the shared utility was called (handles availability checking)
        # Note: requests_module should be the actual requests module now that it's installed
        import requests

        mock_ensure.assert_called_once_with(
            "installed-model",
            "http://localhost:11434",
            requests_module=requests,
            status_channel="prompt_splitter_status",
        )

    def test_parse_lora_tags_standard_loras(self):
        """parse_lora_tags should correctly extract standard LoRA tags."""
        prompt = "woman in dress <lora:beautiful:0.8> running <lora:style-anime:1.0>"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)

        self.assertEqual(len(standard_loras), 2)
        self.assertEqual(len(wanloras), 0)

        # Check first LoRA
        self.assertEqual(standard_loras[0]["name"], "beautiful")
        self.assertEqual(standard_loras[0]["strength"], "0.8")
        self.assertEqual(standard_loras[0]["type"], "lora")
        self.assertEqual(standard_loras[0]["tag"], "<lora:beautiful:0.8>")

        # Check second LoRA
        self.assertEqual(standard_loras[1]["name"], "style-anime")
        self.assertEqual(standard_loras[1]["strength"], "1.0")
        self.assertEqual(standard_loras[1]["type"], "lora")
        self.assertEqual(standard_loras[1]["tag"], "<lora:style-anime:1.0>")

    def test_parse_lora_tags_wanloras(self):
        """parse_lora_tags should correctly extract WanLoRA tags."""
        prompt = "dancing <wanlora:motion-blur:0.5> smoothly <wanlora:cinematic:1.2>"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)

        self.assertEqual(len(standard_loras), 0)
        self.assertEqual(len(wanloras), 2)

        # Check first WanLoRA
        self.assertEqual(wanloras[0]["name"], "motion-blur")
        self.assertEqual(wanloras[0]["strength"], "0.5")
        self.assertEqual(wanloras[0]["type"], "wanlora")
        self.assertEqual(wanloras[0]["tag"], "<wanlora:motion-blur:0.5>")

        # Check second WanLoRA
        self.assertEqual(wanloras[1]["name"], "cinematic")
        self.assertEqual(wanloras[1]["strength"], "1.2")
        self.assertEqual(wanloras[1]["type"], "wanlora")
        self.assertEqual(wanloras[1]["tag"], "<wanlora:cinematic:1.2>")

    def test_parse_lora_tags_mixed(self):
        """parse_lora_tags should correctly handle both standard and WanLoRA tags."""
        prompt = "woman <lora:style:0.8> dancing <wanlora:motion:1.0> in garden"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)

        self.assertEqual(len(standard_loras), 1)
        self.assertEqual(len(wanloras), 1)

        self.assertEqual(standard_loras[0]["name"], "style")
        self.assertEqual(wanloras[0]["name"], "motion")

    def test_parse_lora_tags_complex_names(self):
        """parse_lora_tags should handle LoRA names with spaces and special characters."""
        prompt = "test <lora:My Complex Name v2.1:0.75> and <wanlora:Special-Character_Name:1.0>"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)

        self.assertEqual(len(standard_loras), 1)
        self.assertEqual(len(wanloras), 1)

        self.assertEqual(standard_loras[0]["name"], "My Complex Name v2.1")
        self.assertEqual(standard_loras[0]["strength"], "0.75")

        self.assertEqual(wanloras[0]["name"], "Special-Character_Name")
        self.assertEqual(wanloras[0]["strength"], "1.0")

    def test_remove_all_lora_tags(self):
        """_remove_all_lora_tags should remove all LoRA and WanLoRA tags."""
        prompt = "woman <lora:style:0.8> dancing <wanlora:motion:1.0> in garden"
        clean_prompt = self.node._remove_all_lora_tags(prompt)

        self.assertEqual(clean_prompt, "woman dancing in garden")
        self.assertNotIn("<lora:", clean_prompt)
        self.assertNotIn("<wanlora:", clean_prompt)

    def test_split_prompt_no_lora_tags(self):
        """split_prompt should work normally when no LoRA tags are present."""
        input_prompt = "woman dancing gracefully"

        with patch.object(
            self.node, "_call_ollama", return_value=("woman dancing", "woman dances")
        ):
            with patch.object(self.node, "_ensure_model_available"):
                image_prompt, wan_prompt, wan_prompt_high, wan_prompt_low, analysis = (
                    self.node.split_prompt(input_prompt)
                )

        self.assertEqual(image_prompt, "woman dancing")
        self.assertEqual(wan_prompt, "woman dances")
        self.assertNotIn("<lora:", image_prompt)
        self.assertNotIn("<wanlora:", wan_prompt)

    def test_extract_and_remove_trigger_words(self):
        """_extract_and_remove_trigger_words should find and remove trigger words."""
        # Mock the metadata loader
        mock_metadata = {"civitai": {"trainedWords": ["beautiful", "style-anime"]}}

        with patch("nodes.prompt_splitter_node.get_metadata_loader") as mock_get_loader:
            mock_loader = mock_get_loader.return_value
            mock_loader.load_metadata.return_value = mock_metadata
            mock_loader.extract_trigger_words.return_value = [
                "beautiful",
                "style-anime",
            ]

            lora_list = [
                {
                    "name": "test-lora",
                    "strength": "1.0",
                    "type": "lora",
                    "tag": "<lora:test-lora:1.0>",
                }
            ]
            prompt = "beautiful woman with style-anime features"

            clean_prompt, trigger_words = self.node._extract_and_remove_trigger_words(
                prompt, lora_list
            )

            self.assertEqual(clean_prompt, "woman with features")
            self.assertIn("beautiful", trigger_words)
            self.assertIn("style-anime", trigger_words)

    def test_extract_verbatim_directives(self):
        """_extract_verbatim_directives should find and extract verbatim text."""
        prompt = "woman dancing (image: overwatch, ana) gracefully (video: she jumps up and down)"

        clean_prompt, image_verbatim, video_verbatim = self.node._extract_verbatim_directives(
            prompt
        )

        # Prompt should have wrapper syntax removed but content preserved
        expected_prompt = "woman dancing overwatch, ana gracefully she jumps up and down"
        self.assertEqual(clean_prompt, expected_prompt)
        self.assertEqual(len(image_verbatim), 1)
        self.assertEqual(len(video_verbatim), 1)
        self.assertEqual(image_verbatim[0], "overwatch, ana")
        self.assertEqual(video_verbatim[0], "she jumps up and down")

    def test_extract_verbatim_directives_multiple(self):
        """_extract_verbatim_directives should handle multiple directives of same type."""
        prompt = (
            "(image: character1) dancing (image: outfit: dress) and "
            "(video: motion1) then (video: motion2)"
        )

        clean_prompt, image_verbatim, video_verbatim = self.node._extract_verbatim_directives(
            prompt
        )

        # Prompt should have wrapper syntax removed but content preserved
        expected_prompt = "character1 dancing outfit: dress and motion1 then motion2"
        self.assertEqual(clean_prompt, expected_prompt)
        self.assertEqual(len(image_verbatim), 2)
        self.assertEqual(len(video_verbatim), 2)
        self.assertIn("character1", image_verbatim)
        self.assertIn("outfit: dress", image_verbatim)
        self.assertIn("motion1", video_verbatim)
        self.assertIn("motion2", video_verbatim)

    def test_extract_verbatim_directives_empty_content(self):
        """_extract_verbatim_directives should ignore empty directives."""
        prompt = "woman dancing (image: ) gracefully (video:   )"

        clean_prompt, image_verbatim, video_verbatim = self.node._extract_verbatim_directives(
            prompt
        )

        # Empty directives should be removed entirely
        expected_prompt = "woman dancing gracefully"
        self.assertEqual(clean_prompt, expected_prompt)
        self.assertEqual(len(image_verbatim), 0)
        self.assertEqual(len(video_verbatim), 0)

    def test_split_prompt_with_verbatim_directives(self):
        """split_prompt should handle verbatim directives correctly."""
        input_prompt = (
            "woman dancing (image: overwatch, ana) gracefully (video: she jumps up and down)"
        )

        # Mock _call_ollama to return clean responses (verbatim content added back separately)
        with patch.object(
            self.node,
            "_call_ollama",
            return_value=("woman dancing gracefully", "woman dances"),
        ):
            with patch.object(self.node, "_ensure_model_available"):
                image_prompt, wan_prompt, wan_prompt_high, wan_prompt_low, analysis = (
                    self.node.split_prompt(input_prompt)
                )

        # Should have verbatim content added back deterministically
        self.assertIn("overwatch, ana", image_prompt)
        self.assertIn("she jumps up and down", wan_prompt)

        # Should have base content from LLM
        self.assertIn("woman dancing gracefully", image_prompt)
        self.assertIn("woman dances", wan_prompt)

    def test_wanlora_tags_are_translated_to_lora_in_wan_prompt(self):
        """WanLoRA tags should be translated to <lora:...:...> in the wan_prompt output."""
        input_prompt = "subject <wanlora:motion:1.0> action"
        # Patch _call_ollama to return a base wan_prompt
        with patch.object(
            self.node,
            "_call_ollama",
            return_value=("image content", "video content"),
        ):
            with patch.object(self.node, "_ensure_model_available"):
                image_prompt, wan_prompt, wan_prompt_high, wan_prompt_low, analysis = (
                    self.node.split_prompt(input_prompt)
                )
        # The wan_prompt should contain <lora:motion:1.0> and NOT <wanlora:motion:1.0>
        self.assertIn("<lora:motion:1.0>", wan_prompt)
        self.assertNotIn("<wanlora:motion:1.0>", wan_prompt)
        # The image_prompt should not contain the wanlora tag
        self.assertNotIn("<wanlora:motion:1.0>", image_prompt)
        self.assertNotIn("<lora:motion:1.0>", image_prompt)

    def test_extract_lora_examples(self):
        """_extract_lora_examples should extract examples from LoRA metadata."""
        loras = [
            {"name": "test_lora", "tag": "<lora:test_lora:0.8>"},
            {"name": "nonexistent_lora", "tag": "<lora:nonexistent_lora:0.5>"},
        ]

        # Mock metadata with examples
        mock_metadata = {
            "civitai": {
                "images": [
                    {"meta": {"prompt": "beautiful woman, detailed face, professional lighting"}},
                    {"meta": {"prompt": "stunning portrait, high quality, cinematic"}},
                ]
            }
        }

        with patch("nodes.prompt_splitter_node.get_metadata_loader") as mock_loader:
            mock_loader_instance = mock_loader.return_value
            # Return metadata for test_lora, None for nonexistent_lora
            mock_loader_instance.load_metadata.side_effect = lambda name: (
                mock_metadata if name == "test_lora" else None
            )

            with patch("nodes.prompt_splitter_node.extract_example_prompts") as mock_extract:
                mock_extract.return_value = [
                    "beautiful woman, detailed face, professional lighting",
                    "stunning portrait, high quality, cinematic",
                ]

                examples, descriptions = self.node._extract_lora_examples(loras)

        # Should have examples for test_lora but not nonexistent_lora
        self.assertIn("test_lora", examples)
        self.assertNotIn("nonexistent_lora", examples)
        self.assertEqual(len(examples["test_lora"]), 2)

    def test_extract_lora_examples_with_descriptions(self):
        """_extract_lora_examples should extract both examples and descriptions."""
        loras = [{"name": "test_lora", "tag": "<lora:test_lora:0.8>"}]

        mock_metadata = {
            "modelDescription": "Test LoRA description",
            "civitai": {
                "model": {"description": "Detailed LoRA info"},
                "images": [{"meta": {"prompt": "test prompt"}}],
            },
        }

        with patch("nodes.prompt_splitter_node.get_metadata_loader") as mock_loader:
            mock_loader_instance = mock_loader.return_value
            mock_loader_instance.load_metadata.return_value = mock_metadata

            with patch("nodes.prompt_splitter_node.extract_example_prompts") as mock_extract:
                mock_extract.return_value = ["test prompt"]

                examples, descriptions = self.node._extract_lora_examples(loras)

        self.assertIn("test_lora", examples)
        self.assertIn("test_lora", descriptions)
        self.assertIn("Test LoRA description", descriptions["test_lora"])

    def test_parse_plain_text_response(self):
        """_parse_plain_text_response should parse non-JSON LLM responses."""
        # Test case 1: Normal plain text format
        content1 = """IMAGE_PROMPT: woman, 4k, detailed, masterpiece

WAN_PROMPT: woman dancing, fluid movement, graceful"""

        image_prompt, wan_prompt = self.node._parse_plain_text_response(content1)

        self.assertEqual(image_prompt, "woman, 4k, detailed, masterpiece")
        self.assertEqual(wan_prompt, "woman dancing, fluid movement, graceful")

        # Test case 2: Multiline content
        content2 = """IMAGE_PROMPT: woman, detailed face,
professional lighting, studio quality

WAN_PROMPT: The woman moves gracefully
through the scene with fluid motion"""

        image_prompt2, wan_prompt2 = self.node._parse_plain_text_response(content2)

        self.assertEqual(
            image_prompt2, "woman, detailed face, professional lighting, studio quality"
        )
        self.assertEqual(
            wan_prompt2,
            "The woman moves gracefully through the scene with fluid motion",
        )

        # Test case 3: Empty or missing sections
        content3 = "IMAGE_PROMPT: just an image prompt"

        image_prompt3, wan_prompt3 = self.node._parse_plain_text_response(content3)

        self.assertEqual(image_prompt3, "just an image prompt")
        self.assertEqual(wan_prompt3, "")

    def test_create_contextualized_system_prompt(self):
        """_create_contextualized_system_prompt should enhance base prompt with LoRA examples."""
        lora_examples = {
            "StyleLoRA": [
                "artistic woman, oil painting style",
                "vintage portrait, sepia tones",
            ],
            "MotionLoRA": [
                "dancing figure, fluid movement",
                "running athlete, dynamic pose",
            ],
        }
        lora_descriptions = {
            "StyleLoRA": "A LoRA for artistic oil painting styles and vintage effects",
            "MotionLoRA": "Specialized LoRA for dynamic movement and action poses",
        }

        enhanced_prompt = self.node._create_contextualized_system_prompt(
            lora_examples, lora_descriptions
        )

        # Should contain base prompt content
        self.assertIn("IMAGE_PROMPT", enhanced_prompt)
        self.assertIn("WAN_PROMPT", enhanced_prompt)

        # Should contain LoRA context sections
        self.assertIn("LoRA CONTEXT", enhanced_prompt)
        self.assertIn("StyleLoRA", enhanced_prompt)
        self.assertIn("MotionLoRA", enhanced_prompt)
        self.assertIn("artistic woman, oil painting style", enhanced_prompt)
        self.assertIn("dancing figure, fluid movement", enhanced_prompt)
        self.assertIn("artistic oil painting styles", enhanced_prompt)  # From description

    def test_create_contextualized_system_prompt_no_examples(self):
        """_create_contextualized_system_prompt should return base prompt when no examples."""
        enhanced_prompt = self.node._create_contextualized_system_prompt({}, {})

        # Should be identical to base prompt
        self.assertEqual(enhanced_prompt, self.node._SYSTEM_PROMPT)

    def test_lora_analysis_output_contains_examples(self):
        """split_prompt should return analysis output with LoRA examples."""
        input_prompt = (
            "woman dancing <lora:style:0.8> beautiful face <wanlora:motion:1.0> graceful movement"
        )

        # Mock metadata loader and examples
        with patch("nodes.prompt_splitter_node.get_metadata_loader") as mock_loader:
            mock_loader_instance = mock_loader.return_value
            mock_loader_instance.load_metadata.return_value = {"some": "metadata"}
            mock_loader_instance.extract_trigger_words.return_value = [
                "trigger1",
                "trigger2",
            ]

            with patch("nodes.prompt_splitter_node.extract_example_prompts") as mock_extract:
                mock_extract.return_value = [
                    "example prompt 1 for this lora",
                    "example prompt 2 showing usage",
                ]

                with patch.object(
                    self.node,
                    "_call_ollama",
                    return_value=("woman dancing gracefully", "woman dances"),
                ):
                    with patch.object(self.node, "_ensure_model_available"):
                        image_prompt, wan_prompt, wan_prompt_high, wan_prompt_low, analysis = (
                            self.node.split_prompt(input_prompt)
                        )

        # Parse the JSON analysis
        import json

        analysis_data = json.loads(analysis)

        # Should contain processing success
        self.assertTrue(analysis_data["processing_successful"])

        # Should contain LoRA examples
        self.assertIn("loras_used", analysis_data)
        self.assertIn("image_loras", analysis_data["loras_used"])
        self.assertIn("video_loras", analysis_data["loras_used"])

        # Check that examples are included
        image_loras = analysis_data["loras_used"]["image_loras"]
        if image_loras:
            self.assertIn("examples_fed_to_llm", image_loras[0])
            self.assertIn("trigger_words", image_loras[0])

        video_loras = analysis_data["loras_used"]["video_loras"]
        if video_loras:
            self.assertIn("examples_fed_to_llm", video_loras[0])
            self.assertIn("trigger_words", video_loras[0])

        # Should contain metadata about processing
        self.assertIn("total_examples_used", analysis_data)
        self.assertIn("model_used", analysis_data)

    def test_split_wan_prompt_by_high_low_with_mixed_tags(self):
        """_split_wan_prompt_by_high_low should separate HIGH/LOW lora tags using Ollama classification."""
        # Test prompt with HIGH/LOW pair and unrelated single LoRA
        # Note: wanlora tags have been converted to lora tags at this point in the real flow
        wan_prompt = "dancing robot in the city <lora:character_high:0.8> <lora:character_low:0.6> <lora:unrelated_lora:0.5>"

        # Mock Ollama classification
        mock_classifications = {
            ("character_high", "character_low"): {
                "high_lora": "character_high",
                "low_lora": "character_low",
                "reasoning": "high vs low naming pattern",
            }
        }

        with patch(
            "nodes.lora_metadata_utils.classify_lora_pairs_with_ollama",
            return_value=mock_classifications,
        ):
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(wan_prompt)

        # HIGH prompt should have base + HIGH lora tag from the pair
        self.assertIn("dancing robot in the city", wan_high)
        self.assertIn("<lora:character_high:0.8>", wan_high)
        self.assertNotIn("<lora:character_low:0.6>", wan_high)

        # LOW prompt should have base + LOW lora tag from the pair
        self.assertIn("dancing robot in the city", wan_low)
        self.assertIn("<lora:character_low:0.6>", wan_low)
        self.assertNotIn("<lora:character_high:0.8>", wan_low)

        # Both should contain the unrelated single LoRA (no good fuzzy match found)
        self.assertIn("<lora:unrelated_lora:0.5>", wan_high)
        self.assertIn("<lora:unrelated_lora:0.5>", wan_low)

    def test_split_wan_prompt_by_high_low_with_dash_patterns(self):
        """_split_wan_prompt_by_high_low should handle dash-separated HIGH/LOW lora patterns."""
        # Note: wanlora tags have been converted to lora tags at this point in the real flow
        wan_prompt = (
            "futuristic scene <lora:Wan22-I2V-HIGH-Robot:0.7> <lora:Wan22-I2V-LOW-Robot:0.7>"
        )

        # Mock Ollama classification
        mock_classifications = {
            ("Wan22-I2V-HIGH-Robot", "Wan22-I2V-LOW-Robot"): {
                "high_lora": "Wan22-I2V-HIGH-Robot",
                "low_lora": "Wan22-I2V-LOW-Robot",
                "reasoning": "HIGH vs LOW indicator in name",
            }
        }

        with patch(
            "nodes.lora_metadata_utils.classify_lora_pairs_with_ollama",
            return_value=mock_classifications,
        ):
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(wan_prompt)

        # HIGH prompt should contain only HIGH lora tag
        self.assertIn("<lora:Wan22-I2V-HIGH-Robot:0.7>", wan_high)
        self.assertNotIn("<lora:Wan22-I2V-LOW-Robot:0.7>", wan_high)

        # LOW prompt should contain only LOW lora tag
        self.assertIn("<lora:Wan22-I2V-LOW-Robot:0.7>", wan_low)
        self.assertNotIn("<lora:Wan22-I2V-HIGH-Robot:0.7>", wan_low)

        # Both should contain base prompt
        self.assertIn("futuristic scene", wan_high)
        self.assertIn("futuristic scene", wan_low)

    def test_split_wan_prompt_by_high_low_with_single_letters(self):
        """_split_wan_prompt_by_high_low should handle single letter H/L patterns via Ollama classification."""
        # Test single letter patterns using Ollama classification
        wan_prompt = "test scene <lora:Model-22-H-e8:1.0> <lora:Model-22-L-e8:0.5>"

        # Mock Ollama classification
        mock_classifications = {
            ("Model-22-H-e8", "Model-22-L-e8"): {
                "high_lora": "Model-22-H-e8",
                "low_lora": "Model-22-L-e8",
                "reasoning": "H vs L letter indicator",
            }
        }

        with patch(
            "nodes.lora_metadata_utils.classify_lora_pairs_with_ollama",
            return_value=mock_classifications,
        ):
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(wan_prompt)

        # HIGH prompt should contain only HIGH lora tag
        self.assertIn("<lora:Model-22-H-e8:1.0>", wan_high)
        self.assertNotIn("<lora:Model-22-L-e8:0.5>", wan_high)

        # LOW prompt should contain only LOW lora tag
        self.assertIn("<lora:Model-22-L-e8:0.5>", wan_low)
        self.assertNotIn("<lora:Model-22-H-e8:1.0>", wan_low)

        # Both should contain base prompt
        self.assertIn("test scene", wan_high)
        self.assertIn("test scene", wan_low)

    def test_split_wan_prompt_by_high_low_with_no_pairs(self):
        """_split_wan_prompt_by_high_low should handle prompts with no HIGH/LOW lora tags."""
        # Note: wanlora tags have been converted to lora tags at this point in the real flow
        wan_prompt = "simple scene <lora:style1:0.8> <lora:character:0.6>"

        # Mock empty Ollama classification (no pairs found)
        with patch("nodes.lora_metadata_utils.classify_lora_pairs_with_ollama", return_value={}):
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(wan_prompt)

        # Both outputs should be identical (base + all single lora tags)
        self.assertEqual(wan_high, wan_low)
        self.assertIn("simple scene", wan_high)
        self.assertIn("<lora:style1:0.8>", wan_high)  # Regular lora tag preserved
        self.assertIn("<lora:character:0.6>", wan_high)  # Single lora tag in both

    def test_split_wan_prompt_by_high_low_with_ollama_failure_fallback(self):
        """_split_wan_prompt_by_high_low should treat LoRAs as singles when Ollama fails."""
        wan_prompt = "test scene <lora:character_high:0.8> <lora:character_low:0.6>"

        # Mock Ollama returning empty classifications (failure case)
        with patch("nodes.lora_metadata_utils.classify_lora_pairs_with_ollama", return_value={}):
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(wan_prompt)

        # When Ollama fails, both LoRAs should be treated as single LoRAs (included in both outputs)
        self.assertIn("<lora:character_high:0.8>", wan_high)
        self.assertIn("<lora:character_low:0.6>", wan_high)

        # Both LoRAs should also be in LOW prompt
        self.assertIn("<lora:character_high:0.8>", wan_low)
        self.assertIn("<lora:character_low:0.6>", wan_low)

    def test_high_low_split_integration_with_mocked_shared_function(self):
        """Test HIGH/LOW splitting integration with mocked shared function."""
        wan_prompt = "test scene <lora:Model-22-H-e8:1> <lora:Model-22-L-e8:0.5>"

        # Mock the shared function to return expected results
        mock_high = "test scene <lora:Model-22-H-e8:1>"
        mock_low = "test scene <lora:Model-22-L-e8:0.5>"

        with patch(
            "nodes.lora_metadata_utils.split_prompt_by_lora_high_low_with_ollama",
            return_value=(mock_high, mock_low),
        ) as mock_split:
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(wan_prompt)

        # Verify the node correctly delegates to shared function with advanced mode (always True for PromptSplitterNode)
        mock_split.assert_called_once_with(wan_prompt, use_ollama=True)

        # Verify the node correctly returns results from shared function
        self.assertEqual(wan_high, mock_high)
        self.assertEqual(wan_low, mock_low)

    def test_high_low_split_with_advanced_mode_enabled(self):
        """Test HIGH/LOW splitting with advanced matching mode enabled."""
        wan_prompt = "test scene <lora:character_high:0.8>"

        # Mock the shared function to simulate finding missing LOW pair
        mock_high = "test scene <lora:character_high:0.8>"
        mock_low = "test scene <lora:character_low:0.6>"

        with patch(
            "nodes.lora_metadata_utils.split_prompt_by_lora_high_low_with_ollama",
            return_value=(mock_high, mock_low),
        ) as mock_split:
            wan_high, wan_low = self.node._split_wan_prompt_by_high_low(
                wan_prompt, use_advanced_matching=True
            )

        # Verify the node calls shared function with Ollama enabled
        mock_split.assert_called_once_with(wan_prompt, use_ollama=True)

        # Verify the results
        self.assertEqual(wan_high, mock_high)
        self.assertEqual(wan_low, mock_low)


if __name__ == "__main__":
    unittest.main()
