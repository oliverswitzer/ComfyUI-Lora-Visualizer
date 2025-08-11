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
        with patch.object(
            self.node, "_call_ollama", return_value=("image", "video")
        ) as mock_call:
            with patch.object(self.node, "_ensure_model_available") as mock_ensure:
                image, wan = self.node.split_prompt("A test prompt")
        self.assertEqual(image, "image")
        self.assertEqual(wan, "video")
        mock_call.assert_called_once()
        mock_ensure.assert_called_once()

    def test_split_prompt_returns_empty_when_ollama_empty(self):
        """When _call_ollama returns empty, split_prompt should return empty strings."""
        with patch.object(self.node, "_call_ollama", return_value=("", "")):
            with patch.object(self.node, "_ensure_model_available"):
                image, wan = self.node.split_prompt("A test prompt")
        self.assertEqual(image, "")
        self.assertEqual(wan, "")

    def test_default_model_is_used_when_none(self):
        """If model_name is None, the default model should be used."""
        used = {}

        def fake_call(
            prompt, model_name, api_url, system_prompt
        ):  # pylint: disable=unused-argument
            used["model"] = model_name
            return ("x", "y")

        with patch.object(self.node, "_call_ollama", side_effect=fake_call):
            with patch.object(self.node, "_ensure_model_available"):
                self.node.split_prompt("Prompt without model", model_name=None)
        # Expect the default model name defined in the node to be used
        self.assertEqual(used["model"], "nollama/mythomax-l2-13b:Q4_K_M")

    def test_ensure_model_called_once(self):
        """_ensure_model_available should be called exactly once per split call."""
        with patch.object(self.node, "_call_ollama", return_value=("a", "b")):
            with patch.object(self.node, "_ensure_model_available") as mock_ensure:
                self.node.split_prompt("Another test prompt")
        mock_ensure.assert_called_once()

    def test_override_model_name(self):
        """Providing a model_name should override the default."""
        used = {}

        def fake_call(
            prompt, model_name, api_url, system_prompt
        ):  # pylint: disable=unused-argument
            used["model"] = model_name
            return ("image", "wan")

        with patch.object(self.node, "_call_ollama", side_effect=fake_call):
            with patch.object(self.node, "_ensure_model_available"):
                self.node.split_prompt("Test override", model_name="custom-model")
        self.assertEqual(used["model"], "custom-model")

    def test_ensure_model_download_called_when_missing(self):
        """_ensure_model_available should delegate to the shared utility."""
        node = PromptSplitterNode()

        # Patch the shared utility function directly
        with patch(
            "nodes.prompt_splitter_node._shared_ensure_model_available"
        ) as mock_ensure:
            node._ensure_model_available("test-model", "http://localhost:11434")

        # Verify that the shared utility was called with correct parameters
        mock_ensure.assert_called_once_with(
            "test-model",
            "http://localhost:11434",
            requests_module=None,
            status_channel="prompt_splitter_status",
        )

    def test_ensure_model_no_download_when_present(self):
        """_ensure_model_available should delegate to shared utility regardless of availability."""
        node = PromptSplitterNode()

        # Patch the shared utility function directly
        with patch(
            "nodes.prompt_splitter_node._shared_ensure_model_available"
        ) as mock_ensure:
            node._ensure_model_available("installed-model", "http://localhost:11434")

        # Verify that the shared utility was called (handles availability checking)
        mock_ensure.assert_called_once_with(
            "installed-model",
            "http://localhost:11434",
            requests_module=None,
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

    def test_split_prompt_with_lora_tags(self):
        """split_prompt should parse LoRA tags and add them to appropriate outputs."""
        input_prompt = "woman dancing <lora:style:0.8> gracefully <wanlora:motion:1.0>"

        # Mock the _call_ollama to return clean responses
        with patch.object(
            self.node,
            "_call_ollama",
            return_value=("woman dancing gracefully", "woman dances"),
        ):
            with patch.object(self.node, "_ensure_model_available"):
                image_prompt, wan_prompt = self.node.split_prompt(input_prompt)

        # LoRA should be added to image prompt
        self.assertIn("<lora:style:0.8>", image_prompt)
        self.assertNotIn("<wanlora:motion:1.0>", image_prompt)

        # WanLoRA should be added to video prompt
        self.assertIn("<wanlora:motion:1.0>", wan_prompt)
        self.assertNotIn("<lora:style:0.8>", wan_prompt)

        # Base content should be preserved
        self.assertIn("woman dancing gracefully", image_prompt)
        self.assertIn("woman dances", wan_prompt)

    def test_split_prompt_no_lora_tags(self):
        """split_prompt should work normally when no LoRA tags are present."""
        input_prompt = "woman dancing gracefully"

        with patch.object(
            self.node, "_call_ollama", return_value=("woman dancing", "woman dances")
        ):
            with patch.object(self.node, "_ensure_model_available"):
                image_prompt, wan_prompt = self.node.split_prompt(input_prompt)

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

    def test_split_prompt_with_trigger_words(self):
        """split_prompt should handle trigger words correctly."""
        input_prompt = "beautiful woman <lora:style:0.8> dancing with motion-blur <wanlora:motion:1.0>"

        # Mock metadata for LoRAs
        mock_image_metadata = {"civitai": {"trainedWords": ["beautiful"]}}
        mock_video_metadata = {"civitai": {"trainedWords": ["motion-blur"]}}

        with patch("nodes.prompt_splitter_node.get_metadata_loader") as mock_get_loader:
            mock_loader = mock_get_loader.return_value

            def mock_load_metadata(lora_name):
                if lora_name == "style":
                    return mock_image_metadata
                elif lora_name == "motion":
                    return mock_video_metadata
                return None

            def mock_extract_trigger_words(metadata):
                if metadata == mock_image_metadata:
                    return ["beautiful"]
                elif metadata == mock_video_metadata:
                    return ["motion-blur"]
                return []

            mock_loader.load_metadata.side_effect = mock_load_metadata
            mock_loader.extract_trigger_words.side_effect = mock_extract_trigger_words

            # Mock _call_ollama to return clean responses
            with patch.object(
                self.node,
                "_call_ollama",
                return_value=("woman dancing", "woman dances"),
            ):
                with patch.object(self.node, "_ensure_model_available"):
                    image_prompt, wan_prompt = self.node.split_prompt(input_prompt)

            # Should have LoRA tag and trigger word
            self.assertIn("<lora:style:0.8>", image_prompt)
            self.assertIn("beautiful", image_prompt)

            # Should have WanLoRA tag and trigger word
            self.assertIn("<wanlora:motion:1.0>", wan_prompt)
            self.assertIn("motion-blur", wan_prompt)

            # Should not have wrong type tags
            self.assertNotIn("<wanlora:", image_prompt)
            self.assertNotIn("<lora:", wan_prompt)

    def test_extract_verbatim_directives(self):
        """_extract_verbatim_directives should find and extract verbatim text."""
        prompt = "woman dancing (image: overwatch, ana) gracefully (video: she jumps up and down)"

        clean_prompt, image_verbatim, video_verbatim = (
            self.node._extract_verbatim_directives(prompt)
        )

        # Prompt should have wrapper syntax removed but content preserved
        expected_prompt = (
            "woman dancing overwatch, ana gracefully she jumps up and down"
        )
        self.assertEqual(clean_prompt, expected_prompt)
        self.assertEqual(len(image_verbatim), 1)
        self.assertEqual(len(video_verbatim), 1)
        self.assertEqual(image_verbatim[0], "overwatch, ana")
        self.assertEqual(video_verbatim[0], "she jumps up and down")

    def test_extract_verbatim_directives_multiple(self):
        """_extract_verbatim_directives should handle multiple directives of same type."""
        prompt = "(image: character1) dancing (image: outfit: dress) and (video: motion1) then (video: motion2)"

        clean_prompt, image_verbatim, video_verbatim = (
            self.node._extract_verbatim_directives(prompt)
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

        clean_prompt, image_verbatim, video_verbatim = (
            self.node._extract_verbatim_directives(prompt)
        )

        # Empty directives should be removed entirely
        expected_prompt = "woman dancing gracefully"
        self.assertEqual(clean_prompt, expected_prompt)
        self.assertEqual(len(image_verbatim), 0)
        self.assertEqual(len(video_verbatim), 0)

    def test_split_prompt_with_verbatim_directives(self):
        """split_prompt should handle verbatim directives correctly."""
        input_prompt = "woman dancing (image: overwatch, ana) gracefully (video: she jumps up and down)"

        # Mock _call_ollama to return responses with verbatim content included by LLM
        with patch.object(
            self.node,
            "_call_ollama",
            return_value=(
                "woman dancing gracefully, overwatch, ana",
                "woman dances, she jumps up and down",
            ),
        ):
            with patch.object(self.node, "_ensure_model_available"):
                image_prompt, wan_prompt = self.node.split_prompt(input_prompt)

        # Should have verbatim content included by LLM
        self.assertIn("overwatch, ana", image_prompt)
        self.assertIn("she jumps up and down", wan_prompt)

        # Should have base content from LLM
        self.assertIn("woman dancing gracefully", image_prompt)
        self.assertIn("woman dances", wan_prompt)

    def test_split_prompt_with_loras_and_verbatim(self):
        """split_prompt should handle both LoRAs and verbatim directives together."""
        input_prompt = "(image: overwatch, ana) woman <lora:style:0.8> dancing (video: she jumps) <wanlora:motion:1.0>"

        # Mock metadata loader for LoRAs
        with patch("nodes.prompt_splitter_node.get_metadata_loader") as mock_get_loader:
            mock_loader = mock_get_loader.return_value
            mock_loader.load_metadata.return_value = None
            mock_loader.extract_trigger_words.return_value = []

            # Mock _call_ollama to return responses with verbatim content included by LLM
            with patch.object(
                self.node,
                "_call_ollama",
                return_value=(
                    "woman dancing, overwatch, ana",
                    "woman dances, she jumps",
                ),
            ):
                with patch.object(self.node, "_ensure_model_available"):
                    image_prompt, wan_prompt = self.node.split_prompt(input_prompt)

        # Should have verbatim content included by LLM
        self.assertIn("overwatch, ana", image_prompt)
        self.assertIn("she jumps", wan_prompt)

        # Should have LoRA tags
        self.assertIn("<lora:style:0.8>", image_prompt)
        self.assertIn("<wanlora:motion:1.0>", wan_prompt)

        # Should have base content from LLM
        self.assertIn("woman dancing", image_prompt)
        self.assertIn("woman dances", wan_prompt)


if __name__ == "__main__":
    unittest.main()
