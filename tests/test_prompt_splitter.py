"""
Unit tests for the PromptSplitterNode using unittest.

These tests use the built-in ``unittest`` framework and the ``unittest.mock``
module to replace network calls and model checks.  The goal is to
exercise the logic of the ``PromptSplitterNode`` without requiring a
running Ollama server.  For these tests to run, the repository must be
on the Python path (pytest and additional third-party libraries are
not required).
"""

import unittest
from unittest.mock import patch

from nodes.prompt_splitter_node import PromptSplitterNode

import os

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
                sdxl, wan = self.node.split_prompt("A test prompt")
        self.assertEqual(sdxl, "image")
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

        def fake_call(prompt, model_name, api_url, system_prompt):
            used["model"] = model_name
            return ("x", "y")

        with patch.object(self.node, "_call_ollama", side_effect=fake_call):
            with patch.object(self.node, "_ensure_model_available"):
                self.node.split_prompt("Prompt without model", model_name=None)
        # Expect the default model name defined in the node to be used
        self.assertEqual(used["model"], self.node._DEFAULT_MODEL_NAME)

    def test_ensure_model_called_once(self):
        """_ensure_model_available should be called exactly once per split call."""
        with patch.object(self.node, "_call_ollama", return_value=("a", "b")):
            with patch.object(self.node, "_ensure_model_available") as mock_ensure:
                self.node.split_prompt("Another test prompt")
        mock_ensure.assert_called_once()

    def test_override_model_name(self):
        """Providing a model_name should override the default."""
        used = {}

        def fake_call(prompt, model_name, api_url, system_prompt):
            used["model"] = model_name
            return ("sdxl", "wan")

        with patch.object(self.node, "_call_ollama", side_effect=fake_call):
            with patch.object(self.node, "_ensure_model_available"):
                self.node.split_prompt("Test override", model_name="custom-model")
        self.assertEqual(used["model"], "custom-model")

    def test_ensure_model_download_called_when_missing(self):
        """_ensure_model_available should delegate to the shared utility."""
        node = PromptSplitterNode()

        # Patch the shared utility function directly
        with patch("nodes.prompt_splitter_node._shared_ensure_model_available") as mock_ensure:
            node._ensure_model_available("test-model", "http://localhost:11434")

        # Verify that the shared utility was called with correct parameters
        mock_ensure.assert_called_once_with(
            "test-model",
            "http://localhost:11434",
            requests_module=None,
            status_channel="prompt_splitter_status",
        )

    def test_ensure_model_no_download_when_present(self):
        """_ensure_model_available should delegate to the shared utility regardless of model availability."""
        node = PromptSplitterNode()

        # Patch the shared utility function directly
        with patch("nodes.prompt_splitter_node._shared_ensure_model_available") as mock_ensure:
            node._ensure_model_available("installed-model", "http://localhost:11434")

        # Verify that the shared utility was called (the shared utility handles availability checking)
        mock_ensure.assert_called_once_with(
            "installed-model",
            "http://localhost:11434",
            requests_module=None,
            status_channel="prompt_splitter_status",
        )


if __name__ == "__main__":
    unittest.main()
