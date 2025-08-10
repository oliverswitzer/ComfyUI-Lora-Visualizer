"""Tests for the LoRA Prompt Composer node."""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Patch ComfyUI dependencies before importing the node.  The test
# harness injects mocks for folder_paths and server so that the node
# can be instantiated without requiring a full ComfyUI runtime.
import sys
import os

os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")
sys.modules.setdefault("folder_paths", MagicMock())
sys.modules.setdefault("server", MagicMock())

# Add the parent directory so that the node can be imported
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from nodes.lora_prompt_composer_node import LoRAPromptComposerNode


class TestLoRAPromptComposerNode(unittest.TestCase):
    """Unit tests for the LoRAPromptComposerNode."""

    def setUp(self):
        # Determine the path to the fixtures folder
        self.fixtures_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures"
        )
        # Patch folder_paths.get_folder_paths to return our fixtures directory
        self.folder_patcher = patch(
            "folder_paths.get_folder_paths", return_value=[self.fixtures_path]
        )
        self.folder_patcher.start()
        # Patch PromptServer to avoid any frontend interactions
        self.prompt_patcher = patch("server.PromptServer")
        self.prompt_patcher.start()
        # Instantiate the node
        self.node = LoRAPromptComposerNode()

    def tearDown(self):
        self.folder_patcher.stop()
        self.prompt_patcher.stop()

    def test_compose_prompt_returns_llm_output(self):
        """Ensure that compose_prompt returns the text from _call_ollama."""
        expected_prompt = "A combined prompt"
        # Patch _ensure_model_available to no-op
        with patch.object(self.node, "_ensure_model_available", return_value=None):
            # Patch _call_ollama to return our expected prompt
            with patch.object(
                self.node, "_call_ollama", return_value=expected_prompt
            ) as call_mock:
                result = self.node.compose_prompt(num_wan_loras=1, num_image_loras=1)
                # The result is a tuple with one element
                self.assertEqual(result, (expected_prompt,))
                # Ensure _call_ollama was called once
                call_mock.assert_called_once()

    def test_compose_prompt_uses_default_model(self):
        """Verify that the default model name is used when none is supplied."""
        # Capture the model used by _ensure_model_available
        captured = {}

        def fake_ensure(model, api_url):
            captured["model"] = model
            return None

        with patch.object(
            self.node, "_ensure_model_available", side_effect=fake_ensure
        ):
            # Patch _call_ollama to avoid network
            with patch.object(self.node, "_call_ollama", return_value="prompt"):
                # Invoke without model_name override
                _ = self.node.compose_prompt(
                    num_wan_loras=1, num_image_loras=1, model_name=None
                )
                # Ensure the captured model equals the node's default
                self.assertEqual(captured.get("model"), self.node._DEFAULT_MODEL_NAME)

    def test_compose_prompt_message_contains_loras(self):
        """Ensure the message sent to Ollama contains the correct LoRA lists."""
        captured = {}

        def fake_call(user_message, model_name, api_url, system_prompt):
            # Save the user_message for inspection and return a dummy prompt
            captured["user_message"] = user_message
            return "combined prompt"

        # Skip model availability checks
        with patch.object(self.node, "_ensure_model_available", return_value=None):
            with patch.object(self.node, "_call_ollama", side_effect=fake_call):
                # Use counts larger than available to ensure all LoRAs are listed
                _ = self.node.compose_prompt(num_wan_loras=2, num_image_loras=3)
                # Retrieve the message captured
                message = captured.get("user_message")
                self.assertIsNotNone(message)
                # The message contains a JSON string after a newline
                # Extract JSON payload from the message
                try:
                    json_str = message.split("\n", 1)[1]
                except IndexError:
                    self.fail("User message does not contain JSON payload")
                data = json.loads(json_str)
                # There should be one video LoRA and at least two image LoRAs from fixtures
                video_loras = data.get("video_loras", [])
                image_loras = data.get("image_loras", [])
                # Our fixtures contain exactly one Wan LoRA and two image LoRAs
                self.assertEqual(len(video_loras), 1)
                self.assertEqual(len(image_loras), 2)
                # Check that names match the fixture filenames
                video_names = {e["name"] for e in video_loras}
                image_names = {e["name"] for e in image_loras}
                self.assertIn("DetailAmplifier wan480p v1.0", video_names)
                self.assertIn("Woman877.v2", image_names)
                self.assertIn("illustriousXLv01_stabilizer_v1.198", image_names)


if __name__ == "__main__":
    unittest.main()
