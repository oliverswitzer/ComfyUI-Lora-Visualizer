"""
Tests for the LoRA Analysis domain classes.

These tests cover the vision-based analysis functionality including
metadata extraction, image downloading, and Ollama vision API interactions.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Mock ComfyUI dependencies before importing
os.environ.setdefault("COMFYUI_SKIP_LORA_ANALYSIS", "1")
sys.modules.setdefault("folder_paths", MagicMock())
sys.modules.setdefault("server", MagicMock())

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from nodes.lora_utils import MetadataExtractor, ExampleData
from nodes.lora_analysis.image_analyzer import ImageAnalyzer


class TestMetadataExtractor(unittest.TestCase):
    """Test the MetadataExtractor class."""

    def setUp(self):
        self.extractor = MetadataExtractor()

        # Sample metadata like what comes from CivitAI
        self.sample_metadata = {
            "civitai": {
                "images": [
                    {
                        "url": "https://example.com/image1.jpg",
                        "meta": {"prompt": "a beautiful woman, detailed"},
                    },
                    {
                        "url": "https://example.com/image2.jpg",
                        "meta": {"prompt": "landscape scene, mountains"},
                    },
                    {
                        "url": "https://example.com/image3.jpg",
                        "meta": {"prompt": "a beautiful woman, detailed"},  # Duplicate
                    },
                ],
                "trainedWords": ["beautiful", "detailed", "style"],
            },
            "modelDescription": "A LoRA for generating beautiful portraits",
        }

    def test_extract_example_data_basic(self):
        """Test basic extraction of example data."""
        examples = self.extractor.extract_example_data(self.sample_metadata)

        self.assertEqual(len(examples), 2)  # Should deduplicate
        self.assertIsInstance(examples[0], ExampleData)
        self.assertEqual(examples[0].prompt, "a beautiful woman, detailed")
        self.assertEqual(examples[0].url, "https://example.com/image1.jpg")
        self.assertEqual(examples[1].prompt, "landscape scene, mountains")

    def test_extract_example_data_empty_metadata(self):
        """Test extraction with empty or malformed metadata."""
        self.assertEqual(self.extractor.extract_example_data({}), [])
        self.assertEqual(self.extractor.extract_example_data({"civitai": {}}), [])
        self.assertEqual(
            self.extractor.extract_example_data({"civitai": {"images": []}}), []
        )

    def test_extract_example_data_max_examples(self):
        """Test that max_examples parameter is respected."""
        # Add more examples to test the limit
        large_metadata = {
            "civitai": {
                "images": [
                    {
                        "url": f"https://example.com/image{i}.jpg",
                        "meta": {"prompt": f"prompt {i}"},
                    }
                    for i in range(10)
                ]
            }
        }

        examples = self.extractor.extract_example_data(large_metadata, max_examples=3)
        self.assertEqual(len(examples), 3)

    def test_extract_lora_description(self):
        """Test extraction of LoRA description."""
        description = self.extractor.extract_lora_description(self.sample_metadata)
        self.assertEqual(description, "A LoRA for generating beautiful portraits")

        # Test with missing description
        self.assertIsNone(self.extractor.extract_lora_description({}))
        self.assertIsNone(
            self.extractor.extract_lora_description({"modelDescription": ""})
        )

    def test_extract_trigger_words(self):
        """Test extraction of trigger words."""
        words = self.extractor.extract_trigger_words(self.sample_metadata)
        self.assertEqual(words, ["beautiful", "detailed", "style"])

        # Test with missing trigger words
        self.assertEqual(self.extractor.extract_trigger_words({}), [])
        self.assertEqual(self.extractor.extract_trigger_words({"civitai": {}}), [])

    def test_extract_example_prompts_legacy(self):
        """Test the legacy method for backward compatibility."""
        prompts = self.extractor.extract_example_prompts(self.sample_metadata)
        self.assertEqual(len(prompts), 2)
        self.assertIn("a beautiful woman, detailed", prompts)
        self.assertIn("landscape scene, mountains", prompts)


class TestImageAnalyzer(unittest.TestCase):
    """Test the ImageAnalyzer class."""

    def setUp(self):
        self.analyzer = ImageAnalyzer()

        # Mock requests module
        self.mock_requests = Mock()
        self.analyzer.requests_module = self.mock_requests

        # Sample example data
        self.sample_examples = [
            ExampleData("portrait of a woman", "https://example.com/image1.jpg"),
            ExampleData("landscape scene", "https://example.com/image2.jpg"),
        ]

    def test_download_image_success(self):
        """Test successful image download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status.return_value = None
        self.mock_requests.get.return_value = mock_response

        result = self.analyzer.download_image("https://example.com/image.jpg")

        self.assertEqual(result, b"fake_image_data")
        self.mock_requests.get.assert_called_once()

    def test_download_image_invalid_url(self):
        """Test download with invalid URL."""
        self.assertIsNone(self.analyzer.download_image("not_a_url"))
        self.assertIsNone(self.analyzer.download_image(""))

    def test_download_image_non_image_content(self):
        """Test download with non-image content type."""
        mock_response = Mock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html>Not an image</html>"
        mock_response.raise_for_status.return_value = None
        self.mock_requests.get.return_value = mock_response

        result = self.analyzer.download_image("https://example.com/notimage.html")
        self.assertIsNone(result)

    def test_download_image_request_error(self):
        """Test download with request error."""
        self.mock_requests.get.side_effect = Exception("Network error")

        result = self.analyzer.download_image("https://example.com/image.jpg")
        self.assertIsNone(result)

    def test_download_example_images(self):
        """Test downloading images for multiple examples."""

        # Mock successful download for first image, failure for second
        def mock_download(url):
            if "image1" in url:
                return b"image1_data"
            return None

        with patch.object(self.analyzer, "download_image", side_effect=mock_download):
            result = self.analyzer.download_example_images(self.sample_examples)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].image_bytes, b"image1_data")

    @patch("nodes.lora_analysis.image_analyzer.ensure_model_available")
    def test_analyze_with_vision_success(self, mock_ensure_model):
        """Test successful vision analysis."""
        # Prepare examples with image data
        examples = [ExampleData("test prompt", "https://example.com/image.jpg")]
        examples[0].image_bytes = b"fake_image_data"

        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": (
                    '{"when_to_use": "Test usage", '
                    '"example_prompts_and_analysis": '
                    '[{"prompt": "test", "analysis": "test analysis"}]}'
                )
            }
        }
        mock_response.raise_for_status.return_value = None
        self.mock_requests.post.return_value = mock_response

        result = self.analyzer.analyze_with_vision(
            examples, "Test description", ["trigger1", "trigger2"], "Test system prompt"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["when_to_use"], "Test usage")
        self.assertEqual(len(result["example_prompts_and_analysis"]), 1)

        # Verify API call was made correctly
        self.mock_requests.post.assert_called_once()
        call_args = self.mock_requests.post.call_args
        payload = call_args[1]["json"]

        self.assertEqual(payload["model"], self.analyzer.DEFAULT_VISION_MODEL)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn("images", payload["messages"][1])

    def test_analyze_with_vision_no_images(self):
        """Test vision analysis with no valid images."""
        examples = [ExampleData("test prompt", "https://example.com/image.jpg")]
        # No image_bytes set

        result = self.analyzer.analyze_with_vision(
            examples, None, [], "Test system prompt"
        )

        self.assertIsNone(result)

    def test_parse_analysis_response_success(self):
        """Test parsing valid JSON response."""
        response_text = (
            '{"when_to_use": "Test usage", "example_prompts_and_analysis": []}'
        )

        result = self.analyzer._parse_analysis_response(response_text)

        self.assertIsNotNone(result)
        self.assertEqual(result["when_to_use"], "Test usage")

    def test_parse_analysis_response_with_markdown(self):
        """Test parsing response with markdown formatting."""
        response_text = """```json
{"when_to_use": "Test usage", "example_prompts_and_analysis": []}
```"""

        result = self.analyzer._parse_analysis_response(response_text)

        self.assertIsNotNone(result)
        self.assertEqual(result["when_to_use"], "Test usage")

    def test_parse_analysis_response_invalid_json(self):
        """Test parsing invalid JSON response with fallback."""
        result = self.analyzer._parse_analysis_response("Not valid JSON")
        # Should return fallback response instead of None
        self.assertIsNotNone(result)
        self.assertIn("when_to_use", result)
        self.assertIn("example_prompts_and_analysis", result)

        result = self.analyzer._parse_analysis_response("")
        self.assertIsNone(result)

    def test_parse_analysis_response_missing_fields(self):
        """Test parsing response with missing required fields."""
        # Missing example_prompts_and_analysis
        result = self.analyzer._parse_analysis_response('{"when_to_use": "Test"}')
        self.assertIsNone(result)

        # Wrong field types
        result = self.analyzer._parse_analysis_response(
            '{"when_to_use": 123, "example_prompts_and_analysis": "not_list"}'
        )
        self.assertIsNone(result)


class TestExampleData(unittest.TestCase):
    """Test the ExampleData class."""

    def test_example_data_creation(self):
        """Test creation and basic properties of ExampleData."""
        example = ExampleData("  test prompt  ", "  https://example.com/image.jpg  ")

        self.assertEqual(example.prompt, "test prompt")
        self.assertEqual(example.url, "https://example.com/image.jpg")
        self.assertIsNone(example.image_bytes)

    def test_example_data_repr(self):
        """Test string representation of ExampleData."""
        example = ExampleData(
            "A very long prompt that should be truncated",
            "https://example.com/image.jpg",
        )
        repr_str = repr(example)

        self.assertIn("ExampleData", repr_str)
        self.assertIn("A very long prompt that should be truncated"[:50], repr_str)


if __name__ == "__main__":
    unittest.main()
