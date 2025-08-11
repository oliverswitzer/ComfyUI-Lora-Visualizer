"""
Image Analyzer for LoRA Analysis
================================

Handles downloading images from URLs and analyzing them using Ollama's
vision models. Follows the official Ollama API format for multimodal requests.
"""

from __future__ import annotations

import json
import base64
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

try:
    import requests  # type: ignore[import]
except Exception:  # pylint: disable=broad-exception-caught
    requests = None

from ..lora_utils import ExampleData
from ..ollama_utils import ensure_model_available


class ImageAnalyzer:
    """Handles image downloading and vision-based analysis using Ollama."""

    DEFAULT_VISION_MODEL = "llava:latest"
    DEFAULT_TIMEOUT = 30
    DEFAULT_ANALYSIS_TIMEOUT = 120

    def __init__(self, api_url: str = "http://localhost:11434/api/chat"):
        self.api_url = api_url
        self.requests_module = requests

    def download_image(self, url: str, timeout: int = None) -> Optional[bytes]:
        """Download an image from a URL and return its bytes.

        Args:
            url: The image URL to download
            timeout: Request timeout in seconds

        Returns:
            Image bytes if successful, None if failed
        """
        if not self.requests_module or not url:
            return None

        timeout = timeout or self.DEFAULT_TIMEOUT

        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return None

            headers = {
                "User-Agent": "ComfyUI-LoRA-Visualizer/1.3.0",
                "Accept": "image/*",
            }

            response = self.requests_module.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Basic content type check
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                print(f"Warning: URL returned non-image content type: {content_type}")
                return None

            return response.content

        except Exception as e:
            print(f"Failed to download image from {url}: {e}")
            return None

    def download_example_images(self, examples: List[ExampleData]) -> List[ExampleData]:
        """Download images for a list of examples.

        Args:
            examples: List of ExampleData objects with URLs

        Returns:
            List of ExampleData objects with image_bytes populated (if successful)
        """
        downloaded_examples = []

        for example in examples:
            print(f"Downloading image from: {example.url}")
            image_bytes = self.download_image(example.url)

            if image_bytes:
                example.image_bytes = image_bytes
                downloaded_examples.append(example)
            else:
                print(f"Failed to download image, skipping: {example.url}")

        return downloaded_examples

    def _prepare_context_message(
        self,
        lora_description: Optional[str],
        trigger_words: List[str],
        num_examples: int,
    ) -> str:
        """Prepare the context message for vision analysis."""
        context_parts = []

        if lora_description:
            context_parts.append(f"LoRA Description: {lora_description}")

        if trigger_words:
            context_parts.append(f"Trigger Words: {', '.join(trigger_words)}")

        context_parts.append(
            f"I will show you {num_examples} example images with their prompts:"
        )

        return "\n\n".join(context_parts)

    def analyze_with_vision(
        self,
        examples: List[ExampleData],
        lora_description: Optional[str],
        trigger_words: List[str],
        system_prompt: str,
        model_name: str = None,
        timeout: int = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyze images using Ollama vision model.

        Args:
            examples: List of ExampleData with image_bytes populated
            lora_description: Optional description from metadata
            trigger_words: List of trigger words
            system_prompt: System instructions for the model
            model_name: Vision model to use (defaults to DEFAULT_VISION_MODEL)
            timeout: Analysis timeout in seconds

        Returns:
            Analysis dict with 'when_to_use' and 'example_prompts_and_analysis' or None
        """
        if not examples or not self.requests_module:
            return None

        model_name = model_name or self.DEFAULT_VISION_MODEL
        timeout = timeout or self.DEFAULT_ANALYSIS_TIMEOUT

        # Filter examples that have image data
        valid_examples = [ex for ex in examples if ex.image_bytes]
        if not valid_examples:
            print("No valid images available for vision analysis")
            return None

        try:
            # Ensure the vision model is available
            ensure_model_available(
                model_name, self.api_url, requests_module=self.requests_module
            )

            # Build context message
            user_content = self._prepare_context_message(
                lora_description, trigger_words, len(valid_examples)
            )

            # Add prompt information for each image
            for i, example in enumerate(valid_examples):
                user_content += f"\n\nExample {i+1} - Prompt: {example.prompt}"

            # Encode images to base64 (Ollama format)
            encoded_images = []
            for example in valid_examples:
                encoded = base64.b64encode(example.image_bytes).decode("utf-8")
                encoded_images.append(encoded)

            # Construct message in Ollama's expected format
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                    "images": encoded_images,  # Official Ollama format for vision
                },
            ]

            # Prepare API payload
            payload = {"model": model_name, "messages": messages, "stream": False}

            # Send request to Ollama
            response = self.requests_module.post(
                self.api_url, json=payload, timeout=timeout
            )
            response.raise_for_status()

            # Parse response according to Ollama's format
            data = response.json()
            if isinstance(data, dict) and "message" in data:
                content = data["message"].get("content", "")
                return self._parse_analysis_response(content)

            return None

        except Exception as e:
            print(f"Vision analysis failed: {e}")
            return None

    def _parse_analysis_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the model's JSON response into structured analysis data.

        Args:
            response_text: Raw text response from the model

        Returns:
            Parsed analysis dict or None if parsing failed
        """
        if not response_text:
            return None

        try:
            # Clean up the response (remove any markdown formatting)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            result = json.loads(cleaned)

            # Validate required fields
            when_to_use = result.get("when_to_use")
            examples = result.get("example_prompts_and_analysis")

            if isinstance(when_to_use, str) and isinstance(examples, list):
                return {
                    "when_to_use": when_to_use.strip(),
                    "example_prompts_and_analysis": examples,
                }

        except Exception as e:
            print(f"Failed to parse analysis response: {e}")
            print(f"Response was: {response_text[:200]}...")

        return None
