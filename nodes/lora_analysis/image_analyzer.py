"""
Image Analyzer for LoRA Analysis
================================

Handles downloading images from URLs and analyzing them using Ollama's
vision models. Follows the official Ollama API format for multimodal requests.
"""

from __future__ import annotations

import json
import base64
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

try:
    import requests  # type: ignore[import]
except Exception:  # pylint: disable=broad-exception-caught
    requests = None

from ..lora_utils import ExampleData
from ..ollama_utils import ensure_model_available
from ..logging_utils import log, log_error, log_warning


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
            return self._process_download(url, timeout)
        except Exception as e:
            log_error(f"Failed to download image from {url}: {e}")
            return None

    def _process_download(self, url: str, timeout: int) -> Optional[bytes]:
        """Process the actual download and content type handling."""
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
        if content_type.startswith("image/"):
            return response.content

        if content_type.startswith("video/") or url.endswith((".mp4", ".mov", ".avi")):
            return self._handle_video_content(url, response.content)

        log_warning(f"URL returned non-image/non-video content type: {content_type}")
        return None

    def _handle_video_content(self, url: str, video_bytes: bytes) -> Optional[bytes]:
        """Handle video content by extracting the first frame."""
        log(f"Detected video content, extracting first frame from: {url}")
        frame_bytes = self._extract_video_frame(video_bytes)
        if frame_bytes:
            log("Successfully converted video to image frame")
            return frame_bytes

        log_warning(f"Failed to extract frame from video: {url}")
        return None

    def _extract_video_frame(self, video_bytes: bytes) -> Optional[bytes]:
        """
        Extract the first frame from a video file using ffmpeg.

        Args:
            video_bytes: The video file content as bytes

        Returns:
            The first frame as JPEG bytes, or None if extraction fails
        """
        try:
            # Create temporary files for input video and output image
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_temp:
                video_temp.write(video_bytes)
                video_temp_path = video_temp.name

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_temp:
                image_temp_path = image_temp.name

            try:
                # Use ffmpeg to extract the first frame
                cmd = [
                    "ffmpeg",
                    "-i",
                    video_temp_path,  # Input video
                    "-vf",
                    "select=eq(n\\,0)",  # Select first frame
                    "-vframes",
                    "1",  # Output only 1 frame
                    "-f",
                    "image2",  # Output format
                    "-y",  # Overwrite output file
                    image_temp_path,
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30, check=False
                )

                if result.returncode == 0:
                    # Read the extracted frame
                    with open(image_temp_path, "rb") as f:
                        frame_bytes = f.read()
                    log(
                        f"Successfully extracted first frame from video "
                        f"({len(frame_bytes)} bytes)"
                    )
                    return frame_bytes

                log_error(f"ffmpeg failed: {result.stderr}")
                return None

            finally:
                # Clean up temporary files
                try:
                    os.unlink(video_temp_path)
                    os.unlink(image_temp_path)
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            log_error("ffmpeg timeout while extracting video frame")
            return None
        except Exception as e:
            log_error(f"Failed to extract video frame: {e}")
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
            log(f"Downloading image from: {example.url}")
            image_bytes = self.download_image(example.url)

            if image_bytes:
                example.image_bytes = image_bytes
                downloaded_examples.append(example)
            else:
                log_warning(f"Failed to download image, skipping: {example.url}")

        return downloaded_examples

    def _prepare_context_message(
        self,
        lora_description: Optional[str],
        trigger_words: List[str],
        num_examples: int,
    ) -> str:
        """Prepare the context message for vision analysis."""
        context_parts = []

        context_parts.append(
            "IMPORTANT: Base your analysis primarily on the textual metadata provided below."
        )
        context_parts.append(
            "The example images are supplementary - if they vary significantly, "
            "rely on the description and trigger words."
        )

        if lora_description:
            context_parts.append(f"LoRA Description: {lora_description}")
        else:
            context_parts.append("LoRA Description: Not provided")

        if trigger_words:
            context_parts.append(f"Trigger Words: {', '.join(trigger_words)}")
        else:
            context_parts.append("Trigger Words: Not provided")

        context_parts.append(
            f"I will show you {num_examples} example images with their prompts. "
            f"Use these to supplement your understanding, but prioritize the textual "
            f"metadata above."
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
            log_warning("No valid images available for vision analysis")
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
            for i, example in enumerate(valid_examples):
                if example.image_bytes:
                    encoded = base64.b64encode(example.image_bytes).decode("utf-8")
                    encoded_images.append(encoded)
                    log(
                        f"Successfully encoded image {i+1}: {len(example.image_bytes)} bytes -> "
                        f"{len(encoded)} chars"
                    )
                else:
                    log_warning(f"Example {i+1} has no image data, skipping")

            if not encoded_images:
                log_error("No valid images to send to vision model")
                return None

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

            # Log request details
            log(
                f"Sending vision request with {len(encoded_images)} images to {self.api_url}"
            )
            log(f"Total payload size: ~{len(str(payload))} characters")

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
            log_error(f"Vision analysis failed: {e}")
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
            log_error(f"Failed to parse analysis response: {e}")
            log_error(f"Response was: {response_text[:200]}...")

            # Try to create a fallback response from natural language
            fallback = self._create_fallback_response(response_text)
            if fallback:
                log_warning("Using fallback parser for non-JSON response")
                return fallback

        return None

    def _create_fallback_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Create a fallback structured response from natural language text.

        This handles cases where the vision model doesn't return JSON format.

        Args:
            response_text: Natural language response from the model

        Returns:
            Structured response dict or None if extraction fails
        """
        if not response_text or len(response_text.strip()) < 10:
            return None

        try:
            # Clean up the response
            cleaned = response_text.strip()

            # Create a basic structured response from the natural language
            return {
                "when_to_use": (
                    f"This LoRA appears to be designed for generating images with specific visual "
                    f"characteristics. Based on the model's analysis: {cleaned[:300]}..."
                ),
                "example_prompts_and_analysis": [
                    {
                        "prompt": "Analysis based on natural language response",
                        "analysis": cleaned[:500]
                        + ("..." if len(cleaned) > 500 else ""),
                    }
                ],
            }

        except Exception as e:
            log_error(f"Failed to create fallback response: {e}")
            return None
