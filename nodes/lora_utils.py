"""
Helper functions for working with LoRA metadata and tags
=======================================================

This module centralizes common routines for loading LoRA metadata,
parsing LoRA tags from prompt text and extracting useful fields
for display or further processing.  Centralizing this logic
reduces duplication across different nodes (visualizer, composer,
preprocessor) and makes it easier to unit test these operations
independently.

Functions
---------

``get_loras_folder()``
    Returns the path to the first LoRA folder configured in ComfyUI
    via ``folder_paths.get_folder_paths('loras')``.  Returns
    ``None`` if the helper is unavailable or no paths are found.

``parse_lora_tags(prompt_text)``
    Parse standard LoRA (``<lora:name:strength>``) and wan LoRA
    (``<wanlora:name:strength>``) tags from a prompt string.  Returns
    two lists of dictionaries with ``name``, ``strength``, ``type``
    and ``tag`` keys.

``load_lora_metadata(loras_folder, lora_name)``
    Load the ``*.metadata.json`` file for a given LoRA name from the
    provided folder.  Returns a dictionary on success or ``None``
    if the file cannot be found or parsed.

``extract_lora_info(lora_data, metadata)``
    Combine parsed LoRA tag information with optional metadata into a
    single dictionary suitable for display.  Fields include trigger
    words, preview URLs, example images, model description and base
    model.  Missing fields are represented as ``None`` or empty
    lists.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any

try:
    import folder_paths  # type: ignore  # pylint: disable=import-error
except Exception:  # pylint: disable=broad-exception-caught
    folder_paths = None

from .logging_utils import log_error


class ExampleData:  # pylint: disable=too-few-public-methods
    """Represents an example image with its prompt and URL."""

    def __init__(self, prompt: str, url: str):
        self.prompt = prompt.strip()
        self.url = url.strip()
        self.image_bytes: Optional[bytes] = None

    def __repr__(self) -> str:
        return f"ExampleData(prompt='{self.prompt[:50]}...', url='{self.url}')"


class MetadataExtractor:
    """Extracts LoRA metadata and example data from JSON files."""

    @staticmethod
    def extract_example_data(
        metadata: Dict[str, Any], max_examples: int = 4
    ) -> List[ExampleData]:
        """Extract example prompts and image URLs from LoRA metadata.

        Args:
            metadata: The metadata dictionary loaded from a `.metadata.json` file
            max_examples: Maximum number of examples to extract

        Returns:
            List of ExampleData objects with prompts and URLs
        """
        examples = []
        seen_prompts = set()

        try:
            images = metadata.get("civitai", {}).get("images", [])
            if not isinstance(images, list):
                return examples

            for image in images:
                if not isinstance(image, dict):
                    continue

                # Extract prompt from nested structure
                prompt = None
                if "meta" in image and isinstance(image["meta"], dict):
                    prompt = image["meta"].get("prompt")

                # Extract image URL
                url = image.get("url")

                if prompt and isinstance(prompt, str) and url and isinstance(url, str):

                    prompt = prompt.strip()
                    url = url.strip()

                    # Skip duplicates and empty values
                    if prompt and url and prompt not in seen_prompts:
                        examples.append(ExampleData(prompt, url))
                        seen_prompts.add(prompt)

                        if len(examples) >= max_examples:
                            break

        except Exception as e:
            log_error(f"Error extracting example data: {e}")

        return examples

    @staticmethod
    def extract_lora_description(metadata: Dict[str, Any]) -> Optional[str]:
        """Extract the LoRA description from metadata.

        Args:
            metadata: The metadata dictionary

        Returns:
            Description text or None if not found
        """
        description = metadata.get("modelDescription")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None

    @staticmethod
    def extract_trigger_words(metadata: Dict[str, Any]) -> List[str]:
        """Extract trigger words from metadata.

        Args:
            metadata: The metadata dictionary

        Returns:
            List of trigger words (may be empty)
        """
        try:
            words = metadata.get("civitai", {}).get("trainedWords", [])
            if isinstance(words, list):
                return [str(word).strip() for word in words if word]
        except Exception:
            pass
        return []

    @staticmethod
    def extract_example_prompts(metadata: Dict[str, Any]) -> List[str]:
        """Legacy method for extracting just prompts (for backward compatibility).

        Args:
            metadata: The metadata dictionary

        Returns:
            List of example prompts
        """
        examples = MetadataExtractor.extract_example_data(metadata)
        return [ex.prompt for ex in examples]


def get_loras_folder() -> Optional[str]:
    """Return the configured loras folder path or ``None``.

    This helper uses ComfyUI's ``folder_paths.get_folder_paths('loras')``
    to determine the location of LoRA metadata files.  If the
    ``folder_paths`` module is unavailable (e.g. during testing) or no
    folder is configured, ``None`` is returned.
    """
    if folder_paths is None:
        return None
    try:
        paths = folder_paths.get_folder_paths("loras")
        return paths[0] if paths else None
    except Exception:
        return None


def parse_lora_tags(
    prompt_text: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Parse LoRA tags from the given prompt text.

    This function looks for substrings of the form ``<lora:name:strength>``
    and ``<wanlora:name:strength>`` in the prompt.  It splits each tag
    on the last colon to separate the name from the strength, trims
    whitespace, and returns two lists of dictionaries representing
    standard and wan LoRA tags respectively.

    Args:
        prompt_text: The user-provided prompt string.

    Returns:
        A tuple ``(standard_loras, wanloras)``.  Each element is a list
        where each item is a dict with keys ``name``, ``strength``,
        ``type`` and ``tag``.
    """
    standard_loras: List[Dict[str, str]] = []
    wanloras: List[Dict[str, str]] = []
    # Use non-greedy patterns to capture content inside the tags
    lora_pattern = r"<lora:(.+?)>"
    wanlora_pattern = r"<wanlora:(.+?)>"
    # Find standard LoRA tags
    for match in re.finditer(lora_pattern, prompt_text):
        content = match.group(1).strip()
        # Find last colon separating name and strength
        idx = content.rfind(":")
        if idx > 0:
            name = content[:idx].strip()
            strength = content[idx + 1 :].strip()
            standard_loras.append(
                {
                    "name": name,
                    "strength": strength,
                    "type": "lora",
                    "tag": match.group(0),
                }
            )
    # Find wan LoRA tags
    for match in re.finditer(wanlora_pattern, prompt_text):
        content = match.group(1).strip()
        idx = content.rfind(":")
        if idx > 0:
            name = content[:idx].strip()
            strength = content[idx + 1 :].strip()
            wanloras.append(
                {
                    "name": name,
                    "strength": strength,
                    "type": "wanlora",
                    "tag": match.group(0),
                }
            )
    return standard_loras, wanloras


def load_lora_metadata(
    loras_folder: Optional[str], lora_name: str
) -> Optional[Dict[str, Any]]:
    """Load the metadata JSON for a given LoRA name from a folder.

    Tries to locate a file named ``{name}.metadata.json`` or
    ``{name}.safetensors.metadata.json`` in the provided folder.
    Returns the parsed JSON dictionary on success or ``None`` if the
    file is missing or cannot be parsed.

    Args:
        loras_folder: Path to the LoRA directory.  If ``None``,
            metadata loading is skipped and ``None`` is returned.
        lora_name: Base name of the LoRA, without extension.

    Returns:
        Parsed metadata dictionary or ``None``.
    """
    if not loras_folder:
        return None
    # Build possible file paths
    candidates = [
        os.path.join(loras_folder, f"{lora_name}.metadata.json"),
        os.path.join(loras_folder, f"{lora_name}.safetensors.metadata.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log_error(f"Error loading metadata for {lora_name}: {e}")
                return None
    return None


def extract_lora_info(
    lora_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract display information from parsed LoRA tag and optional metadata.

    Combines the tag information (name, strength, type, raw tag) with
    metadata fields when available.  If metadata is provided, the
    following keys are extracted:

    * ``trainedWords`` from ``meta['civitai']`` → ``trigger_words``
    * ``modelId`` from ``meta['civitai']`` → ``civitai_url``
    * ``preview_url`` → ``preview_url``
    * ``images`` from ``meta['civitai']`` → ``example_images``
    * ``images`` from ``meta['civitai']`` → ``example_data`` (ExampleData objects)
    * ``model_name``, ``modelDescription``, ``base_model``,
      ``preview_nsfw_level`` for additional fields.

    Args:
        lora_data: Dictionary with keys ``name``, ``strength``, ``type`` and ``tag``.
        metadata: Parsed metadata dictionary or ``None``.

    Returns:
        A new dictionary merging lora_data with extracted metadata fields.
    """
    info = {
        "name": lora_data.get("name"),
        "strength": lora_data.get("strength"),
        "type": lora_data.get("type"),
        "tag": lora_data.get("tag"),
        "trigger_words": [],
        "preview_url": None,
        "example_images": [],
        "example_data": [],
        "model_description": None,
        "base_model": None,
        "nsfw_level": 0,
        "civitai_url": None,
    }
    if not metadata:
        return info
    try:
        civ = metadata.get("civitai", {})
        # Use MetadataExtractor for consistent parsing
        info["trigger_words"] = MetadataExtractor.extract_trigger_words(metadata)
        # civitai URL
        model_id = civ.get("modelId")
        if model_id:
            info["civitai_url"] = f"https://civitai.com/models/{model_id}"
        # preview URL
        if metadata.get("preview_url"):
            info["preview_url"] = metadata["preview_url"]
        # example images - keep both legacy dict format and new ExampleData format
        images = civ.get("images")
        if isinstance(images, list):
            # Legacy format for backward compatibility
            examples = []
            for img in images:
                try:
                    examples.append(
                        {
                            "url": img.get("url"),
                            "width": img.get("width", 0),
                            "height": img.get("height", 0),
                            "nsfw_level": img.get("nsfwLevel", 1),
                            "type": img.get("type", "image"),
                            "meta": img.get("meta", {}),
                        }
                    )
                except Exception:
                    pass
            info["example_images"] = examples

            # New structured format using ExampleData (convert to dicts for JSON serialization)
            example_data = MetadataExtractor.extract_example_data(metadata)
            info["example_data"] = [
                {"prompt": ex.prompt, "url": ex.url} for ex in example_data
            ]
        # additional meta fields
        if "model_name" in metadata:
            info["model_name"] = metadata["model_name"]
        if "modelDescription" in metadata:
            info["model_description"] = metadata["modelDescription"]
        if "base_model" in metadata:
            info["base_model"] = metadata["base_model"]
        if "preview_nsfw_level" in metadata:
            info["nsfw_level"] = metadata["preview_nsfw_level"]
    except Exception:
        pass
    return info
