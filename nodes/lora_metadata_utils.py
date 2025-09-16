"""
Shared utilities for parsing LoRA metadata files.

This module provides common functionality for loading and extracting information
from LoRA .metadata.json files, used by both the visualizer and prompt splitter nodes.
"""

import glob
import json
import os
import re
from typing import Any, Optional

try:
    import folder_paths  # type: ignore[import]
except ImportError:
    # folder_paths may not be available during testing
    folder_paths = None

from .logging_utils import log, log_debug, log_error


class LoRAMetadataLoader:
    """Handles loading and parsing LoRA metadata files."""

    def __init__(self):
        """Initialize with ComfyUI's LoRA folder path if available."""
        self.loras_folder = None
        if folder_paths:
            lora_paths = folder_paths.get_folder_paths("loras")
            if lora_paths:
                self.loras_folder = lora_paths[0]

    def load_metadata(self, lora_name: str) -> Optional[dict[str, Any]]:
        """
        Load metadata for a LoRA from its .metadata.json file.
        Searches recursively through subdirectories.

        Args:
            lora_name: Name of the LoRA (without extension)

        Returns:
            Dict containing metadata or None if not found
        """
        if not self.loras_folder:
            log_error(f"LoRA folder not available, cannot load metadata for {lora_name}")
            return None

        # Search recursively for metadata files with different naming patterns
        patterns = [
            os.path.join(self.loras_folder, "**", f"{lora_name}.metadata.json"),
            os.path.join(self.loras_folder, "**", f"{lora_name}.safetensors.metadata.json"),
        ]

        for pattern in patterns:
            for metadata_path in glob.glob(pattern, recursive=True):
                try:
                    with open(metadata_path, encoding="utf-8") as f:
                        log_debug(f"Loaded metadata for LoRA: {lora_name}")
                        return json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    log_error(f"Error loading metadata for {lora_name}: {e}")
                    continue

        log(f"No metadata file found for LoRA: {lora_name}")
        return None

    def extract_trigger_words(self, metadata: Optional[dict[str, Any]]) -> list[str]:
        """
        Extract trigger words from LoRA metadata.

        Args:
            metadata: Loaded metadata dict or None

        Returns:
            List of trigger words, empty if none found
        """
        if not metadata:
            return []

        # Extract from civitai.trainedWords field
        try:
            if "civitai" in metadata and "trainedWords" in metadata["civitai"]:
                trained_words = metadata["civitai"]["trainedWords"]
                if isinstance(trained_words, list):
                    return [word.strip() for word in trained_words if word and word.strip()]
        except (KeyError, TypeError):
            pass

        return []

    def is_video_lora(self, metadata: Optional[dict[str, Any]]) -> bool:
        """
        Determine if a LoRA is for video generation based on base model.

        Args:
            metadata: Loaded metadata dict or None

        Returns:
            True if this is a video LoRA (WAN), False for image LoRA
        """
        if not metadata:
            return False

        # Check base_model field for video indicators
        base_model = metadata.get("base_model", "").lower()
        if "wan" in base_model or "video" in base_model or "i2v" in base_model:
            return True

        # Also check civitai.baseModel field
        try:
            civitai = metadata.get("civitai")
            if isinstance(civitai, dict):
                civitai_base = civitai.get("baseModel", "").lower()
            else:
                civitai_base = ""
            if "wan" in civitai_base or "video" in civitai_base or "i2v" in civitai_base:
                return True
        except (AttributeError, TypeError):
            pass

        return False

    def get_lora_info(self, lora_name: str) -> dict[str, Any]:
        """
        Get comprehensive information about a LoRA including trigger words and type.

        Args:
            lora_name: Name of the LoRA

        Returns:
            Dict with trigger_words, is_video_lora, and metadata fields
        """
        metadata = self.load_metadata(lora_name)
        trigger_words = self.extract_trigger_words(metadata)
        is_video = self.is_video_lora(metadata)

        return {
            "trigger_words": trigger_words,
            "is_video_lora": is_video,
            "metadata": metadata,
        }


# Global instance for reuse
_metadata_loader = None


def get_metadata_loader() -> LoRAMetadataLoader:
    """Get a shared instance of the metadata loader."""
    global _metadata_loader
    if _metadata_loader is None:
        _metadata_loader = LoRAMetadataLoader()
    return _metadata_loader


def load_lora_metadata(lora_name: str) -> Optional[dict[str, Any]]:
    """Convenience function to load metadata for a LoRA."""
    return get_metadata_loader().load_metadata(lora_name)


def get_lora_trigger_words(lora_name: str) -> list[str]:
    """Convenience function to get trigger words for a LoRA."""
    loader = get_metadata_loader()
    metadata = loader.load_metadata(lora_name)
    return loader.extract_trigger_words(metadata)


def is_video_lora(lora_name: str) -> bool:
    """Convenience function to check if a LoRA is for video generation."""
    loader = get_metadata_loader()
    metadata = loader.load_metadata(lora_name)
    return loader.is_video_lora(metadata)


def parse_lora_tags(prompt_text: str) -> tuple[list[dict], list[dict]]:
    """
    Parse LoRA tags from prompt text.

    Args:
        prompt_text: Text containing LoRA tags

    Returns:
        Tuple of (standard_loras, wanloras) where each is a list of dicts
        containing name, strength, type, and tag information.
    """
    standard_loras = []
    wanloras = []

    # Pattern for both LoRA types: capture everything inside the tags
    # Both handle names with spaces and special characters the same way
    lora_pattern = r"<lora:(.+?)>"
    wanlora_pattern = r"<wanlora:(.+?)>"

    # Find standard LoRA tags
    for match in re.finditer(lora_pattern, prompt_text):
        content = match.group(1).strip()
        # Split by last colon to separate name from strength
        last_colon_index = content.rfind(":")
        if last_colon_index > 0:
            name = content[:last_colon_index].strip()
            strength = content[last_colon_index + 1 :].strip()

            standard_loras.append(
                {
                    "name": name,
                    "strength": strength,
                    "type": "lora",
                    "tag": match.group(0),
                }
            )

    # Find wanlora tags (same logic as standard LoRAs)
    for match in re.finditer(wanlora_pattern, prompt_text):
        content = match.group(1).strip()
        # Split by last colon to separate name from strength
        last_colon_index = content.rfind(":")
        if last_colon_index > 0:
            name = content[:last_colon_index].strip()
            strength = content[last_colon_index + 1 :].strip()

            wanloras.append(
                {
                    "name": name,
                    "strength": strength,
                    "type": "wanlora",
                    "tag": match.group(0),
                }
            )

    return standard_loras, wanloras


def discover_all_loras() -> dict[str, dict[str, Any]]:
    """
    Discover all LoRAs in the ComfyUI LoRA directory and load their metadata.
    Recursively scans all subdirectories for LoRA files.

    Returns:
        Dict mapping LoRA name to its metadata and info
    """
    loader = get_metadata_loader()
    if not loader.loras_folder:
        log_error("LoRA folder not available for discovery")
        return {}

    loras = {}

    # Find all .safetensors files recursively
    pattern = os.path.join(loader.loras_folder, "**", "*.safetensors")
    for file_path in glob.glob(pattern, recursive=True):
        lora_name = os.path.basename(file_path).replace(".safetensors", "")

        # Get full info including metadata
        lora_info = loader.get_lora_info(lora_name)
        if lora_info["metadata"]:  # Only include LoRAs with metadata
            # Add directory and full path info
            rel_path = os.path.relpath(os.path.dirname(file_path), loader.loras_folder)
            lora_info["directory"] = rel_path if rel_path != "." else ""
            lora_info["full_path"] = file_path
            loras[lora_name] = lora_info

    log(f"Discovered {len(loras)} LoRAs with metadata (including nested directories)")
    return loras


def extract_embeddable_content(metadata: dict[str, Any]) -> str:
    """
    Extract text content from LoRA metadata for embedding generation.
    Title/filename words are repeated to increase their semantic weight.

    Args:
        metadata: LoRA metadata dictionary

    Returns:
        Combined text suitable for embedding
    """
    content_parts = []
    lora_name = metadata.get("file_name", "unknown")

    log_debug(f"🔍 Extracting embeddable content for LoRA: {lora_name}")

    # PRIORITY 1: File name (extract and repeat key words)
    file_name = metadata.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        log_debug(f"  📁 File name: '{file_name}'")
        # Extract meaningful words from filename (remove version numbers, common prefixes)
        title_words = re.findall(r"[a-zA-Z]{3,}", file_name.lower())
        log_debug(f"  🔤 Extracted title words: {title_words}")
        # Filter out common non-descriptive words
        filtered_words = [
            w
            for w in title_words
            if w
            not in ["lora", "wan", "lownoise", "highnoise", "safetensors", "i2v", "t2v", "version"]
        ]
        log_debug(f"  ✨ Filtered title words: {filtered_words}")
        # Repeat important title words 3x for higher semantic weight
        for word in filtered_words:
            content_parts.extend([word] * 3)
            log_debug(f"  🔁 Added '{word}' x3 for high priority")

    # PRIORITY 2: Model name and description
    model_name = metadata.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        log_debug(f"  📝 Model name: '{model_name}'")
        content_parts.append(model_name)
        # Also extract and repeat key words from model name
        model_words = re.findall(r"[a-zA-Z]{3,}", model_name.lower())
        filtered_model_words = [w for w in model_words if w not in ["lora", "for", "wan", "the"]]
        log_debug(f"  🎯 Model words added: {filtered_model_words}")
        content_parts.extend(filtered_model_words)

    model_description = metadata.get("modelDescription")
    if isinstance(model_description, str) and model_description.strip():
        # Clean HTML tags from description
        description = re.sub(r"<[^>]+>", "", model_description)
        content_parts.append(description)

    # Civitai model info
    civitai = metadata.get("civitai")
    if isinstance(civitai, dict):
        # Model name and description
        model = civitai.get("model")
        if isinstance(model, dict):
            model_name = model.get("name")
            if isinstance(model_name, str) and model_name.strip():
                content_parts.append(model_name)
            model_desc = model.get("description")
            if isinstance(model_desc, str) and model_desc.strip():
                description = re.sub(r"<[^>]+>", "", model_desc)
                content_parts.append(description)
            tags = model.get("tags")
            if isinstance(tags, list):
                content_parts.extend([str(tag) for tag in tags if tag])
        # Training words
        trained_words = civitai.get("trainedWords")
        if isinstance(trained_words, list):
            content_parts.extend([str(word) for word in trained_words if word])
    # If civitai is not a dict, skip safely

    # Tags from top level
    tags = metadata.get("tags")
    if isinstance(tags, list):
        content_parts.extend([str(tag) for tag in tags if tag])

    # Combine and clean
    combined = " ".join(str(part) for part in content_parts if part)
    final_content = combined.strip()

    preview = final_content[:100]
    suffix = "..." if len(final_content) > 100 else ""
    log_debug(f"  📊 Final embeddable content ({len(final_content)} chars): '{preview}{suffix}'")
    log_debug(f"  📈 Word count: {len(final_content.split())} words")

    return final_content


def extract_model_description(metadata: dict[str, Any]) -> str:
    """
    Extract model description from LoRA metadata.

    Args:
        metadata: LoRA metadata dictionary

    Returns:
        Combined description text from all available sources
    """
    if not metadata:
        return ""

    description_parts = []

    # Extract from modelDescription field
    if "modelDescription" in metadata and metadata["modelDescription"]:
        description_parts.append(metadata["modelDescription"].strip())

    # Extract from civitai model description
    if "civitai" in metadata and "model" in metadata["civitai"]:
        if (
            "description" in metadata["civitai"]["model"]
            and metadata["civitai"]["model"]["description"]
        ):
            description_parts.append(metadata["civitai"]["model"]["description"].strip())

    # Join all parts and limit length for context
    full_description = " ".join(description_parts)

    # Truncate if too long to keep manageable for LLM context
    if len(full_description) > 800:
        full_description = full_description[:800].rstrip()

    return full_description


def extract_example_prompts(metadata: dict[str, Any], limit: int = 5) -> list[str]:
    """
    Extract example prompts from LoRA metadata for style analysis.

    Args:
        metadata: LoRA metadata dictionary
        limit: Maximum number of prompts to extract

    Returns:
        List of example prompt texts
    """
    prompts = []

    if not metadata:
        return prompts

    if "civitai" in metadata and "images" in metadata["civitai"]:
        for image in metadata["civitai"]["images"]:
            if "meta" in image and image["meta"] is not None and "prompt" in image["meta"]:
                prompt = image["meta"]["prompt"]
                if isinstance(prompt, str) and prompt.strip():
                    prompts.append(prompt.strip())
                    if len(prompts) >= limit:
                        break

    return prompts


def classify_lora_type(metadata: dict[str, Any]) -> str:
    """
    Classify LoRA as image or video generation type.

    Args:
        metadata: LoRA metadata dictionary

    Returns:
        "image", "video", or "unknown"
    """
    base_model = metadata.get("base_model", "").lower()

    # Check for video indicators
    video_keywords = ["wan", "video", "i2v", "wan video"]
    if any(keyword in base_model for keyword in video_keywords):
        return "video"

    # Check civitai base model
    if "civitai" in metadata:
        civitai = metadata.get("civitai")
        if isinstance(civitai, dict):
            civitai_base = civitai.get("baseModel", "").lower()
        else:
            civitai_base = ""
        if any(keyword in civitai_base for keyword in video_keywords):
            return "video"

    # Check for image model indicators
    image_keywords = ["sdxl", "sd1.5", "sd 1.5", "flux", "illustrious", "noobai"]
    if any(keyword in base_model for keyword in image_keywords):
        return "image"

    if "civitai" in metadata:
        civitai = metadata.get("civitai")
        if isinstance(civitai, dict):
            civitai_base = civitai.get("baseModel", "").lower()
        else:
            civitai_base = ""
        if any(keyword in civitai_base for keyword in image_keywords):
            return "image"

    return "unknown"


def is_wan_2_2_lora(metadata: dict[str, Any]) -> bool:
    """
    Determine if a LoRA is specifically for WAN 2.2 (requires high/low pairing).

    Args:
        metadata: LoRA metadata dictionary

    Returns:
        True if this is a WAN 2.2 LoRA that needs high/low pairing
    """
    if not metadata:
        return False

    wan_2_2_indicators = [
        "wan2.2",
        "wan 2.2",
        "wanv2.2",
        "wan v2.2",
        "wan_2.2",
        "wan video 2.2",  # Matches "Wan Video 2.2 I2V-A14B"
        "i2v-a14b",  # Matches the specific model architecture
        "video 2.2",  # Matches "Video 2.2" part
    ]

    # Check base_model field for WAN 2.2 indicators
    base_model = metadata.get("base_model", "").lower()
    for indicator in wan_2_2_indicators:
        if indicator in base_model:
            return True

    # Check civitai base model
    civitai = metadata.get("civitai")
    if isinstance(civitai, dict):
        civitai_base = civitai.get("baseModel", "").lower()
        for indicator in wan_2_2_indicators:
            if indicator in civitai_base:
                return True

    # Check model name and description for WAN 2.2 mentions
    model_name = metadata.get("model_name", "").lower()
    for indicator in wan_2_2_indicators:
        if indicator in model_name:
            return True

    # Check civitai model name
    if isinstance(civitai, dict):
        model = civitai.get("model")
        if isinstance(model, dict):
            civitai_model_name = model.get("name", "").lower()
            for indicator in wan_2_2_indicators:
                if indicator in civitai_model_name:
                    return True

    # Check filename for WAN 2.2 indicators (important pattern!)
    filename = metadata.get("file_name", "").lower()
    for indicator in wan_2_2_indicators:
        if indicator in filename:
            return True

    return False


def extract_recommended_weight(metadata: dict[str, Any]) -> float:
    """
    Extract recommended weight/strength for a LoRA from its metadata.

    Args:
        metadata: LoRA metadata dictionary

    Returns:
        Recommended weight (default 0.8 if not found)
    """
    # Look for weight recommendations in description
    description_text = extract_model_description(metadata)

    # Look for weight patterns like "0.7", "weight: 0.8", "strength 0.6-0.9"
    weight_patterns = [
        r"weight[:\s]+([0-9]*\.?[0-9]+)",
        r"strength[:\s]+([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*strength",
        r"use.*?([0-9]*\.?[0-9]+)",
        r"best.*?([0-9]*\.?[0-9]+)",
    ]

    for pattern in weight_patterns:
        matches = re.findall(pattern, description_text, re.IGNORECASE)
        if matches:
            try:
                weight = float(matches[0])
                # Sanity check - weights should be reasonable
                if 0.1 <= weight <= 2.0:
                    return weight
            except ValueError:
                continue

    # Default weight based on LoRA type
    lora_type = classify_lora_type(metadata)
    if lora_type == "video":
        return 0.6  # Video LoRAs often need lower weights
    else:
        return 0.8  # Standard default for image LoRAs
