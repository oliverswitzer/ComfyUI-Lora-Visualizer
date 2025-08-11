"""
Shared utilities for parsing LoRA metadata files.

This module provides common functionality for loading and extracting information
from LoRA .metadata.json files, used by both the visualizer and prompt splitter nodes.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple

try:
    import folder_paths  # type: ignore[import]
except ImportError:
    # folder_paths may not be available during testing
    folder_paths = None

from .logging_utils import log, log_error


class LoRAMetadataLoader:
    """Handles loading and parsing LoRA metadata files."""

    def __init__(self):
        """Initialize with ComfyUI's LoRA folder path if available."""
        self.loras_folder = None
        if folder_paths:
            lora_paths = folder_paths.get_folder_paths("loras")
            if lora_paths:
                self.loras_folder = lora_paths[0]

    def load_metadata(self, lora_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a LoRA from its .metadata.json file.

        Args:
            lora_name: Name of the LoRA (without extension)

        Returns:
            Dict containing metadata or None if not found
        """
        if not self.loras_folder:
            log_error(
                f"LoRA folder not available, cannot load metadata for {lora_name}"
            )
            return None

        # Try different metadata file naming patterns
        metadata_paths = [
            os.path.join(self.loras_folder, f"{lora_name}.metadata.json"),
            os.path.join(self.loras_folder, f"{lora_name}.safetensors.metadata.json"),
        ]

        for metadata_path in metadata_paths:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        log(f"Loaded metadata for LoRA: {lora_name}")
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    log_error(f"Error loading metadata for {lora_name}: {e}")
                    return None

        log(f"No metadata file found for LoRA: {lora_name}")
        return None

    def extract_trigger_words(self, metadata: Optional[Dict[str, Any]]) -> List[str]:
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
                    return [
                        word.strip() for word in trained_words if word and word.strip()
                    ]
        except (KeyError, TypeError):
            pass

        return []

    def is_video_lora(self, metadata: Optional[Dict[str, Any]]) -> bool:
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
            civitai_base = metadata.get("civitai", {}).get("baseModel", "").lower()
            if (
                "wan" in civitai_base
                or "video" in civitai_base
                or "i2v" in civitai_base
            ):
                return True
        except (AttributeError, TypeError):
            pass

        return False

    def get_lora_info(self, lora_name: str) -> Dict[str, Any]:
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


def load_lora_metadata(lora_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function to load metadata for a LoRA."""
    return get_metadata_loader().load_metadata(lora_name)


def get_lora_trigger_words(lora_name: str) -> List[str]:
    """Convenience function to get trigger words for a LoRA."""
    loader = get_metadata_loader()
    metadata = loader.load_metadata(lora_name)
    return loader.extract_trigger_words(metadata)


def is_video_lora(lora_name: str) -> bool:
    """Convenience function to check if a LoRA is for video generation."""
    loader = get_metadata_loader()
    metadata = loader.load_metadata(lora_name)
    return loader.is_video_lora(metadata)


def parse_lora_tags(prompt_text: str) -> Tuple[List[Dict], List[Dict]]:
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
