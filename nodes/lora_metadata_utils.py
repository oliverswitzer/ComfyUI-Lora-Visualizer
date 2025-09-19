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

from .logging_utils import log, log_debug, log_error, log_warning


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


def split_prompt_by_lora_high_low(prompt_text: str) -> tuple[str, str]:
    """
    Simple mode: Extract base prompt and treat ALL LoRAs as singles.

    NO pattern matching, NO fuzzy matching, NO classification.
    Just extract LoRA tags and include them in both outputs.

    Args:
        prompt_text: Input prompt with LoRA tags

    Returns:
        Tuple of (high_prompt, low_prompt) where:
        - Both outputs are identical
        - Both contain base prompt + all LoRA tags
        - Base prompt text excludes all LoRA tags
    """
    log_debug(f"Simple mode: treating all LoRAs as singles: '{prompt_text}'")

    # Extract base prompt and all LoRA tags
    base_prompt_parts = []
    all_lora_tags = []

    # Split the prompt by lora tags and regular content
    parts = re.split(r"(<lora:[^>]+>)", prompt_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("<lora:") and part.endswith(">"):
            all_lora_tags.append(part)
            log_debug(f"Found LoRA tag: {part}")
        else:
            # This is regular prompt content (text, not LoRA tags)
            base_prompt_parts.append(part)

    # Build the base prompt (text content only, no LoRA tags)
    base_prompt = " ".join(base_prompt_parts).strip()

    # Simple mode: ALL LoRA tags go to both outputs (treated as singles)
    # Build HIGH prompt: base + all lora tags
    high_prompt_parts = [base_prompt] + all_lora_tags
    high_prompt = " ".join(part for part in high_prompt_parts if part.strip())

    # Build LOW prompt: base + all lora tags (identical to HIGH in simple mode)
    low_prompt_parts = [base_prompt] + all_lora_tags
    low_prompt = " ".join(part for part in low_prompt_parts if part.strip())

    log_debug(f"HIGH prompt: '{high_prompt}'")
    log_debug(f"LOW prompt: '{low_prompt}'")

    log(
        f"Simple LoRA split - All {len(all_lora_tags)} LoRA tags treated as singles (included in both outputs)"
    )

    return high_prompt, low_prompt


def find_lora_pairs_in_prompt(prompt_text: str) -> str:
    """
    Find and pair LoRA tags in a prompt using fuzzy string matching.

    This function extracts LoRA tags from the prompt, finds the best matching pairs
    from available LoRAs on the system using string similarity, and stitches the
    results back into the prompt deterministically.

    This approach uses pure string similarity without making assumptions about naming
    patterns, making it flexible for any LoRA naming convention.

    Args:
        prompt_text: Input prompt containing LoRA tags

    Returns:
        Modified prompt with paired LoRA tags added
    """
    import re

    from rapidfuzz import fuzz, process

    log_debug("Finding LoRA pairs in prompt with rapidfuzz")

    # Extract all LoRA tags from the prompt
    lora_pattern = r"<lora:([^>]+)>"
    lora_matches = re.findall(lora_pattern, prompt_text)

    if not lora_matches:
        log_debug("  No LoRA tags found in prompt")
        return prompt_text

    log_debug(f"  Found {len(lora_matches)} LoRA tags: {lora_matches}")

    # Get all available LoRAs from the system
    all_loras = discover_all_loras()
    available_lora_names = list(all_loras.keys())

    if not available_lora_names:
        log_debug("  No LoRAs available on system")
        return prompt_text

    # Extract LoRA names from the prompt (without weights)
    lora_names_in_prompt = []
    for lora_tag in lora_matches:
        lora_name = lora_tag.split(":")[0]  # Get name before weight
        lora_names_in_prompt.append(lora_name)

    log_debug(f"  LoRA names in prompt: {lora_names_in_prompt}")

    pairs_to_add = []

    for lora_name in lora_names_in_prompt:
        # Use rapidfuzz to find the most similar LoRA
        # Exclude the current LoRA and any already in the prompt
        candidates = [
            name
            for name in available_lora_names
            if name != lora_name and name not in lora_names_in_prompt
        ]

        if not candidates:
            continue

        # Use fuzzy matching to find the best candidate
        # Use a high threshold to ensure we only get very similar LoRAs
        result = process.extractOne(
            lora_name,
            candidates,
            scorer=fuzz.ratio,
            score_cutoff=60,  # Only accept matches with >60% similarity
        )

        if result:
            best_match, score, _ = result
            # Check if this pair is not already being added
            if best_match not in pairs_to_add:
                pairs_to_add.append(best_match)
                log(f"    Rapidfuzz found pair: {lora_name} <-> {best_match} (score: {score}%)")

    # Add paired LoRAs to the prompt deterministically
    if pairs_to_add:
        log(f"Adding {len(pairs_to_add)} paired LoRAs to prompt: {pairs_to_add}")

        # Insert new LoRA tags at the end of existing LoRA tags in the prompt
        new_lora_tags = []
        for pair_name in pairs_to_add:
            # Use a default weight of 1.0 for paired LoRAs
            new_lora_tags.append(f"<lora:{pair_name}:1.0>")

        # Find the position to insert (after the last LoRA tag)
        last_lora_pos = 0
        for match in re.finditer(lora_pattern, prompt_text):
            last_lora_pos = match.end()

        if last_lora_pos > 0:
            # Insert after the last LoRA tag
            result = (
                prompt_text[:last_lora_pos]
                + " "
                + " ".join(new_lora_tags)
                + prompt_text[last_lora_pos:]
            )
        else:
            # Fallback: add at the beginning
            result = " ".join(new_lora_tags) + " " + prompt_text

        return result.strip()

    return prompt_text


def find_lora_pair(lora_name: str, available_lora_names: list[str]) -> Optional[str]:
    """
    Find the best matching LoRA pair using fuzzy string similarity.

    Uses string similarity to find the most similar LoRA name from the available
    list, without making assumptions about naming patterns. This makes it flexible
    for any LoRA naming convention.

    Args:
        lora_name: Name of the LoRA to find a pair for
        available_lora_names: List of available LoRA names to search through

    Returns:
        Best matching LoRA name or None if no suitable match found
    """
    from rapidfuzz import fuzz, process

    log_debug(f"Finding LoRA pair for: {lora_name}")

    # Exclude the input LoRA from candidates
    candidates = [name for name in available_lora_names if name != lora_name]

    if not candidates:
        return None

    # Use fuzzy matching to find the best candidate
    result = process.extractOne(
        lora_name,
        candidates,
        scorer=fuzz.ratio,
        score_cutoff=60,  # Only accept matches with >60% similarity
    )

    if result:
        best_match, score, _ = result
        log_debug(f"Found LoRA pair: {lora_name} <-> {best_match} (score: {score}%)")
        return best_match

    return None


# Keep the old function name for backward compatibility
def find_lora_high_low_pair(lora_name: str, available_lora_names: list[str]) -> Optional[str]:
    """
    DEPRECATED: Use find_lora_pair() instead.

    This function is kept for backward compatibility.
    """
    return find_lora_pair(lora_name, available_lora_names)


def split_prompt_by_lora_high_low_with_ollama(
    prompt_text: str, use_ollama: bool = True
) -> tuple[str, str]:
    """
    Split a prompt into HIGH and LOW specific versions using configurable matching strategies.

    Two modes available:
    - Simple mode (use_ollama=False): Pattern-based extraction of existing HIGH/LOW LoRAs only
    - Advanced mode (use_ollama=True): Fuzzy matching + Ollama to find and classify LoRA pairs

    Args:
        prompt_text: Input prompt with LoRA tags
        use_ollama: Whether to use advanced fuzzy matching + Ollama classification (defaults to True)

    Returns:
        Tuple of (high_prompt, low_prompt) where:
        - Simple mode: Extracts existing HIGH/LOW LoRAs based on naming patterns
        - Advanced mode: Finds pairs via fuzzy matching and classifies with Ollama
        - Single LoRAs (no pairs found): Included in both outputs
        - Base prompt text excludes all LoRA tags
    """
    if use_ollama:
        # Advanced mode: Use fuzzy matching + Ollama classification
        return _split_with_fuzzy_matching_and_ollama(prompt_text)
    else:
        # Simple mode: Use pattern-based extraction (same as split_prompt_by_lora_high_low)
        return split_prompt_by_lora_high_low(prompt_text)


def _split_with_fuzzy_matching_and_ollama(prompt_text: str) -> tuple[str, str]:
    """
    Advanced splitting using fuzzy matching + Ollama classification.

    This is the original implementation that was in split_prompt_by_lora_high_low_with_ollama.
    """
    import re

    log_debug(f"Input prompt for advanced HIGH/LOW split: '{prompt_text}'")

    # Extract the base prompt (without lora tags) and lora tags separately
    base_prompt_parts = []
    high_lora_tags = []
    low_lora_tags = []
    single_lora_tags = []

    # First pass: collect all LoRA names from the prompt
    lora_names_in_prompt = []
    lora_pattern = r"<lora:([^>]+)>"
    for match in re.finditer(lora_pattern, prompt_text):
        tag_content = match.group(1)
        lora_name = tag_content.split(":")[0]  # Get name before weight
        lora_names_in_prompt.append(lora_name)

    log_debug(f"Found LoRA names in prompt: {lora_names_in_prompt}")

    # Split the prompt by lora tags and regular content
    parts = re.split(r"(<lora:[^>]+>)", prompt_text)

    # Collect candidate pairs for classification
    candidate_pairs = []
    lora_tag_mapping = {}  # Map lora name to full tag

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("<lora:") and part.endswith(">"):
            # This is a lora tag - extract name and store mapping
            tag_content = part[6:-1]  # Remove <lora: and >
            tag_name = tag_content.split(":")[0]  # Get name before first colon
            lora_tag_mapping[tag_name] = part

            # Use fuzzy matching to find LoRA pairs
            pair_name = find_lora_pair_fuzzy(tag_name, lora_names_in_prompt)

            if pair_name:
                # We found a pair - add to candidates for classification
                pair_tuple = tuple(sorted([tag_name, pair_name]))  # Ensure consistent ordering
                if pair_tuple not in [tuple(sorted([p[0], p[1]])) for p in candidate_pairs]:
                    candidate_pairs.append((tag_name, pair_name))
                    log_debug(f"Found LoRA pair for classification: {tag_name} <-> {pair_name}")
            else:
                # Single lora (no pair found) - include in both outputs
                single_lora_tags.append(part)
                log_debug(f"Classified as single lora (no pair found): {part}")
        else:
            # This is regular prompt content (text, not LoRA tags)
            base_prompt_parts.append(part)

    # Classify HIGH/LOW for all candidate pairs using Ollama
    if candidate_pairs:
        classifications = classify_lora_pairs_with_ollama(candidate_pairs)

        # Apply classifications
        for lora1, lora2 in candidate_pairs:
            lora1_tag = lora_tag_mapping[lora1]
            lora2_tag = lora_tag_mapping[lora2]

            classification = classifications.get((lora1, lora2))
            if not classification:
                # Fallback: treat as single LoRAs (include both in both outputs) if Ollama classification failed
                log_debug(f"Ollama classification failed for {lora1} vs {lora2}, treating as single LoRAs")
                single_lora_tags.extend([lora1_tag, lora2_tag])
                continue

            # Apply Ollama classification
            if classification["high_lora"] == lora1:
                high_lora_tags.append(lora1_tag)
                low_lora_tags.append(lora2_tag)
                log_debug(f"Ollama classified {lora1} as HIGH, {lora2} as LOW")
            else:
                high_lora_tags.append(lora2_tag)
                low_lora_tags.append(lora1_tag)
                log_debug(f"Ollama classified {lora2} as HIGH, {lora1} as LOW")

    # Build the base prompt (text content only, no LoRA tags)
    base_prompt = " ".join(base_prompt_parts).strip()

    # Build HIGH prompt: base + HIGH lora tags + single lora tags
    high_prompt_parts = [base_prompt] + high_lora_tags + single_lora_tags
    high_prompt = " ".join(part for part in high_prompt_parts if part.strip())

    # Build LOW prompt: base + LOW lora tags + single lora tags
    low_prompt_parts = [base_prompt] + low_lora_tags + single_lora_tags
    low_prompt = " ".join(part for part in low_prompt_parts if part.strip())

    log_debug(f"HIGH output: '{high_prompt}'")
    log_debug(f"LOW output: '{low_prompt}'")

    log_debug(
        f"Advanced HIGH/LOW split - HIGH lora tags: {len(high_lora_tags)}, LOW lora tags: {len(low_lora_tags)}, Single lora tags: {len(single_lora_tags)}"
    )

    return high_prompt, low_prompt


def find_lora_pair_fuzzy(lora_name: str, lora_names_in_prompt: list[str]) -> Optional[str]:
    """
    Find the best matching LoRA pair using fuzzy string similarity.

    Args:
        lora_name: Name of the LoRA to find a pair for
        lora_names_in_prompt: List of LoRA names in the current prompt

    Returns:
        Best matching LoRA name or None if no suitable match found
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        log_debug(f"rapidfuzz not available, cannot find fuzzy pairs for {lora_name}")
        return None

    # Exclude the current LoRA from candidates
    candidates = [name for name in lora_names_in_prompt if name != lora_name]

    if not candidates:
        return None

    # Use fuzzy matching to find the best candidate
    result = process.extractOne(
        lora_name,
        candidates,
        scorer=fuzz.ratio,
        score_cutoff=60,  # Only accept matches with >60% similarity
    )

    if result:
        best_match, score, _ = result
        log_debug(f"Fuzzy found LoRA pair: {lora_name} <-> {best_match} (score: {score}%)")
        return best_match

    return None




def classify_lora_pairs_with_ollama(
    candidate_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """
    Use Ollama to classify which LoRA in each pair is HIGH vs LOW.

    Args:
        candidate_pairs: List of tuples containing LoRA name pairs

    Returns:
        Dict mapping (lora1, lora2) tuples to classification results with keys:
        - "high_lora": Name of the LoRA that should be considered HIGH
        - "low_lora": Name of the LoRA that should be considered LOW
        - "reasoning": Brief explanation of the classification
    """
    if not candidate_pairs:
        return {}

    try:
        # Import shared Ollama utilities
        from .ollama_utils import call_ollama_chat as _shared_call_ollama_chat
        from .ollama_utils import ensure_model_available as _shared_ensure_model_available

        # Import requests for Ollama communication
        try:
            import requests
        except ImportError:
            log_error("requests module not available for Ollama communication")
            return {}

        # Ensure qwen-coder model is available
        model_name = "qwen-coder:7b"
        api_url = "http://localhost:11434/api/chat"

        try:
            log_debug(f"Ensuring {model_name} is available for HIGH/LOW classification")
            _shared_ensure_model_available(
                model_name,
                api_url,
                requests_module=requests,
                status_channel="lora_metadata_status",
            )
        except Exception as e:
            log_error(f"Failed to ensure {model_name} is available: {e}")
            raise Exception(
                f"Cannot classify LoRA pairs: {model_name} model is not available. "
                "Please install it with: ollama pull qwen-coder:7b"
            ) from e

        # Create structured prompt for classification
        system_prompt = create_ollama_classification_prompt()

        # Format pairs for Ollama
        pairs_text = "\n".join(
            [f"Pair {i + 1}: {pair[0]} vs {pair[1]}" for i, pair in enumerate(candidate_pairs)]
        )
        user_prompt = f"Classify these LoRA pairs:\n\n{pairs_text}"

        log_debug(f"Sending {len(candidate_pairs)} pairs to Ollama for HIGH/LOW classification")

        try:
            response = _shared_call_ollama_chat(
                system_prompt,
                user_prompt,
                model_name=model_name,
                api_url=api_url,
                timeout=60,
                requests_module=requests,
            )

            if not response:
                log_error("Ollama returned empty response for HIGH/LOW classification")
                return {}

            # Parse structured response
            classifications = parse_ollama_classification_response(response, candidate_pairs)
            log_debug(f"Successfully classified {len(classifications)} LoRA pairs using Ollama")
            return classifications

        except Exception as e:
            log_error(f"Error during Ollama HIGH/LOW classification: {e}")
            return {}

    except ImportError as e:
        log_error(f"Failed to import Ollama utilities: {e}")
        return {}


def create_ollama_classification_prompt() -> str:
    """Create the system prompt for Ollama HIGH/LOW classification."""
    return """You are a LoRA classification expert. Your task is to analyze LoRA model names and determine which one in each pair should be considered HIGH vs LOW for WAN 2.2 video generation.

HIGH LoRAs typically:
- Have "high", "H", "highnoise", or similar indicators in the name
- Are designed for high noise/strength applications
- May have version suffixes that suggest higher intensity

LOW LoRAs typically:
- Have "low", "L", "lownoise", or similar indicators in the name
- Are designed for low noise/strength applications
- May have version suffixes that suggest lower intensity

IMPORTANT: Base your classification ONLY on the LoRA names provided. Do not make assumptions about content or purpose beyond what the names clearly indicate.

For each pair, output a JSON object with this exact structure:
{
  "classifications": [
    {
      "pair_index": 1,
      "high_lora": "name_of_high_lora",
      "low_lora": "name_of_low_lora",
      "reasoning": "brief explanation of classification"
    }
  ]
}

Examples:

Input Pairs:
Pair 1: NSFW-22-H-e8 vs NSFW-22-L-e8
Pair 2: character_highnoise vs character_lownoise

Expected Output:
{
  "classifications": [
    {
      "pair_index": 1,
      "high_lora": "NSFW-22-H-e8",
      "low_lora": "NSFW-22-L-e8",
      "reasoning": "H indicator suggests high noise/strength"
    },
    {
      "pair_index": 2,
      "high_lora": "character_highnoise",
      "low_lora": "character_lownoise",
      "reasoning": "highnoise vs lownoise clearly indicates intensity levels"
    }
  ]
}"""


def parse_ollama_classification_response(
    response: str, candidate_pairs: list[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    """
    Parse the structured JSON response from Ollama classification.

    Args:
        response: Raw JSON response from Ollama
        candidate_pairs: Original list of candidate pairs for validation

    Returns:
        Dict mapping pair tuples to classification results
    """
    try:
        # Clean response - remove markdown code blocks if present
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        # Parse JSON
        data = json.loads(response_clean)

        if "classifications" not in data:
            log_error("Ollama response missing 'classifications' key")
            return {}

        classifications = {}

        for classification in data["classifications"]:
            pair_index = classification.get("pair_index", 0) - 1  # Convert to 0-based index

            if pair_index < 0 or pair_index >= len(candidate_pairs):
                log_warning(f"Invalid pair_index {pair_index + 1} in Ollama response")
                continue

            pair = candidate_pairs[pair_index]
            high_lora = classification.get("high_lora", "")
            low_lora = classification.get("low_lora", "")
            reasoning = classification.get("reasoning", "")

            # Validate that high/low LoRAs match the original pair
            if high_lora not in pair or low_lora not in pair:
                log_warning(f"Ollama classification doesn't match original pair: {pair}")
                continue

            if high_lora == low_lora:
                log_warning(f"Ollama classified both LoRAs the same: {high_lora}")
                continue

            classifications[pair] = {
                "high_lora": high_lora,
                "low_lora": low_lora,
                "reasoning": reasoning,
            }

            log_debug(
                f"Ollama classified {pair}: HIGH={high_lora}, LOW={low_lora}, Reason={reasoning}"
            )

        return classifications

    except json.JSONDecodeError as e:
        log_error(f"Failed to parse Ollama classification response as JSON: {e}")
        log_debug(f"Raw response: {response[:500]}...")
        return {}
    except Exception as e:
        log_error(f"Error parsing Ollama classification response: {e}")
        return {}
