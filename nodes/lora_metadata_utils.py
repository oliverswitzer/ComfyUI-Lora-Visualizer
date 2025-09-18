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


def split_prompt_by_lora_high_low(prompt_text: str) -> tuple[str, str]:
    """
    Split a prompt into HIGH and LOW specific versions based on LoRA tags.

    This function extracts the existing high/low splitting logic from PromptSplitterNode
    to make it reusable across different nodes.

    High LoRA tags contain: "high", "HIGH", "hn", "HN" in their name
    Low LoRA tags contain: "low", "LOW", "ln", "LN" in their name

    Args:
        prompt_text: Input prompt with LoRA tags

    Returns:
        Tuple of (high_prompt, low_prompt) where:
        - high_prompt: Contains ONLY HIGH lora tags + base prompt text
        - low_prompt: Contains ONLY LOW lora tags + base prompt text
        - Base prompt text excludes all LoRA tags
        - Single LoRA tags (without high/low/hn/ln) are included in both outputs
    """
    log_debug(f"Splitting prompt by HIGH/LOW LoRA tags: '{prompt_text}'")

    base_prompt_parts = []
    high_lora_tags = []
    low_lora_tags = []
    single_lora_tags = []

    # Split the prompt by lora tags and regular content
    parts = re.split(r"(<lora:[^>]+>)", prompt_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("<lora:") and part.endswith(">"):
            # This is a lora tag - classify it by name
            tag_content = part[6:-1]  # Remove <lora: and >
            tag_name = tag_content.split(":")[0]  # Get name before first colon

            tag_name_lower = tag_name.lower()
            log_debug(f"Checking LoRA tag name: '{tag_name}' (lowercase: '{tag_name_lower}')")

            # Enhanced matching for HIGH LoRA tags: high, HIGH, hn, HN
            # Use word boundaries to avoid false positives like "highlight" matching "high"
            if any(
                pattern in tag_name_lower.replace("_", " ").replace("-", " ").split()
                for pattern in ["high", "hn"]
            ):
                high_lora_tags.append(part)
                log_debug(f"Classified as HIGH LoRA: {part}")
            # Enhanced matching for LOW LoRA tags: low, LOW, ln, LN
            # Use word boundaries to avoid false positives like "lowlight" matching "low"
            elif any(
                pattern in tag_name_lower.replace("_", " ").replace("-", " ").split()
                for pattern in ["low", "ln"]
            ):
                low_lora_tags.append(part)
                log_debug(f"Classified as LOW LoRA: {part}")
            else:
                # Single lora (not part of HIGH/LOW pair) - include in both outputs
                single_lora_tags.append(part)
                log_debug(f"Classified as single LoRA (included in both): {part}")
        else:
            # This is regular prompt content (text, not LoRA tags)
            base_prompt_parts.append(part)

    # Build the base prompt (text content only, no LoRA tags)
    base_prompt = " ".join(base_prompt_parts).strip()

    # Build HIGH prompt: base + HIGH lora tags + single lora tags
    high_prompt_parts = [base_prompt] + high_lora_tags + single_lora_tags
    high_prompt = " ".join(part for part in high_prompt_parts if part.strip())

    # Build LOW prompt: base + LOW lora tags + single lora tags
    low_prompt_parts = [base_prompt] + low_lora_tags + single_lora_tags
    low_prompt = " ".join(part for part in low_prompt_parts if part.strip())

    log_debug(f"HIGH prompt: '{high_prompt}'")
    log_debug(f"LOW prompt: '{low_prompt}'")

    log(
        f"LoRA split - HIGH tags: {len(high_lora_tags)}, "
        f"LOW tags: {len(low_lora_tags)}, "
        f"Single tags: {len(single_lora_tags)}"
    )

    return high_prompt, low_prompt


def find_lora_pairs_in_prompt_with_ollama(
    prompt_text: str,
    model_name: str = "qwen2.5-coder:7b",
    api_url: str = "http://localhost:11434/api/chat",
) -> str:
    """
    Find and pair HIGH/LOW LoRA tags in a prompt using Ollama for intelligent matching.

    This function extracts LoRA tags from the prompt, uses Ollama to intelligently find
    their HIGH/LOW pairs from available LoRAs, and stitches the results back into the
    prompt deterministically.

    Args:
        prompt_text: Input prompt containing LoRA tags
        model_name: Ollama model to use for analysis
        api_url: Ollama API URL

    Returns:
        Modified prompt with paired LoRA tags added
    """
    import re
    from .ollama_utils import call_ollama_chat, ensure_model_available

    log_debug(f"Finding LoRA pairs in prompt with Ollama")

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

    # Find HIGH/LOW LoRAs that need pairing
    high_low_loras_in_prompt = []
    for lora_tag in lora_matches:
        lora_name = lora_tag.split(":")[0]  # Get name before weight
        if any(pattern in lora_name.lower() for pattern in ["high", "low", "hn", "ln"]):
            high_low_loras_in_prompt.append(lora_name)

    if not high_low_loras_in_prompt:
        log_debug("  No HIGH/LOW/HN/LN LoRAs found in prompt")
        return prompt_text

    log_debug(f"  Found HIGH/LOW LoRAs to pair: {high_low_loras_in_prompt}")

    # Filter available LoRAs to only potential pairs
    potential_pairs = [
        name
        for name in available_lora_names
        if any(pattern in name.lower() for pattern in ["high", "low", "hn", "ln"])
    ]

    if not potential_pairs:
        log_debug("  No potential HIGH/LOW pairs available on system")
        return prompt_text

    # Use Ollama to find pairs
    pairs_to_add = []

    for lora_name in high_low_loras_in_prompt:
        try:
            # Create a focused prompt for this specific LoRA
            system_prompt = """You are an expert at LoRA (Low-Rank Adaptation) pairing for AI generation.

Find the best matching HIGH/LOW or HN/LN pair for the given LoRA from the available list.

Rules:
- HIGH pairs with LOW (and vice versa)
- HN pairs with LN (and vice versa)
- Look for exact name matches except for the high/low/hn/ln part
- Respond with ONLY the LoRA name, or "NONE" if no suitable pair exists

Examples:
- "character_high" pairs with "character_low"
- "style_hn" pairs with "style_ln"
- "Wan22-I2V-HIGH-Fantasy" pairs with "Wan22-I2V-LOW-Fantasy"
"""

            candidates_list = "\n".join(
                [f"- {name}" for name in potential_pairs if name != lora_name]
            )

            user_prompt = f"""Find the best pair for: {lora_name}

Available LoRAs:
{candidates_list}

Best pair name:"""

            # Ensure model is available
            ensure_model_available(model_name, api_url, status_channel="lora_pairing_status")

            # Call Ollama
            response = call_ollama_chat(
                system_prompt,
                user_prompt,
                model_name=model_name,
                api_url=api_url,
                timeout=15,
            )

            if not response or response.strip().upper() == "NONE":
                log_debug(f"    Ollama found no pair for {lora_name}")
                continue

            # Clean and validate response
            pair_name = response.strip()

            # Try to match response to available LoRA names
            if pair_name in potential_pairs and pair_name != lora_name:
                # Check if this pair is not already in the prompt
                if not any(pair_name in tag for tag in lora_matches):
                    pairs_to_add.append(pair_name)
                    log(f"    Ollama found pair: {lora_name} <-> {pair_name}")
                else:
                    log_debug(f"    Pair {pair_name} already in prompt, skipping")
            else:
                # Try fallback matching
                fallback_pair = _fallback_string_pairing(lora_name, potential_pairs)
                if fallback_pair and not any(fallback_pair in tag for tag in lora_matches):
                    pairs_to_add.append(fallback_pair)
                    log(f"    Fallback pair found: {lora_name} <-> {fallback_pair}")

        except Exception as e:
            log_error(f"    Error finding pair for {lora_name}: {e}")
            # Try fallback
            fallback_pair = _fallback_string_pairing(lora_name, potential_pairs)
            if fallback_pair and not any(fallback_pair in tag for tag in lora_matches):
                pairs_to_add.append(fallback_pair)
                log(f"    Fallback pair found: {lora_name} <-> {fallback_pair}")

    # Add paired LoRAs to the prompt deterministically
    if pairs_to_add:
        log(f"Adding {len(pairs_to_add)} paired LoRAs to prompt: {pairs_to_add}")

        # Insert new LoRA tags at the end of existing LoRA tags in the prompt
        # This keeps the prompt structure deterministic
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


def _fallback_string_pairing(lora_name: str, candidate_names: list[str]) -> Optional[str]:
    """
    Fallback string-based pairing when Ollama is unavailable.

    This is a simplified version of the old logic for emergency use only.
    """
    log_debug(f"Using fallback string pairing for: {lora_name}")

    name_lower = lora_name.lower()

    # Simple high/low replacement with word boundary awareness
    # to avoid false positives like "highlight" -> "lowlight"
    def try_replacement(original: str, find_word: str, replace_word: str) -> Optional[str]:
        """Try to replace a word while preserving case and avoiding false positives."""
        import re

        # Use word boundaries to avoid substring matches
        words = original.replace("_", " ").replace("-", " ").split()

        for i, word in enumerate(words):
            if word.lower() == find_word.lower():
                # Preserve case pattern
                if word.isupper():
                    replacement = replace_word.upper()
                elif word.istitle():
                    replacement = replace_word.capitalize()
                else:
                    replacement = replace_word.lower()

                words[i] = replacement
                # Reconstruct with original separators
                result = original
                for orig_word, new_word in zip(
                    original.replace("_", " ").replace("-", " ").split(), words
                ):
                    result = result.replace(orig_word, new_word, 1)
                return result

        return None

    # Try each pattern
    patterns = [
        ("high", "low"),
        ("low", "high"),
        ("hn", "ln"),
        ("ln", "hn"),
    ]

    for find_word, replace_word in patterns:
        target = try_replacement(lora_name, find_word, replace_word)
        if target and target in candidate_names:
            log(f"Fallback pairing found: {lora_name} <-> {target}")
            return target

    return None


# Keep the old function name for backward compatibility but mark it as deprecated
def find_lora_high_low_pair(lora_name: str, available_lora_names: list[str]) -> Optional[str]:
    """
    DEPRECATED: Use find_lora_high_low_pair_with_ollama() instead.

    This function is kept for backward compatibility but will use simple string matching.
    """
    log_debug(
        "DEPRECATED: Using find_lora_high_low_pair (string-based). Consider upgrading to Ollama-based pairing."
    )
    return _fallback_string_pairing(lora_name, available_lora_names)
