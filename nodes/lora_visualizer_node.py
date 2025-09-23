"""
LoRA Visualizer Node Implementation
Parses prompts for LoRA tags and displays metadata, thumbnails, and example images.
"""

import time
from typing import Optional

import folder_paths

from .lora_metadata_utils import get_metadata_loader, parse_lora_tags


class LoRAVisualizerNode:
    """
    A ComfyUI node that visualizes LoRA metadata from prompt text.

    Features:
    - Parses standard LoRA tags: <lora:name:strength>
    - Parses custom wanlora tags: <wanlora:name:strength>
    - Shows trigger words, strength, thumbnails
    - Hover to view all example images
    """

    CATEGORY = "conditioning"
    DESCRIPTION = """Analyzes prompt text to extract and visualize LoRA information with metadata.

• Parses all LoRA tags: <lora:name:strength> and <wanlora:name:strength>
• Automatically detects WAN/video LoRAs by examining metadata
• Displays thumbnails, trigger words, and base models
• Shows scalable previews with hover galleries
• Supports both image and video LoRAs with appropriate visual indicators
• Requires ComfyUI LoRA Manager for metadata"""

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input schema for this ComfyUI node."""
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter your prompt with LoRA tags here...",
                        "tooltip": (
                            "Input text containing LoRA tags like <lora:MyLora:0.8> or "
                            "<wanlora:MyWanLora:1.0>. The node will automatically detect and "
                            "visualize all LoRA references with their metadata."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("raw_lora_info", "original_prompt")
    OUTPUT_TOOLTIPS = (
        (
            "Raw metadata information about detected LoRAs in a structured format "
            "for debugging and analysis."
        ),
        "The original prompt text passed through unchanged for downstream processing.",
    )
    FUNCTION = "visualize_loras"
    OUTPUT_NODE = True

    def __init__(self):
        self.loras_folder = (
            folder_paths.get_folder_paths("loras")[0]
            if folder_paths.get_folder_paths("loras")
            else None
        )
        self.metadata_loader = get_metadata_loader()

    def parse_lora_tags(self, prompt_text: str) -> tuple[list[dict], list[dict]]:
        """
        Parse LoRA tags from prompt text using shared parsing logic.

        Returns:
            Tuple of (standard_loras, wanloras) where each is a list of dicts
            containing name, strength, and type information.
        """
        return parse_lora_tags(prompt_text)

    def load_metadata(self, lora_name: str) -> Optional[dict]:
        """
        Load metadata for a LoRA from its .metadata.json file.
        Uses the shared metadata loader that searches recursively.

        Args:
            lora_name: Name of the LoRA (without extension)

        Returns:
            Dict containing metadata or None if not found
        """
        return self.metadata_loader.load_metadata(lora_name)

    def extract_lora_info(self, lora_data: dict, metadata: Optional[dict]) -> dict:
        """
        Extract relevant information from LoRA data and metadata.

        Args:
            lora_data: Parsed LoRA information (name, strength, type)
            metadata: Loaded metadata dict or None

        Returns:
            Dict with extracted information for display
        """
        info = {
            "name": lora_data["name"],
            "strength": lora_data["strength"],
            "type": lora_data["type"],
            "tag": lora_data["tag"],
            "trigger_words": [],
            "preview_url": None,
            "example_images": [],
            "model_description": None,
            "base_model": None,
            "nsfw_level": 0,
            "civitai_url": None,
        }

        if metadata:
            # Extract trigger words
            civitai_data = metadata.get("civitai")
            if civitai_data and "trainedWords" in civitai_data:
                info["trigger_words"] = civitai_data["trainedWords"]

            # Extract Civitai URL
            if civitai_data and "modelId" in civitai_data:
                model_id = civitai_data["modelId"]
                info["civitai_url"] = f"https://civitai.com/models/{model_id}"

            # Extract preview image
            if "preview_url" in metadata:
                info["preview_url"] = metadata["preview_url"]

            # Extract example images
            if civitai_data and "images" in civitai_data:
                info["example_images"] = [
                    {
                        "url": img["url"],
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                        "nsfw_level": img.get("nsfwLevel", 1),
                        "type": img.get(
                            "type", "video" if img["url"].endswith(".mp4") else "image"
                        ),
                        "meta": img.get("meta", {}),  # Include full metadata for prompts
                    }
                    for img in civitai_data["images"]
                ]

            # Extract model info
            if "model_name" in metadata:
                info["model_name"] = metadata["model_name"]
            if "modelDescription" in metadata:
                info["model_description"] = metadata["modelDescription"]
            if "base_model" in metadata:
                info["base_model"] = metadata["base_model"]
            if "preview_nsfw_level" in metadata:
                info["nsfw_level"] = metadata["preview_nsfw_level"]

        return info

    def format_lora_info(self, loras_info: list[dict], title: str) -> str:
        """
        Format LoRA information for display.

        Args:
            loras_info: List of LoRA info dicts
            title: Title for this section

        Returns:
            Formatted string for display
        """
        if not loras_info:
            return f"{title}: None found\n"

        result = f"{title} ({len(loras_info)} found):\n"
        result += "=" * 50 + "\n"

        for i, lora in enumerate(loras_info, 1):
            result += f"\n{i}. {lora['name']} (strength: {lora['strength']})\n"
            result += f"   Tag: {lora['tag']}\n"

            if lora["trigger_words"]:
                result += f"   Trigger words: {', '.join(lora['trigger_words'])}\n"
            else:
                result += "   Trigger words: Not available\n"

            if lora["base_model"]:
                result += f"   Base model: {lora['base_model']}\n"

            if lora["preview_url"]:
                result += "   Preview: Available\n"
            else:
                result += "   Preview: Not available\n"

            if lora["example_images"]:
                result += f"   Example images: {len(lora['example_images'])} available\n"
            else:
                result += "   Example images: Not available\n"

            result += "\n"

        return result

    def visualize_loras(self, prompt_text: str, **kwargs) -> tuple[str, str]:
        """
        Main function that processes the prompt and returns LoRA information.

        Args:
            prompt_text: Input prompt text containing LoRA tags

        Returns:
            Tuple of (raw_lora_info, original_prompt)
        """
        if not prompt_text.strip():
            return ("No prompt text provided.", prompt_text)

        # Parse LoRA tags from prompt
        standard_loras, wanloras = self.parse_lora_tags(prompt_text)

        # Debug logging
        # Note: Verbose debug logs removed to reduce noise

        if not standard_loras and not wanloras:
            return ("No LoRA tags found in prompt.", prompt_text)

        # Process standard LoRAs
        standard_loras_info = []
        for lora_data in standard_loras:
            metadata = self.load_metadata(lora_data["name"])
            info = self.extract_lora_info(lora_data, metadata)
            standard_loras_info.append(info)

        # Process wanloras
        wanloras_info = []
        for lora_data in wanloras:
            metadata = self.load_metadata(lora_data["name"])
            info = self.extract_lora_info(lora_data, metadata)
            wanloras_info.append(info)

        # Try to get ComfyUI node ID from kwargs (ComfyUI passes unique_id during execution)
        node_id = (
            kwargs.get("unique_id")
            or kwargs.get("node_id")
            or kwargs.get("extra_pnginfo", {}).get("workflow", {}).get("nodes", [{}])[0].get("id")
        )
        if not node_id:
            # Create a unique identifier based on object ID + timestamp
            node_id = f"{id(self)}_{int(time.time() * 1000)}"

        print(f"DEBUG: LoRA Visualizer node_id: {node_id}, available kwargs: {list(kwargs.keys())}")

        # Store visualization data for frontend access
        self.last_lora_data = {
            "standard_loras": standard_loras_info,
            "wanloras": wanloras_info,
            "prompt": prompt_text,
            "node_id": str(node_id),
        }

        # Send data to frontend via server message
        try:
            from server import PromptServer

            message_data = {"node_id": str(node_id), "data": self.last_lora_data}
            PromptServer.instance.send_sync("lora_visualization_data", message_data)
        except Exception as e:
            print(f"Failed to send LoRA visualization data: {e}")

        # Create raw metadata output for debugging/analysis
        raw_metadata = {
            "total_loras_found": len(standard_loras) + len(wanloras),
            "standard_loras_count": len(standard_loras),
            "wanloras_count": len(wanloras),
            "standard_loras": [
                {
                    "name": lora["name"],
                    "strength": lora["strength"],
                    "tag": lora["tag"],
                    "trigger_words": lora.get("trigger_words", []),
                    "base_model": lora.get("base_model", "Unknown"),
                    "civitai_url": lora.get("civitai_url"),
                    "preview_url": lora.get("preview_url", ""),
                    "example_images_count": len(lora.get("example_images", [])),
                    "example_images_sample": [
                        img.get("url", "") for img in lora.get("example_images", [])
                    ][:3],  # First 3 URLs for debugging
                    "has_metadata": bool(lora.get("preview_url") or lora.get("trigger_words")),
                }
                for lora in standard_loras_info
            ],
            "wanloras": [
                {
                    "name": lora["name"],
                    "strength": lora["strength"],
                    "tag": lora["tag"],
                    "trigger_words": lora.get("trigger_words", []),
                    "base_model": lora.get("base_model", "Unknown"),
                    "civitai_url": lora.get("civitai_url"),
                    "preview_url": lora.get("preview_url", ""),
                    "example_images_count": len(lora.get("example_images", [])),
                    "example_images_sample": [
                        img.get("url", "") for img in lora.get("example_images", [])
                    ][:3],  # First 3 URLs for debugging
                    "has_metadata": bool(lora.get("preview_url") or lora.get("trigger_words")),
                }
                for lora in wanloras_info
            ],
        }

        # Convert to readable string format
        import json as json_module

        raw_info_output = json_module.dumps(raw_metadata, indent=2, ensure_ascii=False)

        return (raw_info_output, prompt_text)
