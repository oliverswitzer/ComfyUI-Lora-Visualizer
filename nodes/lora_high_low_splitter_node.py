"""
LoRA High/Low Splitter Node Implementation
-----------------------------------------

Simple node that takes a prompt with LoRA tags and splits it into two prompts:
- High prompt: Contains only HIGH/HN LoRA tags + base prompt text
- Low prompt: Contains only LOW/LN LoRA tags + base prompt text

This node extracts and reuses the high/low splitting logic from the existing
PromptSplitterNode, making it available as a standalone operation without
requiring Ollama or AI processing.

Features:
- Splits prompts based on LoRA tag names containing high/low/hn/ln
- Single LoRA tags (without high/low patterns) are included in both outputs
- Preserves original prompt structure and formatting
- Compatible with WAN 2.2 workflows and any other high/low LoRA usage
"""

from .logging_utils import log
from .lora_metadata_utils import split_prompt_by_lora_high_low


class LoRAHighLowSplitterNode:
    """Split a prompt into high and low LoRA variants for WAN 2.2 and similar workflows."""

    CATEGORY = "conditioning"
    DESCRIPTION = """Splits a prompt with LoRA tags into HIGH and LOW variants.

    Takes a prompt containing LoRA tags and creates two outputs:
    • High prompt: Contains only HIGH/HN LoRA tags + base text
    • Low prompt: Contains only LOW/LN LoRA tags + base text

    Perfect for WAN 2.2 workflows that require separate high and low noise prompts.
    LoRA tags without high/low/hn/ln in their name are included in both outputs.

    Supported patterns:
    • HIGH, high, HN, hn in LoRA names → High output
    • LOW, low, LN, ln in LoRA names → Low output
    • Other LoRA tags → Both outputs
    """

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("high_prompt", "low_prompt")
    OUTPUT_TOOLTIPS = (
        "Prompt with HIGH/HN LoRA tags only + single LoRA tags",
        "Prompt with LOW/LN LoRA tags only + single LoRA tags",
    )
    OUTPUT_NODE = True
    FUNCTION = "split_high_low"

    @classmethod
    def INPUT_TYPES(cls):
        """Define required inputs for this node."""
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "woman dancing <lora:style_high:0.8> <lora:motion_low:0.6> in garden",
                        "tooltip": (
                            "Input prompt with LoRA tags to be split.\n"
                            "LoRA tags with 'high/HIGH/hn/HN' → high_prompt output\n"
                            "LoRA tags with 'low/LOW/ln/LN' → low_prompt output\n"
                            "Other LoRA tags → included in both outputs"
                        ),
                    },
                ),
            },
        }

    def split_high_low(self, prompt_text: str) -> tuple[str, str]:
        """
        Split the prompt into high and low LoRA variants.

        Args:
            prompt_text: Input prompt with LoRA tags

        Returns:
            Tuple of (high_prompt, low_prompt)
        """
        if not prompt_text or not prompt_text.strip():
            log("LoRA High/Low Splitter: Empty input prompt, returning empty results")
            return "", ""

        log("LoRA High/Low Splitter: Splitting prompt by HIGH/LOW LoRA tags")

        # Use the shared splitting logic with HN/LN support
        high_prompt, low_prompt = split_prompt_by_lora_high_low(prompt_text)

        log("LoRA High/Low Splitter: Split completed")
        log(f"  High prompt length: {len(high_prompt)} characters")
        log(f"  Low prompt length: {len(low_prompt)} characters")

        return high_prompt.strip(), low_prompt.strip()
