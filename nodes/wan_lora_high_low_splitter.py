"""
WAN LoRA High/Low Splitter Node Implementation
---------------------------------------------

Uses Ollama to determine which LoRA tags are HIGH and which are LOW,
then uses regex to split the prompt accordingly.

Features:
- LLM analyzes LoRA tags to classify them as HIGH or LOW
- High prompt: original prompt with only HIGH LoRA tags
- Low prompt: original prompt with only LOW LoRA tags
- Simple regex-based splitting after LLM classification
"""

import json
import re

from .lora_metadata_utils import find_lora_relative_path, parse_lora_tag
from .ollama_utils import call_ollama_chat as _shared_call_ollama_chat
from .ollama_utils import (
    ensure_model_available as _shared_ensure_model_available,
)

try:
    import requests  # type: ignore[import]
except Exception:
    requests = None


class WANLoRAHighLowSplitterNode:
    """Split a prompt into WAN 2.2 HIGH and LOW specific prompts using Ollama."""

    CATEGORY = "conditioning"
    DESCRIPTION = """Splits a prompt containing WAN 2.2 LoRA tags into HIGH and LOW variants.

    Uses Ollama to intelligently split a prompt containing mixed HIGH/LOW LoRA tags:

    • High prompt: Original prompt with only HIGH LoRA tags
    • Low prompt: Original prompt with only LOW LoRA tags
    • Uses qwen2.5-coder:7b model for structured output
    """

    # Default configuration
    _DEFAULT_MODEL_NAME = "qwen2.5-coder:7b"
    _DEFAULT_API_URL = "http://localhost:11434/api/chat"

    # Simple system prompt for LoRA tag classification
    _SYSTEM_PROMPT = """
You are analyzing LoRA tags in a prompt to classify them as HIGH or LOW for WAN 2.2.

Given a list of LoRA tags from a prompt, classify each as either "HIGH" or "LOW" based on the tag name.
Tags with "HIGH" in the name should be classified as HIGH.
Tags with "LOW" in the name should be classified as LOW.
Tags without HIGH/LOW in the name should be classified as "SINGLE" (will be included in both outputs).

Return JSON with this structure:
{{
  "high_tags": ["<lora:example_HIGH:0.8>", "<lora:example_2_H:0.8>", "<lora:example_3_HN:0.8>"],
  "low_tags": ["<lora:example_LOW:0.6>", "<lora:example_2_L:0.6>", "<lora:example_3_LN:0.6>"],
  "single_tags": ["<lora:other:1.0>"]
}}

Lora names often match very closely, with one word or even an abbreviated few characters being different. Those differences indicate which LoRA is HIGH and which is LOW.
You will see "HN" and "LN" sometimes, indicating that the LoRA is a HIGH NOISE (HN) or LOW NOISE (LN) version of the same LoRA.
There may be other variations of that naming convention such as H/L, or different casings of high/low.
Please use your best judgment to classify the LoRA tags based on that information.

Possible LoRA scenarios for high and low tag names:
- "<lora:style_HIGH>" and "<lora:style_LOW>"
--> {{
    "high_tags": ["<lora:style_HIGH>"],
    "low_tags": ["<lora:style_LOW>"]
}}
- "<lora:example_H>" and "<lora:example_L>"
--> {{
    "high_tags": ["<lora:example_H>"],
    "low_tags": ["<lora:example_L>"]
}}
- "<lora:some-lora-WAN22-T2V-HN>" and "<lora:some-lora-WAN22-T2V-LN>"
--> {{
    "high_tags": ["<lora:some-lora-WAN22-T2V-HN>"],
    "low_tags": ["<lora:some-lora-WAN22-T2V-LN>"],
}}
- "<lora:noise_I2V_HN>" and "<lora:noise_I2V_LN>"
--> {{
    "high_tags": ["<lora:noise_I2V_HN>"],
    "low_tags": ["<lora:noise_I2V_LN>"]
}}
- "<lora:noise_Wan22_HN>" and "<lora:noise_Wan22_LN>"
--> {{
    "high_tags": ["<lora:noise_Wan22_HN>"],
    "low_tags": ["<lora:noise_Wan22_LN>"]
}}
- "<lora:effect_High>" and "<lora:effect_Low>"
--> {{
    "high_tags": ["<lora:effect_High>"],
    "low_tags": ["<lora:effect_Low>"]
}}
- "<lora:pattern_h>" and "<lora:pattern_l>"
--> {{
    "high_tags": ["<lora:pattern_h>"],
    "low_tags": ["<lora:pattern_l>"]
}}
- "<lora:texture_H>" and "<lora:texture_L>"
--> {{
    "high_tags": ["<lora:texture_H>"],
    "low_tags": ["<lora:texture_L>"]
}}

Other acronyms you may see in the LoRA names but can ignore:
- T2V = Text to Video
- I2V = Image to Video
- Wan22 = WAN 2.2
"""

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("high_prompt", "low_prompt", "analysis")
    OUTPUT_TOOLTIPS = (
        "High noise prompt with HIGH LoRA tags for WAN 2.2",
        "Low noise prompt with LOW LoRA tags for WAN 2.2",
        "JSON analysis of LoRA tag distribution",
    )
    OUTPUT_NODE = True
    FUNCTION = "split_wan_prompt"

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
                        "placeholder": "woman dancing <lora:style_HIGH:0.8> detailed face <lora:style_LOW:0.6>",
                        "tooltip": (
                            "Prompt containing WAN 2.2 HIGH/LOW LoRA tag pairs.\n"
                            "HIGH tags: <lora:name_HIGH:strength> → high noise prompt\n"
                            "LOW tags: <lora:name_LOW:strength> → low noise prompt\n"
                            "Single LoRAs (no HIGH/LOW) → included in both prompts"
                        ),
                    },
                ),
            },
        }

    def _extract_lora_tags(self, prompt_text: str) -> list[str]:
        """Extract all LoRA tags from the prompt."""
        lora_pattern = r"<lora:[^>]+>"
        return re.findall(lora_pattern, prompt_text)

    def _classify_lora_tags_with_llm(self, lora_tags: list[str]) -> dict:
        """Use LLM to classify LoRA tags into HIGH, LOW categories."""
        lora_tags_text = "\n".join(lora_tags)

        content = _shared_call_ollama_chat(
            self._SYSTEM_PROMPT,
            f"Classify these LoRA tags:\n{lora_tags_text}",
            model_name=self._DEFAULT_MODEL_NAME,
            api_url=self._DEFAULT_API_URL,
            timeout=30,
            requests_module=requests,
        )

        # Clean and parse JSON
        content_clean = content.strip().replace("```json", "").replace("```", "")
        result = json.loads(content_clean)
        return {
            "high_tags": result.get("high_tags", []),
            "low_tags": result.get("low_tags", []),
        }

    def split_wan_prompt(self, prompt_text: str) -> tuple[str, str, str]:
        """Split a WAN 2.2 prompt into HIGH and LOW variants."""
        # Extract LoRA tags
        lora_tags = self._extract_lora_tags(prompt_text)

        # Use LLM to classify HIGH/LOW tags
        _shared_ensure_model_available(
            self._DEFAULT_MODEL_NAME, self._DEFAULT_API_URL, requests_module=requests
        )
        classification = self._classify_lora_tags_with_llm(lora_tags)

        # Remove ALL LoRA tags to get base prompt
        base_prompt = prompt_text
        for tag in lora_tags:
            base_prompt = base_prompt.replace(tag, "")
        base_prompt = re.sub(r"\s+", " ", base_prompt).strip()

        # Build prompts
        high_prompt = f"{base_prompt} {' '.join(classification['high_tags'])}".strip()
        low_prompt = f"{base_prompt} {' '.join(classification['low_tags'])}".strip()

        # Create enhanced analysis with detailed LoRA information
        analysis = self._create_detailed_analysis(
            classification["high_tags"], classification["low_tags"], base_prompt
        )
        return high_prompt, low_prompt, json.dumps(analysis)

    def _create_detailed_analysis(
        self, high_tags: list[str], low_tags: list[str], base_prompt: str
    ) -> dict:
        """Create detailed analysis with LoRA paths and strengths."""
        analysis = {"prompt_no_lora_tags": base_prompt}

        # Process HIGH tags
        for i, tag in enumerate(high_tags, 1):
            lora_info = parse_lora_tag(tag)
            rel_path = find_lora_relative_path(lora_info["name"])

            analysis[f"high_lora_{i}"] = {
                "tag": tag,
                "strength": lora_info["strength"],
                "rel_path": rel_path,
            }

        # Process LOW tags
        for i, tag in enumerate(low_tags, 1):
            lora_info = parse_lora_tag(tag)
            rel_path = find_lora_relative_path(lora_info["name"])

            analysis[f"low_lora_{i}"] = {
                "tag": tag,
                "strength": lora_info["strength"],
                "rel_path": rel_path,
            }

        return analysis
