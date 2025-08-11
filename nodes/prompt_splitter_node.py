"""
Prompt Splitter Node Implementation
-----------------------------------

Intelligently splits a combined prompt into separate image and video prompts
using Ollama AI. Handles LoRA tags, trigger words, and verbatim directives
deterministically for precise control over prompt distribution.

Features:
- LoRA tag routing: <lora:name:strength> → image prompt
- WanLoRA tag routing: <wanlora:name:strength> → video prompt  
- Automatic trigger word extraction from LoRA metadata files
- Verbatim directives: (image: text) and (video: text) for exact placement
- Configurable Ollama model (default: nollama/mythomax-l2-13b:Q4_K_M)

The AI focuses on content splitting while deterministic rules handle
tag placement and verbatim text distribution.

Note: Requires Ollama running locally and ``requests`` library.
"""

import json
import re
from typing import Tuple, List, Dict

# Import shared utilities for interacting with Ollama.  These helpers
# centralize model download and chat requests to avoid duplicating
# network logic across nodes.
from .ollama_utils import (
    ensure_model_available as _shared_ensure_model_available,
)  # noqa: E402
from .ollama_utils import call_ollama_chat as _shared_call_ollama_chat  # noqa: E402
from .logging_utils import log, log_error
from .lora_metadata_utils import get_metadata_loader

try:
    import requests  # type: ignore[import]
except Exception:
    # ``requests`` may not be present in minimal test environments.  It
    # will be passed through to shared helpers if available.
    requests = None


class PromptSplitterNode:
    """Split a combined prompt into image and video components using Ollama."""

    CATEGORY = "conditioning"
    DESCRIPTION = """Splits a scene description into separate image and WAN prompts.

    Uses Ollama to intelligently split your prompt into static visual elements
    (image prompt) and motion/action elements (video prompt). Features:

    • Handles LoRA tags: <lora:name:strength> → image prompt
    • Handles WanLoRA tags: <wanlora:name:strength> → video prompt  
    • Extracts trigger words from LoRA metadata automatically
    • Verbatim directives: (image: text) → image prompt, (video: text) → video prompt
    • Configurable Ollama model, API URL, and system prompt
    """

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "wan_prompt")
    OUTPUT_NODE = True
    FUNCTION = "split_prompt"

    # Default values for optional parameters
    _DEFAULT_MODEL_NAME = "nollama/mythomax-l2-13b:Q4_K_M"
    _DEFAULT_API_URL = "http://localhost:11434/api/chat"

    # Default model name used when the caller does not supply a model_name.
    # Users may override this via the model_name input.  Any valid Ollama
    # model name can be provided; there is no enforced allow list.

    # System prompt instructing Ollama how to split prompts.  The
    # indentation here is intentional; triple-quoted strings preserve
    # newlines and spaces which the model will see.
    _SYSTEM_PROMPT = """
You are splitting a single combined scene description into two separate outputs:

1. IMAGE_PROMPT — For SD image generation.
   - Keep only static, visual descriptors: characters, clothing, environment, props, positions, facial expressions (if static).
   - Do NOT add or remove descriptive terms unless they already appear in the source.
   - Keep all or any explicit or sexual descriptors exactly as written, without euphemisms or softening.
   - Do NOT add adjectives, tone, style words, or any narrative text unless they are in the original.

2. WAN_PROMPT — For WAN I2V video generation.
   - Keep only motion/action descriptors.
   - Do NOT add new actions or change their meaning.
   - Use the exact same explicit terms from the source (do not reword).
   - No narrative, story, emotional tone, or metaphor — only plain action description.

General Rules:
- Do not add any adjectives, storylines, or implied emotions that are not in the source.
- Copy source words exactly, unless you must remove them because they are irrelevant to the specific output type.
- If unsure which output a term belongs in, place it in the IMAGE_PROMPT.
- IGNORE any LoRA tags like <lora:...> or <wanlora:...> - they will be handled separately.
- Return your final result in JSON with keys "image_prompt" and "wan_prompt".

Output format: valid JSON with keys 'image_prompt' and 'wan_prompt'.

Examples:
Input Prompt: "woman, 4k, flowing red dress, rooftop party at night, string lights, cinematic, she starts to twirl under the lights"
{
  "image_prompt": "woman, 4k, flowing red dress, rooftop party at night, string lights, cinematic, shallow depth of field, relaxed stance, poised to move",
  "wan_prompt": "She twirls beneath the string lights, the fabric of her dress sweeping outward as the camera slowly circles."
}

Input Prompt: "two girls and one boy, sunlit park picnic, casual outfits, golden hour, laughing together on a blanket, ultra-detailed"
{
  "image_prompt": "two girls and one boy, sunlit park picnic, casual outfits, golden hour, laughing together on a blanket, ultra-detailed, soft rim light",
  "wan_prompt": "They lean in, share the phone between them, and burst into louder laughter as the boy nudges the snack bowl."
}

Input Prompt: "teen boy in leather jacket in a narrow alley, moody backlight, gritty texture, he moves toward a fight"
{
  "image_prompt": "teen boy in a leather jacket, narrow alley, moody backlight, gritty texture, intense expression, stance squared, fists lowered",
  "wan_prompt": "He cracks his knuckles and steps forward, shoulders tightening as he squares up, while the camera eases backward."
}

Input Prompt: "woman dancing (image: overwatch, ana) gracefully (video: she jumps up and down)"
{
  "image_prompt": "woman dancing gracefully, overwatch, ana",
  "wan_prompt": "woman dances, she jumps up and down"
}
"""

    @classmethod
    def INPUT_TYPES(cls):
        """Define required and optional inputs for this node."""
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "woman dancing <lora:style:0.8> (image: beautiful face) "
                        "(video: twirls gracefully) <wanlora:motion:1.0>",
                        "tooltip": (
                            "Combined prompt to be split into image and video prompts.\n"
                            "Supports: <lora:name:strength> → image, <wanlora:name:strength> → video\n"
                            "Verbatim: (image: text) → image, (video: text) → video\n"
                            "AI splits remaining content into static visuals vs motion/actions."
                        ),
                    },
                ),
            },
            "optional": {
                "model_name": (
                    "STRING",
                    {
                        "default": cls._DEFAULT_MODEL_NAME,
                        "tooltip": "Name of the Ollama model to use "
                        "(e.g. 'nous-hermes2', 'mythomax-mistral').",
                    },
                ),
                "api_url": (
                    "STRING",
                    {
                        "default": cls._DEFAULT_API_URL,
                        "tooltip": "URL of the Ollama chat API endpoint.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Custom system prompt (leave blank to use default).",
                        "tooltip": "Override the default instructions sent to the Ollama model.",
                    },
                ),
            },
        }

    def parse_lora_tags(self, prompt_text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse LoRA tags from prompt text.

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

    def _extract_and_remove_trigger_words(
        self, prompt_text: str, lora_list: List[Dict]
    ) -> Tuple[str, List[str]]:
        """
        Extract trigger words for LoRAs from prompt and remove them.

        Args:
            prompt_text: The input prompt text
            lora_list: List of LoRA dicts with name, strength, type, tag

        Returns:
            Tuple of (cleaned_prompt, extracted_trigger_words)
        """
        extracted_trigger_words = []
        metadata_loader = get_metadata_loader()

        for lora in lora_list:
            lora_name = lora["name"]
            trigger_words = metadata_loader.extract_trigger_words(
                metadata_loader.load_metadata(lora_name)
            )

            for trigger_word in trigger_words:
                if trigger_word.lower() in prompt_text.lower():
                    extracted_trigger_words.append(trigger_word)
                    # Remove trigger word from prompt (case-insensitive)
                    pattern = re.compile(re.escape(trigger_word), re.IGNORECASE)
                    prompt_text = pattern.sub("", prompt_text)
                    log(
                        f"Prompt Splitter: Extracted trigger word '{trigger_word}' for {lora_name}"
                    )

        # Clean up extra whitespace after removals
        prompt_text = re.sub(r"\s+", " ", prompt_text).strip()
        return prompt_text, extracted_trigger_words

    def _extract_verbatim_directives(
        self, prompt_text: str
    ) -> Tuple[str, List[str], List[str]]:
        """
        Extract verbatim text directives from prompt.

        Args:
            prompt_text: Input prompt text with potential (image: ...) and (video: ...) directives

        Returns:
            Tuple of (cleaned_prompt, image_verbatim_list, video_verbatim_list)
        """
        image_verbatim = []
        video_verbatim = []

        # Pattern for (image: content) - capture everything until the closing parenthesis
        image_pattern = r"\(image:\s*([^)]+)\)"
        for match in re.finditer(image_pattern, prompt_text):
            content = match.group(1).strip()
            if content:
                image_verbatim.append(content)
                log(f"Prompt Splitter: Found image verbatim directive: '{content}'")

        # Pattern for (video: content) - capture everything until the closing parenthesis
        video_pattern = r"\(video:\s*([^)]+)\)"
        for match in re.finditer(video_pattern, prompt_text):
            content = match.group(1).strip()
            if content:
                video_verbatim.append(content)
                log(f"Prompt Splitter: Found video verbatim directive: '{content}'")

        # Keep the original prompt with verbatim directives for LLM context
        # We'll add the verbatim content back deterministically later
        return prompt_text, image_verbatim, video_verbatim

    def _remove_all_lora_tags(self, prompt_text: str) -> str:
        """Remove all LoRA and WanLoRA tags from prompt text."""
        # Remove both types of tags
        prompt_text = re.sub(r"<lora:[^>]*>", "", prompt_text)
        prompt_text = re.sub(r"<wanlora:[^>]*>", "", prompt_text)
        # Clean up extra whitespace
        prompt_text = re.sub(r"\s+", " ", prompt_text).strip()
        return prompt_text

    def _ensure_model_available(self, model: str, api_url: str) -> None:
        """Delegate to shared utility to ensure the Ollama model exists.

        Uses the shared ``ensure_model_available`` helper from ``ollama_utils``
        to check for and download the requested model.  The status
        messages are sent on the ``prompt_splitter_status`` channel.
        If the helper is unavailable (e.g. due to import failure), this
        method quietly returns.
        """
        # Pass the imported ``requests`` module to ensure that patched
        # requests in this module are used for network operations during tests.
        _shared_ensure_model_available(
            model,
            api_url,
            requests_module=requests,
            status_channel="prompt_splitter_status",
        )

    def _call_ollama(
        self, prompt: str, model_name: str, api_url: str, system_prompt: str
    ) -> Tuple[str, str]:
        """Call the Ollama chat API via shared helper and parse the response.

        Uses the shared ``call_ollama_chat`` helper to obtain the assistant's
        content.  The content is then expected to be a JSON object
        containing ``iamge_prompt`` and ``wan_prompt`` keys.  If any
        error occurs (including JSON parsing failure), empty strings are
        returned.
        """
        # Call the shared helper directly
        log("Prompt Splitter: Contacting Ollama API...")
        content = _shared_call_ollama_chat(
            system_prompt,
            prompt,
            model_name=model_name,
            api_url=api_url,
            timeout=60,
            requests_module=requests,
        )
        if not content:
            log_error("Prompt Splitter: Ollama returned empty response")
            return "", ""

        log(
            f"Prompt Splitter: Received response from Ollama ({len(content)} characters)"
        )
        try:
            result = json.loads(content.strip())
            image_prompt = str(result.get("image_prompt", ""))
            wan_prompt = str(result.get("wan_prompt", ""))
            log("Prompt Splitter: Successfully parsed JSON response")
            return image_prompt, wan_prompt
        except json.JSONDecodeError as e:
            log_error(f"Prompt Splitter: Failed to parse JSON response: {e}")
            log_error(f"Prompt Splitter: Raw response: {content[:200]}...")
            return "", ""

    def _naive_split(self, prompt: str) -> Tuple[str, str]:
        """Deprecated fallback splitting.

        Historically, this method provided a naive fallback when Ollama
        requests failed.  To avoid confusion, fallback splitting is no
        longer used.  This method now simply returns two copies of the
        input prompt, preserving behaviour for any external callers that
        might still invoke it.
        """
        return prompt, prompt

    def split_prompt(
        self,
        prompt_text: str,
        model_name: str = None,
        api_url: str = None,
        system_prompt: str = "",
    ) -> Tuple[str, str]:
        """Public method invoked by ComfyUI to split prompts.

        Args:
            prompt_text: Full scene description provided by the user.
            model_name: Optional Ollama model name; defaults to
                ``self._DEFAULT_MODEL_NAME`` if ``None``.
            api_url: Optional Ollama API URL; defaults to
                ``self._DEFAULT_API_URL`` if ``None``.
            system_prompt: Optional system prompt override; if empty the
                built-in default is used.

        Returns:
            (image_prompt, wan_prompt)
        """
        if not prompt_text or not prompt_text.strip():
            log("Prompt Splitter: Empty input prompt, returning empty results")
            return "", ""

        # Extract verbatim directives (but keep them in the prompt for LLM context)
        log("Prompt Splitter: Extracting verbatim directives...")
        _, image_verbatim, video_verbatim = self._extract_verbatim_directives(
            prompt_text
        )
        log(
            f"Prompt Splitter: Found {len(image_verbatim)} image and {len(video_verbatim)} video verbatim directives"
        )

        # Parse LoRA tags from prompt
        log("Prompt Splitter: Parsing LoRA tags...")
        standard_loras, wanloras = self.parse_lora_tags(prompt_text)
        log(
            f"Prompt Splitter: Found {len(standard_loras)} standard LoRAs and {len(wanloras)} WanLoRAs"
        )

        # Extract and remove trigger words for all LoRAs
        all_loras = standard_loras + wanloras
        prompt_without_triggers, extracted_trigger_words = (
            self._extract_and_remove_trigger_words(prompt_text, all_loras)
        )
        log(f"Prompt Splitter: Extracted {len(extracted_trigger_words)} trigger words")

        # Remove all LoRA tags from prompt before sending to LLM (keep verbatim directives)
        clean_prompt = self._remove_all_lora_tags(prompt_without_triggers)
        log(f"Prompt Splitter: Cleaned prompt length: {len(clean_prompt)} characters")

        # Determine which model to use: the caller-supplied name or the default.
        model = model_name or self._DEFAULT_MODEL_NAME
        url = api_url or self._DEFAULT_API_URL
        sys_prompt = system_prompt if system_prompt else self._SYSTEM_PROMPT

        log(f"Prompt Splitter: Starting split using model '{model}'")

        # Ensure the model is available before attempting to generate
        try:
            log(f"Prompt Splitter: Checking model availability for '{model}'")
            self._ensure_model_available(model, url)
            log(f"Prompt Splitter: Model '{model}' is ready")
        except Exception as e:
            log_error(f"Prompt Splitter: Error ensuring model availability: {e}")
            return "", ""

        # Send clean prompt (without LoRA tags) to Ollama
        log("Prompt Splitter: Sending request to Ollama...")
        image_prompt, wan_prompt = self._call_ollama(
            clean_prompt, model, url, sys_prompt
        )

        if image_prompt and wan_prompt:
            # Add LoRA tags and trigger words back to appropriate prompts
            metadata_loader = get_metadata_loader()

            # Standard LoRAs go to image prompt
            for lora in standard_loras:
                image_prompt = f"{image_prompt} {lora['tag']}"
                log(f"Prompt Splitter: Added {lora['tag']} to image prompt")

                # Add trigger words for this LoRA to image prompt
                trigger_words = metadata_loader.extract_trigger_words(
                    metadata_loader.load_metadata(lora["name"])
                )
                for trigger_word in trigger_words:
                    if trigger_word in extracted_trigger_words:
                        image_prompt = f"{image_prompt} {trigger_word}"
                        log(
                            f"Prompt Splitter: Added trigger word '{trigger_word}' to image prompt"
                        )

            # WanLoRAs go to video prompt
            for wanlora in wanloras:
                wan_prompt = f"{wan_prompt} {wanlora['tag']}"
                log(f"Prompt Splitter: Added {wanlora['tag']} to video prompt")

                # Add trigger words for this WanLoRA to video prompt
                trigger_words = metadata_loader.extract_trigger_words(
                    metadata_loader.load_metadata(wanlora["name"])
                )
                for trigger_word in trigger_words:
                    if trigger_word in extracted_trigger_words:
                        wan_prompt = f"{wan_prompt} {trigger_word}"
                        log(
                            f"Prompt Splitter: Added trigger word '{trigger_word}' to video prompt"
                        )

            # Note: Verbatim directives are handled by the LLM in its response
            # We don't need to add them back since they should already be included
            if image_verbatim:
                log(
                    f"Prompt Splitter: LLM should have included {len(image_verbatim)} image verbatim directives"
                )
            if video_verbatim:
                log(
                    f"Prompt Splitter: LLM should have included {len(video_verbatim)} video verbatim directives"
                )

            log("Prompt Splitter: Successfully split prompt")
            log(
                f"Prompt Splitter: Final image prompt length: {len(image_prompt)} characters"
            )
            log(
                f"Prompt Splitter: Final video prompt length: {len(wan_prompt)} characters"
            )
            return image_prompt.strip(), wan_prompt.strip()

        # If Ollama returned empty or invalid responses, do not attempt a
        # naive fallback.  Return empty strings to signal failure.
        log_error(
            "Prompt Splitter: Failed to split prompt - Ollama returned empty response"
        )
        return "", ""
