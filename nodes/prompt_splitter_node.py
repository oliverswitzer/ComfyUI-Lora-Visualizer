"""
Prompt Splitter Node Implementation
-----------------------------------

This node takes a single natural language prompt (which may include
LoRA tags such as ``<lora:name:strength>`` or ``<wanlora:name:strength>``)
and splits it into two distinct prompts: one suitable for Stable
Diffusion XL (SDXL) image generation and another appropriate for WAN
video generation. The split is performed by delegating to an Ollama
model running locally via its chat API.

The default system prompt used for the Ollama call encodes the
recommendations developed with the user earlier in this conversation.
It instructs the model to preserve LoRA and WanLoRA tags exactly as
given, avoid duplicating sentences across the two prompts, and treat
the SDXL prompt as a single still frame while allowing the WAN prompt
to describe motion.  Example input/output is included for clarity.

Users may override the Ollama model name, API URL or system prompt at
runtime via optional inputs.  The node will always return a pair of
strings (image_prompt, wan_prompt) ready for downstream conditioning
nodes.

Note: This node requires the ``requests`` library at runtime.  If
``requests`` is not available in your Python environment, install it
with ``pip install requests`` or add it to your ComfyUI environment's
requirements file.
"""

import json
from typing import Tuple

# Import shared utilities for interacting with Ollama.  These helpers
# centralize model download and chat requests to avoid duplicating
# network logic across nodes.
from .ollama_utils import (
    ensure_model_available as _shared_ensure_model_available,
)  # noqa: E402
from .ollama_utils import call_ollama_chat as _shared_call_ollama_chat  # noqa: E402
from .logging_utils import log_error

try:
    import requests  # type: ignore[import]
except Exception:
    # ``requests`` may not be present in minimal test environments.  It
    # will be passed through to shared helpers if available.
    requests = None


class PromptSplitterNode:
    """Split a combined prompt into image and video components using Ollama."""

    CATEGORY = "conditioning"
    DESCRIPTION = """Splits a scene description into separate SDXL and WAN prompts.

    This node sends your prompt to a local Ollama model with a system
    prompt that explains how to construct a still image prompt and a
    corresponding video prompt.  It preserves any LoRA or WanLoRA tags
    and returns the two prompts as strings.  Optional inputs allow you
    to override the Ollama model, API URL and system prompt.
    """

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "wan_prompt")
    OUTPUT_NODE = True
    FUNCTION = "split_prompt"

    # Default values for optional parameters
    _DEFAULT_MODEL_NAME = "nous-hermes2"
    _DEFAULT_API_URL = "http://localhost:11434/api/chat"

    # Default model name used when the caller does not supply a model_name.
    # Users may override this via the model_name input.  Any valid Ollama
    # model name can be provided; there is no enforced allow list.

    # System prompt instructing Ollama how to split prompts.  The
    # indentation here is intentional; triple-quoted strings preserve
    # newlines and spaces which the model will see.
    _SYSTEM_PROMPT = """
You are a creative prompt engineer for a two-stage pipeline:
1) Stable Diffusion XL (SDXL) generates a still image,
2) WAN Image-to-Video (I2V) animates it.

You will receive a single free-form scene prompt (not guaranteed to be vetted).
Transform it into TWO coordinated outputs:

1) SDXL Prompt (first frame):
- Describe a single, detailed still moment that will be the FIRST FRAME of the video.
- Focus on subject(s), pose, apparel, background, composition, lighting, camera/framing.
- ALLOW keyword/CSV-style tokens (e.g., 'woman, 4k, red dress, ultra-detailed, rim light') and quality tokens ('masterpiece', 'best quality').
- Preserve <lora:name:strength> exactly as given, and preserve any provided trigger words for those LoRAs.
- NEVER include <wanlora:...> tags here — they belong only in the WAN prompt.
- If a motion verb is present, bias toward moving it to the WAN prompt unless the SDXL LoRA(s) clearly depict an action pose that requires it for context.
- Do not add descriptive elements, emotions, or modifiers that were not explicitly present in the input prompt.

2) WAN Prompt (motion):
- Describe a short, coherent motion sequence that naturally continues from the SDXL still.
- Written in clear, natural sentences (NOT keyword/CSV lists). Avoid quality/resolution tokens (e.g., 4k, HDR, masterpiece).
- Include camera motion and temporal phrasing if relevant.
- Preserve <lora:...> and <wanlora:...> exactly as given. All <wanlora:...> tags must appear here if they exist in the input.
- Avoid copying full sentences from the SDXL prompt; instead, expand the described moment into an immediate action.
- Do not inject emotions, intensifiers, or subjective tone words (e.g., 'passionately', 'joyfully') unless they were explicitly stated in the input.

Consistency & Constraints:
- Maintain subject count, genders, and relationships exactly as described.
- Do not add or remove characters, species, or props unless the input clearly implies them.
- If the input is ambiguous, choose a reasonable interpretation and still output both prompts.
- The SDXL prompt should stand alone as a great still; the WAN prompt should read as a natural continuation.
- Preserve any provided trigger words for LoRAs exactly. (Note: external logic in the calling system should identify these trigger words.)

Output format: valid JSON with keys 'sdxl_prompt' and 'wan_prompt'.

Examples:
Input Prompt: "woman, 4k, flowing red dress, rooftop party at night, string lights, cinematic, <lora:reddress:1.0> she starts to twirl under the lights <wanlora:dance:0.8>"
{
  "sdxl_prompt": "woman, 4k, flowing red dress, rooftop party at night, string lights, cinematic, shallow depth of field, relaxed stance, poised to move, <lora:reddress:1.0>",
  "wan_prompt": "She twirls beneath the string lights, the fabric of her dress sweeping outward as the camera slowly circles. <wanlora:dance:0.8>"
}

Input Prompt: "two girls and one boy, sunlit park picnic, casual outfits, golden hour, laughing together on a blanket, ultra-detailed"
{
  "sdxl_prompt": "two girls and one boy, sunlit park picnic, casual outfits, golden hour, laughing together on a blanket, ultra-detailed, soft rim light",
  "wan_prompt": "They lean in, share the phone between them, and burst into louder laughter as the boy nudges the snack bowl."
}

Input Prompt: "teen boy in leather jacket <lora:badboy:1.2> in a narrow alley, moody backlight, gritty texture, he moves toward a fight <wanlora:fightscene:0.8>"
{
  "sdxl_prompt": "teen boy in a leather jacket <lora:badboy:1.2>, narrow alley, moody backlight, gritty texture, intense expression, stance squared, fists lowered",
  "wan_prompt": "He cracks his knuckles and steps forward, shoulders tightening as he squares up, while the camera eases backward. <wanlora:fightscene:0.8>"
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
                        "placeholder": "Enter the full prompt describing your scene, "
                        "including any LoRA tags...",
                        "tooltip": (
                            "Combined prompt to be split into image and video prompts.\n"
                            "Include <lora:...> or <wanlora:...> tags if needed; "
                            "they will be preserved."
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
        containing ``sdxl_prompt`` and ``wan_prompt`` keys.  If any
        error occurs (including JSON parsing failure), empty strings are
        returned.
        """
        # Call the shared helper directly
        content = _shared_call_ollama_chat(
            system_prompt,
            prompt,
            model_name=model_name,
            api_url=api_url,
            timeout=60,
            requests_module=requests,
        )
        if not content:
            return "", ""
        try:
            result = json.loads(content.strip())
            sdxl_prompt = str(result.get("sdxl_prompt", ""))
            wan_prompt = str(result.get("wan_prompt", ""))
            return sdxl_prompt, wan_prompt
        except json.JSONDecodeError:
            # Content is not valid JSON; treat as failure
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
            return "", ""
        # Determine which model to use: the caller-supplied name or the default.
        model = model_name or self._DEFAULT_MODEL_NAME
        url = api_url or self._DEFAULT_API_URL
        sys_prompt = system_prompt if system_prompt else self._SYSTEM_PROMPT
        # Ensure the model is available before attempting to generate
        try:
            self._ensure_model_available(model, url)
        except Exception as e:
            log_error(f"Error ensuring model availability: {e}")
        # Try contacting Ollama first
        sdxl, wan = self._call_ollama(prompt_text, model, url, sys_prompt)
        if sdxl and wan:
            return sdxl, wan
        # If Ollama returned empty or invalid responses, do not attempt a
        # naive fallback.  Return empty strings to signal failure.
        return "", ""
