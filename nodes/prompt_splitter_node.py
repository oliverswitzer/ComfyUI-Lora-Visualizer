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

from .logging_utils import log, log_debug, log_error
from .lora_metadata_utils import (
    extract_example_prompts,
    extract_model_description,
    get_metadata_loader,
    parse_lora_tags,
)
from .ollama_utils import call_ollama_chat as _shared_call_ollama_chat  # noqa: E402

# Import shared utilities for interacting with Ollama.  These helpers
# centralize model download and chat requests to avoid duplicating
# network logic across nodes.
from .ollama_utils import (
    ensure_model_available as _shared_ensure_model_available,
)  # noqa: E402

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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "wan_prompt", "lora_analysis")
    OUTPUT_TOOLTIPS = (
        "Image prompt with LoRA tags, trigger words, and static visual elements",
        "Video prompt with WanLoRA tags, trigger words, and motion/action elements",
        "JSON analysis of LoRAs used and their examples that were fed to the LLM",
    )
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

💡 Best Practices for IMAGE_PROMPT (SD image generation):
- Use a keyword/CSV-style format, separated by commas.
- Keep descriptors short, concrete, and visual — e.g., "woman, 4k, red dress, rooftop at night, string lights".
- Preserve ordering so the main subject comes first, followed by descriptors, environment, style cues.
- Avoid writing full sentences or sequential events — this is a still image, not a scene description.
- Keep detail density high (pose, clothing, props, lighting) but ONLY from source text.
- Do NOT insert adjectives or style tags unless they already appear in the input.

---

1. IMAGE_PROMPT — For SD image generation.
   - Keep only static, visual descriptors: characters, clothing, environment, props, positions, facial expressions (if static).
   - Do NOT add or remove descriptive terms unless they already appear in the source.
   - Keep all descriptors exactly as written, without euphemisms or softening.
   - Do NOT add adjectives, tone, style words, or any narrative text unless they are in the original.

2. WAN_PROMPT — For WAN I2V video generation.
   - Keep only motion/action descriptors.
   - Do NOT add new actions or change their meaning.
   - Use the exact same terms from the source (do not reword).
   - No narrative, story, emotional tone, or metaphor — only plain action description.

General Rules:
- Do not add any adjectives, storylines, or implied emotions that are not in the source.
- Copy source words exactly, unless you must remove them because they are irrelevant to the specific output type.
- If unsure which output a term belongs in, place it in the IMAGE_PROMPT.
- IGNORE any LoRA tags like <lora:...> or <wanlora:...> - they will be handled separately.
- IGNORE verbatim directive content - it has been extracted and will be added back separately.
- Return your final result in JSON with keys "image_prompt" and "wan_prompt".

Output format: valid JSON with keys 'image_prompt' and 'wan_prompt'.

Examples:

Input Prompt: "woman, 4k, flowing red dress, rooftop party at night, string lights, cinematic, she starts to twirl under the lights"
{
  "image_prompt": "woman, 4k, flowing red dress, rooftop party at night, string lights, cinematic, shallow depth of field, relaxed stance, poised to move",
  "wan_prompt": "the woman twirls slowly under the string lights, her dress moving with the spin"
}

Input Prompt: "two girls and one boy, sunlit park picnic, casual outfits, golden hour, laughing together on a blanket, ultra-detailed"
{
  "image_prompt": "two girls and one boy, sunlit park picnic, casual outfits, golden hour, laughing together on a blanket, ultra-detailed, soft rim light",
  "wan_prompt": "they lean in closer on the blanket, laughing and talking together in the sun"
}

Input Prompt: "teen boy in leather jacket in a narrow alley, moody backlight, gritty texture, he moves toward a fight"
{
  "image_prompt": "teen boy in a leather jacket, narrow alley, moody backlight, gritty texture, intense expression, stance squared, fists lowered",
  "wan_prompt": "he cracks his knuckles and steps toward the other person, ready to fight"
}

Input Prompt: "woman dancing overwatch, ana gracefully she jumps up and down"
{
  "image_prompt": "woman dancing gracefully",
  "wan_prompt": "the woman dances, then jumps up and down several times"
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
                            "Supports: <lora:name:strength> → image, "
                            "<wanlora:name:strength> → video\n"
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

    def parse_lora_tags(self, prompt_text: str) -> tuple[list[dict], list[dict]]:
        """
        Parse LoRA tags from prompt text using shared parsing logic.

        Returns:
            Tuple of (standard_loras, wanloras) where each is a list of dicts
            containing name, strength, type, and tag information.
        """
        return parse_lora_tags(prompt_text)

    def _extract_and_remove_trigger_words(
        self, prompt_text: str, lora_list: list[dict]
    ) -> tuple[str, list[str]]:
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
                    log(f"Prompt Splitter: Extracted trigger word '{trigger_word}' for {lora_name}")

        # Clean up extra whitespace after removals
        prompt_text = re.sub(r"\s+", " ", prompt_text).strip()
        return prompt_text, extracted_trigger_words

    def _extract_lora_examples(
        self, loras: list[dict[str, str]]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """
        Extract example prompts and descriptions from LoRA metadata to ground LLM behavior.

        Args:
            loras: List of LoRA dictionaries with 'name' and 'tag' keys

        Returns:
            Tuple of (lora_examples dict, lora_descriptions dict) mapping LoRA names to their data
        """
        metadata_loader = get_metadata_loader()
        lora_examples = {}
        lora_descriptions = {}

        for lora in loras:
            lora_name = lora["name"]
            try:
                metadata = metadata_loader.load_metadata(lora_name)
                if not metadata:
                    continue

                # Extract and process examples
                self._process_lora_examples(metadata, lora_name, lora_examples)
                # Extract model description
                self._process_lora_description(metadata, lora_name, lora_descriptions)

            except Exception as e:
                log_debug(f"Prompt Splitter: Could not extract data for LoRA '{lora_name}': {e}")
                continue

        return lora_examples, lora_descriptions

    def _process_lora_examples(self, metadata: dict, lora_name: str, lora_examples: dict):
        """Process and extract examples for a LoRA."""
        examples = extract_example_prompts(metadata, limit=3)
        if not examples:
            return

        # Clean examples: remove any existing LoRA tags and limit length
        cleaned_examples = []
        for example in examples:
            cleaned = self._clean_example(example)
            cleaned_examples.append(cleaned)

        if cleaned_examples:
            lora_examples[lora_name] = cleaned_examples
            log_debug(f"Prompt Splitter: Found {len(cleaned_examples)} examples for '{lora_name}'")

    def _clean_example(self, example: str) -> str:
        """Clean an example prompt by removing LoRA tags and truncating if needed."""
        # Remove LoRA tags from examples
        cleaned = re.sub(r"<(?:lora|wanlora):[^>]+>", "", example)
        cleaned = " ".join(cleaned.split())  # Remove extra whitespace
        # Truncate length to keep prompt manageable
        if len(cleaned) > 500:
            cleaned = cleaned[:500].rstrip()
        return cleaned

    def _process_lora_description(self, metadata: dict, lora_name: str, lora_descriptions: dict):
        """Process and extract description for a LoRA."""
        description = extract_model_description(metadata)
        if description:
            lora_descriptions[lora_name] = description
            log_debug(
                f"Prompt Splitter: Found description for '{lora_name}': {len(description)} chars"
            )

    def _create_contextualized_system_prompt(
        self, lora_examples: dict[str, list[str]], lora_descriptions: dict[str, str]
    ) -> str:
        """
        Create a system prompt with LoRA examples and descriptions to ground the LLM.

        Args:
            lora_examples: Dict mapping LoRA names to their example prompts
            lora_descriptions: Dict mapping LoRA names to their model descriptions

        Returns:
            Enhanced system prompt with LoRA context
        """
        base_prompt = self._SYSTEM_PROMPT

        # If no examples or descriptions found, return base prompt
        if not lora_examples and not lora_descriptions:
            return base_prompt

        # Build LoRA context section
        context_section = "\n\n--- LoRA CONTEXT ---\n"
        context_section += (
            "Use this LoRA information to understand their intended use and style:\n\n"
        )

        # Include descriptions first
        if lora_descriptions:
            for lora_name, description in lora_descriptions.items():
                context_section += f"'{lora_name}' LoRA:\n"
                truncated_desc = description[:300] + ("..." if len(description) > 300 else "")
                context_section += f"  Purpose: {truncated_desc}\n\n"

        # Include examples
        if lora_examples:
            context_section += "--- LoRA USAGE EXAMPLES ---\n"
            for lora_name, examples in lora_examples.items():
                context_section += f"'{lora_name}' examples:\n"
                for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples per LoRA
                    context_section += f"  {i}. {example}\n"
                context_section += "\n"

        context_section += (
            "IMPORTANT: When splitting prompts, maintain the same style, terminology, "
            "and content patterns shown in these examples and descriptions. "
            "Avoid adding creative flourishes or story elements not present in the "
            "original input or this context.\n"
        )

        # Insert context section before the final examples in the base prompt
        insertion_point = base_prompt.find("Examples:")
        if insertion_point != -1:
            enhanced_prompt = (
                base_prompt[:insertion_point] + context_section + base_prompt[insertion_point:]
            )
        else:
            # Fallback: append to end
            enhanced_prompt = base_prompt + context_section

        return enhanced_prompt

    def _extract_verbatim_directives(self, prompt_text: str) -> tuple[str, list[str], list[str]]:
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
                log_debug(f"Prompt Splitter: Found image verbatim directive: '{content}'")

        # Pattern for (video: content) - capture everything until the closing parenthesis
        video_pattern = r"\(video:\s*([^)]+)\)"
        for match in re.finditer(video_pattern, prompt_text):
            content = match.group(1).strip()
            if content:
                video_verbatim.append(content)
                log_debug(f"Prompt Splitter: Found video verbatim directive: '{content}'")

        # Remove the wrapper syntax but keep the content for LLM context
        # Replace (image: content) with just content
        prompt_without_wrappers = re.sub(image_pattern, r"\1", prompt_text)
        prompt_without_wrappers = re.sub(video_pattern, r"\1", prompt_without_wrappers)

        # Clean up extra whitespace
        prompt_without_wrappers = re.sub(r"\s+", " ", prompt_without_wrappers).strip()

        return prompt_without_wrappers, image_verbatim, video_verbatim

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
    ) -> tuple[str, str]:
        """Call the Ollama chat API via shared helper and parse the response.

        Uses the shared ``call_ollama_chat`` helper to obtain the assistant's
        content.  The content is then expected to be a JSON object
        containing ``iamge_prompt`` and ``wan_prompt`` keys.  If any
        error occurs (including JSON parsing failure), empty strings are
        returned.
        """
        # Call the shared helper directly
        log("Prompt Splitter: Contacting Ollama API...")
        try:
            content = _shared_call_ollama_chat(
                system_prompt,
                prompt,
                model_name=model_name,
                api_url=api_url,
                timeout=60,
                requests_module=requests,
            )
        except Exception as e:
            log_error(f"Prompt Splitter: Error calling Ollama: {e}")
            if "Connection" in str(e) or "refused" in str(e):
                raise Exception(
                    "Cannot connect to Ollama. Please ensure Ollama is installed and running. "
                    "Run 'ollama serve' in a terminal and try again. "
                    "Visit https://ollama.ai for installation instructions."
                ) from e
            raise Exception(
                f"Ollama API error: {e}. Check your Ollama configuration and try again."
            ) from e

        if not content:
            log_error("Prompt Splitter: Ollama returned empty response")
            raise Exception(
                "Ollama returned empty response. The AI model may be overloaded "
                "or experiencing issues."
            )

        log(f"Prompt Splitter: Received response from Ollama ({len(content)} characters)")

        # Check for common Ollama error patterns before attempting to parse
        content_lower = content.lower()
        if any(
            error_pattern in content_lower
            for error_pattern in [
                "error",
                "failed",
                "downloading",
                "pulling",
                "not found",
                "invalid",
                "connection refused",
                "timeout",
                "404",
                "500",
                "502",
                "503",
                "manifest unknown",
            ]
        ):
            log_error(f"Prompt Splitter: Ollama returned error response: {content[:300]}...")
            if "downloading" in content_lower or "pulling" in content_lower:
                raise Exception(
                    "Model is being downloaded by Ollama. Please wait for the download to complete "
                    "and try again. This may take several minutes depending on model size."
                ) from None
            elif "not found" in content_lower or "manifest unknown" in content_lower:
                raise Exception(
                    "Model not found in Ollama. Please ensure the model is available or "
                    "use a different model name. You may need to run: ollama pull <model-name>"
                ) from None
            else:
                raise Exception(
                    f"Ollama API error: {content[:200]}... "
                    f"Check your Ollama configuration and model availability."
                ) from None

        try:
            result = json.loads(content.strip())
            image_prompt = str(result.get("image_prompt", ""))
            wan_prompt = str(result.get("wan_prompt", ""))
            log("Prompt Splitter: Successfully parsed JSON response")
            return image_prompt, wan_prompt
        except json.JSONDecodeError as e:
            log_debug(f"Prompt Splitter: Failed to parse JSON response: {e}")
            log_debug(f"Prompt Splitter: Raw response: {content[:200]}...")

            # Fallback: try to parse plain text format
            log("Prompt Splitter: Attempting fallback parsing for plain text format...")
            image_prompt, wan_prompt = self._parse_plain_text_response(content)

            if image_prompt or wan_prompt:
                log("Prompt Splitter: Successfully parsed plain text response")
                return image_prompt, wan_prompt
            else:
                log_error("Prompt Splitter: Could not parse response in any format")
                log_error(f"Prompt Splitter: Full response content: {content}")
                raise Exception(
                    "Invalid response from AI model. The AI model returned malformed "
                    "data that could not be parsed as JSON or plain text. "
                    f"Full response: {content[:200]}... \n\n"
                    "This usually indicates the model doesn't understand the prompt format. "
                    "The 'nollama/mythomax-l2-13b:Q4_K_M' model may not be compatible. "
                    "Try these better options:\n"
                    "• llama3.1:8b (recommended)\n"
                    "• llama3.2:3b (faster)\n"
                    "• qwen2.5:7b (good at instructions)\n\n"
                    "Run: ollama pull llama3.1:8b"
                ) from None

    def _parse_plain_text_response(self, content: str) -> tuple[str, str]:
        """
        Parse plain text response that contains IMAGE_PROMPT: and WAN_PROMPT: sections.

        Args:
            content: Raw text response from LLM

        Returns:
            Tuple of (image_prompt, wan_prompt)
        """
        image_prompt = ""
        wan_prompt = ""

        # Split the content into lines for processing
        lines = content.strip().split("\n")
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()

            # Check for section headers
            if line.startswith("IMAGE_PROMPT:"):
                # Save previous section
                if current_section == "WAN_PROMPT":
                    wan_prompt = " ".join(current_content).strip()
                elif current_section == "IMAGE_PROMPT":
                    image_prompt = " ".join(current_content).strip()

                # Start new section
                current_section = "IMAGE_PROMPT"
                current_content = []
                # Get content after the colon
                after_colon = line[len("IMAGE_PROMPT:") :].strip()
                if after_colon:
                    current_content.append(after_colon)

            elif line.startswith("WAN_PROMPT:"):
                # Save previous section
                if current_section == "IMAGE_PROMPT":
                    image_prompt = " ".join(current_content).strip()
                elif current_section == "WAN_PROMPT":
                    wan_prompt = " ".join(current_content).strip()

                # Start new section
                current_section = "WAN_PROMPT"
                current_content = []
                # Get content after the colon
                after_colon = line[len("WAN_PROMPT:") :].strip()
                if after_colon:
                    current_content.append(after_colon)

            elif current_section and line:
                # Add to current section if we're in one
                current_content.append(line)

        # Save the last section
        if current_section == "IMAGE_PROMPT":
            image_prompt = " ".join(current_content).strip()
        elif current_section == "WAN_PROMPT":
            wan_prompt = " ".join(current_content).strip()

        # If that didn't work, try a more flexible approach
        if not image_prompt and not wan_prompt:
            log_debug("Prompt Splitter: Trying flexible parsing for non-standard format...")

            # Look for any mention of "image" and "video/wan" in the response
            content_lower = content.lower()

            # Try to extract anything that looks like it could be prompts
            # This is a fallback for models that don't follow the exact format
            if "for the image" in content_lower or "image prompt" in content_lower:
                # Try to extract text after "image" mentions
                import re

                image_patterns = [
                    r"image[:\s]+([^\n]+)",
                    r"for the image[:\s]+([^\n]+)",
                    r"image prompt[:\s]+([^\n]+)",
                ]
                for pattern in image_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        image_prompt = match.group(1).strip()
                        break

            if (
                "for the video" in content_lower
                or "wan prompt" in content_lower
                or "video prompt" in content_lower
            ):
                # Try to extract text after "video/wan" mentions
                video_patterns = [
                    r"video[:\s]+([^\n]+)",
                    r"for the video[:\s]+([^\n]+)",
                    r"wan prompt[:\s]+([^\n]+)",
                    r"video prompt[:\s]+([^\n]+)",
                ]
                for pattern in video_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        wan_prompt = match.group(1).strip()
                        break

            # Last resort: if the response is relatively short and looks like a prompt,
            # assume it's an image prompt
            if not image_prompt and not wan_prompt and len(content.strip()) < 200:
                # If it looks like a prompt (comma-separated, descriptive), use it as image prompt
                if "," in content and not content.count("\n") > 3:
                    image_prompt = content.strip()
                    log_debug("Prompt Splitter: Using entire response as image prompt (fallback)")

        log_debug(
            f"Parsed plain text - Image: {len(image_prompt)} chars, WAN: {len(wan_prompt)} chars"
        )
        return image_prompt, wan_prompt

    def _send_progress_update(self, progress: float, message: str) -> None:
        """Send progress update to ComfyUI's standard progress system.

        Args:
            progress: Progress value between 0.0 and 1.0
            message: Status message to display
        """
        try:
            from server import PromptServer

            # Use the standard ComfyUI progress format
            progress_data = {"node": str(id(self)), "value": progress, "max": 1.0}
            PromptServer.instance.send_sync("progress", progress_data)

            # Also log to console for debugging
            log(f"Progress {int(progress * 100)}%: {message}")

        except Exception as e:
            # Don't fail the entire operation if progress update fails
            log_error(f"Failed to send progress update: {e}")

    def _naive_split(self, prompt: str) -> tuple[str, str]:
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
    ) -> tuple[str, str, str]:
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
            (image_prompt, wan_prompt, lora_analysis)
        """
        if not prompt_text or not prompt_text.strip():
            log("Prompt Splitter: Empty input prompt, returning empty results")
            return "", "", "{}"

        # Send initial progress update
        self._send_progress_update(0.1, "Starting prompt analysis...")

        # Extract verbatim directives and remove wrapper syntax for LLM context
        log("Prompt Splitter: Extracting verbatim directives...")
        prompt_without_wrappers, image_verbatim, video_verbatim = self._extract_verbatim_directives(
            prompt_text
        )
        log(
            f"Prompt Splitter: Found {len(image_verbatim)} image and "
            f"{len(video_verbatim)} video verbatim directives"
        )

        self._send_progress_update(0.2, "Parsing LoRA tags and trigger words...")

        # Parse LoRA tags from original prompt (before wrapper removal)
        log("Prompt Splitter: Parsing LoRA tags...")
        standard_loras, wanloras = self.parse_lora_tags(prompt_text)
        log(
            f"Prompt Splitter: Found {len(standard_loras)} standard LoRAs and "
            f"{len(wanloras)} WanLoRAs"
        )

        # Extract and remove trigger words for all LoRAs (use prompt without wrappers)
        all_loras = standard_loras + wanloras
        prompt_without_triggers, extracted_trigger_words = self._extract_and_remove_trigger_words(
            prompt_without_wrappers, all_loras
        )
        log_debug(f"Prompt Splitter: Extracted {len(extracted_trigger_words)} trigger words")

        # Remove all LoRA tags from prompt before sending to LLM
        clean_prompt = self._remove_all_lora_tags(prompt_without_triggers)
        log_debug(f"Prompt Splitter: Cleaned prompt length: {len(clean_prompt)} characters")

        # Extract LoRA examples and descriptions to ground LLM behavior
        log_debug("Prompt Splitter: Extracting LoRA examples and descriptions for context...")
        lora_examples, lora_descriptions = self._extract_lora_examples(all_loras)

        # Determine which model to use: the caller-supplied name or the default.
        model = model_name or self._DEFAULT_MODEL_NAME
        url = api_url or self._DEFAULT_API_URL

        # Create contextualized system prompt with LoRA examples and descriptions
        if system_prompt:
            # User provided custom system prompt, use as-is
            sys_prompt = system_prompt
        else:
            # Use our enhanced system prompt with LoRA context
            sys_prompt = self._create_contextualized_system_prompt(lora_examples, lora_descriptions)
            if lora_examples or lora_descriptions:
                total_examples = sum(len(examples) for examples in lora_examples.values())
                total_descriptions = len(lora_descriptions)
                log(
                    f"Prompt Splitter: Enhanced system prompt with {total_examples} examples "
                    f"and {total_descriptions} descriptions from LoRAs"
                )

        log(f"Prompt Splitter: Starting split using model '{model}'")
        self._send_progress_update(0.3, f"Checking Ollama model '{model}'...")

        # Ensure the model is available before attempting to generate
        try:
            log_debug(f"Prompt Splitter: Checking model availability for '{model}'")
            self._ensure_model_available(model, url)
            log_debug(f"Prompt Splitter: Model '{model}' is ready")
        except Exception as e:
            log_error(f"Prompt Splitter: Error ensuring model availability: {e}")
            if "Connection" in str(e) or "refused" in str(e):
                raise Exception(
                    "Cannot connect to Ollama. Please ensure Ollama is installed and running. "
                    "Try these steps:\n"
                    "1. Install Ollama from https://ollama.ai\n"
                    "2. Start Ollama: 'ollama serve'\n"
                    "3. Pull your model: 'ollama pull nollama/mythomax-l2-13b:Q4_K_M'\n"
                    "4. Verify it's running: 'ollama list'"
                ) from e
            raise Exception(
                f"Model availability error: {e}. Check that Ollama is properly "
                f"configured and the model name is correct."
            ) from e

        self._send_progress_update(0.6, "Generating prompt splits with AI...")

        # Send clean prompt (without LoRA tags) to Ollama
        log("Prompt Splitter: Sending request to Ollama...")
        image_prompt, wan_prompt = self._call_ollama(clean_prompt, model, url, sys_prompt)

        if image_prompt and wan_prompt:
            self._send_progress_update(0.8, "Reassembling prompts with LoRA tags...")

            # Add LoRA tags and trigger words back to appropriate prompts
            metadata_loader = get_metadata_loader()

            # Standard LoRAs go to image prompt
            for lora in standard_loras:
                image_prompt = f"{image_prompt} {lora['tag']}"
                log_debug(f"Prompt Splitter: Added {lora['tag']} to image prompt")

                # Add trigger words for this LoRA to image prompt
                trigger_words = metadata_loader.extract_trigger_words(
                    metadata_loader.load_metadata(lora["name"])
                )
                for trigger_word in trigger_words:
                    if trigger_word in extracted_trigger_words:
                        image_prompt = f"{image_prompt} {trigger_word}"
                        log(f"Prompt Splitter: Added trigger word '{trigger_word}' to image prompt")

            # WanLoRAs go to video prompt
            for wanlora in wanloras:
                lora_tag = re.sub(r"<wanlora:([^>]+)>", r"<lora:\1>", wanlora["tag"])
                wan_prompt = f"{wan_prompt} {lora_tag}"
                log_debug(
                    f"Prompt Splitter: Added {lora_tag} to video prompt (converted from WanLoRA tag back to LoRA tag)"
                )

                # Add trigger words for this WanLoRA to video prompt
                trigger_words = metadata_loader.extract_trigger_words(
                    metadata_loader.load_metadata(wanlora["name"])
                )
                for trigger_word in trigger_words:
                    if trigger_word in extracted_trigger_words:
                        wan_prompt = f"{wan_prompt} {trigger_word}"
                        log(f"Prompt Splitter: Added trigger word '{trigger_word}' to video prompt")

            # Add verbatim directives back to appropriate prompts deterministically
            for verbatim in image_verbatim:
                image_prompt = f"{image_prompt} {verbatim}"
                log_debug(f"Prompt Splitter: Added image verbatim: '{verbatim}'")

            for verbatim in video_verbatim:
                wan_prompt = f"{wan_prompt} {verbatim}"
                log_debug(f"Prompt Splitter: Added video verbatim: '{verbatim}'")

            self._send_progress_update(1.0, "Prompt splitting completed!")

            # Create LoRA analysis output
            analysis_data = {
                "scene_description": clean_prompt,
                "loras_used": {
                    "image_loras": [
                        {
                            "name": lora["name"],
                            "tag": lora["tag"],
                            "examples_fed_to_llm": lora_examples.get(lora["name"], []),
                            "trigger_words": (
                                metadata_loader.extract_trigger_words(
                                    metadata_loader.load_metadata(lora["name"])
                                )
                                if metadata_loader.load_metadata(lora["name"])
                                else []
                            ),
                        }
                        for lora in standard_loras
                    ],
                    "video_loras": [
                        {
                            "name": lora["name"],
                            "tag": lora["tag"],
                            "examples_fed_to_llm": lora_examples.get(lora["name"], []),
                            "trigger_words": (
                                metadata_loader.extract_trigger_words(
                                    metadata_loader.load_metadata(lora["name"])
                                )
                                if metadata_loader.load_metadata(lora["name"])
                                else []
                            ),
                        }
                        for lora in wanloras
                    ],
                },
                "total_examples_used": sum(len(examples) for examples in lora_examples.values()),
                "verbatim_directives": {
                    "image_verbatim": image_verbatim,
                    "video_verbatim": video_verbatim,
                },
                "model_used": model,
                "processing_successful": True,
            }

            analysis_output = json.dumps(analysis_data, indent=2, ensure_ascii=False)

            log("Prompt Splitter: Successfully split prompt")
            log(f"Prompt Splitter: Final image prompt length: {len(image_prompt)} characters")
            log(f"Prompt Splitter: Final video prompt length: {len(wan_prompt)} characters")
            return image_prompt.strip(), wan_prompt.strip(), analysis_output

        # If Ollama returned empty or invalid responses, do not attempt a
        # naive fallback.  Return empty strings to signal failure.
        log_error("Prompt Splitter: Failed to split prompt - Ollama returned empty response")

        # Create error analysis output
        json.dumps(
            {
                "error": "AI model returned empty response",
                "processing_successful": False,
            }
        )

        raise Exception(
            "AI model returned empty response. The AI model may be overloaded or "
            "experiencing issues. Try again or use a different model."
        )
