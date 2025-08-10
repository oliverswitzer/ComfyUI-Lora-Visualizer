"""
LoRA Prompt Composer Node Implementation
--------------------------------------

This node inspects all installed LoRA metadata files and, with the help
of a locally running Ollama model, composes a single creative prompt
that combines multiple image‐generation LoRAs and video (WAN) LoRAs
together.  It is designed to automate the pairing of LoRAs and to
include their trigger words and tags directly in the generated prompt.

Features:

* Scans the ``models/loras`` folder for ``*.metadata.json`` files.
* Classifies each LoRA as an image LoRA or a video LoRA based on its
  ``base_model`` field (video LoRAs typically have a base model that
  contains ``"wan"`` or ``"video"``).
* Extracts trigger words from the ``trainedWords`` list in the
  metadata; if no trained words are available, falls back to the
  filename as a trigger.
* Passes a summary of available LoRAs to an Ollama model, asking it to
  select up to N video LoRAs and up to M image LoRAs and to return a
  single prompt string that uses all chosen LoRAs.  The model is
  instructed to include both the LoRA trigger words and the LoRA tags
  (<wanvideo:name:1.0> or <lora:name:1.0>) in the prompt.
* Automatically downloads the chosen Ollama model if it is not present
  locally via the ``/api/pull`` endpoint, emitting status messages to
  the ComfyUI frontend.

The node exposes two integer inputs to control how many video and
image LoRAs should be combined.  Additional optional inputs allow the
user to specify the Ollama model, API URL, and system prompt.  The
default system prompt outlines how the model should behave.

Note: This node requires the ``requests`` library at runtime.  If
``requests`` is not available in your Python environment, install it
with ``pip install requests`` or add it to your ComfyUI environment's
requirements file.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Optional, Tuple

# Attempt to import requests so we can pass it through to shared
# helpers.  If requests is not available, it will remain None and
# shared helpers will fall back to their own imports.
try:
    import requests  # type: ignore[import]
except Exception:
    requests = None

try:
    # Import shared utilities for Ollama interactions
    from .ollama_utils import ensure_model_available, send_chat  # type: ignore
except Exception:
    ensure_model_available = None  # type: ignore
    send_chat = None  # type: ignore

# ComfyUI imports are optional for testing.  When running under
# pytest these modules are patched/mocked before import.
try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None

try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None


class LoRAPromptComposerNode:
    """Compose a combined prompt from available LoRAs using Ollama."""

    CATEGORY = "conditioning"
    DESCRIPTION = """Composes a scene prompt that uses multiple image and video LoRAs.

    This node examines all installed LoRA metadata files, extracts
    trigger words, and delegates to a local Ollama model to choose
    compatible LoRAs and craft a single prompt.  The resulting prompt
    contains both <lora:...> and <wanvideo:...> tags with strength 1.0
    and includes the trigger words for each selected LoRA.  Users can
    configure how many video and image LoRAs to include as well as
    override the model, API URL, and system prompt.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_NODE = True
    FUNCTION = "compose_prompt"

    # Default configuration values.  These can be overridden via
    # optional inputs on the node.  The default model should be an
    # instruction‐tuned model capable of following the system prompt.
    _DEFAULT_MODEL_NAME = "nous-hermes2"
    _DEFAULT_API_URL = "http://localhost:11434/api/chat"

    # System prompt that instructs the LLM to select and combine LoRAs.
    # It explains the format of the lists provided in the user message
    # and how to produce the final prompt.
    _SYSTEM_PROMPT = (
        "You are an expert prompt engineer for generative media models.\n\n"
        "You will receive two lists of LoRAs: `video_loras` and `image_loras`.\n"
        "Each entry has a `name`, a `trigger`, and a list of `tags`.\n"
        "You will also receive two integers `max_video_count` and `max_image_count`.\n"
        "Your task is to choose up to `max_video_count` video LoRAs and up to\n"
        "`max_image_count` image LoRAs that complement each other thematically\n"
        "(for example, similar subjects or styles based on the tags).  Then\n"
        "compose a single scene prompt that incorporates all selected LoRAs.\n"
        "The prompt should be written in natural language, describing a\n"
        "coherent scene or action that uses the trigger words of each\n"
        "selected LoRA.  For each selected video LoRA include a tag of\n"
        "the form `<wanvideo:NAME:1.0>` and for each selected image LoRA\n"
        "include a tag of the form `<lora:NAME:1.0>`.  Replace NAME with\n"
        "the exact value of the `name` field from the LoRA entry.\n"
        "Place each tag somewhere in the prompt so that it flows naturally.\n"
        "Also include the trigger words themselves in the prompt.\n"
        "Do not explain your reasoning.  Output only the final prompt as\n"
        "plain text with no JSON and no extra commentary."
    )

    def __init__(self) -> None:
        # Determine the loras folder via ComfyUI's folder_paths helper.
        # When running outside of ComfyUI (e.g. during tests), this
        # attribute may remain None; tests should patch folder_paths.
        self.loras_folder: Optional[str] = None
        if folder_paths is not None:
            try:
                paths = folder_paths.get_folder_paths("loras")
                if paths:
                    self.loras_folder = paths[0]
            except Exception:
                # If folder_paths fails, leave loras_folder unset
                self.loras_folder = None

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs including counts and optional overrides."""
        return {
            "required": {},
            "optional": {
                "num_wan_loras": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Maximum number of video (WAN) LoRAs to combine."
                }),
                "num_image_loras": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Maximum number of image LoRAs to combine."
                }),
                "model_name": ("STRING", {
                    "default": cls._DEFAULT_MODEL_NAME,
                    "tooltip": "Name of the Ollama model to use for composing the prompt."
                }),
                "api_url": ("STRING", {
                    "default": cls._DEFAULT_API_URL,
                    "tooltip": "URL of the Ollama chat API endpoint."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Custom system prompt (leave blank to use default).",
                    "tooltip": "Override the default system instructions sent to the Ollama model."
                }),
            },
        }

    # ------------------------------------------------------------------
    # Helper methods for interacting with Ollama
    #
    def _ensure_model_available(self, model: str, api_url: str) -> None:
        """Ensure that the requested model is installed via shared utility.

        Delegates to ``ensure_model_available`` from ``ollama_utils`` with
        the channel ``lora_prompt_composer_status``.  If the helper
        cannot be imported, no action is taken.
        """
        if ensure_model_available is None:
            return
        # Pass the imported ``requests`` module so that tests can
        # monkey‑patch network calls on this module.  The status
        # channel identifies this node for frontend updates.
        ensure_model_available(
            model,
            api_url,
            requests_module=requests,
            status_channel="lora_prompt_composer_status",
        )

    def _call_ollama(self, user_message: str, model_name: str, api_url: str, system_prompt: str) -> str:
        """Call the Ollama chat API via shared helper and return raw content.

        Uses ``send_chat`` from ``ollama_utils`` to send the system and
        user messages.  Returns the assistant's reply or an empty
        string on failure.
        """
        if send_chat is None:
            return ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        # Pass through the imported ``requests`` module so that test
        # patches on this module take effect.
        return send_chat(
            model_name,
            api_url,
            messages,
            timeout=60,
            requests_module=requests,
        )

    # ------------------------------------------------------------------
    # Helper methods for loading and classifying LoRAs
    #
    def _load_all_lora_metadata(self) -> List[Dict[str, any]]:
        """Load metadata for all LoRAs found in the loras folder.

        Returns a list of dicts representing the parsed JSON from each
        ``*.metadata.json`` file.  If ``self.loras_folder`` is None or
        the folder does not exist, returns an empty list.
        """
        result: List[Dict[str, any]] = []
        folder = self.loras_folder
        if not folder or not os.path.isdir(folder):
            return result
        for fname in os.listdir(folder):
            if not fname.endswith(".metadata.json"):
                continue
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Attempt to load corresponding analysis file, if it exists.  The
                    # analysis file is expected to have the same base name with
                    # ``.analyzed.metadata.json`` inserted before the extension.
                    analysis_path = path.replace(
                        ".metadata.json", ".analyzed.metadata.json"
                    )
                    if os.path.exists(analysis_path):
                        try:
                            with open(analysis_path, "r", encoding="utf-8") as af:
                                analysis_data = json.load(af)
                            # Only attach analysis if non-empty dict
                            if isinstance(analysis_data, dict) and analysis_data:
                                data["analysis"] = analysis_data
                        except Exception as ae:
                            # If analysis cannot be loaded, silently skip
                            print(f"LoRAPromptComposerNode: failed to load analysis for {fname}: {ae}")
                    result.append(data)
            except Exception as e:
                print(f"LoRAPromptComposerNode: failed to load metadata from {fname}: {e}")
                continue
        return result

    def _extract_trigger(self, meta: Dict[str, any], file_name: str) -> str:
        """Extract a trigger word from metadata or fallback to file name.

        The trigger word is taken from the first item in the
        ``trainedWords`` list under ``meta['civitai']``, if present and
        non-empty.  Otherwise, the file name itself (without extension)
        is used with spaces replaced by underscores.  The trigger
        returned is lowercased.

        Args:
            meta: Metadata dictionary for a LoRA.
            file_name: The ``file_name`` field from the metadata.

        Returns:
            A string representing the trigger word.
        """
        # Try trainedWords
        try:
            trained = meta.get("civitai", {}).get("trainedWords", [])
            if trained:
                # Use first non-empty string
                for word in trained:
                    if isinstance(word, str) and word.strip():
                        return word.strip().lower()
        except Exception:
            pass
        # Fallback: derive from file_name (remove extension, replace spaces)
        trigger = file_name.rsplit(".", 1)[0]  # remove extension if any
        trigger = trigger.replace(" ", "_").replace("/", "_")
        return trigger.lower()

    def _classify_loras(self, metas: List[Dict[str, any]]) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
        """Separate metadata into image and video LoRA lists.

        Determines whether a LoRA is for video generation by inspecting
        the ``base_model`` string in its metadata.  If the base model
        contains the substring ``"wan"`` or ``"video"`` (case
        insensitive), the LoRA is classified as a video LoRA; otherwise
        it is considered an image LoRA.  Each returned entry contains
        the name, trigger, tags and original metadata.

        Args:
            metas: List of metadata dictionaries.

        Returns:
            A tuple of (video_loras, image_loras) where each element is
            a list of dictionaries with keys ``name``, ``trigger``,
            ``tags``, and ``meta``.
        """
        video_list: List[Dict[str, any]] = []
        image_list: List[Dict[str, any]] = []
        for meta in metas:
            file_name = meta.get("file_name") or meta.get("model_name") or ""
            base = meta.get("base_model", "")
            # Determine type based on base_model
            is_video = False
            if isinstance(base, str):
                lb = base.lower()
                if "wan" in lb or "video" in lb:
                    is_video = True
            # Extract tags list (lowercase)
            tags = []
            try:
                model_info = meta.get("civitai", {}).get("model", {})
                tags = model_info.get("tags", [])
            except Exception:
                pass
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).lower() for t in tags if isinstance(t, str)]
            # Determine trigger
            trigger = self._extract_trigger(meta, file_name)
            entry = {
                "name": file_name,
                "trigger": trigger,
                "tags": tags,
                "meta": meta,
            }
            # If analysis is present in the meta, attach it to the entry.  The
            # analysis dict may contain keys such as 'when_to_use' and
            # 'example_prompts_and_analysis'.  Only attach if it is a dict.
            analysis = meta.get("analysis")
            if isinstance(analysis, dict) and analysis:
                entry["analysis"] = analysis
            if is_video:
                video_list.append(entry)
            else:
                image_list.append(entry)
        return video_list, image_list

    # ------------------------------------------------------------------
    # Main node function
    #
    def compose_prompt(
        self,
        num_wan_loras: int = 1,
        num_image_loras: int = 1,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        system_prompt: str = "",
    ) -> Tuple[str]:
        """Compose a combined prompt using available LoRAs.

        This public method is called by ComfyUI.  It loads all LoRA
        metadata, builds a message for the Ollama model instructing it
        to select up to the specified numbers of video and image LoRAs,
        and returns the resulting prompt.  If no metadata is found or
        the Ollama call fails, an empty string is returned.

        Args:
            num_wan_loras: Maximum number of video LoRAs to include.
            num_image_loras: Maximum number of image LoRAs to include.
            model_name: Optional override for the Ollama model.  If
                omitted, the default is used.
            api_url: Optional override for the Ollama API URL.
            system_prompt: Optional override for the system prompt.

        Returns:
            A single string containing the combined prompt.
        """
        # Load metadata
        metas = self._load_all_lora_metadata()
        if not metas:
            return ("",)
        # Classify LoRAs into video and image categories
        video_loras, image_loras = self._classify_loras(metas)
        # Prepare the message to send to the model.  We include the
        # entire list of available LoRAs and specify the maximum
        # selection counts.  The counts are clamped to be at least 0.
        max_video = max(0, int(num_wan_loras))
        max_image = max(0, int(num_image_loras))
        # Build lists for video and image LoRAs.  Include any analysis data
        # present on the entry so that the LLM has richer context.  Each
        # analysis dict may contain fields like 'when_to_use' and
        # 'example_prompts_and_analysis'.  If no analysis is available,
        # the 'analysis' key is omitted to reduce payload size.
        video_entries = []
        for e in video_loras:
            item = {"name": e["name"], "trigger": e["trigger"], "tags": e["tags"]}
            analysis = e.get("analysis")
            if isinstance(analysis, dict) and analysis:
                item["analysis"] = analysis
            video_entries.append(item)
        image_entries = []
        for e in image_loras:
            item = {"name": e["name"], "trigger": e["trigger"], "tags": e["tags"]}
            analysis = e.get("analysis")
            if isinstance(analysis, dict) and analysis:
                item["analysis"] = analysis
            image_entries.append(item)
        user_payload = {
            "video_loras": video_entries,
            "image_loras": image_entries,
            "max_video_count": max_video,
            "max_image_count": max_image,
        }
        # Compose user message as a JSON string plus a short instruction.
        # We embed the JSON in the user message because the system
        # prompt already instructs the assistant how to interpret it.
        user_message = (
            "Here is the available LoRA data and selection limits as JSON. "
            "Please choose and compose a prompt accordingly.\n" + json.dumps(user_payload, ensure_ascii=False)
        )
        # Determine model and API URL
        model = model_name or self._DEFAULT_MODEL_NAME
        url = api_url or self._DEFAULT_API_URL
        sys_prompt = system_prompt if system_prompt else self._SYSTEM_PROMPT
        # Ensure model availability
        try:
            self._ensure_model_available(model, url)
        except Exception as e:
            print(f"LoRAPromptComposerNode: error ensuring model availability: {e}")
        # Call Ollama
        prompt = self._call_ollama(user_message, model, url, sys_prompt)
        # Return as a single-element tuple as required by ComfyUI
        return (prompt,)