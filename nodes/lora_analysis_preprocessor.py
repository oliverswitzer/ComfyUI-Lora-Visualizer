"""
LoRA Analysis Preprocessor
-------------------------

This module contains helper functions to perform an offline analysis of
LoRA metadata files.  The goal of the analysis is to provide richer
context for prompt composition without sending large amounts of raw
metadata to the prompt generation LLM at runtime.  When executed, the
preprocessor reads existing ``*.metadata.json`` files, extracts up to
four example prompts per LoRA (if available), and delegates to an
instruction‑tuned Ollama model to produce a summary.  The summary
describes when to use the LoRA (a high‑level purpose) and provides an
analysis of each example prompt.  The results are written to a new
file with the suffix ``.analyzed.metadata.json`` in the same
directory as the original metadata.

The structure of the analysis JSON is:

.. code-block:: json

    {
      "when_to_use": "...",
      "example_prompts_and_analysis": [
        {"prompt": "...", "analysis": "..."},
        ...
      ]
    }

The ``when_to_use`` field should succinctly describe the LoRA's
subject, style, or scenarios where it excels.  The
``example_prompts_and_analysis`` field contains each example prompt and
the LLM's interpretation of what the resulting image likely depicts.

The preprocessor is designed to be run lazily at ComfyUI start‑up or
invoked manually.  If the analysis file already exists for a given
LoRA, the preprocessor skips reprocessing that file.  Network
operations to Ollama are optional and can be disabled via the
``requests`` import; in test environments, ``requests`` may be
patched or unavailable.

Example usage::

    from nodes.lora_analysis_preprocessor import analyze_all_loras
    analyze_all_loras("/path/to/models/loras", model_name="nous-hermes2")

Note that this module does not register any ComfyUI nodes; it is
intended for internal use by other modules.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

try:
    import requests  # type: ignore[import]
except Exception:
    requests = None

try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None

# ---------------------------------------------------------------------------
# System prompt for LoRA analysis

_ANALYSIS_SYSTEM_PROMPT = (
    "You are analyzing the usage of a LoRA for a generative AI system.\n\n"
    "You will be given a list of example prompts that users have used with this LoRA.\n"
    "From these prompts, infer what kind of subjects, styles or scenarios the LoRA excels at.\n"
    "Write two fields in JSON format: \n"
    "1. 'when_to_use': A short paragraph describing the purpose of this LoRA and when it should be used.\n"
    "2. 'example_prompts_and_analysis': A list of objects where each object has the keys 'prompt' and 'analysis'.\n"
    "   For each example prompt provided, repeat the prompt in the 'prompt' field and in 'analysis' describe what the\n"
    "   resulting image likely depicts, focusing on subject, style, lighting, mood, and any distinctive features.\n"
    "Do not include any code fences.  Output strictly valid JSON with keys 'when_to_use' and 'example_prompts_and_analysis'.\n"
    "Do not mention the LoRA name or any file names; describe content in generic terms."
)


def _call_ollama_for_analysis(
    example_prompts: List[str], model_name: str, api_url: str
) -> Optional[Dict[str, any]]:
    """Send the example prompts to Ollama and parse the analysis response.

    This helper constructs a chat request containing the system prompt and
    the user content (the example prompts), calls the Ollama chat API,
    and attempts to parse the returned content as JSON.  On failure
    (e.g. network issues, JSON decode errors), None is returned.

    Args:
        example_prompts: A list of up to four example prompts from the LoRA metadata.
        model_name: The name of the Ollama model to use.
        api_url: The chat API URL (e.g. "http://localhost:11434/api/chat").

    Returns:
        A dictionary with keys 'when_to_use' and 'example_prompts_and_analysis', or None on failure.
    """
    if requests is None:
        print("LoRA analysis: 'requests' library not available; skipping analysis.")
        return None
    # Build user message containing the example prompts as JSON for clarity.
    user_content = {"example_prompts": example_prompts}
    messages = [
        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(api_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Extract assistant content; structure may vary by Ollama version
        if isinstance(data, dict):
            if "message" in data:
                content = data["message"].get("content", "")
            elif "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
            else:
                content = ""
        else:
            content = ""
        content = content.strip()
        if not content:
            return None
        # Attempt to parse JSON from the assistant content
        try:
            result = json.loads(content)
            when_to_use = result.get("when_to_use")
            examples = result.get("example_prompts_and_analysis")
            if isinstance(when_to_use, str) and isinstance(examples, list):
                return {
                    "when_to_use": when_to_use.strip(),
                    "example_prompts_and_analysis": examples,
                }
            else:
                return None
        except Exception as parse_err:
            print(
                f"LoRA analysis: JSON parse error: {parse_err}; content was: {content}"
            )
            return None
    except Exception as e:
        print(f"LoRA analysis: error contacting Ollama: {e}")
        return None


def _extract_example_prompts(meta: Dict[str, any]) -> List[str]:
    """Extract up to four example prompts from the LoRA metadata.

    The metadata may contain example prompts under ``meta['civitai']['images']``.
    Each image entry's ``meta.prompt`` field is used as an example.  The
    function returns the first four unique prompts found.  Any
    duplicate or missing values are skipped.

    Args:
        meta: The parsed metadata dictionary for a single LoRA.

    Returns:
        A list of up to four example prompts.
    """
    examples: List[str] = []
    try:
        images = meta.get("civitai", {}).get("images", [])
        if isinstance(images, list):
            for image in images:
                p = None
                try:
                    p = image.get("meta", {}).get("prompt")
                except Exception:
                    p = None
                if isinstance(p, str) and p.strip():
                    prompt_str = p.strip()
                    # Avoid duplicates
                    if prompt_str not in examples:
                        examples.append(prompt_str)
                if len(examples) >= 4:
                    break
    except Exception:
        pass
    return examples


def analyze_all_loras(
    loras_folder: str,
    model_name: str = "nous-hermes2",
    api_url: str = "http://localhost:11434/api/chat",
    status_channel: Optional[str] = "lora_analysis_status",
) -> None:
    """Perform analysis on all LoRA metadata files in the given folder.

    This function iterates over all ``*.metadata.json`` files in
    ``loras_folder``.  For each metadata file, if a corresponding
    ``*.analyzed.metadata.json`` file is not present, the function
    extracts example prompts, sends them to an Ollama model for
    summarization, and writes the result.  Basic status messages are
    emitted via the ComfyUI PromptServer if available.

    Args:
        loras_folder: Path to the folder containing LoRA metadata files.
        model_name: Name of the Ollama model to use for analysis.
        api_url: URL of the Ollama chat API.
        status_channel: Optional message channel used to send status
            updates to the ComfyUI frontend.  If None, no messages are
            sent.
    """
    if not loras_folder or not os.path.isdir(loras_folder):
        return
    # Determine base URL for status messages
    for fname in os.listdir(loras_folder):
        if not fname.endswith(".metadata.json"):
            continue
        meta_path = os.path.join(loras_folder, fname)
        # The analysis file name: insert `.analyzed` before `.metadata.json`
        analyzed_path = meta_path.replace(".metadata.json", ".analyzed.metadata.json")
        # Skip if analysis already exists
        if os.path.exists(analyzed_path):
            continue
        # Load metadata
        meta: Dict[str, any] = {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"LoRA analysis: failed to read {fname}: {e}")
            continue
        # Extract example prompts
        prompts = _extract_example_prompts(meta)
        if not prompts:
            # Nothing to analyze; write empty analysis to avoid reprocessing
            try:
                with open(analyzed_path, "w", encoding="utf-8") as out:
                    json.dump({}, out)
            except Exception:
                pass
            continue
        # Inform user that analysis is starting
        if status_channel and PromptServer is not None:
            try:
                PromptServer.instance.send_sync(
                    status_channel,
                    {"status": f"Analyzing LoRA '{fname}' for usage summary..."},
                )
            except Exception:
                pass
        # Call Ollama to get analysis
        result = _call_ollama_for_analysis(prompts, model_name, api_url)
        if result is None:
            # Write empty analysis to avoid repeated attempts
            try:
                with open(analyzed_path, "w", encoding="utf-8") as out:
                    json.dump({}, out)
            except Exception:
                pass
            continue
        # Write analysis to file
        try:
            with open(analyzed_path, "w", encoding="utf-8") as out:
                json.dump(result, out, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"LoRA analysis: failed to write {analyzed_path}: {e}")
            continue
        # Notify completion
        if status_channel and PromptServer is not None:
            try:
                PromptServer.instance.send_sync(
                    status_channel, {"status": f"Finished analyzing LoRA '{fname}'."}
                )
            except Exception:
                pass
