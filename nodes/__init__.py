"""Expose node classes for ComfyUI.

This module provides dictionaries mapping class names to node classes and
human‑friendly display names.  ComfyUI uses these mappings to discover
custom nodes when the package is loaded.  You can add additional
nodes here by importing the class and updating ``NODE_CLASS_MAPPINGS``
and ``NODE_DISPLAY_NAME_MAPPINGS`` accordingly.
"""

from .lora_visualizer_node import LoRAVisualizerNode  # type: ignore
from .prompt_splitter_node import PromptSplitterNode  # type: ignore
from .lora_prompt_composer_node import LoRAPromptComposerNode  # type: ignore

# Import analysis preprocessor for background LoRA analysis.  This import
# is optional; if the module or function is unavailable, analysis will
# simply be skipped.  See below for invocation.
try:
    from .lora_analysis_preprocessor import analyze_all_loras  # type: ignore
except Exception:
    analyze_all_loras = None  # type: ignore


# Maps internal names to node classes.  The keys become the type names
# shown in JSON saved graphs.  Keep them stable to avoid breaking
# existing workflows.
NODE_CLASS_MAPPINGS = {
    "LoRAVisualizerNode": LoRAVisualizerNode,
    "PromptSplitterNode": PromptSplitterNode,
    "LoRAPromptComposerNode": LoRAPromptComposerNode,
}


# Human‑readable names displayed in the ComfyUI editor.  Update as
# necessary to improve clarity.
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAVisualizerNode": "LoRA Visualizer",
    "PromptSplitterNode": "Prompt Splitter (Ollama)",
    "LoRAPromptComposerNode": "LoRA Prompt Composer (Ollama)",
}


__all__ = [
    "LoRAVisualizerNode",
    "PromptSplitterNode",
    "LoRAPromptComposerNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# ---------------------------------------------------------------------------
# Background LoRA analysis at import time


# When the module is imported within a running ComfyUI session, attempt to
# analyse all LoRA metadata files in the configured "loras" directory.
# This is executed in a daemon thread so it does not block the main
# application startup.  If analysis is disabled via the environment
# variable ``COMFYUI_SKIP_LORA_ANALYSIS``, or if the analysis preprocessor
# is not available, this step is skipped.
def _trigger_background_lora_analysis():
    """Spawn a background thread to analyse LoRA metadata if configured."""
    import threading
    import os

    # Respect environment variable override
    if os.environ.get("COMFYUI_SKIP_LORA_ANALYSIS"):
        return
    # Ensure preprocessor is available
    if analyze_all_loras is None:
        return
    # Use folder_paths to locate LoRA directory
    try:
        import folder_paths  # type: ignore

        paths = folder_paths.get_folder_paths("loras")
        if not paths:
            return
        lora_dir = paths[0]
    except Exception:
        return
    # Determine a default model for analysis (use PromptSplitterNode's default)
    default_model = PromptSplitterNode._DEFAULT_MODEL_NAME
    api_url = PromptSplitterNode._DEFAULT_API_URL

    # Start analysis in a separate thread
    def worker():
        try:
            analyze_all_loras(
                lora_dir,
                model_name=default_model,
                api_url=api_url,
                status_channel="lora_analysis_status",
            )
        except Exception as e:
            # Print errors but do not propagate
            print(f"Background LoRA analysis error: {e}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


_trigger_background_lora_analysis()
