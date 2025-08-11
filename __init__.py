try:
    from .nodes.lora_visualizer_node import LoRAVisualizerNode
    from .nodes.prompt_splitter_node import PromptSplitterNode
    from .nodes.prompt_composer_node import PromptComposerNode
except ImportError:
    # Fallback for when this module is imported directly (e.g., during testing)
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from nodes.lora_visualizer_node import LoRAVisualizerNode
    from nodes.prompt_splitter_node import PromptSplitterNode
    from nodes.prompt_composer_node import PromptComposerNode

NODE_CLASS_MAPPINGS = {
    "LoRAVisualizer": LoRAVisualizerNode,
    "PromptSplitter": PromptSplitterNode,
    "PromptComposer": PromptComposerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAVisualizer": "LoRA Visualizer",
    "PromptSplitter": "Prompt Splitter",
    "PromptComposer": "LoRA Prompt Composer",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
