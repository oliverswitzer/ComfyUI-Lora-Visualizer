from .nodes.lora_visualizer_node import LoRAVisualizerNode

NODE_CLASS_MAPPINGS = {
    "LoRAVisualizer": LoRAVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAVisualizer": "LoRA Visualizer",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]