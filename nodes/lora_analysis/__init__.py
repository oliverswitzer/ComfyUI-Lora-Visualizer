"""
LoRA Analysis Subdomain
======================

This subdomain handles vision-based analysis of LoRA metadata using
Ollama's multimodal capabilities. It follows bounded context principles
with clear separation of concerns:

- `image_analyzer.py`: Vision analysis and image processing

Metadata extraction is handled by the shared `lora_utils.py` module.
Ollama API interactions are handled by the shared `ollama_utils.py` module.
The main entry point is `lora_analysis.py` in the parent directory.
"""

from .image_analyzer import ImageAnalyzer

__all__ = ["ImageAnalyzer"]
