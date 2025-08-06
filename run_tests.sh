#!/bin/bash
# Simple script to run tests with proper PYTHONPATH

export PYTHONPATH="/Users/oliverswitzer/workspace/LLM-video-stuff/ComfyUI/custom_nodes/lora-visualizer"
pytest tests/test_lora_parsing.py -v