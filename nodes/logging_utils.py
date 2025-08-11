"""
Centralized logging utilities for ComfyUI-Lora-Visualizer.

This module provides a consistent logging interface for all components
of the LoRA Visualizer custom node.
"""


def log(message: str) -> None:
    """
    Log a message with the ComfyUI-Lora-Visualizer prefix.

    Args:
        message: The message to log
    """
    print(f"[ComfyUI-Lora-Visualizer] {message}")


def log_debug(message: str) -> None:
    """
    Log a debug message with DEBUG prefix.

    Args:
        message: The debug message to log
    """
    print(f"[ComfyUI-Lora-Visualizer] DEBUG: {message}")


def log_warning(message: str) -> None:
    """
    Log a warning message with WARNING prefix.

    Args:
        message: The warning message to log
    """
    print(f"[ComfyUI-Lora-Visualizer] WARNING: {message}")


def log_error(message: str) -> None:
    """
    Log an error message with ERROR prefix.

    Args:
        message: The error message to log
    """
    print(f"[ComfyUI-Lora-Visualizer] ERROR: {message}")
