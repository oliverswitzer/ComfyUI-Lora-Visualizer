"""
Example usage of the LoRA High/Low Splitter Node

This demonstrates how the new LoRAHighLowSplitterNode works with various
LoRA naming patterns, including support for HN/LN naming conventions.
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from nodes.lora_high_low_splitter_node import LoRAHighLowSplitterNode


def main():
    """Demonstrate the LoRA High/Low Splitter functionality."""

    # Initialize the node
    splitter = LoRAHighLowSplitterNode()

    # Example 1: Basic HIGH/LOW pattern
    print("=== Example 1: Basic HIGH/LOW Pattern ===")
    prompt1 = "woman dancing <lora:style_high:0.8> <lora:motion_low:0.6> in garden"
    high_prompt1, low_prompt1 = splitter.split_high_low(prompt1)

    print(f"Input:  {prompt1}")
    print(f"High:   {high_prompt1}")
    print(f"Low:    {low_prompt1}")
    print()

    # Example 2: HN/LN pattern support
    print("=== Example 2: HN/LN Pattern Support ===")
    prompt2 = "robot walking <lora:character_hn:0.7> <lora:background_ln:0.5> through city"
    high_prompt2, low_prompt2 = splitter.split_high_low(prompt2)

    print(f"Input:  {prompt2}")
    print(f"High:   {high_prompt2}")
    print(f"Low:    {low_prompt2}")
    print()

    # Example 3: Mixed patterns with single LoRAs
    print("=== Example 3: Mixed Patterns with Single LoRAs ===")
    prompt3 = ("futuristic scene <lora:Style_HIGH:0.8> "
               "<lora:Motion_low:0.6> <lora:General_Style:0.7> "
               "<lora:Effect_HN:0.5> <lora:Bg_ln:0.4>")
    high_prompt3, low_prompt3 = splitter.split_high_low(prompt3)

    print(f"Input:  {prompt3}")
    print(f"High:   {high_prompt3}")
    print(f"Low:    {low_prompt3}")
    print()

    # Example 4: WAN 2.2 realistic example
    print("=== Example 4: WAN 2.2 Realistic Example ===")
    prompt4 = ("cyberpunk cityscape <lora:Wan22-I2V-HIGH-Cyberpunk:0.7> "
               "with neon lights <lora:Wan22-I2V-LOW-Cyberpunk:0.7> "
               "and flying cars <lora:General-SciFi:0.5>")
    high_prompt4, low_prompt4 = splitter.split_high_low(prompt4)

    print(f"Input:  {prompt4}")
    print(f"High:   {high_prompt4}")
    print(f"Low:    {low_prompt4}")
    print()

    # Example 5: Edge case - avoiding false positives
    print("=== Example 5: Edge Case - Avoiding False Positives ===")
    prompt5 = ("character <lora:highlight_effect:0.8> "
               "with <lora:lowlight_shadows:0.6> "
               "and <lora:character_high:0.7>")
    high_prompt5, low_prompt5 = splitter.split_high_low(prompt5)

    print(f"Input:  {prompt5}")
    print(f"High:   {high_prompt5}")
    print(f"Low:    {low_prompt5}")
    print()

    print("=== Summary ===")
    print("The LoRA High/Low Splitter Node:")
    print("• Splits prompts based on LoRA tag names containing high/low/hn/ln")
    print("• Includes single LoRA tags (without high/low patterns) in both outputs")
    print("• Uses word boundaries to avoid false positives like 'highlight' → 'high'")
    print("• Perfect for WAN 2.2 workflows and any high/low LoRA usage")
    print("• Reuses proven logic from the existing PromptSplitterNode")


if __name__ == "__main__":
    main()