#!/usr/bin/env python3
"""
Test script to validate rapidfuzz-based LoRA pairing against WAN 2.2 LoRAs.

This script tests the find_lora_pairs_in_prompt_with_rapidfuzz function
against all LoRAs in the ../ComfyUI/models/loras/wan/wan2.2 directory
to ensure it works correctly with real LoRA files.

Usage:
    python test_lora_pairing.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add the project root to Python path so we can import our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our actual LoRA pairing functions from the project
from nodes.lora_metadata_utils import find_lora_pair, find_lora_pairs_in_prompt


def find_lora_files(directory: str) -> List[str]:
    """
    Find all LoRA files in the specified directory.

    Args:
        directory: Path to search for LoRA files

    Returns:
        List of LoRA filenames (without extensions)
    """
    lora_path = Path(directory)
    if not lora_path.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return []

    lora_files = []
    for file_path in lora_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in [".safetensors", ".pt", ".ckpt"]:
            lora_files.append(file_path.stem)  # Filename without extension

    return sorted(lora_files)


def test_individual_pairing_function(lora_names: List[str]) -> Tuple[Dict[str, str], Set[str]]:
    """
    Test our actual find_lora_pair function with a list of LoRA names.

    Args:
        lora_names: List of LoRA names to test pairing with

    Returns:
        Tuple of (matched_pairs, unmatched_loras)
    """
    matched_pairs = {}
    unmatched_loras = set(lora_names)

    for lora_name in lora_names:
        # Use our actual project function
        pair = find_lora_pair(lora_name, lora_names)
        if pair:
            matched_pairs[lora_name] = pair
            # Remove both from unmatched set
            unmatched_loras.discard(lora_name)
            unmatched_loras.discard(pair)

    return matched_pairs, unmatched_loras


def test_prompt_pairing_function(lora_names: List[str]) -> Dict[str, str]:
    """
    Test our actual find_lora_pairs_in_prompt function.

    Args:
        lora_names: List of LoRA names to create test prompts with

    Returns:
        Dictionary showing original prompt -> enhanced prompt mappings
    """
    prompt_results = {}

    # Test with individual LoRA prompts
    for lora_name in lora_names:
        test_prompt = f"a beautiful woman dancing <lora:{lora_name}:0.8> in a garden"

        # Mock the discover_all_loras function by creating a temporary mock
        import unittest.mock

        with unittest.mock.patch("nodes.lora_metadata_utils.discover_all_loras") as mock_discover:
            # Create mock LoRA database from our file list
            mock_loras = {name: {"metadata": {}} for name in lora_names}
            mock_discover.return_value = mock_loras

            enhanced_prompt = find_lora_pairs_in_prompt(test_prompt)

            if enhanced_prompt != test_prompt:
                prompt_results[test_prompt] = enhanced_prompt

    return prompt_results


def analyze_pairing_results(matched_pairs: Dict[str, str], unmatched_loras: Set[str]) -> None:
    """
    Analyze and display the results of LoRA pairing.

    Args:
        matched_pairs: Dictionary of LoRA -> paired LoRA mappings
        unmatched_loras: Set of LoRAs that didn't find pairs
    """
    print("=" * 80)
    print("LORA PAIRING ANALYSIS RESULTS")
    print("=" * 80)

    # Convert pairs to unique pairs (avoid duplicates like A->B and B->A)
    unique_pairs = set()
    for lora, pair in matched_pairs.items():
        pair_tuple = tuple(sorted([lora, pair]))
        unique_pairs.add(pair_tuple)

    print(f"Total LoRAs analyzed: {len(matched_pairs) + len(unmatched_loras)}")
    print(f"LoRAs with pairs found: {len(matched_pairs)}")
    print(f"Unique pairs discovered: {len(unique_pairs)}")
    print(f"LoRAs without pairs: {len(unmatched_loras)}")
    print()

    if unique_pairs:
        print("SUCCESSFULLY PAIRED LORAS:")
        print("-" * 50)
        for i, (lora1, lora2) in enumerate(sorted(unique_pairs), 1):
            print(f"{i:2d}. {lora1}")
            print(f"    <-> {lora2}")
            print()

    if unmatched_loras:
        print("LORAS WITHOUT PAIRS:")
        print("-" * 50)
        for i, lora in enumerate(sorted(unmatched_loras), 1):
            print(f"{i:2d}. {lora}")
    else:
        print("All LoRAs found pairs!")

    print()
    print("=" * 80)


def main():
    """Main function to run the LoRA pairing validation test."""
    print("Starting LoRA Pairing Validation Test")
    print("=" * 80)

    # Path to WAN 2.2 LoRAs
    wan_lora_directory = "../ComfyUI/models/loras/wan/wan2.2"

    print(f"Searching for LoRAs in: {wan_lora_directory}")

    # Find all LoRA files
    lora_names = find_lora_files(wan_lora_directory)

    if not lora_names:
        print("[ERROR] No LoRA files found in the specified directory.")
        print("Please check that the directory exists and contains LoRA files.")
        return

    print(f"Found {len(lora_names)} LoRA files")
    print()

    # Show all found LoRAs
    print("LORA FILES DISCOVERED:")
    print("-" * 50)
    for i, lora in enumerate(lora_names, 1):
        print(f"{i:2d}. {lora}")
    print()

    # Test individual pairing function
    print("Testing individual LoRA pairing function (find_lora_pair)...")
    matched_pairs, unmatched_loras = test_individual_pairing_function(lora_names)

    # Analyze and display results
    analyze_pairing_results(matched_pairs, unmatched_loras)

    # Test prompt-based pairing function
    print("\nTesting prompt-based LoRA pairing function (find_lora_pairs_in_prompt)...")
    prompt_results = test_prompt_pairing_function(lora_names)

    if prompt_results:
        print("PROMPT ENHANCEMENT RESULTS:")
        print("-" * 50)
        for i, (original, enhanced) in enumerate(prompt_results.items(), 1):
            print(f"{i:2d}. Original: {original}")
            print(f"    Enhanced: {enhanced}")
            print()
    else:
        print("No prompt enhancements were made.")
    print(f"Prompts enhanced: {len(prompt_results)} out of {len(lora_names)} tested")

    # Additional analysis for debugging
    if matched_pairs:
        from rapidfuzz import fuzz

        print("DETAILED PAIRING ANALYSIS:")
        print("-" * 50)
        for lora, pair in sorted(matched_pairs.items()):
            # Calculate similarity score for transparency
            similarity = fuzz.ratio(lora, pair)
            print(f"{lora}")
            print(f"  -> {pair} (similarity: {similarity}%)")
        print()

    print("LoRA pairing validation test completed!")


if __name__ == "__main__":
    main()
