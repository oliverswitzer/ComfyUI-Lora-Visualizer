#!/usr/bin/env python3
"""
Generate ONE massive ComfyUI workflow containing ALL WAN 2.2 HIGH/LOW LoRA pairs.

This script creates a single workflow with all pairs arranged in parallel,
each with their own text input → splitter → 3 show text nodes.
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Define a simplified version of extract_example_prompts
def extract_example_prompts(metadata: Dict, limit: int = 3) -> List[str]:
    """Extract example prompts from LoRA metadata."""
    examples = []

    try:
        # Try civitai.images first
        if "civitai" in metadata and "images" in metadata["civitai"]:
            images = metadata["civitai"]["images"]
            if isinstance(images, list):
                for img in images[:limit]:
                    if isinstance(img, dict) and "meta" in img:
                        meta = img["meta"]
                        if "prompt" in meta and meta["prompt"]:
                            examples.append(str(meta["prompt"]).strip())
                        elif "generationProcess" in meta and meta["generationProcess"] == "txt2img":
                            for key in ["parameters", "prompt_text", "text"]:
                                if key in meta and meta[key]:
                                    examples.append(str(meta[key]).strip())
                                    break

        # Also try civitai.trainedWords as examples
        if len(examples) < limit and "civitai" in metadata:
            if "trainedWords" in metadata["civitai"]:
                trained_words = metadata["civitai"]["trainedWords"]
                if isinstance(trained_words, list) and trained_words:
                    examples.append(", ".join(trained_words[:3]))

    except Exception as e:
        print(f"Error extracting examples: {e}")

    return examples[:limit]


def load_metadata(metadata_path: str) -> Optional[Dict]:
    """Load metadata from a JSON file."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        return None


def find_lora_pairs(lora_dir: str) -> List[Tuple[str, str, str, str]]:
    """Find HIGH/LOW LoRA pairs in the directory."""
    lora_dir = Path(lora_dir)
    if not lora_dir.exists():
        print(f"Directory {lora_dir} does not exist")
        return []

    safetensors_files = list(lora_dir.glob("*.safetensors"))
    pairs = []
    processed_bases = set()

    for file in safetensors_files:
        base_name = file.stem
        if base_name in processed_bases:
            continue

        base_lower = base_name.lower()
        high_file = None
        low_file = None

        if any(pattern in base_lower for pattern in ['high', 'hn', '_h_', '_h.']):
            high_file = file
            low_base = base_name
            low_base = re.sub(r'[Hh]igh', 'Low', low_base)
            low_base = re.sub(r'HIGH', 'LOW', low_base)
            low_base = re.sub(r'[Hh][Nn]', 'LN', low_base)
            low_base = re.sub(r'_[Hh]_', '_L_', low_base)
            low_base = re.sub(r'_[Hh]\.', '_L.', low_base)

            low_path = lora_dir / f"{low_base}.safetensors"
            if low_path.exists():
                low_file = low_path

        elif any(pattern in base_lower for pattern in ['low', 'ln', '_l_', '_l.']):
            low_file = file
            high_base = base_name
            high_base = re.sub(r'[Ll]ow', 'High', high_base)
            high_base = re.sub(r'LOW', 'HIGH', high_base)
            high_base = re.sub(r'[Ll][Nn]', 'HN', high_base)
            high_base = re.sub(r'_[Ll]_', '_H_', high_base)
            high_base = re.sub(r'_[Ll]\.', '_H.', high_base)

            high_path = lora_dir / f"{high_base}.safetensors"
            if high_path.exists():
                high_file = high_path

        if high_file and low_file:
            pair_base = re.sub(r'[_\-\s]*(high|low|hn|ln)[_\-\s]*', '', base_name, flags=re.IGNORECASE)
            pair_base = re.sub(r'[_\-\s]+', '_', pair_base).strip('_')

            metadata_file = high_file.with_suffix('.metadata.json')
            if not metadata_file.exists():
                metadata_file = low_file.with_suffix('.metadata.json')

            pairs.append((pair_base, high_file.name, low_file.name, metadata_file.name if metadata_file.exists() else None))

            processed_bases.add(high_file.stem)
            processed_bases.add(low_file.stem)

    return pairs


def get_example_prompt(metadata_path: str) -> str:
    """Extract an example prompt from metadata."""
    if not metadata_path or not os.path.exists(metadata_path):
        return "beautiful woman, detailed"

    metadata = load_metadata(metadata_path)
    if not metadata:
        return "beautiful woman, detailed"

    examples = extract_example_prompts(metadata, limit=1)
    if examples:
        example = examples[0]
        example = re.sub(r'<(?:lora|wanlora):[^>]+>', '', example)
        example = re.sub(r'\s+', ' ', example).strip()
        return example if example else "beautiful woman, detailed"

    return "beautiful woman, detailed"


def create_mega_workflow(pairs: List[Tuple[str, str, str, str]], lora_dir: str) -> Dict:
    """Create one massive workflow with all LoRA pairs."""

    workflow = {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "groupNodes": {},
            "ue_links": [],
            "ds": {"scale": 0.5, "offset": [5000, 3000]},
            "frontendVersion": "1.25.11",
            "VHS_latentpreview": True,
            "VHS_latentpreviewrate": 0,
            "VHS_MetadataImage": True,
            "VHS_KeepIntermediate": True
        },
        "version": 0.4
    }

    node_id = 1
    link_id = 1

    # Arrange pairs in a grid: 4 pairs per row, with each pair taking up significant space
    pairs_per_row = 4
    pair_width = 1200  # Width per pair (text input + splitter + 3 show text nodes)
    pair_height = 400  # Height per pair
    start_x = -5000
    start_y = -2500

    for i, (pair_base, high_lora, low_lora, metadata_file) in enumerate(pairs):
        print(f"Adding pair {i+1}/{len(pairs)}: {pair_base}")

        # Calculate position
        row = i // pairs_per_row
        col = i % pairs_per_row
        base_x = start_x + (col * pair_width)
        base_y = start_y + (row * pair_height)

        # Get example prompt
        metadata_path = os.path.join(lora_dir, metadata_file) if metadata_file else None
        example_prompt = get_example_prompt(metadata_path)

        # Remove .safetensors extension for LoRA tags
        high_name = high_lora.replace('.safetensors', '')
        low_name = low_lora.replace('.safetensors', '')
        full_prompt = f"{example_prompt} <lora:{high_name}:1.0> <lora:{low_name}:1.0>"

        # 1. Text input node
        text_node = {
            "id": node_id,
            "type": "ttN text",
            "pos": [base_x, base_y],
            "size": [400, 150],
            "flags": {},
            "order": i * 4,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "text", "type": "STRING", "links": [link_id]}],
            "properties": {
                "cnr_id": "comfyui_tinyterranodes",
                "ver": "2.0.9",
                "Node name for S&R": "ttN text"
            },
            "widgets_values": [full_prompt]
        }
        workflow["nodes"].append(text_node)
        text_node_id = node_id
        text_link_id = link_id
        node_id += 1
        link_id += 1

        # 2. WAN LoRA High/Low Splitter node
        splitter_node = {
            "id": node_id,
            "type": "WANLoRAHighLowSplitter",
            "pos": [base_x + 450, base_y],
            "size": [300, 150],
            "flags": {},
            "order": i * 4 + 1,
            "mode": 0,
            "inputs": [
                {
                    "name": "prompt_text",
                    "type": "STRING",
                    "widget": {"name": "prompt_text"},
                    "link": text_link_id
                }
            ],
            "outputs": [
                {"name": "high_prompt", "type": "STRING", "links": [link_id]},
                {"name": "low_prompt", "type": "STRING", "links": [link_id + 1]},
                {"name": "analysis", "type": "STRING", "links": [link_id + 2]}
            ],
            "properties": {
                "cnr_id": "comfyui-lora-visualizer",
                "ver": "be9066028e8151fde79736d128408de6baa63527",
                "Node name for S&R": "WANLoRAHighLowSplitter"
            },
            "widgets_values": [""]
        }
        workflow["nodes"].append(splitter_node)
        splitter_node_id = node_id
        high_link_id = link_id
        low_link_id = link_id + 1
        analysis_link_id = link_id + 2
        node_id += 1
        link_id += 3

        # 3. High prompt ShowText node
        high_show_node = {
            "id": node_id,
            "type": "ShowText|pysssss",
            "pos": [base_x + 800, base_y - 50],
            "size": [250, 80],
            "flags": {},
            "order": i * 4 + 2,
            "mode": 0,
            "inputs": [{"name": "text", "type": "STRING", "link": high_link_id}],
            "outputs": [{"name": "STRING", "shape": 6, "type": "STRING", "links": None}],
            "properties": {
                "cnr_id": "comfyui-custom-scripts",
                "ver": "1.2.5",
                "Node name for S&R": "ShowText|pysssss"
            }
        }
        workflow["nodes"].append(high_show_node)
        node_id += 1

        # 4. Low prompt ShowText node
        low_show_node = {
            "id": node_id,
            "type": "ShowText|pysssss",
            "pos": [base_x + 800, base_y + 50],
            "size": [250, 80],
            "flags": {},
            "order": i * 4 + 3,
            "mode": 0,
            "inputs": [{"name": "text", "type": "STRING", "link": low_link_id}],
            "outputs": [{"name": "STRING", "shape": 6, "type": "STRING", "links": None}],
            "properties": {
                "cnr_id": "comfyui-custom-scripts",
                "ver": "1.2.5",
                "Node name for S&R": "ShowText|pysssss"
            }
        }
        workflow["nodes"].append(low_show_node)
        node_id += 1

        # 5. Analysis ShowText node
        analysis_show_node = {
            "id": node_id,
            "type": "ShowText|pysssss",
            "pos": [base_x + 800, base_y + 150],
            "size": [250, 80],
            "flags": {},
            "order": i * 4 + 4,
            "mode": 0,
            "inputs": [{"name": "text", "type": "STRING", "link": analysis_link_id}],
            "outputs": [{"name": "STRING", "shape": 6, "type": "STRING", "links": None}],
            "properties": {
                "cnr_id": "comfyui-custom-scripts",
                "ver": "1.2.5",
                "Node name for S&R": "ShowText|pysssss"
            }
        }
        workflow["nodes"].append(analysis_show_node)
        node_id += 1

        # Add links for this pair
        workflow["links"].extend([
            [text_link_id, text_node_id, 0, splitter_node_id, 0, "STRING"],
            [high_link_id, splitter_node_id, 0, high_show_node["id"], 0, "STRING"],
            [low_link_id, splitter_node_id, 1, low_show_node["id"], 0, "STRING"],
            [analysis_link_id, splitter_node_id, 2, analysis_show_node["id"], 0, "STRING"]
        ])

    workflow["last_node_id"] = node_id - 1
    workflow["last_link_id"] = link_id - 1

    return workflow


def main():
    import sys
    """Main function to generate the mega workflow."""
    lora_dir = "../ComfyUI/models/loras/wan/wan2.2"
    output_file = sys.argv[1] if len(sys.argv) > 1 else "mega_wan_workflow.json"

    print("Scanning for WAN 2.2 HIGH/LOW LoRA pairs...")
    pairs = find_lora_pairs(lora_dir)

    if not pairs:
        print("No HIGH/LOW LoRA pairs found!")
        return

    print(f"Found {len(pairs)} LoRA pairs. Creating mega workflow...")

    # Create the massive workflow
    workflow = create_mega_workflow(pairs, lora_dir)

    # Save workflow
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)

    print(f"Generated mega workflow: {output_file}")
    print(f"Contains {len(pairs)} LoRA pairs, each with:")
    print(f"  - Text input with both HIGH and LOW LoRA tags")
    print(f"  - WAN LoRA High/Low Splitter node")
    print(f"  - Three ShowText nodes (high, low, analysis)")
    print(f"  - Total nodes: {len(workflow['nodes'])}")
    print(f"  - Total links: {len(workflow['links'])}")
    print(f"\nArranged in a 4-column grid for easy viewing!")


if __name__ == "__main__":
    main()