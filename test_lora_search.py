#!/usr/bin/env python3
"""
Simple interactive LoRA search tester.

This script lets you test different search queries against your LoRA database
to see which LoRAs match and their relevance scores.

Usage:
    python test_lora_search.py

Then enter search queries when prompted!
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


def find_comfyui_lora_folder():
    """Try to automatically find ComfyUI's LoRA folder."""
    possible_paths = [
        os.path.expanduser("~/ComfyUI/models/loras"),
        "C:/ComfyUI/models/loras",
        "../ComfyUI/models/loras",
        "../../ComfyUI/models/loras",
        "./ComfyUI/models/loras",
        "../models/loras",
        "../../models/loras",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def mock_folder_paths(lora_folder: str):
    """Mock ComfyUI's folder_paths for testing."""

    class MockFolderPaths:
        @staticmethod
        def get_folder_paths(folder_type):
            if folder_type == "loras":
                return [lora_folder]
            return []

    return MockFolderPaths()


def setup_prompt_composer(lora_folder: str):
    """Initialize the prompt composer with mocked ComfyUI context."""
    from nodes import lora_metadata_utils
    from nodes import prompt_composer_node

    # Mock folder_paths
    mock_fp = mock_folder_paths(lora_folder)
    original_folder_paths_1 = lora_metadata_utils.folder_paths
    original_folder_paths_2 = getattr(prompt_composer_node, "folder_paths", None)

    lora_metadata_utils.folder_paths = mock_fp
    if hasattr(prompt_composer_node, "folder_paths"):
        prompt_composer_node.folder_paths = mock_fp

    try:
        from nodes.prompt_composer_node import PromptComposerNode

        print("🔧 Initializing prompt composer...")
        composer = PromptComposerNode()

        print("🧮 Loading embeddings...")
        success = composer._initialize_embeddings()

        if not success:
            print("❌ Failed to initialize embeddings!")
            return None, None, None

        # Count types
        image_count = video_count = 0
        directories = set()

        for name, info in composer._lora_database.items():
            from nodes.lora_metadata_utils import classify_lora_type

            lora_type = classify_lora_type(info["metadata"])
            if lora_type == "image":
                image_count += 1
            elif lora_type == "video":
                video_count += 1

            directory = info.get("directory", "")
            if directory:
                directories.add(directory)

        print(f"✅ Loaded {len(composer._lora_database)} LoRAs:")
        print(f"   🖼️  {image_count} image LoRAs")
        print(f"   🎥 {video_count} video LoRAs")
        if directories:
            print(f"   📂 Directories: {sorted(directories)}")
        else:
            print(f"   📂 All in root directory")

        return composer, original_folder_paths_1, original_folder_paths_2

    except Exception as e:
        print(f"❌ Error setting up composer: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def search_loras(composer, query: str, lora_type: str = "both", max_results: int = 5):
    """Search for LoRAs matching the query."""
    results = {"image": [], "video": []}

    if lora_type in ["both", "image"]:
        try:
            image_results = composer._find_relevant_loras(query, "image", max_results, 1.2)
            # Enhance results with full path info from database
            for result in image_results:
                lora_name = result.get("name")
                if lora_name in composer._lora_database:
                    db_info = composer._lora_database[lora_name]
                    result["full_path"] = db_info.get("full_path", "Unknown")
                    result["directory"] = db_info.get("directory", "")
            results["image"] = image_results
        except Exception as e:
            print(f"❌ Error searching image LoRAs: {e}")

    if lora_type in ["both", "video"]:
        try:
            video_results = composer._find_relevant_loras(query, "video", max_results, 1.2)
            # Enhance results with full path info from database
            for result in video_results:
                lora_name = result.get("name")
                if lora_name in composer._lora_database:
                    db_info = composer._lora_database[lora_name]
                    result["full_path"] = db_info.get("full_path", "Unknown")
                    result["directory"] = db_info.get("directory", "")
            results["video"] = video_results
        except Exception as e:
            print(f"❌ Error searching video LoRAs: {e}")

    return results


def print_results(query: str, results: dict):
    """Print search results in a nice format."""
    print(f"\n🔍 Results for: '{query}'")
    print("=" * 50)

    for lora_type, loras in results.items():
        if not loras:
            continue

        emoji = "🖼️ " if lora_type == "image" else "🎥"
        print(f"\n{emoji} {lora_type.upper()} LoRAs ({len(loras)} found):")

        for i, lora in enumerate(loras, 1):
            name = lora.get("name", "Unknown")
            score = lora.get("relevance_score", 0)
            directory = lora.get("directory", "")
            weight = lora.get("recommended_weight", "N/A")
            full_path = lora.get("full_path", "Unknown")

            print(f"  {i:2d}. {name}")
            print(f"      📊 Score: {score:.4f}")
            print(f"      📂 Path:  {full_path}")
            if directory:
                print(f"      📁 SubDir: {directory}")
            print(f"      ⚖️  Weight: {weight}")


def main():
    print("🎯 Interactive LoRA Search Tester")
    print("=" * 50)

    # Find LoRA folder
    lora_folder = find_comfyui_lora_folder()
    if not lora_folder:
        lora_folder = input("Enter path to ComfyUI/models/loras folder: ").strip()
        if not os.path.exists(lora_folder):
            print("❌ Folder not found!")
            return

    print(f"📁 Using LoRA folder: {lora_folder}")

    # Setup prompt composer
    composer, orig1, orig2 = setup_prompt_composer(lora_folder)
    if not composer:
        return

    try:
        print("\n" + "=" * 50)
        print("🚀 Ready to search! Enter queries to test.")
        print("Commands:")
        print("  • Just type a query (searches both image and video)")
        print("  • 'image: query' - search only image LoRAs")
        print("  • 'video: query' - search only video LoRAs")
        print("  • 'quit' or 'exit' - exit the program")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n🔍 Search query: ").strip()

                if not user_input or user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                # Parse input
                if user_input.startswith("image:"):
                    query = user_input[6:].strip()
                    lora_type = "image"
                elif user_input.startswith("video:"):
                    query = user_input[6:].strip()
                    lora_type = "video"
                else:
                    query = user_input
                    lora_type = "both"

                if not query:
                    print("❌ Empty query!")
                    continue

                # Search
                results = search_loras(composer, query, lora_type)
                print_results(query, results)

                # Quick stats
                total_found = len(results.get("image", [])) + len(results.get("video", []))
                if total_found == 0:
                    print(
                        "\n💡 Tip: Try different keywords like 'dance', 'motion', 'woman', 'character', etc."
                    )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    finally:
        # Restore original folder_paths
        if orig1 is not None:
            from nodes import lora_metadata_utils

            lora_metadata_utils.folder_paths = orig1
        if orig2 is not None:
            from nodes import prompt_composer_node

            prompt_composer_node.folder_paths = orig2


if __name__ == "__main__":
    main()
