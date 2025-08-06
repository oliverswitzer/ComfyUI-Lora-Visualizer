# LoRA Visualizer - ComfyUI Custom Node

A ComfyUI custom node that parses prompt text for LoRA tags (wan and image gen) and visualizes their metadata, including trigger words, strength values, thumbnail previews, and example images.

![Just the node](docs/images/node-in-workflow.png)
![Just the node](docs/images/node-hover.png)
![Just the node](docs/images/just-node.png)


## Features

- **✅ Consistent LoRA Parsing**: Backend Python parsing handles both standard `<lora:name:strength>` and custom `<wanlora:name:strength>` tags with identical logic
- **✅ Complex Name Support**: Handles LoRA names with spaces, colons, and special characters (e.g., `<lora:Detail Enhancer v2.0: Professional Edition:0.8>`)
- **✅ Visual Thumbnails**: Displays actual LoRA preview images loaded from metadata files
- **✅ Metadata Integration**: Shows trigger words, model information, and base model details from ComfyUI LoRA Manager
- **✅ Separate Visual Lists**: Standard LoRAs (blue theme) and WanLoRAs (orange theme) displayed in distinct, color-coded sections
- **✅ Canvas-based Rendering**: Properly integrated with ComfyUI's node system using custom widget drawing
- **✅ Hover Gallery**: Hover over thumbnails to see trigger words and example images
- **✅ Backend-Frontend Architecture**: Python handles parsing and logic, JavaScript handles visualization
- **✅ Comprehensive Testing**: Unit tests cover edge cases and complex name parsing

## Installation

1. Clone or download this repository to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url> lora-visualizer
   ```

2. Restart ComfyUI to load the custom node

3. The node will appear in the ComfyUI node menu under **conditioning** → **LoRA Visualizer**

## Requirements

- **ComfyUI LoRA Manager**: This node depends on the ComfyUI LoRA Manager custom node being installed, as it uses the metadata files that the LoRA Manager downloads and maintains.

## Usage

1. Add the **LoRA Visualizer** node to your workflow
2. Enter a prompt containing LoRA tags in the `prompt_text` input field
3. The node will automatically parse and display information about each LoRA found

### Supported LoRA Tag Formats

- **Standard LoRAs**: `<lora:model_name:strength>`
  - Example: `<lora:landscape_v1:0.8>`
  
- **WanLoRAs**: `<wanlora:model_name:strength>`
  - Example: `<wanlora:Woman877.v2:1.0>`

### Example Prompt

```
A beautiful portrait <lora:realistic_skin:0.7> of <wanlora:Woman877.v2:0.8> woman standing in a garden, highly detailed
```

This will display:
- **Standard LoRAs**: realistic_skin (strength: 0.7)
- **WanLoRAs**: Woman877.v2 (strength: 0.8)

## Metadata File Requirements

The node looks for metadata files in the `models/loras/` directory with the naming pattern:
- `{lora_name}.metadata.json`
- `{lora_name}.safetensors.metadata.json`

These files should contain metadata in the format used by the ComfyUI LoRA Manager, including:
- `civitai.trainedWords`: Array of trigger words
- `preview_url`: Path to thumbnail image
- `civitai.images`: Array of example images
- `base_model`: Base model information
- `model_name`: Display name of the model

### Example Metadata Structure

```json
{
  "file_name": "Woman877.v2",
  "model_name": "Photorealistic AI Influencer – Woman877",
  "preview_url": "/path/to/Woman877.v2.webp",
  "base_model": "SDXL 1.0",
  "civitai": {
    "trainedWords": ["woman877"],
    "images": [
      {
        "url": "https://example.com/image1.jpg",
        "width": 768,
        "height": 1152,
        "nsfwLevel": 1
      }
    ]
  }
}
```

## Output

The node provides two outputs:

1. **lora_info** (STRING): A formatted text report with detailed information about all found LoRAs
2. **processed_prompt** (STRING): The original prompt text (can be modified in future versions)

## Features in Development

- **Image Gallery**: Full implementation of hover-to-view example images
- **Interactive Controls**: Click to copy trigger words, adjust strengths
- **Filtering Options**: Filter by base model, NSFW level, etc.
- **Export Options**: Export LoRA information in various formats

## Testing

Run the test suite to verify functionality:

```bash
./run_tests.sh
```

## File Structure

```
lora-visualizer/
├── __init__.py                     # Node registration
├── README.md                       # This file
├── nodes/
│   └── lora_visualizer_node.py     # Main node implementation
├── web/
│   └── lora_visualizer.js          # Frontend visualization
└── tests/
    └── test_lora_parsing.py        # Unit tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Changelog

### v1.0.0
- Initial release
- Basic LoRA tag parsing
- Metadata visualization
- Thumbnail display
- Separate lists for standard and WanLoRAs