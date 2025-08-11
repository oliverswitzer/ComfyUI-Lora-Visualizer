# LoRA Visualizer - ComfyUI Custom Node

A ComfyUI custom node that parses prompt text for LoRA tags (wan and image gen, using `<lora:yourtag:1.0>` syntax and `<wanlora:yourtag:1.0>` syntax respectively) and visualizes their metadata, including trigger words, strength values, thumbnail previews, and example images.

![LoRA Visualizer node in workflow](docs/images/node-in-workflow.png)

![LoRA Visualizer node interface](docs/images/just-node.png)

![LoRA Visualizer hover interaction](docs/images/node-hover.png)


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
- **Filtering Options**: Filter by base model, content level, etc.
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

## Publishing to ComfyUI Registry

This node can be published to the [ComfyUI Registry](https://registry.comfy.org) for easy installation by users.

### Setup for Publishing

1. **Create a Publisher Account**: Go to [Comfy Registry](https://registry.comfy.org) and create a publisher account
2. **Get Your Publisher ID**: Find your publisher ID (after the `@` symbol) on your profile page
3. **Update pyproject.toml**: Add your Publisher ID to the `PublisherId` field in `pyproject.toml`
4. **Create API Key**: Generate an API key for your publisher in the registry
5. **Set GitHub Secret**: Add your API key as `REGISTRY_ACCESS_TOKEN` in your GitHub repository secrets (Settings → Secrets and Variables → Actions → New Repository Secret)

### Automated Release Workflow

The project uses **conventional commits** for automatic semantic versioning. The **"Release and Publish"** GitHub Action automatically determines the next version based on your commit messages:

#### Commit Message Format:
- `fix: description` → **patch** version bump (1.0.0 → 1.0.1)
- `feat: description` → **minor** version bump (1.0.0 → 1.1.0)
- `BREAKING CHANGE:` in commit body → **major** version bump (1.0.0 → 2.0.0)

#### Release Process:
1. **Make commits** using conventional format
2. **Go to Actions** → "Release and Publish to ComfyUI Registry"
3. **Click "Run workflow"**
4. **Add changelog** (optional)
5. **Choose dry run** to preview without releasing

This workflow automatically:
- ✅ Analyzes commit messages since last release
- ✅ Calculates appropriate version bump
- ✅ Updates version in `pyproject.toml`
- ✅ Creates git tag (e.g., `v1.1.0`)
- ✅ Creates GitHub release with changelog
- ✅ Publishes to ComfyUI Registry

#### Example Commit Messages:
```bash
git commit -m "fix: resolve parsing issue with special characters"
git commit -m "feat: add support for custom LoRA tags"
git commit -m "feat: new visualization mode

BREAKING CHANGE: removes old API methods"
```

### Manual Publishing

For quick republishing without version changes:
1. **Go to Actions** → "Release and Publish to ComfyUI Registry"
2. **Click "Run workflow"**
3. **Select "publish_only"** from the action type dropdown
4. **Click "Run workflow"**

Alternatively, use the ComfyUI CLI: `comfy node publish`

For more details, see the [ComfyUI Registry Publishing Guide](https://docs.comfy.org/registry/publishing).

## Development

### Prerequisites

- [PDM](https://pdm.fming.dev/latest/) for dependency management
- Python 3.8+ (same as ComfyUI requirement)

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/oliverswitzer/ComfyUI-Lora-Visualizer.git
   cd ComfyUI-Lora-Visualizer
   ```

2. **Install development dependencies**:
   ```bash
   pdm install
   ```
   This creates a virtual environment and installs pytest, black, and pylint.

### Running Tests

pdm run test

### Code Quality

**Format code with Black**:
```bash
pdm run format
```

**Lint with Pylint**:
```bash
pdm run lint
```

**Run all checks (format + lint + test)**:
```bash
pdm run check
```

### Test Structure

- **`tests/test_lora_parsing.py`**: Main test suite
- **`tests/fixtures/`**: Sample metadata files for testing
- **`conftest.py`**: Test configuration and ComfyUI mocking

Tests cover:
- LoRA tag parsing (standard and WanLoRA formats)
- Metadata extraction and processing
- Civitai URL generation
- Edge cases and error handling

### Available PDM Scripts

| Command | Description |
|---------|-------------|
| `pdm run format` | Format code with Black |
| `pdm run lint` | Lint code with Pylint |
| `pdm run check` | Run format + lint (tests via ./run_tests.sh) |

Note: For tests, use `./run_tests.sh` due to import complexities with ComfyUI's module structure.

### Adding New Tests

1. Add test methods to `TestLoRAVisualizerNode` class
2. Use fixture files in `tests/fixtures/` for realistic data
3. Mock ComfyUI dependencies (already set up in `conftest.py`)
4. Run tests to ensure everything passes

Example test:
```python
def test_new_feature(self):
    """Test description."""
    # Setup
    test_data = {...}
    
    # Execute
    result = self.node.some_method(test_data)
    
    # Assert
    self.assertEqual(result, expected_value)
```

## License

[Add your license here]

## Changelog

### v1.0.0
- Initial release
- Basic LoRA tag parsing
- Metadata visualization
- Thumbnail display
- Separate lists for standard and WanLoRAs