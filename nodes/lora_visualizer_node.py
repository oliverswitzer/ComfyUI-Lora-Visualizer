"""
LoRA Visualizer Node Implementation
Parses prompts for LoRA tags and displays metadata, thumbnails, and example images.
"""

import os
import json
import re
import folder_paths
from typing import Dict, List, Tuple, Optional, Any
from server import PromptServer
from aiohttp import web


class LoRAVisualizerNode:
    """
    A ComfyUI node that visualizes LoRA metadata from prompt text.
    
    Features:
    - Parses standard LoRA tags: <lora:name:strength>
    - Parses custom wanlora tags: <wanlora:name:strength>  
    - Shows trigger words, strength, thumbnails
    - Hover to view all example images
    """
    
    CATEGORY = "conditioning"
    DESCRIPTION = """Analyzes prompt text to extract and visualize LoRA information with metadata.
    
• Parses standard LoRA tags: <lora:name:strength>
• Parses custom WanLoRA tags: <wanlora:name:strength>
• Displays thumbnails, trigger words, and base models
• Shows scalable previews with hover galleries
• Supports both image and video LoRAs
• Requires ComfyUI LoRA Manager for metadata"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt with LoRA tags here...",
                    "tooltip": "Input text containing LoRA tags like <lora:MyLora:0.8> or <wanlora:MyWanLora:1.0>. The node will automatically detect and visualize all LoRA references with their metadata."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("raw_lora_info", "original_prompt")
    OUTPUT_TOOLTIPS = (
        "Raw metadata information about detected LoRAs in a structured format for debugging and analysis.",
        "The original prompt text passed through unchanged for downstream processing."
    )
    FUNCTION = "visualize_loras"
    OUTPUT_NODE = True
    
    def __init__(self):
        self.loras_folder = folder_paths.get_folder_paths("loras")[0] if folder_paths.get_folder_paths("loras") else None
        
    def parse_lora_tags(self, prompt_text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse LoRA tags from prompt text.
        
        Returns:
            Tuple of (standard_loras, wanloras) where each is a list of dicts
            containing name, strength, and type information.
        """
        standard_loras = []
        wanloras = []
        
        # Pattern for both LoRA types: capture everything inside the tags
        # Both handle names with spaces and special characters the same way
        lora_pattern = r'<lora:(.+?)>'
        wanlora_pattern = r'<wanlora:(.+?)>'
        
        # Find standard LoRA tags
        for match in re.finditer(lora_pattern, prompt_text):
            content = match.group(1).strip()
            # Split by last colon to separate name from strength
            last_colon_index = content.rfind(':')
            if last_colon_index > 0:
                name = content[:last_colon_index].strip()
                strength = content[last_colon_index + 1:].strip()
                
                standard_loras.append({
                    'name': name,
                    'strength': strength,
                    'type': 'lora',
                    'tag': match.group(0)
                })
        
        # Find wanlora tags (same logic as standard LoRAs)
        for match in re.finditer(wanlora_pattern, prompt_text):
            content = match.group(1).strip()
            # Split by last colon to separate name from strength
            last_colon_index = content.rfind(':')
            if last_colon_index > 0:
                name = content[:last_colon_index].strip()
                strength = content[last_colon_index + 1:].strip()
                
                wanloras.append({
                    'name': name,
                    'strength': strength,
                    'type': 'wanlora',
                    'tag': match.group(0)
                })
            
        return standard_loras, wanloras
    
    def load_metadata(self, lora_name: str) -> Optional[Dict]:
        """
        Load metadata for a LoRA from its .metadata.json file.
        
        Args:
            lora_name: Name of the LoRA (without extension)
            
        Returns:
            Dict containing metadata or None if not found
        """
        if not self.loras_folder:
            return None
            
        metadata_path = os.path.join(self.loras_folder, f"{lora_name}.metadata.json")
        
        if not os.path.exists(metadata_path):
            # Try with .safetensors extension in name
            metadata_path = os.path.join(self.loras_folder, f"{lora_name}.safetensors.metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
            
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading metadata for {lora_name}: {e}")
            return None
    
    def extract_lora_info(self, lora_data: Dict, metadata: Optional[Dict]) -> Dict:
        """
        Extract relevant information from LoRA data and metadata.
        
        Args:
            lora_data: Parsed LoRA information (name, strength, type)
            metadata: Loaded metadata dict or None
            
        Returns:
            Dict with extracted information for display
        """
        info = {
            'name': lora_data['name'],
            'strength': lora_data['strength'],
            'type': lora_data['type'],
            'tag': lora_data['tag'],
            'trigger_words': [],
            'preview_url': None,
            'example_images': [],
            'model_description': None,
            'base_model': None,
            'nsfw_level': 0,
        }
        
        if metadata:
            # Extract trigger words
            if 'civitai' in metadata and 'trainedWords' in metadata['civitai']:
                info['trigger_words'] = metadata['civitai']['trainedWords']
            
            # Extract preview image
            if 'preview_url' in metadata:
                info['preview_url'] = metadata['preview_url']
            
            # Extract example images
            if 'civitai' in metadata and 'images' in metadata['civitai']:
                info['example_images'] = [
                    {
                        'url': img['url'],
                        'width': img.get('width', 0),
                        'height': img.get('height', 0),
                        'nsfw_level': img.get('nsfwLevel', 1)
                    }
                    for img in metadata['civitai']['images']
                ]
            
            # Extract model info
            if 'model_name' in metadata:
                info['model_name'] = metadata['model_name']
            if 'modelDescription' in metadata:
                info['model_description'] = metadata['modelDescription']
            if 'base_model' in metadata:
                info['base_model'] = metadata['base_model']
            if 'preview_nsfw_level' in metadata:
                info['nsfw_level'] = metadata['preview_nsfw_level']
                
        return info
    
    def format_lora_info(self, loras_info: List[Dict], title: str) -> str:
        """
        Format LoRA information for display.
        
        Args:
            loras_info: List of LoRA info dicts
            title: Title for this section
            
        Returns:
            Formatted string for display
        """
        if not loras_info:
            return f"{title}: None found\n"
            
        result = f"{title} ({len(loras_info)} found):\n"
        result += "=" * 50 + "\n"
        
        for i, lora in enumerate(loras_info, 1):
            result += f"\n{i}. {lora['name']} (strength: {lora['strength']})\n"
            result += f"   Tag: {lora['tag']}\n"
            
            if lora['trigger_words']:
                result += f"   Trigger words: {', '.join(lora['trigger_words'])}\n"
            else:
                result += f"   Trigger words: Not available\n"
                
            if lora['base_model']:
                result += f"   Base model: {lora['base_model']}\n"
                
            if lora['preview_url']:
                result += f"   Preview: Available\n"
            else:
                result += f"   Preview: Not available\n"
                
            if lora['example_images']:
                result += f"   Example images: {len(lora['example_images'])} available\n"
            else:
                result += f"   Example images: Not available\n"
                
            result += "\n"
            
        return result
    
    def visualize_loras(self, prompt_text: str) -> Tuple[str, str]:
        """
        Main function that processes the prompt and returns LoRA information.
        
        Args:
            prompt_text: Input prompt text containing LoRA tags
            
        Returns:
            Tuple of (raw_lora_info, original_prompt)
        """
        if not prompt_text.strip():
            return ("No prompt text provided.", prompt_text)
        
        # Parse LoRA tags from prompt
        standard_loras, wanloras = self.parse_lora_tags(prompt_text)
        
        # Debug logging
        print(f"DEBUG: Parsed {len(standard_loras)} standard LoRAs: {standard_loras}")
        print(f"DEBUG: Parsed {len(wanloras)} WanLoRAs: {wanloras}")
        
        if not standard_loras and not wanloras:
            return ("No LoRA tags found in prompt.", prompt_text)
        
        # Process standard LoRAs
        standard_loras_info = []
        for lora_data in standard_loras:
            metadata = self.load_metadata(lora_data['name'])
            info = self.extract_lora_info(lora_data, metadata)
            standard_loras_info.append(info)
        
        # Process wanloras
        wanloras_info = []
        for lora_data in wanloras:
            metadata = self.load_metadata(lora_data['name'])
            info = self.extract_lora_info(lora_data, metadata)
            wanloras_info.append(info)
        
        # Store visualization data for frontend access
        self.last_lora_data = {
            'standard_loras': standard_loras_info,
            'wanloras': wanloras_info,
            'prompt': prompt_text
        }
        
        # Send data to frontend via server message
        try:
            from server import PromptServer
            message_data = {
                "node_id": str(id(self)),
                "data": self.last_lora_data
            }
            print(f"DEBUG: Sending LoRA data to frontend: {message_data}")
            PromptServer.instance.send_sync("lora_visualization_data", message_data)
        except Exception as e:
            print(f"Failed to send LoRA visualization data: {e}")
        
        # Create raw metadata output for debugging/analysis
        raw_metadata = {
            "total_loras_found": len(standard_loras) + len(wanloras),
            "standard_loras_count": len(standard_loras),
            "wanloras_count": len(wanloras),
            "standard_loras": [
                {
                    "name": lora['name'],
                    "strength": lora['strength'],
                    "tag": lora['tag'],
                    "trigger_words": lora.get('trigger_words', []),
                    "base_model": lora.get('base_model', 'Unknown'),
                    "has_metadata": bool(lora.get('preview_url') or lora.get('trigger_words'))
                }
                for lora in standard_loras_info
            ],
            "wanloras": [
                {
                    "name": lora['name'],
                    "strength": lora['strength'],
                    "tag": lora['tag'],
                    "trigger_words": lora.get('trigger_words', []),
                    "base_model": lora.get('base_model', 'Unknown'),
                    "has_metadata": bool(lora.get('preview_url') or lora.get('trigger_words'))
                }
                for lora in wanloras_info
            ]
        }
        
        # Convert to readable string format
        import json
        raw_info_output = json.dumps(raw_metadata, indent=2, ensure_ascii=False)
        
        return (raw_info_output, prompt_text)


# Add API endpoint for frontend to fetch LoRA metadata
@PromptServer.instance.routes.get("/lora_metadata/{lora_name}")
async def get_lora_metadata(request):
    """API endpoint to get LoRA metadata for a specific LoRA"""
    lora_name = request.match_info.get('lora_name')
    
    if not lora_name:
        return web.json_response({"error": "No LoRA name provided"}, status=400)
    
    # Get loras folder path
    loras_folder = folder_paths.get_folder_paths("loras")[0] if folder_paths.get_folder_paths("loras") else None
    
    if not loras_folder:
        return web.json_response({"error": "LoRAs folder not found"}, status=404)
    
    # Try to find metadata file
    metadata_path = os.path.join(loras_folder, f"{lora_name}.metadata.json")
    
    if not os.path.exists(metadata_path):
        # Try with .safetensors extension in name
        metadata_path = os.path.join(loras_folder, f"{lora_name}.safetensors.metadata.json")
    
    if not os.path.exists(metadata_path):
        return web.json_response({"error": "Metadata not found"}, status=404)
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return web.json_response(metadata)
    except (json.JSONDecodeError, IOError) as e:
        return web.json_response({"error": f"Failed to load metadata: {str(e)}"}, status=500)