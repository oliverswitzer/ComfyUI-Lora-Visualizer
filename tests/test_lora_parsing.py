"""
Tests for LoRA Visualizer Node functionality
"""

import unittest
import sys
import os
import json
from unittest.mock import Mock, patch

# Mock ComfyUI dependencies before importing
sys.modules['folder_paths'] = Mock()
sys.modules['server'] = Mock()
sys.modules['aiohttp'] = Mock()

# Add current directory to import our nodes
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from nodes.lora_visualizer_node import LoRAVisualizerNode


class TestLoRAVisualizerNode(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock folder_paths to return None for loras folder
        with patch('folder_paths.get_folder_paths', return_value=[]):
            self.node = LoRAVisualizerNode()
    
    def test_parse_standard_lora_tags(self):
        """Test parsing of standard LoRA tags."""
        prompt = "A beautiful landscape <lora:landscape_v1:0.8> with mountains"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 1)
        self.assertEqual(len(wanloras), 0)
        self.assertEqual(standard_loras[0]['name'], 'landscape_v1')
        self.assertEqual(standard_loras[0]['strength'], '0.8')
        self.assertEqual(standard_loras[0]['type'], 'lora')
    
    def test_parse_wanlora_tags(self):
        """Test parsing of wanlora tags."""
        prompt = "Portrait of a woman <wanlora:Woman877.v2:1.0> smiling"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 0)
        self.assertEqual(len(wanloras), 1)
        self.assertEqual(wanloras[0]['name'], 'Woman877.v2')
        self.assertEqual(wanloras[0]['strength'], '1.0')
        self.assertEqual(wanloras[0]['type'], 'wanlora')
    
    def test_parse_mixed_lora_tags(self):
        """Test parsing of both standard and wanlora tags."""
        prompt = "Portrait <lora:style_v1:0.5> of woman <wanlora:Woman877.v2:0.8> in park"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 1)
        self.assertEqual(len(wanloras), 1)
        self.assertEqual(standard_loras[0]['name'], 'style_v1')
        self.assertEqual(wanloras[0]['name'], 'Woman877.v2')

    def test_parse_wanlora_with_spaces(self):
        """Test parsing of wanlora tags with spaces and special characters."""
        prompt = "test <wanlora:DetailAmplifier wan480p v1.0:1> more text"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 0)
        self.assertEqual(len(wanloras), 1)
        
        wanlora = wanloras[0]
        self.assertEqual(wanlora['name'], 'DetailAmplifier wan480p v1.0')
        self.assertEqual(wanlora['strength'], '1')
        self.assertEqual(wanlora['type'], 'wanlora')
        self.assertEqual(wanlora['tag'], '<wanlora:DetailAmplifier wan480p v1.0:1>')

    def test_parse_complex_wanlora_names(self):
        """Test parsing complex wanlora names with multiple colons."""
        prompt = "test <wanlora:Model Name: v2.0: Enhanced Edition:0.5> more text"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 0)
        self.assertEqual(len(wanloras), 1)
        
        wanlora = wanloras[0]
        self.assertEqual(wanlora['name'], 'Model Name: v2.0: Enhanced Edition')
        self.assertEqual(wanlora['strength'], '0.5')
        self.assertEqual(wanlora['type'], 'wanlora')
        self.assertEqual(wanlora['tag'], '<wanlora:Model Name: v2.0: Enhanced Edition:0.5>')

    def test_parse_lora_with_spaces(self):
        """Test parsing of standard LoRA tags with spaces and special characters."""
        prompt = "test <lora:Detail Enhancer v2.0: Professional Edition:0.8> more text"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 1)
        self.assertEqual(len(wanloras), 0)
        
        lora = standard_loras[0]
        self.assertEqual(lora['name'], 'Detail Enhancer v2.0: Professional Edition')
        self.assertEqual(lora['strength'], '0.8')
        self.assertEqual(lora['type'], 'lora')
        self.assertEqual(lora['tag'], '<lora:Detail Enhancer v2.0: Professional Edition:0.8>')

    def test_parse_consistent_handling(self):
        """Test that both LoRA types handle complex names consistently."""
        prompt = "test <lora:Style: Modern Art v1.0:0.7> and <wanlora:Character: Anime Girl v2.1:0.9> together"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 1)
        self.assertEqual(len(wanloras), 1)
        
        # Both should parse complex names with colons correctly
        lora = standard_loras[0]
        self.assertEqual(lora['name'], 'Style: Modern Art v1.0')
        self.assertEqual(lora['strength'], '0.7')
        
        wanlora = wanloras[0]
        self.assertEqual(wanlora['name'], 'Character: Anime Girl v2.1')
        self.assertEqual(wanlora['strength'], '0.9')
    
    def test_parse_multiple_loras(self):
        """Test parsing multiple LoRA tags of the same type."""
        prompt = "A scene <lora:style1:0.5> with <lora:style2:0.3> and <lora:style3:1.0>"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 3)
        self.assertEqual(len(wanloras), 0)
        
        names = [lora['name'] for lora in standard_loras]
        self.assertIn('style1', names)
        self.assertIn('style2', names)
        self.assertIn('style3', names)
    
    def test_parse_no_lora_tags(self):
        """Test handling of prompts with no LoRA tags."""
        prompt = "A simple prompt with no LoRA tags"
        standard_loras, wanloras = self.node.parse_lora_tags(prompt)
        
        self.assertEqual(len(standard_loras), 0)
        self.assertEqual(len(wanloras), 0)
    
    def test_extract_lora_info_no_metadata(self):
        """Test extracting LoRA info when no metadata is available."""
        lora_data = {
            'name': 'test_lora',
            'strength': '0.8',
            'type': 'lora',
            'tag': '<lora:test_lora:0.8>'
        }
        
        info = self.node.extract_lora_info(lora_data, None)
        
        self.assertEqual(info['name'], 'test_lora')
        self.assertEqual(info['strength'], '0.8')
        self.assertEqual(info['type'], 'lora')
        self.assertEqual(info['trigger_words'], [])
        self.assertIsNone(info['preview_url'])
        self.assertEqual(info['example_images'], [])
    
    def test_extract_lora_info_with_metadata(self):
        """Test extracting LoRA info with metadata."""
        lora_data = {
            'name': 'Woman877.v2',
            'strength': '1.0',
            'type': 'wanlora',
            'tag': '<wanlora:Woman877.v2:1.0>'
        }
        
        # Mock metadata similar to the provided example
        metadata = {
            'civitai': {
                'trainedWords': ['woman877'],
                'images': [
                    {
                        'url': 'https://example.com/image1.jpg',
                        'width': 768,
                        'height': 1152,
                        'nsfwLevel': 1
                    }
                ]
            },
            'model_name': 'Test Model',
            'base_model': 'SDXL 1.0',
            'preview_url': '/path/to/preview.webp',
            'preview_nsfw_level': 1
        }
        
        info = self.node.extract_lora_info(lora_data, metadata)
        
        self.assertEqual(info['name'], 'Woman877.v2')
        self.assertEqual(info['trigger_words'], ['woman877'])
        self.assertEqual(info['preview_url'], '/path/to/preview.webp')
        self.assertEqual(len(info['example_images']), 1)
        self.assertEqual(info['base_model'], 'SDXL 1.0')
        self.assertEqual(info['nsfw_level'], 1)
    
    def test_format_lora_info_empty(self):
        """Test formatting empty LoRA info."""
        result = self.node.format_lora_info([], "Test Section")
        self.assertIn("Test Section: None found", result)
    
    def test_format_lora_info_with_data(self):
        """Test formatting LoRA info with data."""
        loras_info = [{
            'name': 'test_lora',
            'strength': '0.8',
            'tag': '<lora:test_lora:0.8>',
            'trigger_words': ['test', 'word'],
            'base_model': 'SDXL 1.0',
            'preview_url': '/path/to/preview.jpg',
            'example_images': [{'url': 'test1.jpg'}, {'url': 'test2.jpg'}]
        }]
        
        result = self.node.format_lora_info(loras_info, "Test Section")
        
        self.assertIn("Test Section (1 found)", result)
        self.assertIn("test_lora", result)
        self.assertIn("strength: 0.8", result)
        self.assertIn("test, word", result)
        self.assertIn("SDXL 1.0", result)
        self.assertIn("2 available", result)


class TestVisualizeLoras(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock folder_paths to return None for loras folder
        with patch('folder_paths.get_folder_paths', return_value=[]):
            self.node = LoRAVisualizerNode()
    
    def test_visualize_empty_prompt(self):
        """Test handling of empty prompt."""
        result, processed_prompt = self.node.visualize_loras("")
        
        self.assertEqual(result, "No prompt text provided.")
        self.assertEqual(processed_prompt, "")
    
    def test_visualize_no_loras(self):
        """Test handling of prompt with no LoRAs."""
        prompt = "A simple prompt with no LoRA tags"
        result, processed_prompt = self.node.visualize_loras(prompt)
        
        self.assertEqual(result, "No LoRA tags found in prompt.")
        self.assertEqual(processed_prompt, prompt)
    
    def test_visualize_with_loras(self):
        """Test visualization with LoRA tags."""
        prompt = "Portrait <lora:style_v1:0.5> of woman <wanlora:Woman877.v2:0.8>"
        result, processed_prompt = self.node.visualize_loras(prompt)
        
        # Parse JSON result
        import json
        metadata = json.loads(result)
        
        self.assertEqual(metadata["total_loras_found"], 2)
        self.assertEqual(metadata["standard_loras_count"], 1)
        self.assertEqual(metadata["wanloras_count"], 1)
        self.assertEqual(metadata["standard_loras"][0]["name"], "style_v1")
        self.assertEqual(metadata["wanloras"][0]["name"], "Woman877.v2")
        self.assertEqual(processed_prompt, prompt)


if __name__ == '__main__':
    unittest.main()