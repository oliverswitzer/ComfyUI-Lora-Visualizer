"""
Pytest configuration for LoRA Visualizer tests
"""

import sys
from unittest.mock import Mock

# Mock ComfyUI dependencies before any imports
def pytest_configure():
    """Configure pytest with necessary mocks"""
    
    # Mock folder_paths
    mock_folder_paths = Mock()
    mock_folder_paths.get_folder_paths.return_value = ['/fake/loras/path']
    sys.modules['folder_paths'] = mock_folder_paths
    
    # Mock server
    mock_server = Mock()
    mock_prompt_server = Mock()
    mock_prompt_server.instance = Mock()
    mock_prompt_server.instance.send_sync = Mock()
    mock_server.PromptServer = mock_prompt_server
    sys.modules['server'] = mock_server
    
    # Mock aiohttp
    sys.modules['aiohttp'] = Mock()
    mock_web = Mock()
    sys.modules['aiohttp.web'] = mock_web