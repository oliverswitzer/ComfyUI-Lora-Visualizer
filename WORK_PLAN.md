# HIGH/LOW LoRA Splitting Refactor Work Plan

## Overview
Refactor HIGH/LOW LoRA splitting to have predictable default behavior with optional advanced matching.

## Tasks

### 1. Add Boolean Flag to Control Behavior
- [x] Add `find_matching_high_low_lora` parameter to `PromptSplitterNode`
- [x] Add `find_matching_high_low_lora` parameter to `LoRAHighLowSplitterNode`
- [x] Set default to `false` for predictable behavior
- [x] Add comprehensive documentation explaining both modes
- [x] Update node tooltips and descriptions

**Mode 1 (Default: `false`)**: Simple extraction/separation of existing HIGH/LOW LoRAs in prompt
**Mode 2 (`true`)**: Advanced fuzzy matching + Ollama to find and add missing pairs

### 2. Move Unit Tests to Proper Location
- [x] Move `split_prompt_by_lora_high_low_with_ollama` tests from `test_prompt_splitter.py` to `test_lora_metadata_utils.py`
- [x] Move `classify_lora_pairs_with_ollama` tests to `test_lora_metadata_utils.py`
- [x] Move `find_lora_pair_fuzzy` tests to `test_lora_metadata_utils.py`
- [x] ~~Move `determine_high_low_lexicographic` tests to `test_lora_metadata_utils.py`~~ (function removed - was flawed)
- [x] Move `parse_ollama_classification_response` tests to `test_lora_metadata_utils.py`
- [x] Add comprehensive test coverage for all shared utility functions

### 3. Update Node Tests with Mocks
- [x] Update `test_prompt_splitter.py` to mock shared functions instead of testing through them
- [x] Update `test_lora_high_low_splitter.py` to mock shared functions
- [x] Focus node tests on node-specific behavior and integration
- [x] Remove redundant tests that now exist in `test_lora_metadata_utils.py`

### 4. Implement Two Operation Modes
- [x] Update `split_prompt_by_lora_high_low_with_ollama` to handle both modes
- [x] Create simple extraction mode (pattern-based, no fuzzy matching)
- [x] Keep advanced mode (fuzzy matching + Ollama classification)
- [x] Ensure backward compatibility
- [x] Update function signatures and documentation
- [x] Remove `find_matching_high_low_lora` param from PromptSplitterNode (opaque to users)

### 5. Final Validation
- [x] Run `pdm format`
- [x] Run `pdm lint`
- [x] Run `pdm test` (all tests pass)
- [x] Verify both nodes work correctly in both modes
- [x] Test edge cases and error handling

## Expected Benefits
- ✅ Predictable default behavior (no surprises)
- ✅ Advanced features available when needed
- ✅ Proper test architecture (shared utilities tested directly)
- ✅ Clear separation of concerns
- ✅ No real Ollama calls in tests

---
*This file will be deleted before merging*