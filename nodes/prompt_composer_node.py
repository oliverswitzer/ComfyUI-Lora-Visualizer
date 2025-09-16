"""
LoRA Prompt Composer Node Implementation

A ComfyUI node that takes natural language scene descriptions and intelligently
composes prompts with optimal LoRA combinations, weights, and trigger words.
"""

import json
import re
from typing import Any, Optional

from .logging_utils import log, log_debug, log_error
from .lora_metadata_utils import (
    classify_lora_type,
    discover_all_loras,
    extract_embeddable_content,
    extract_example_prompts,
    extract_recommended_weight,
)


class PromptComposerNode:
    """
    A ComfyUI node that composes prompts with optimal LoRA combinations.

    Features:
    - Natural language scene description input
    - Intelligent LoRA discovery (separate image and video)
    - Automatic weight optimization
    - Trigger word integration
    - Style mimicry from example prompts
    """

    CATEGORY = "conditioning"
    DESCRIPTION = """Composes prompts with optimal LoRA combinations from natural language.

This node

• Takes natural language scene descriptions as input
• Discovers relevant image and video LoRAs separately
• Generates complete prompts with proper LoRA tags and weights
• Includes appropriate trigger words from LoRA metadata
• Mimics writing patterns from successful example prompts
• Supports configurable limits for image and video LoRAs
• Optimized for creative content workflows
"""

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input schema for this ComfyUI node."""
        return {
            "required": {
                "scene_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Describe your scene in natural language...",
                        "tooltip": (
                            "Natural language description of the desired scene. "
                            "The node will find matching LoRAs and compose an "
                            "optimal prompt."
                        ),
                    },
                ),
                "max_image_loras": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 10,
                        "tooltip": (
                            "Maximum number of image LoRAs to include in the composed prompt"
                        ),
                    },
                ),
                "max_video_loras": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 5,
                        "tooltip": (
                            "Maximum number of video LoRAs to include in the composed prompt"
                        ),
                    },
                ),
            },
            "optional": {
                "content_boost": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.5,
                        "max": 2.0,
                        "tooltip": (
                            "Boost factor for content-specific LoRAs (character, pose, etc.)"
                        ),
                    },
                ),
                "style_preference": (
                    ["technical", "artistic", "natural"],
                    {
                        "default": "natural",
                        "tooltip": "Prompt writing style preference",
                    },
                ),
                "image_lora_dir_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "e.g., characters, styles",
                        "tooltip": (
                            "Optional subdirectory path within models/loras to limit image LoRA search. "
                            "Leave empty to search all directories."
                        ),
                    },
                ),
                "wan_lora_dir_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "e.g., motion, video",
                        "tooltip": (
                            "Optional subdirectory path within models/loras to limit video LoRA search. "
                            "Leave empty to search all directories."
                        ),
                    },
                ),
                "default_lora_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Default weight for all LoRAs (overrides metadata recommendations)",
                    },
                ),
                "low_lora_weight_offset": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": (
                            "Amount to reduce LOW LoRA weights by (e.g., 0.2 means HIGH=1.0, LOW=0.8). "
                            "Only applies to WAN 2.2 LOW LoRAs."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("composed_prompt", "lora_analysis", "metadata_summary")
    OUTPUT_TOOLTIPS = (
        "Complete prompt with LoRA tags, trigger words, and scene description",
        "Detailed analysis of selected LoRAs and their relevance scores",
        "Summary of metadata and processing statistics",
    )
    FUNCTION = "compose_prompt"
    OUTPUT_NODE = True

    def __init__(self):
        """Initialize the prompt composer with lazy-loaded embeddings."""
        self._embeddings_initialized = False
        self._lora_database = {}
        self._embedding_model = None
        self._lora_embeddings = {}

    def _initialize_embeddings(self) -> bool:
        """
        Initialize the embeddings system if not already done.

        Returns:
            True if successful, False otherwise
        """
        if self._embeddings_initialized:
            return True

        try:
            # Use TF-IDF from scikit-learn (much simpler, no extra dependencies)
            from sklearn.feature_extraction.text import TfidfVectorizer

            log("Initializing TF-IDF embeddings model...")

            # Create TF-IDF vectorizer
            self._embedding_model = TfidfVectorizer(
                stop_words="english",
                max_features=1000,  # Limit vocabulary size
                ngram_range=(1, 2),  # Use unigrams and bigrams
                lowercase=True,
                strip_accents="unicode",
            )

            # Discover all LoRAs and prepare embeddings
            log("Discovering LoRAs and generating embeddings...")
            self._lora_database = discover_all_loras()

            # Collect all LoRA text content for TF-IDF fitting
            lora_documents = []
            lora_names = []

            for lora_name, lora_info in self._lora_database.items():
                log_debug(f"Processing LoRA: {lora_name}")
                metadata = lora_info["metadata"]
                if metadata:
                    embeddable_text = extract_embeddable_content(metadata)
                    if embeddable_text and embeddable_text.strip():
                        lora_documents.append(embeddable_text)
                        lora_names.append(lora_name)
                        log_debug(f"  Added text for {lora_name}: '{embeddable_text[:50]}...'")
                    else:
                        log_debug(f"  No embeddable text for {lora_name}")
                else:
                    log_debug(f"  No metadata for {lora_name}")

            if lora_documents:
                log(f"Fitting TF-IDF on {len(lora_documents)} LoRA documents...")
                # Fit TF-IDF on all documents at once
                tfidf_matrix = self._embedding_model.fit_transform(lora_documents)

                # Store embeddings as sparse vectors (much more memory efficient)
                for i, lora_name in enumerate(lora_names):
                    self._lora_embeddings[lora_name] = tfidf_matrix[i]

                log(f"Generated TF-IDF embeddings for {len(lora_names)} LoRAs")
            else:
                log_error("No LoRA documents found for TF-IDF fitting")

            self._embeddings_initialized = True
            log(f"Embeddings initialized for {len(self._lora_embeddings)} LoRAs")
            return True

        except ImportError as e:
            log_error(f"Failed to import required dependencies: {e}")
            log_error("Please install with: pip install scikit-learn")
            return False
        except Exception as e:
            log_error(f"Failed to initialize embeddings: {e}")
            return False

    def _find_wan_lora_pair(self, selected_lora_name: str) -> Optional[str]:
        """
        Find the matching HIGH/LOW pair for a WAN LoRA using simple word replacement.

        Args:
            selected_lora_name: Name of the selected WAN LoRA

        Returns:
            Name of the matching pair LoRA, or None if no pair found
        """
        import re

        log_debug(f"Looking for pair for: {selected_lora_name}")

        name_lower = selected_lora_name.lower()

        # Check if it contains "high" or "low" (case insensitive)
        has_high = "high" in name_lower
        has_low = "low" in name_lower

        log_debug(f"  has_high: {has_high}, has_low: {has_low}")

        if not (has_high or has_low):
            log_debug("  Not a high/low variant, skipping")
            return None  # Not a high/low variant

        # Simple replacement: swap "high" with "low" and vice versa (preserve case)
        if has_high:
            # Replace preserving case: HIGH->LOW, High->Low, high->low
            def replace_high(match):
                original = match.group(0)
                if original.isupper():
                    return "LOW"
                elif original.istitle():
                    return "Low"
                else:
                    return "low"

            pair_name = re.sub(r"high", replace_high, selected_lora_name, flags=re.IGNORECASE)
        else:  # has_low
            # Replace preserving case: LOW->HIGH, Low->High, low->high
            def replace_low(match):
                original = match.group(0)
                if original.isupper():
                    return "HIGH"
                elif original.istitle():
                    return "High"
                else:
                    return "high"

            pair_name = re.sub(r"low", replace_low, selected_lora_name, flags=re.IGNORECASE)

        log_debug(f"  Generated pair name: {pair_name}")

        # Check if the pair exists in available LoRAs
        exists = pair_name in self._lora_database
        log_debug(f"  Pair exists in database: {exists}")

        if exists:
            log(f"WAN 2.2 pair found: {selected_lora_name} <-> {pair_name}")
            return pair_name
        else:
            log_debug(f"  Pair not found. Database has {len(self._lora_database)} LoRAs")

            # Show all LoRAs containing "high" or "low" for debugging
            high_low_loras = [
                name
                for name in self._lora_database.keys()
                if "high" in name.lower() or "low" in name.lower()
            ]
            if high_low_loras:
                log_debug(f"  HIGH/LOW LoRAs in database: {high_low_loras[:10]}")  # Show first 10

            # Log some similar names for debugging
            similar_names = [
                name
                for name in self._lora_database.keys()
                if any(word in name.lower() for word in pair_name.lower().split()[:3])
            ][:5]
            if similar_names:
                log_debug(f"  Similar names in database: {similar_names}")
            return None

    def _apply_wan_2_2_pairing(
        self, video_loras: list[dict[str, Any]], max_count: int
    ) -> list[dict[str, Any]]:
        """
        Post-process video LoRA selection to add WAN 2.2 high/low pairs.
        Respects the max_count limit by treating pairs as single units.

        Args:
            video_loras: List of selected video LoRAs
            max_count: Maximum number of LoRA "units" (pairs count as 1 unit)

        Returns:
            Expanded list with high/low pairs added for WAN 2.2 LoRAs
        """
        from .lora_metadata_utils import is_wan_2_2_lora

        expanded_loras = []
        units_used = 0

        for lora in video_loras:
            if units_used >= max_count:
                log(f"⚠️ Reached max_video_loras limit ({max_count}), skipping remaining LoRAs")
                break

            expanded_loras.append(lora)  # Always add the original

            # Check if this is a WAN 2.2 LoRA
            metadata = lora.get("metadata")
            if metadata and is_wan_2_2_lora(metadata):
                log_debug(f"Found WAN 2.2 LoRA: {lora['name']}")
                # Try to find the matching pair
                pair_name = self._find_wan_lora_pair(lora["name"])

                if pair_name and pair_name in self._lora_database:
                    # Add the pair LoRA with same relevance score and proper structure
                    pair_db_info = self._lora_database[pair_name]
                    pair_metadata = pair_db_info["metadata"]
                    pair_info = {
                        "name": pair_name,
                        "metadata": pair_metadata,
                        "relevance_score": lora.get("relevance_score", 0.0),
                        "recommended_weight": extract_recommended_weight(pair_metadata),
                        "trigger_words": pair_db_info.get("trigger_words", []),
                        "type": "video",
                    }
                    expanded_loras.append(pair_info)

                    log(f"🔗 WAN 2.2 pairing: {lora['name']} + {pair_name}")
                else:
                    log(f"⚠️ WAN 2.2 LoRA {lora['name']} has no matching pair")

            units_used += 1  # Each original LoRA (with or without pair) counts as 1 unit

        return expanded_loras

    def _find_relevant_loras(
        self,
        scene_description: str,
        lora_type: str,
        max_count: int,
        content_boost: float,
        directory_filter: str = "",
    ) -> list[dict[str, Any]]:
        """
        Find relevant LoRAs for the scene description.

        Args:
            scene_description: Natural language description
            lora_type: "image" or "video"
            max_count: Maximum number of LoRAs to return
            content_boost: Boost factor for content-specific LoRAs

        Returns:
            List of relevant LoRA info dicts with scores, ordered by similarity (top N)
        """
        if not self._embeddings_initialized or not self._embedding_model:
            log_error("Embeddings not initialized")
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Clean scene description to remove existing LoRA tags
            clean_scene = re.sub(r"<(?:lora|wanlora):[^>]+>", "", scene_description)
            clean_scene = " ".join(clean_scene.split())  # Remove extra whitespace
            preview = clean_scene[:50] + ("..." if len(clean_scene) > 50 else "")
            log_debug(f"  🧹 Cleaned scene for embedding: '{preview}'")

            # Generate TF-IDF embedding for cleaned scene description
            scene_embedding = self._embedding_model.transform([clean_scene])

            relevant_loras = []

            for lora_name, lora_info in self._lora_database.items():
                metadata = lora_info["metadata"]
                if not metadata:
                    continue

                # Filter by LoRA type
                actual_type = classify_lora_type(metadata)
                if actual_type != lora_type:
                    continue

                # Filter by directory if specified
                if directory_filter:
                    lora_dir = lora_info.get("directory", "")
                    # Normalize path separators for cross-platform compatibility
                    normalized_lora_dir = lora_dir.replace("\\", "/")
                    normalized_filter = directory_filter.replace("\\", "/")
                    log_debug(
                        f"  🔍 Checking directory filter: '{normalized_filter}' vs '{normalized_lora_dir}'"
                    )
                    if not normalized_lora_dir.startswith(normalized_filter):
                        log_debug(
                            f"  📁 Skipping {lora_name}: not in directory '{directory_filter}' (found in '{lora_dir}')"
                        )
                        continue
                    else:
                        log_debug(
                            f"  ✅ {lora_name}: matches directory filter '{directory_filter}'"
                        )

                # Calculate relevance score
                if lora_name in self._lora_embeddings:
                    lora_embedding = self._lora_embeddings[lora_name]

                    # Calculate cosine similarity between sparse matrices
                    similarity = cosine_similarity(scene_embedding, lora_embedding)[0][0]
                    log_debug(f"  📊 {lora_name} base similarity: {similarity:.4f}")

                    # Apply content boost for character/pose LoRAs
                    if self._is_content_lora(metadata):
                        old_similarity = similarity
                        similarity *= content_boost
                        log_debug(
                            f"  🚀 {lora_name} content boost applied: "
                            f"{old_similarity:.4f} → {similarity:.4f}"
                        )
                    else:
                        log_debug(f"  📋 {lora_name} no content boost (not content-specific)")

                    # Apply keyword matching bonus for exact position matches
                    keyword_boost = self._get_keyword_boost(scene_description, metadata, lora_name)
                    if keyword_boost > 1.0:
                        old_similarity = similarity
                        similarity *= keyword_boost
                        log_debug(
                            f"  🎯 {lora_name} keyword boost applied: "
                            f"{old_similarity:.4f} → {similarity:.4f}"
                        )

                    # Add to candidates list (no threshold filtering)
                    relevant_loras.append(
                        {
                            "name": lora_name,
                            "metadata": metadata,
                            "relevance_score": float(similarity),
                            "recommended_weight": extract_recommended_weight(metadata),
                            "trigger_words": lora_info["trigger_words"],
                            "type": lora_type,
                        }
                    )

            # Sort by relevance score and limit
            relevant_loras.sort(key=lambda x: x["relevance_score"], reverse=True)
            return relevant_loras[:max_count]

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            log_error(f"Error finding relevant LoRAs: {e}\nTraceback:\n{tb}")
            return []

    def _is_content_lora(self, metadata: dict[str, Any]) -> bool:
        """
        Determine if a LoRA is content-specific (character, pose, etc.).

        Args:
            metadata: LoRA metadata dictionary

        Returns:
            True if this is a content-specific LoRA
        """
        content_tags = {
            "character",
            "pose",
            "clothing",
            "action",
            "expression",
            "anatomy",
            "body",
            "face",
            "hair",
            "eyes",
            "position",  # Added: position-specific content
            "motion",  # Added: motion-specific content
            "pov",  # Added: POV is content-specific
            "concept",  # Added: concept LoRAs are usually content-specific
        }

        # Check civitai tags
        civitai = metadata.get("civitai")
        if isinstance(civitai, dict):
            model = civitai.get("model")
            if isinstance(model, dict):
                model_tags = model.get("tags", [])
                if isinstance(model_tags, list):
                    if any(
                        isinstance(tag, str) and tag.lower() in content_tags for tag in model_tags
                    ):
                        return True
            # Check if it has trigger words (often indicates character LoRAs)
            trained_words = civitai.get("trainedWords")
            if isinstance(trained_words, list) and any(trained_words):
                return True

        # Check top-level tags
        tags = metadata.get("tags")
        if isinstance(tags, list):
            if any(isinstance(tag, str) and tag.lower() in content_tags for tag in tags):
                return True

        return False

    def _get_keyword_boost(
        self, scene_description: str, metadata: dict[str, Any], lora_name: str
    ) -> float:
        """
        Calculate keyword matching boost for LoRAs with exact position/concept matches.

        Args:
            scene_description: User's scene description
            metadata: LoRA metadata dictionary
            lora_name: Name of the LoRA

        Returns:
            Boost multiplier (1.0 = no boost, >1.0 = boost)
        """
        # Clean the scene description to remove any existing LoRA tags
        clean_scene = re.sub(r"<(?:lora|wanlora):[^>]+>", "", scene_description)
        clean_scene = " ".join(clean_scene.split())  # Remove extra whitespace
        scene_lower = clean_scene.lower()
        boost = 1.0

        log_debug(f"    🔍 Keyword boost check for {lora_name}")
        orig_preview = scene_description[:50] + ("..." if len(scene_description) > 50 else "")
        clean_preview = clean_scene[:50] + ("..." if len(clean_scene) > 50 else "")
        log_debug(f"    📝 Original scene: '{orig_preview}'")
        log_debug(f"    🧹 Cleaned scene: '{clean_preview}'")
        log_debug(f"    📁 LoRA name: '{lora_name.lower()}'")

        # Position-specific keyword matching
        position_keywords = {
            "pose": ["pose", "position", "stance"],
            "style": ["style", "aesthetic", "look"],
            "motion": ["motion", "movement", "action"],
            "pov": ["pov", "point of view", "first person"],
        }

        # Find the best matching position and apply only one boost
        best_boost = 1.0
        best_position = None
        best_source = None

        lora_name_lower = lora_name.lower()
        lora_tags = metadata.get("tags", [])
        lora_tags_lower = (
            [tag.lower() for tag in lora_tags if isinstance(tag, str)]
            if isinstance(lora_tags, list)
            else []
        )
        civitai_tags = []
        civitai = metadata.get("civitai")
        if isinstance(civitai, dict):
            model = civitai.get("model")
            if isinstance(model, dict):
                civitai_tags = [
                    tag.lower() for tag in model.get("tags", []) if isinstance(tag, str)
                ]

        log_debug(f"    📋 LoRA tags: {lora_tags_lower}")
        log_debug(f"    🌐 Civitai tags: {civitai_tags}")

        for position, keywords in position_keywords.items():
            scene_has_keyword = any(keyword in scene_lower for keyword in keywords)
            if not scene_has_keyword:
                continue

            # Check LoRA name (highest priority - 1.5x boost)
            lora_has_keyword = any(keyword in lora_name_lower for keyword in keywords)
            if lora_has_keyword and 1.5 > best_boost:
                best_boost = 1.5
                best_position = position
                best_source = "title"

            # Check LoRA tags (medium priority - 1.3x boost)
            tag_has_keyword = any(keyword in lora_tags_lower for keyword in keywords)
            if tag_has_keyword and 1.3 > best_boost:
                best_boost = 1.3
                best_position = position
                best_source = "tags"

            # Check civitai tags (lowest priority - 1.3x boost)
            civitai_has_keyword = any(keyword in civitai_tags for keyword in keywords)
            if civitai_has_keyword and 1.3 > best_boost:
                best_boost = 1.3
                best_position = position
                best_source = "civitai"

            log_debug(f"    🎯 Position '{position}': Scene has keyword, checking sources...")
            log_debug(f"      📁 Title match: {lora_has_keyword}")
            log_debug(f"      📋 Tag match: {tag_has_keyword}")
            log_debug(f"      🌐 Civitai match: {civitai_has_keyword}")

        # Apply the best boost found
        if best_boost > 1.0:
            boost *= best_boost
            log_debug(
                f"    ✨ Applied {best_boost}x boost for '{best_position}' match in {best_source}!"
            )

        log_debug(f"    📊 Final keyword boost for {lora_name}: {boost}x")
        return boost

    def _analyze_prompt_style(self, example_prompts: list[str]) -> dict[str, Any]:
        """
        Analyze example prompts to extract style patterns.

        Args:
            example_prompts: List of example prompt texts

        Returns:
            Dict with style analysis results
        """
        if not example_prompts:
            return {"patterns": [], "common_terms": [], "structure": "simple"}

        # Extract common patterns
        all_text = " ".join(example_prompts)

        # Find common technical terms
        technical_terms = re.findall(
            r"\b(?:masterpiece|high quality|detailed|cinematic|depth of field|"
            r"bokeh|sharp focus|professional|4k|8k|ultra|highly detailed)\b",
            all_text,
            re.IGNORECASE,
        )

        # Find common artistic terms
        artistic_terms = re.findall(
            r"\b(?:beautiful|gorgeous|stunning|elegant|graceful|ethereal|"
            r"dramatic|moody|atmospheric|dreamy|fantasy|magical)\b",
            all_text,
            re.IGNORECASE,
        )

        # Analyze structure
        avg_length = sum(len(prompt.split()) for prompt in example_prompts) / len(example_prompts)
        structure = "complex" if avg_length > 20 else "simple"

        return {
            "patterns": list(set(technical_terms + artistic_terms)),
            "common_terms": list(set(technical_terms)),
            "artistic_terms": list(set(artistic_terms)),
            "structure": structure,
            "avg_length": avg_length,
        }

    def _compose_final_prompt(
        self,
        scene_description: str,
        image_loras: list[dict[str, Any]],
        video_loras: list[dict[str, Any]],
        style_preference: str,
        default_lora_weight: float = 1.0,
        low_lora_weight_offset: float = 0.2,
    ) -> str:
        """
        Compose the final prompt with LoRA tags and styling.

        Args:
            scene_description: Original scene description
            image_loras: Selected image LoRAs
            video_loras: Selected video LoRAs
            style_preference: Style preference setting
            default_lora_weight: Default weight for all LoRAs
            low_lora_weight_offset: Amount to reduce LOW LoRA weights by

        Returns:
            Composed prompt string
        """
        try:
            log_debug("_compose_final_prompt: Starting composition")
            prompt_parts = []

            # Add LoRA tags at the beginning
            log_debug("_compose_final_prompt: Adding image LoRA tags")
            for _i, lora in enumerate(image_loras):
                weight = default_lora_weight
                tag = f"<lora:{lora['name']}:{weight}>"
                prompt_parts.append(tag)

            log_debug("_compose_final_prompt: Adding video LoRA tags")
            for _i, lora in enumerate(video_loras):
                # Apply LOW LoRA offset for WAN 2.2 LOW LoRAs
                weight = default_lora_weight
                lora_name_lower = lora["name"].lower()
                if "low" in lora_name_lower:
                    weight = max(0.1, weight - low_lora_weight_offset)
                    log_debug(f"Applied LOW offset to {lora['name']}: {weight}")

                tag = f"<wanlora:{lora['name']}:{weight}>"
                prompt_parts.append(tag)
        except Exception as e:
            log_error(f"_compose_final_prompt: Error in LoRA tag generation: {e}")
            raise

        # Collect trigger words
        try:
            log_debug("_compose_final_prompt: Collecting trigger words")
            trigger_words = []
            for _i, lora in enumerate(image_loras + video_loras):
                lora_triggers = lora.get("trigger_words") or []
                if isinstance(lora_triggers, list):
                    trigger_words.extend(lora_triggers)
                else:
                    lora_name = lora.get("name", "unknown")
                    log_error(f"Invalid trigger_words for LoRA {lora_name}: {lora_triggers}")
        except Exception as e:
            log_error(f"_compose_final_prompt: Error collecting trigger words: {e}")
            raise

        # Add trigger words (remove duplicates but preserve order)
        try:
            log_debug("_compose_final_prompt: Deduplicating trigger words")
            seen_triggers = set()
            unique_triggers = []
            for _i, word in enumerate(trigger_words):
                if word and word.lower() not in seen_triggers:
                    unique_triggers.append(word)
                    seen_triggers.add(word.lower())

            log_debug(f"_compose_final_prompt: Found {len(unique_triggers)} unique trigger words")
            if unique_triggers:
                prompt_parts.extend(unique_triggers)

            # Add the cleaned scene description (remove existing LoRA tags)
            clean_scene = re.sub(r"<(?:lora|wanlora):[^>]+>", "", scene_description)
            clean_scene = " ".join(clean_scene.split())  # Remove extra whitespace
            log_debug("_compose_final_prompt: Adding cleaned scene description")
            prompt_parts.append(clean_scene)

            # Analyze style from all selected LoRAs
            log_debug("_compose_final_prompt: Analyzing style from example prompts")
            all_example_prompts = []
            for i, lora in enumerate(image_loras + video_loras):
                lora_name = lora.get("name", "unknown")
                log_debug(f"_compose_final_prompt: Extracting examples from LoRA {i}: {lora_name}")
                metadata = lora.get("metadata")
                log_debug(f"_compose_final_prompt: Metadata type: {type(metadata)}")
                examples = extract_example_prompts(metadata, limit=2)
                example_count = len(examples) if examples else 0
                log_debug(f"_compose_final_prompt: Found {example_count} example prompts")
                if examples:
                    all_example_prompts.extend(examples)
        except Exception as e:
            log_error(
                f"_compose_final_prompt: Error in trigger word processing or style analysis: {e}"
            )
            raise

        style_analysis = self._analyze_prompt_style(all_example_prompts)

        # Add style-appropriate enhancements
        if style_preference == "technical":
            prompt_parts.extend(["masterpiece", "high quality", "detailed"])
        elif style_preference == "artistic":
            if style_analysis["artistic_terms"]:
                prompt_parts.extend(style_analysis["artistic_terms"][:2])
            prompt_parts.extend(["beautiful", "stunning"])
        else:  # natural
            # Use common terms from examples
            if style_analysis["common_terms"]:
                prompt_parts.extend(style_analysis["common_terms"][:2])

        # Join with commas and clean up
        final_prompt = ", ".join(prompt_parts)

        # Clean up duplicate commas and extra spaces
        final_prompt = re.sub(r",\s*,+", ",", final_prompt)
        final_prompt = re.sub(r"\s+", " ", final_prompt).strip()

        return final_prompt

    def _gather_image_lora_context(self, image_loras: list[dict[str, Any]]) -> str:
        """
        Gather all embeddable content, tags, and metadata from selected image LoRAs
        and concatenate them into a single string for use as a query.
        """
        context_parts = []
        for lora in image_loras:
            metadata = lora.get("metadata", {})
            # Embeddable content
            try:
                embeddable = extract_embeddable_content(metadata)
                if embeddable:
                    context_parts.append(str(embeddable))
            except Exception:
                pass
            # Tags
            tags = metadata.get("tags", [])
            if isinstance(tags, list):
                context_parts.extend([str(tag) for tag in tags if tag])
            # Civitai tags
            civitai = metadata.get("civitai")
            if isinstance(civitai, dict):
                model = civitai.get("model")
                if isinstance(model, dict):
                    civitai_tags = model.get("tags", [])
                    if isinstance(civitai_tags, list):
                        context_parts.extend([str(tag) for tag in civitai_tags if tag])
                trained_words = civitai.get("trainedWords")
                if isinstance(trained_words, list):
                    context_parts.extend([str(word) for word in trained_words if word])
            # Name
            if lora.get("name"):
                context_parts.append(str(lora["name"]))
        # Remove duplicates, preserve order
        seen = set()
        context = []
        for part in context_parts:
            if part and part not in seen:
                context.append(part)
                seen.add(part)
        return " ".join(context)

    def compose_prompt(
        self,
        scene_description: str,
        max_image_loras: int = 3,
        max_video_loras: int = 2,
        content_boost: float = 1.2,
        style_preference: str = "natural",
        image_lora_dir_path: str = "",
        wan_lora_dir_path: str = "",
        default_lora_weight: float = 1.0,
        low_lora_weight_offset: float = 0.2,
    ) -> tuple[str, str, str]:
        """
        Main function that composes prompts from scene descriptions.

        Args:
            scene_description: Natural language scene description
            max_image_loras: Maximum number of image LoRAs to include
            max_video_loras: Maximum number of video LoRAs to include
            content_boost: Boost factor for content-specific LoRAs
            style_preference: Style preference ("technical", "artistic", "natural")
            image_lora_dir_path: Optional subdirectory to filter image LoRAs
            wan_lora_dir_path: Optional subdirectory to filter video LoRAs
            default_lora_weight: Default weight for all LoRAs (overrides metadata)
            low_lora_weight_offset: Amount to reduce LOW LoRA weights by

        Returns:
            Tuple of (composed_prompt, lora_analysis, metadata_summary)
        """
        if not scene_description.strip():
            return ("No scene description provided.", "", "")

        # Initialize embeddings if needed
        if not self._initialize_embeddings():
            return (
                "Error: Could not initialize embeddings system.",
                "Embeddings initialization failed. Please install scikit-learn.",
                "Error: Missing dependencies",
            )

        try:
            log(f"Composing prompt for: {scene_description}")

            # Find relevant image LoRAs
            log("Finding relevant image LoRAs...")
            if image_lora_dir_path:
                log(f"Filtering image LoRAs to directory: {image_lora_dir_path}")
            image_loras = self._find_relevant_loras(
                scene_description,
                "image",
                max_image_loras,
                content_boost,
                image_lora_dir_path,
            )
            image_names = [lora.get("name", "unknown") for lora in image_loras]
            log(f"Found {len(image_loras)} image LoRAs: {image_names}")

            # Build context for video LoRA selection
            if image_loras:
                video_query_context = self._gather_image_lora_context(image_loras)
                video_lora_selection_basis = {
                    "basis": "image_loras",
                    "image_loras_used": image_names,
                    "context_string": video_query_context,
                }
                log(
                    f"Using image LoRA context for video LoRA selection: {video_query_context[:100]}..."
                )
            else:
                video_query_context = scene_description
                video_lora_selection_basis = {
                    "basis": "scene_description",
                    "image_loras_used": [],
                    "context_string": scene_description,
                }
                log("No image LoRAs found, using scene description for video LoRA selection.")

            # Find relevant video LoRAs using the new context
            log("Finding relevant video LoRAs...")
            if wan_lora_dir_path:
                log(f"Filtering video LoRAs to directory: {wan_lora_dir_path}")
            video_loras = self._find_relevant_loras(
                video_query_context,
                "video",
                max_video_loras,
                content_boost,
                wan_lora_dir_path,
            )

            # Apply WAN 2.2 high/low pairing
            original_count = len(video_loras)
            video_loras = self._apply_wan_2_2_pairing(video_loras, max_video_loras)
            if len(video_loras) > original_count:
                log(
                    f"WAN 2.2 pairing expanded video LoRAs from {original_count} to {len(video_loras)}"
                )

            video_names = [lora.get("name", "unknown") for lora in video_loras]
            log(f"Found {len(video_loras)} video LoRAs: {video_names}")

            log(f"Found {len(image_loras)} image LoRAs, {len(video_loras)} video LoRAs")

            # Compose the final prompt
            log_debug("Starting prompt composition...")
            composed_prompt = self._compose_final_prompt(
                scene_description,
                image_loras,
                video_loras,
                style_preference,
                default_lora_weight,
                low_lora_weight_offset,
            )
            log_debug("Prompt composition completed successfully")

            # Create analysis output with actual weights used
            def get_actual_weight(lora_name: str, is_video: bool) -> float:
                """Calculate the actual weight used for a LoRA."""
                weight = default_lora_weight
                if is_video and "low" in lora_name.lower():
                    weight = max(0.1, weight - low_lora_weight_offset)
                return weight

            analysis_data = {
                "scene_description": scene_description,
                "image_loras": [
                    {
                        "name": lora["name"],
                        "relevance_score": lora["relevance_score"],
                        "weight": get_actual_weight(lora["name"], False),
                        "recommended_weight": lora["recommended_weight"],
                        "trigger_words": lora["trigger_words"],
                    }
                    for lora in image_loras
                ],
                "video_loras": [
                    {
                        "name": lora["name"],
                        "relevance_score": lora["relevance_score"],
                        "weight": get_actual_weight(lora["name"], True),
                        "recommended_weight": lora["recommended_weight"],
                        "trigger_words": lora["trigger_words"],
                    }
                    for lora in video_loras
                ],
                "style_preference": style_preference,
                "settings": {
                    "content_boost": content_boost,
                    "default_lora_weight": default_lora_weight,
                    "low_lora_weight_offset": low_lora_weight_offset,
                },
                "video_lora_selection_basis": video_lora_selection_basis,
            }

            analysis_output = json.dumps(analysis_data, indent=2, ensure_ascii=False)

            # Create metadata summary
            metadata_summary = {
                "total_loras_discovered": len(self._lora_database),
                "total_loras_with_embeddings": len(self._lora_embeddings),
                "image_loras_found": len(image_loras),
                "video_loras_found": len(video_loras),
                "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
                "processing_successful": True,
                "video_lora_selection_basis": video_lora_selection_basis,
            }

            summary_output = json.dumps(metadata_summary, indent=2)

            log("Prompt composition completed successfully")
            return (composed_prompt, analysis_output, summary_output)

        except Exception as e:
            log_error(f"Error composing prompt: {e}")
            return (
                f"Error: {str(e)}",
                f"Error during prompt composition: {str(e)}",
                json.dumps({"error": str(e), "processing_successful": False}),
            )
