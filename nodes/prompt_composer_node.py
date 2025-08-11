"""
LoRA Prompt Composer Node Implementation

A ComfyUI node that takes natural language scene descriptions and intelligently
composes prompts with optimal LoRA combinations, weights, and trigger words.
"""

import json
import re
from typing import Dict, List, Tuple, Any

from .lora_metadata_utils import (
    discover_all_loras,
    extract_embeddable_content,
    extract_example_prompts,
    classify_lora_type,
    extract_recommended_weight,
)
from .logging_utils import log, log_error


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
                            "Maximum number of image LoRAs to include in the "
                            "composed prompt"
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
                            "Maximum number of video LoRAs to include in the "
                            "composed prompt"
                        ),
                    },
                ),
            },
            "optional": {
                "relevance_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Minimum relevance score for LoRA inclusion (0.0-1.0)",
                    },
                ),
                "content_boost": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.5,
                        "max": 2.0,
                        "tooltip": (
                            "Boost factor for content-specific LoRAs "
                            "(character, pose, etc.)"
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
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer

            log("Initializing sentence-transformers embeddings model...")

            # Load a small, efficient model for content analysis
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Discover all LoRAs and prepare embeddings
            log("Discovering LoRAs and generating embeddings...")
            self._lora_database = discover_all_loras()

            # Generate embeddings for all LoRAs
            for lora_name, lora_info in self._lora_database.items():
                metadata = lora_info["metadata"]
                if metadata:
                    embeddable_text = extract_embeddable_content(metadata)
                    if embeddable_text:
                        embedding = self._embedding_model.encode(embeddable_text)
                        self._lora_embeddings[lora_name] = embedding

            self._embeddings_initialized = True
            log(f"Embeddings initialized for {len(self._lora_embeddings)} LoRAs")
            return True

        except ImportError as e:
            log_error(f"Failed to import sentence-transformers: {e}")
            log_error("Please install with: pip install sentence-transformers")
            return False
        except Exception as e:
            log_error(f"Failed to initialize embeddings: {e}")
            return False

    def _find_relevant_loras(
        self,
        scene_description: str,
        lora_type: str,
        max_count: int,
        relevance_threshold: float,
        content_boost: float,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant LoRAs for the scene description.

        Args:
            scene_description: Natural language description
            lora_type: "image" or "video"
            max_count: Maximum number of LoRAs to return
            relevance_threshold: Minimum relevance score
            content_boost: Boost factor for content-specific LoRAs

        Returns:
            List of relevant LoRA info dicts with scores
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
            log(f"  🧹 Cleaned scene for embedding: '{preview}'")

            # Generate embedding for cleaned scene description
            scene_embedding = self._embedding_model.encode(clean_scene)

            relevant_loras = []

            for lora_name, lora_info in self._lora_database.items():
                metadata = lora_info["metadata"]
                if not metadata:
                    continue

                # Filter by LoRA type
                actual_type = classify_lora_type(metadata)
                if actual_type != lora_type:
                    continue

                # Calculate relevance score
                if lora_name in self._lora_embeddings:
                    lora_embedding = self._lora_embeddings[lora_name]

                    # Calculate cosine similarity
                    similarity = cosine_similarity([scene_embedding], [lora_embedding])[
                        0
                    ][0]
                    log(f"  📊 {lora_name} base similarity: {similarity:.4f}")

                    # Apply content boost for character/pose LoRAs
                    if self._is_content_lora(metadata):
                        old_similarity = similarity
                        similarity *= content_boost
                        log(
                            f"  🚀 {lora_name} content boost applied: "
                            f"{old_similarity:.4f} → {similarity:.4f}"
                        )
                    else:
                        log(f"  📋 {lora_name} no content boost (not content-specific)")

                    # Apply keyword matching bonus for exact position matches
                    keyword_boost = self._get_keyword_boost(
                        scene_description, metadata, lora_name
                    )
                    if keyword_boost > 1.0:
                        old_similarity = similarity
                        similarity *= keyword_boost
                        log(
                            f"  🎯 {lora_name} keyword boost applied: "
                            f"{old_similarity:.4f} → {similarity:.4f}"
                        )

                    # Filter by threshold
                    if similarity >= relevance_threshold:
                        relevant_loras.append(
                            {
                                "name": lora_name,
                                "metadata": metadata,
                                "relevance_score": float(similarity),
                                "recommended_weight": extract_recommended_weight(
                                    metadata
                                ),
                                "trigger_words": lora_info["trigger_words"],
                                "type": lora_type,
                            }
                        )

            # Sort by relevance score and limit
            relevant_loras.sort(key=lambda x: x["relevance_score"], reverse=True)
            return relevant_loras[:max_count]

        except Exception as e:
            log_error(f"Error finding relevant LoRAs: {e}")
            return []

    def _is_content_lora(self, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a LoRA is content-specific (character, pose, etc.).

        Args:
            metadata: LoRA metadata dictionary

        Returns:
            True if this is a content-specific LoRA
        """
        # Check tags for content indicators
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
        if "civitai" in metadata and "model" in metadata["civitai"]:
            model_tags = metadata["civitai"]["model"].get("tags", [])
            if any(tag.lower() in content_tags for tag in model_tags):
                return True

        # Check top-level tags
        if "tags" in metadata:
            if any(tag.lower() in content_tags for tag in metadata["tags"]):
                return True

        # Check if it has trigger words (often indicates character LoRAs)
        if "civitai" in metadata and metadata["civitai"].get("trainedWords"):
            return True

        return False

    def _get_keyword_boost(
        self, scene_description: str, metadata: Dict[str, Any], lora_name: str
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

        log(f"    🔍 Keyword boost check for {lora_name}")
        orig_preview = scene_description[:50] + (
            "..." if len(scene_description) > 50 else ""
        )
        clean_preview = clean_scene[:50] + ("..." if len(clean_scene) > 50 else "")
        log(f"    📝 Original scene: '{orig_preview}'")
        log(f"    🧹 Cleaned scene: '{clean_preview}'")
        log(f"    📁 LoRA name: '{lora_name.lower()}'")

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
        lora_tags_lower = [tag.lower() for tag in lora_tags] if lora_tags else []
        civitai_tags = []
        if "civitai" in metadata and "model" in metadata["civitai"]:
            civitai_tags = [
                tag.lower() for tag in metadata["civitai"]["model"].get("tags", [])
            ]

        log(f"    📋 LoRA tags: {lora_tags_lower}")
        log(f"    🌐 Civitai tags: {civitai_tags}")

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

            log(f"    🎯 Position '{position}': Scene has keyword, checking sources...")
            log(f"      📁 Title match: {lora_has_keyword}")
            log(f"      📋 Tag match: {tag_has_keyword}")
            log(f"      🌐 Civitai match: {civitai_has_keyword}")

        # Apply the best boost found
        if best_boost > 1.0:
            boost *= best_boost
            log(
                f"    ✨ Applied {best_boost}x boost for '{best_position}' match in {best_source}!"
            )

        log(f"    📊 Final keyword boost for {lora_name}: {boost}x")
        return boost

    def _analyze_prompt_style(self, example_prompts: List[str]) -> Dict[str, Any]:
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
        avg_length = sum(len(prompt.split()) for prompt in example_prompts) / len(
            example_prompts
        )
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
        image_loras: List[Dict[str, Any]],
        video_loras: List[Dict[str, Any]],
        style_preference: str,
    ) -> str:
        """
        Compose the final prompt with LoRA tags and styling.

        Args:
            scene_description: Original scene description
            image_loras: Selected image LoRAs
            video_loras: Selected video LoRAs
            style_preference: Style preference setting

        Returns:
            Composed prompt string
        """
        try:
            log("_compose_final_prompt: Starting composition")
            prompt_parts = []

            # Add LoRA tags at the beginning
            log("_compose_final_prompt: Adding image LoRA tags")
            for i, lora in enumerate(image_loras):
                weight = lora["recommended_weight"]
                tag = f"<lora:{lora['name']}:{weight}>"
                prompt_parts.append(tag)

            log("_compose_final_prompt: Adding video LoRA tags")
            for i, lora in enumerate(video_loras):
                weight = lora["recommended_weight"]
                tag = f"<wanlora:{lora['name']}:{weight}>"
                prompt_parts.append(tag)
        except Exception as e:
            log_error(f"_compose_final_prompt: Error in LoRA tag generation: {e}")
            raise

        # Collect trigger words
        try:
            log("_compose_final_prompt: Collecting trigger words")
            trigger_words = []
            for i, lora in enumerate(image_loras + video_loras):

                lora_triggers = lora.get("trigger_words") or []
                if isinstance(lora_triggers, list):
                    trigger_words.extend(lora_triggers)
                else:
                    lora_name = lora.get("name", "unknown")
                    log_error(
                        f"Invalid trigger_words for LoRA {lora_name}: {lora_triggers}"
                    )
        except Exception as e:
            log_error(f"_compose_final_prompt: Error collecting trigger words: {e}")
            raise

        # Add trigger words (remove duplicates but preserve order)
        try:
            log("_compose_final_prompt: Deduplicating trigger words")
            seen_triggers = set()
            unique_triggers = []
            for i, word in enumerate(trigger_words):
                if word and word.lower() not in seen_triggers:
                    unique_triggers.append(word)
                    seen_triggers.add(word.lower())

            log(
                f"_compose_final_prompt: Found {len(unique_triggers)} unique trigger words"
            )
            if unique_triggers:
                prompt_parts.extend(unique_triggers)

            # Add the cleaned scene description (remove existing LoRA tags)
            clean_scene = re.sub(r"<(?:lora|wanlora):[^>]+>", "", scene_description)
            clean_scene = " ".join(clean_scene.split())  # Remove extra whitespace
            log("_compose_final_prompt: Adding cleaned scene description")
            prompt_parts.append(clean_scene)

            # Analyze style from all selected LoRAs
            log("_compose_final_prompt: Analyzing style from example prompts")
            all_example_prompts = []
            for i, lora in enumerate(image_loras + video_loras):
                lora_name = lora.get("name", "unknown")
                log(
                    f"_compose_final_prompt: Extracting examples from LoRA {i}: {lora_name}"
                )
                metadata = lora.get("metadata")
                log(f"_compose_final_prompt: Metadata type: {type(metadata)}")
                examples = extract_example_prompts(metadata, limit=2)
                example_count = len(examples) if examples else 0
                log(f"_compose_final_prompt: Found {example_count} example prompts")
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

    def compose_prompt(
        self,
        scene_description: str,
        max_image_loras: int = 3,
        max_video_loras: int = 2,
        relevance_threshold: float = 0.3,
        content_boost: float = 1.2,
        style_preference: str = "natural",
    ) -> Tuple[str, str, str]:
        """
        Main function that composes prompts from scene descriptions.

        Args:
            scene_description: Natural language scene description
            max_image_loras: Maximum number of image LoRAs to include
            max_video_loras: Maximum number of video LoRAs to include
            relevance_threshold: Minimum relevance score for inclusion
            content_boost: Boost factor for content-specific LoRAs
            style_preference: Style preference ("technical", "artistic", "natural")

        Returns:
            Tuple of (composed_prompt, lora_analysis, metadata_summary)
        """
        if not scene_description.strip():
            return ("No scene description provided.", "", "")

        # Initialize embeddings if needed
        if not self._initialize_embeddings():
            return (
                "Error: Could not initialize embeddings system.",
                "Embeddings initialization failed. Please install sentence-transformers.",
                "Error: Missing dependencies",
            )

        try:
            log(f"Composing prompt for: {scene_description}")

            # Find relevant LoRAs
            log("Finding relevant image LoRAs...")
            image_loras = self._find_relevant_loras(
                scene_description,
                "image",
                max_image_loras,
                relevance_threshold,
                content_boost,
            )
            image_names = [lora.get("name", "unknown") for lora in image_loras]
            log(f"Found {len(image_loras)} image LoRAs: {image_names}")

            log("Finding relevant video LoRAs...")
            video_loras = self._find_relevant_loras(
                scene_description,
                "video",
                max_video_loras,
                relevance_threshold,
                content_boost,
            )
            video_names = [lora.get("name", "unknown") for lora in video_loras]
            log(f"Found {len(video_loras)} video LoRAs: {video_names}")

            log(f"Found {len(image_loras)} image LoRAs, {len(video_loras)} video LoRAs")

            # Compose the final prompt
            log("Starting prompt composition...")
            composed_prompt = self._compose_final_prompt(
                scene_description, image_loras, video_loras, style_preference
            )
            log("Prompt composition completed successfully")

            # Create analysis output
            analysis_data = {
                "scene_description": scene_description,
                "image_loras": [
                    {
                        "name": lora["name"],
                        "relevance_score": lora["relevance_score"],
                        "weight": lora["recommended_weight"],
                        "trigger_words": lora["trigger_words"],
                    }
                    for lora in image_loras
                ],
                "video_loras": [
                    {
                        "name": lora["name"],
                        "relevance_score": lora["relevance_score"],
                        "weight": lora["recommended_weight"],
                        "trigger_words": lora["trigger_words"],
                    }
                    for lora in video_loras
                ],
                "style_preference": style_preference,
                "thresholds": {
                    "relevance_threshold": relevance_threshold,
                    "content_boost": content_boost,
                },
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
