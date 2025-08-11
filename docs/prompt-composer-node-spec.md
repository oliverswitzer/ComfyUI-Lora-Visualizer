# LoRA Prompt Composer Node - Product Requirements Document

## 🎯 **Product Vision**

A ComfyUI custom node that takes a natural language scene description and intelligently composes a production-ready prompt containing optimal combinations of image and video LoRAs with proper weights, trigger words, and writing style based on proven example prompts.

## 🔥 **Problem Statement**

ComfyUI users face significant friction when trying to:
1. **Discover compatible LoRAs**: Finding the right combination of image and video LoRAs for a specific scene
2. **Compose effective prompts**: Manually constructing prompts with proper LoRA tags, weights, and trigger words
3. **Learn from examples**: Understanding effective prompt patterns from successful generations
4. **Balance complexity**: Managing multiple LoRAs without conflicts or diminishing returns

**Current workflow pain points:**
- Manual browsing through hundreds of LoRAs and their metadata
- Trial-and-error with LoRA combinations and strength values
- Inconsistent trigger word usage leading to poor results
- Time-consuming prompt construction for each new scene concept

## 🎪 **User Personas**

### Primary: **Creative Prompter**
- Uses ComfyUI for both image and video generation
- Has extensive LoRA and WAN LoRA collections (50-100)
- Values efficiency and consistent quality
- Wants to focus on creativity, not technical prompt engineering

### Secondary: **Workflow Builder**
- Creates reusable ComfyUI workflows
- Needs predictable, parameterizable prompt generation
- Values node reliability and performance

## 🚀 **Core Features**

### **1. Intelligent LoRA Discovery**
- **Semantic Search**: Natural language scene description → relevant LoRA suggestions
- **Dual-Mode Selection**: Separate discovery for image LoRAs and WAN video LoRAs
- **Compatibility Analysis**: Ensure selected LoRAs work well together
- **Relevance Ranking**: Score and rank LoRAs by relevance to scene description

### **2. Smart Prompt Composition**
- **LoRA Tag Generation**: Automatic `<lora:name:weight>` and `<wanlora:name:weight>` insertion
- **Trigger Word Integration**: Include relevant trigger words from metadata
- **Style Mimicry**: Adopt writing patterns from successful example prompts from each lora
- **Weight Optimization**: Suggest optimal strength values based on metadata and compatibility

### **3. Configurable Output Control**
- **LoRA Limits**: User-configurable max number of image and video LoRAs
- **Quality Thresholds**: Filter LoRAs by relevance score minimums
- **Style Preferences**: Control prompt writing style (technical vs. artistic)
- **No Safety Controls**: Important to use models that allow for all content types 

## 📋 **Functional Requirements**

### **Input Specification**
```python
INPUT_TYPES = {
    "required": {
        "scene_description": ("STRING", {
            "multiline": True,
            "placeholder": "Describe your scene in natural language...",
            "tooltip": "Natural language description of the desired scene"
        }),
        "max_image_loras": ("INT", {
            "default": 3,
            "min": 0,
            "max": 10,
            "tooltip": "Maximum number of image LoRAs to include"
        }),
        "max_video_loras": ("INT", {
            "default": 2, 
            "min": 0,
            "max": 5,
            "tooltip": "Maximum number of video LoRAs to include"
        })
    },
    "optional": {
        "relevance_threshold": ("FLOAT", {
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
            "tooltip": "Minimum relevance score for LoRA inclusion"
        }),
        "content_boost": ("FLOAT", {
            "default": 1.0,
            "min": 0.5,
            "max": 2.0,
            "tooltip": "Boost factor for content-specific LoRAs (character, pose, etc.)"
        }),
        "style_preference": (["technical", "artistic", "natural"], {
            "default": "natural",
            "tooltip": "Prompt writing style"
        })
    }
}
```

### **Output Specification**
```python
RETURN_TYPES = ("STRING", "STRING", "STRING")
RETURN_NAMES = ("composed_prompt", "lora_analysis", "metadata_summary")
OUTPUT_TOOLTIPS = (
    "Complete prompt with LoRA tags, trigger words, and scene description",
    "Detailed analysis of selected LoRAs and their relevance scores",
    "Summary of metadata used in composition"
)
```

### **Core Processing Pipeline**

#### **Phase 1: Scene Analysis**
1. **Parse Input**: Extract key concepts, style indicators, subject matter
2. **Generate Embeddings**: Create vector representation of scene description
3. **Content Classification**: Detect content themes, artistic styles, technical requirements

#### **Phase 2: LoRA Discovery**
1. **Dual Search**: Separate vector searches for image and video LoRAs
2. **Relevance Scoring**: Cosine similarity + metadata boost factors
3. **Compatibility Filtering**: Remove conflicting LoRAs (same character, competing styles)
4. **Ranking & Selection**: Top-K selection respecting user limits

#### **Phase 3: Prompt Composition**
1. **Style Analysis**: Extract patterns from example prompts in selected LoRAs
2. **Trigger Word Extraction**: Collect `trainedWords` from selected LoRAs
3. **Weight Optimization**: Calculate optimal strength values
4. **Prompt Assembly**: Compose final prompt with proper structure

## 🔧 **Technical Architecture**

### **Data Sources**
- **Primary**: ComfyUI LoRA metadata files (`*.metadata.json`)
- **Fallback**: Basic LoRA file scanning and metadata inference
- **Enhancement**: User-curated LoRA descriptions and tags

### **Embeddings Strategy**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (22MB, content-capable)
- **Embedding Targets**:
  - LoRA descriptions (`modelDescription`, `civitai.model.description`)
  - Tags and categories (`tags`, `civitai.model.tags`)
  - Example prompts (`civitai.images[].meta.prompt`)
  - Usage instructions and best practices
- **Storage**: In-memory vector store with persistent caching

### **LoRA Classification**
```python
def classify_lora_type(metadata: Dict) -> str:
    """Classify LoRA as image or video generation."""
    base_model = metadata.get("base_model", "").lower()
    
    if any(keyword in base_model for keyword in ["wan", "video", "i2v"]):
        return "video"
    elif any(keyword in base_model for keyword in ["sdxl", "sd1.5", "flux", "illustrious"]):
        return "image"
    else:
        return "unknown"
```

### **Weight Optimization Algorithm**
```python
def calculate_optimal_weight(lora_metadata: Dict, relevance_score: float) -> float:
    """Calculate optimal LoRA strength based on metadata and relevance."""
    # Base weight from metadata recommendations
    base_weight = extract_recommended_weight(lora_metadata)
    
    # Adjust based on relevance score
    relevance_factor = 0.8 + (relevance_score * 0.4)  # Range: 0.8-1.2
    
    # Apply LoRA-specific modifiers
    if is_character_lora(lora_metadata):
        modifier = 0.9  # Character LoRAs typically need lower weights
    elif is_style_lora(lora_metadata):
        modifier = 1.1  # Style LoRAs can handle higher weights
    else:
        modifier = 1.0
    
    return round(base_weight * relevance_factor * modifier, 2)
```

## 🎨 **Prompt Style Mimicry**

### **Style Pattern Analysis**
Extract and replicate successful prompt patterns:

1. **Structure Analysis**: Identify ordering patterns (subject → style → technical)
2. **Vocabulary Extraction**: Common adjectives, technical terms, artistic styles
3. **Punctuation Patterns**: Comma usage, parentheses, emphasis markers
4. **Length Optimization**: Target length based on successful examples

### **Example Style Templates**
```python
STYLE_TEMPLATES = {
    "artistic": "{subject}, {artistic_style}, {lighting}, {composition}, {quality_terms}",
    "technical": "{subject}, {technical_specs}, {camera_settings}, {post_processing}",
    "natural": "{natural_description}, {mood}, {style_hints}, {quality_boost}"
}
```

## 📊 **Success Metrics**

### **Primary KPIs**
- **User Adoption**: Node usage frequency vs. manual prompt creation
- **Prompt Quality**: User satisfaction ratings with generated prompts
- **Time Savings**: Reduction in prompt composition time
- **Generation Success**: Rate of acceptable outputs on first try

### **Technical Metrics**
- **Response Time**: < 2 seconds for prompt generation
- **Memory Usage**: < 500MB for embeddings database
- **Accuracy**: > 80% relevant LoRA suggestions
- **Compatibility**: > 95% conflict-free LoRA combinations

## 🚧 **Implementation Phases**

### **Phase 1: Core MVP** (Week 1-2)
- Basic scene description → LoRA discovery
- Simple prompt composition with LoRA tags
- Configurable LoRA limits
- Basic vector similarity search

### **Phase 2: Smart Composition** (Week 3-4)
- Trigger word integration
- Weight optimization
- Style pattern analysis
- Compatibility filtering

### **Phase 3: Polish & Enhancement** (Week 5-6)
- Advanced style mimicry
- Performance optimization
- Comprehensive testing
- Documentation and examples

## 🔒 **Constraints & Considerations**

### **Technical Constraints**
- **Memory Limits**: Embeddings database must fit in reasonable RAM
- **Dependency Management**: Minimize external dependencies for easy installation
- **Performance**: Real-time response required for good UX
- **Compatibility**: Must work across different ComfyUI installations

### **Content Requirements**
- **Creative Content Support**: Essential for comprehensive LoRA discovery and composition
  - Embeddings model must understand creative terminology without censorship or filtering
  - Many character LoRAs are designed for creative content (poses, clothing, artistic styles)
  - Style and enhancement LoRAs often target creative scene composition and lighting
  - Video LoRAs frequently add motion dynamics to creative content scenarios
- **Unrestricted Analysis**: Tool must analyze all LoRA metadata regardless of content rating
- **Context-Aware Matching**: Understanding of creative content context improves LoRA relevance
- **Professional Use Case**: Tool serves legitimate creative content creators and artists

### **User Experience**
- **Learning Curve**: Must be intuitive for non-technical users
- **Customization**: Power users need fine-grained control
- **Reliability**: Consistent results for similar inputs
- **Integration**: Seamless workflow integration with existing nodes

## 🔮 **Future Enhancements**

### **Advanced Features**
- **Learning System**: Improve suggestions based on user feedback
- **Prompt Templates**: Pre-built templates for common scenarios
- **Batch Processing**: Generate multiple prompt variations
- **Integration**: Connect with other prompt enhancement tools

### **Community Features**
- **Shared Embeddings**: Community-curated LoRA descriptions
- **Prompt Sharing**: Share successful prompt compositions
- **Collaborative Filtering**: Recommendations based on similar users

---

## 📝 **Acceptance Criteria**

### **Core Functionality**
- [ ] Node accepts natural language scene descriptions
- [ ] Discovers relevant image and video LoRAs within user-specified limits
- [ ] Generates complete prompts with proper LoRA tags and weights
- [ ] Includes appropriate trigger words from LoRA metadata
- [ ] Mimics successful prompt writing patterns

### **Quality Assurance**
- [ ] Response time < 2 seconds for typical queries
- [ ] Memory usage < 500MB for embeddings
- [ ] > 80% user satisfaction with LoRA relevance
- [ ] Zero crashes or exceptions during normal operation
- [ ] Comprehensive error handling and user feedback

### **User Experience**
- [ ] Intuitive interface following ComfyUI patterns
- [ ] Clear tooltips and documentation
- [ ] Configurable parameters with sensible defaults
- [ ] Visual feedback during processing
- [ ] Comprehensive output analysis and metadata
