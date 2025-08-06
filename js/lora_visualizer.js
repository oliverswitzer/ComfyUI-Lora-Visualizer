import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

// LoRA Visualizer Extension for ComfyUI
app.registerExtension({
    name: "LoRAVisualizer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoRAVisualizer") {
            console.debug("Registering LoRAVisualizer node");
            
            // Store original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Override node creation
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                console.debug("LoRAVisualizer node created");
                
                // Add custom visualization widget
                this.addLoRAVisualizationWidget();
                
                // Add mouse event handlers for hover functionality
                this.addMouseEventHandlers();
                
                // Initialize empty data and hover state
                this.loraData = {
                    standard_loras: [],
                    wanloras: [],
                    prompt: ""
                };
                this.thumbnailAreas = [];
                this.currentlyHovered = null;
                this.galleryHovered = false;
                this.hideTimeout = null;
                this.mouseTimeout = null;
                this.loraMetadataCache = {};
                
                // Track if user has manually resized the node
                this.manuallyResized = false;
                
                // Clean up HTML buttons when node is removed
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    this.removeHTMLButtons();
                    if (originalOnRemoved) {
                        originalOnRemoved.apply(this, arguments);
                    }
                };
                
                return result;
            };
            
            // Add context menu option to reset auto-sizing
            const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }
                
                options.push({
                    content: this.manuallyResized ? "Enable Auto-Sizing" : "Auto-Sizing Enabled",
                    callback: () => {
                        if (this.manuallyResized) {
                            this.manuallyResized = false;
                            console.debug("Auto-sizing re-enabled");
                            // Immediately auto-size to content
                            if (this.loraVisualizationWidget) {
                                this.setSize(this.computeSize());
                            }
                        }
                    },
                    disabled: !this.manuallyResized
                });
            };
            
            // Override onResize to detect manual resizing
            const originalOnResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                // Mark that the user has manually resized if size is different from auto-calculated
                if (this.loraVisualizationWidget) {
                    const autoSize = this.loraVisualizationWidget.computeSize(size[0]);
                    if (Math.abs(size[1] - autoSize[1]) > 10) { // Allow some tolerance
                        this.manuallyResized = true;
                        console.debug("Manual resize detected, disabling auto-sizing");
                    }
                }
                
                if (originalOnResize) {
                    return originalOnResize.apply(this, arguments);
                }
            };
            
            // Add method to create visualization widget
            nodeType.prototype.addLoRAVisualizationWidget = function() {
                const widget = this.addWidget("LORA_VIZ", "lora_visualization", "", () => {}, {
                    serialize: false
                });
                
                            widget.computeSize = function(width) {
                return [width, 300]; // Increased height for larger thumbnails
            };
                
                widget.draw = function(ctx, node, widgetWidth, y, height) {
                    // Custom drawing for LoRA visualization
                    node.drawLoRAVisualization(ctx, widgetWidth, y, height);
                };
                
                this.loraVisualizationWidget = widget;
                this.setSize(this.computeSize());
            };
            
            // Add method to setup mouse event handlers
            nodeType.prototype.addMouseEventHandlers = function() {
                // Listen for mouse events on the canvas
                const canvas = app.canvas.canvas;
                
                // Throttle mouse move events to prevent flickering
                let mouseTimeout;
                this.mouseTimeout = mouseTimeout;
                
                canvas.addEventListener('mousemove', (e) => {
                    clearTimeout(this.mouseTimeout);
                    this.mouseTimeout = setTimeout(() => {
                        this.handleMouseMove(e);
                        // Update button positions when canvas moves
                        this.updateHTMLButtonPositions();
                    }, 16); // ~60fps
                });
                
                canvas.addEventListener('mouseleave', (e) => {
                    clearTimeout(this.mouseTimeout);
                    // Don't immediately hide - let the normal logic handle it
                    if (this.hideTimeout) {
                        clearTimeout(this.hideTimeout);
                    }
                    this.hideTimeout = setTimeout(() => {
                        if (!this.galleryHovered) {
                            this.hideHoverGallery();
                            this.currentlyHovered = null;
                        }
                    }, 200);
                });
                
                // Add click event for copy buttons
                canvas.addEventListener('click', (e) => {
                    this.handleCanvasClick(e);
                });
            };
            
            // Canvas click handling is no longer needed since we use HTML buttons
            nodeType.prototype.handleCanvasClick = function(e) {
                // HTML buttons handle their own clicks now
                return;
            };
            
            // Add method to show copy feedback
            nodeType.prototype.showCopyFeedback = function(buttonArea, isError = false) {
                // Create temporary feedback overlay
                const feedback = document.createElement('div');
                feedback.style.cssText = `
                    position: fixed;
                    background: ${isError ? '#ff4444' : '#4CAF50'};
                    color: white;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    z-index: 10000;
                    pointer-events: none;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                `;
                feedback.textContent = isError ? 'Copy failed!' : 'Copied to clipboard!';
                
                // Position near the button
                const rect = app.canvas.canvas.getBoundingClientRect();
                const scale = app.canvas.ds.scale;
                const offsetX = app.canvas.ds.offset[0];
                const offsetY = app.canvas.ds.offset[1];
                const nodeRect = this.getBounding();
                
                const buttonScreenX = rect.left + ((buttonArea.x + nodeRect[0] + offsetX) * scale);
                const buttonScreenY = rect.top + ((buttonArea.y + nodeRect[1] + offsetY) * scale);
                
                feedback.style.left = buttonScreenX + 'px';
                feedback.style.top = (buttonScreenY - 35) + 'px';
                
                document.body.appendChild(feedback);
                
                // Remove after 2 seconds
                setTimeout(() => {
                    if (document.body.contains(feedback)) {
                        document.body.removeChild(feedback);
                    }
                }, 2000);
            };
            
            // Add method to handle mouse movement
            nodeType.prototype.handleMouseMove = function(e) {
                if (!this.thumbnailAreas) return;
                
                const rect = app.canvas.canvas.getBoundingClientRect();
                const canvasX = e.clientX - rect.left;
                const canvasY = e.clientY - rect.top;
                
                // Convert canvas coordinates to node coordinates
                const nodeRect = this.getBounding();
                const scale = app.canvas.ds.scale;
                const offsetX = app.canvas.ds.offset[0];
                const offsetY = app.canvas.ds.offset[1];
                
                // Calculate actual mouse position relative to the node
                const mouseX = (canvasX / scale) - offsetX - nodeRect[0];
                const mouseY = (canvasY / scale) - offsetY - nodeRect[1];
                
                // Check if mouse is over any thumbnail
                let hoveredLora = null;
                for (const area of this.thumbnailAreas) {
                    if (mouseX >= area.x && mouseX <= area.x + area.width &&
                        mouseY >= area.y && mouseY <= area.y + area.height) {
                        hoveredLora = area.lora;
                        break;
                    }
                }
                
                // Clear any pending hide timeout
                if (this.hideTimeout) {
                    clearTimeout(this.hideTimeout);
                    this.hideTimeout = null;
                }
                
                if (hoveredLora) {
                    // Show gallery if hovering over a thumbnail
                    if (hoveredLora !== this.currentlyHovered) {
                        this.currentlyHovered = hoveredLora;
                        this.showHoverGallery(hoveredLora, canvasX, canvasY, this.currentThumbnailSize);
                    }
                    // If we're still hovering over the same LoRA, don't start hide timer
                } else if (this.currentlyHovered && !this.galleryHovered) {
                    // Mouse left thumbnail area - start timer to hide gallery
                    this.hideTimeout = setTimeout(() => {
                        if (!this.galleryHovered) {
                            this.hideHoverGallery();
                            this.currentlyHovered = null;
                        }
                    }, 300);
                }
            };
            
            // Add method to show hover gallery
            nodeType.prototype.showHoverGallery = function(lora, x, y, thumbnailSize = 80) {
                this.hideHoverGallery(); // Hide any existing gallery
                
                const loraKey = `${lora.name}_${lora.type}`;
                const metadata = this.loraMetadataCache && this.loraMetadataCache[loraKey];
                
                if (!metadata) {
                    console.debug(`No metadata found for ${lora.name}`);
                    return;
                }
                
                console.debug(`Showing hover gallery for ${lora.name}`);
                
                // Create gallery popup
                const gallery = document.createElement('div');
                gallery.className = 'lora-hover-gallery';
                gallery.style.cssText = `
                    position: fixed;
                    left: ${x + 10}px;
                    top: ${y + 10}px;
                    background: rgba(0, 0, 0, 0.95);
                    border: 2px solid ${lora.type === 'wanlora' ? '#ff9a4a' : '#4a9eff'};
                    border-radius: 8px;
                    padding: 15px;
                    max-width: 400px;
                    max-height: 300px;
                    overflow-y: auto;
                    z-index: 10000;
                    color: white;
                    font-family: Arial, sans-serif;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    pointer-events: auto;
                `;
                
                // Prevent gallery from disappearing when mouse enters it
                gallery.addEventListener('mouseenter', () => {
                    this.galleryHovered = true;
                    // Clear any pending hide timeout
                    if (this.hideTimeout) {
                        clearTimeout(this.hideTimeout);
                        this.hideTimeout = null;
                    }
                });
                
                gallery.addEventListener('mouseleave', () => {
                    this.galleryHovered = false;
                    
                    // Give a bit of time in case mouse moves back to thumbnail or gallery
                    this.hideTimeout = setTimeout(() => {
                        if (!this.galleryHovered) {
                            this.hideHoverGallery();
                            this.currentlyHovered = null;
                        }
                    }, 200);
                });
                
                // Add title
                const title = document.createElement('div');
                title.style.cssText = `
                    font-weight: bold;
                    font-size: 14px;
                    margin-bottom: 10px;
                    color: ${lora.type === 'wanlora' ? '#ff9a4a' : '#4a9eff'};
                `;
                title.textContent = `${lora.name} (${lora.strength})`;
                gallery.appendChild(title);
                
                // Add base model info
                if (metadata.baseModel) {
                    const baseModelDiv = document.createElement('div');
                    baseModelDiv.style.cssText = `
                        margin-bottom: 10px;
                        font-size: 11px;
                        color: #4CAF50;
                        font-weight: bold;
                    `;
                    baseModelDiv.textContent = `Base Model: ${metadata.baseModel}`;
                    gallery.appendChild(baseModelDiv);
                }
                
                // Add trigger words
                if (metadata.triggerWords && metadata.triggerWords.length > 0) {
                    const triggersDiv = document.createElement('div');
                    triggersDiv.style.cssText = `
                        margin-bottom: 10px;
                        font-size: 12px;
                    `;
                    triggersDiv.innerHTML = `<strong>Trigger Words:</strong> ${metadata.triggerWords.join(', ')}`;
                    gallery.appendChild(triggersDiv);
                } else {
                    const noTriggersDiv = document.createElement('div');
                    noTriggersDiv.style.cssText = `
                        margin-bottom: 10px;
                        font-size: 12px;
                        color: #999;
                    `;
                    noTriggersDiv.textContent = 'No trigger words available';
                    gallery.appendChild(noTriggersDiv);
                }
                
                // Add Civitai link
                if (metadata.civitaiUrl) {
                    const linkDiv = document.createElement('div');
                    linkDiv.style.cssText = `
                        margin-bottom: 10px;
        font-size: 12px;
                    `;
                    
                    const link = document.createElement('a');
                    link.href = metadata.civitaiUrl;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.style.cssText = `
                        color: #2196F3;
                        text-decoration: none;
                        font-weight: bold;
                        display: inline-flex;
                        align-items: center;
                        gap: 4px;
                    `;
                    link.innerHTML = '🔗 View on Civitai';
                    
                    link.onmouseover = () => link.style.textDecoration = 'underline';
                    link.onmouseout = () => link.style.textDecoration = 'none';
                    
                    linkDiv.appendChild(link);
                    gallery.appendChild(linkDiv);
                }
                
                // Add example images
                if (metadata.exampleImages && metadata.exampleImages.length > 0) {
                    const imagesDiv = document.createElement('div');
                    imagesDiv.style.cssText = `
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(${thumbnailSize}px, 1fr));
                        gap: 5px;
                        margin-top: 10px;
                    `;
                    
                    metadata.exampleImages.slice(0, 6).forEach(imageData => {
                        if (imageData.type === 'video' || imageData.url.endsWith('.mp4')) {
                            // Create autoplay video element
                            const videoContainer = document.createElement('div');
                            videoContainer.style.cssText = `
                                width: ${thumbnailSize}px;
                                height: ${thumbnailSize}px;
                                border-radius: 4px;
                                overflow: hidden;
                                position: relative;
                                cursor: pointer;
                            `;
                            
                            const video = document.createElement('video');
                            video.style.cssText = `
                                width: 100%;
                                height: 100%;
                                object-fit: cover;
                                border-radius: 4px;
                            `;
                            video.src = imageData.url;
                            video.autoplay = true;
                            video.muted = true; // Required for autoplay in most browsers
                            video.loop = true;
                            video.playsInline = true; // For mobile compatibility
                            
                            // Add click handler to open full video
                            videoContainer.onclick = () => window.open(imageData.url, '_blank');
                            
                            // Add video label overlay
                            const label = document.createElement('div');
                            label.style.cssText = `
                                position: absolute;
                                bottom: 2px;
                                right: 2px;
                                background: rgba(0,0,0,0.7);
                                color: white;
                                font-size: 8px;
                                padding: 1px 3px;
                                border-radius: 2px;
                                pointer-events: none;
                            `;
                            label.textContent = 'VIDEO';
                            
                            videoContainer.appendChild(video);
                            videoContainer.appendChild(label);
                            imagesDiv.appendChild(videoContainer);
                        } else {
                            // Regular image
                            const img = document.createElement('img');
                            img.style.cssText = `
                                width: ${thumbnailSize}px;
                                height: ${thumbnailSize}px;
                                object-fit: cover;
                                border-radius: 4px;
                                cursor: pointer;
                            `;
                            img.src = imageData.url;
                            img.onclick = () => window.open(imageData.url, '_blank');
                            imagesDiv.appendChild(img);
                        }
                    });
                    
                    gallery.appendChild(imagesDiv);
                } else {
                    const noImagesDiv = document.createElement('div');
                    noImagesDiv.style.cssText = `
                        margin-top: 10px;
                        font-size: 12px;
                        color: #999;
                    `;
                    noImagesDiv.textContent = 'No example images available';
                    gallery.appendChild(noImagesDiv);
                }
                
                document.body.appendChild(gallery);
                this.hoverGallery = gallery;
                this.galleryHovered = false;
            };
            
            // Add method to hide hover gallery
            nodeType.prototype.hideHoverGallery = function() {
                if (this.hoverGallery) {
                    document.body.removeChild(this.hoverGallery);
                    this.hoverGallery = null;
                }
            };
            
            // Add method to draw LoRA visualization
            nodeType.prototype.drawLoRAVisualization = function(ctx, width, y, height) {
                // Clear thumbnail areas and button areas for this redraw
                this.thumbnailAreas = [];
                this.htmlButtonAreas = [];
                this.currentThumbnailSize = null;
                
                // Remove any existing HTML buttons
                this.removeHTMLButtons();
                
                // Use data from backend instead of parsing locally
                const standardLoras = this.loraData.standard_loras || [];
                const wanloras = this.loraData.wanloras || [];
                
                // Only log when we actually have data to draw
                if (standardLoras.length > 0 || wanloras.length > 0) {
                    console.debug(`Drawing LoRAs - Standard: ${standardLoras.length}, WanLoRAs: ${wanloras.length}`);
                }
                
                if (standardLoras.length === 0 && wanloras.length === 0) {
                    ctx.fillStyle = "#666";
                    ctx.font = "12px Arial";
                    ctx.fillText("Execute node to see LoRA visualization...", 10, y + 20);
                    return;
                }
                
                let currentY = y + 15;
                
                // Draw standard LoRAs
                if (standardLoras.length > 0) {
                    ctx.fillStyle = "#4a9eff";
                    ctx.font = "bold 14px Arial";
                    ctx.fillText(`Standard LoRAs (${standardLoras.length})`, 10, currentY);
                    currentY += 20;
                    
                    standardLoras.forEach(lora => {
                        currentY = this.drawLoRAItem(ctx, lora, 20, currentY, width - 30, "#4a9eff");
                    });
                    
                    currentY += 10;
                }
                
                // Draw WanLoRAs
                if (wanloras.length > 0) {
                    ctx.fillStyle = "#ff9a4a";
                    ctx.font = "bold 14px Arial";
                    ctx.fillText(`WanLoRAs (${wanloras.length})`, 10, currentY);
                    currentY += 20;
                    
                    wanloras.forEach(lora => {
                        currentY = this.drawLoRAItem(ctx, lora, 20, currentY, width - 30, "#ff9a4a");
                    });
                }
                
                // Update widget height based on content (only if not manually resized)
                const neededHeight = Math.max(200, currentY - y + 20);
                if (this.loraVisualizationWidget && neededHeight !== this.loraVisualizationWidget.computeSize(width)[1]) {
                    this.loraVisualizationWidget.computeSize = function(w) {
                        return [w, neededHeight];
                    };
                    
                    // Only auto-resize if user hasn't manually resized
                    if (!this.manuallyResized) {
                        this.setSize(this.computeSize());
                    }
                }
                
                // Create HTML buttons after drawing is complete
                // Use a small delay to ensure the canvas has been rendered
                setTimeout(() => {
                    this.createHTMLButtons();
                }, 10);
            };
            
            // Add method to draw individual LoRA item
            nodeType.prototype.drawLoRAItem = function(ctx, lora, x, y, width, accentColor) {
                // Calculate thumbnail size as percentage of node width (20% of available width)
                const thumbSize = Math.min(Math.max(width * 0.20, 60), 150); // Min 60px, max 150px, 20% of width
                const itemHeight = Math.max(thumbSize + 20, 110); // Minimum height for larger text
                
                // Draw background
                ctx.fillStyle = "#333";
                ctx.fillRect(x, y, width, itemHeight);
                
                // Draw accent border
                ctx.fillStyle = accentColor;
                ctx.fillRect(x, y, 3, itemHeight);
                
                // Draw scalable thumbnail
                const thumbX = x + 10;
                const thumbY = y + 10;
                
                // Store thumbnail size for hover gallery reference
                if (!this.currentThumbnailSize) {
                    this.currentThumbnailSize = thumbSize;
                }
                
                ctx.fillStyle = "#444";
                ctx.fillRect(thumbX, thumbY, thumbSize, thumbSize);
                
                // Try to load and draw actual thumbnail if available
                this.loadAndDrawThumbnail(ctx, lora, thumbX, thumbY, thumbSize);
                
                // Store thumbnail area for hover detection
                if (!this.thumbnailAreas) {
                    this.thumbnailAreas = [];
                }
                this.thumbnailAreas.push({
                    lora: lora,
                    x: thumbX,
                    y: thumbY,
                    width: thumbSize,
                    height: thumbSize,
                    nodeX: x,
                    nodeY: y
                });
                
                // Draw LoRA info text
                const textX = thumbX + thumbSize + 10;
                const textStartY = y + 15;
                
                // LoRA name
                ctx.fillStyle = "#fff";
                ctx.font = "bold 16px Arial";
                ctx.fillText(lora.name, textX, textStartY);
                
                // Strength
                ctx.fillStyle = "#ccc";
                ctx.font = "14px Arial";
                ctx.fillText(`Strength: ${lora.strength}`, textX, textStartY + 22);
                
                // Tag
                ctx.fillStyle = "#999";
                ctx.font = "12px monospace";
                const tagText = lora.tag.length > 35 ? lora.tag.substring(0, 35) + "..." : lora.tag;
                ctx.fillText(tagText, textX, textStartY + 40);
                
                // Base model
                if (lora.base_model) {
                    ctx.fillStyle = "#4CAF50";
                    ctx.font = "12px Arial";
                    ctx.fillText(`Base: ${lora.base_model}`, textX, textStartY + 56);
                }
                
                // Trigger words (if loaded) with HTML buttons
                const triggerY = lora.base_model ? textStartY + 74 : textStartY + 56;
                
                if (lora.triggerWords && lora.triggerWords.length > 0) {
                    ctx.fillStyle = "#aaa";
                    ctx.font = "12px Arial";
                    const triggerText = `Triggers: ${lora.triggerWords.join(", ")}`;
                    const maxTriggerText = triggerText.length > 50 ? triggerText.substring(0, 50) + "..." : triggerText;
                    ctx.fillText(maxTriggerText, textX, triggerY);
                    
                    // Store button info for HTML button creation
                    const buttonX = textX + ctx.measureText(maxTriggerText).width + 10;
                    const buttonY = triggerY - 10;
                    
                    if (!this.htmlButtonAreas) {
                        this.htmlButtonAreas = [];
                    }
                    
                    // Store copy button info
                    this.htmlButtonAreas.push({
                        type: 'copy',
                        lora: lora,
                        x: buttonX,
                        y: buttonY,
                        triggerWords: lora.triggerWords
                    });
                    
                    // Store link button info (if metadata available)
                    const loraKey = `${lora.name}_${lora.type}`;
                    const metadata = this.loraMetadataCache && this.loraMetadataCache[loraKey];
                    if (metadata && metadata.civitaiUrl) {
                        this.htmlButtonAreas.push({
                            type: 'link',
                            lora: lora,
                            x: buttonX + 22, // 16px button + 6px gap
                            y: buttonY,
                            civitaiUrl: metadata.civitaiUrl
                        });
                    }
                }
                
                return y + itemHeight + 5; // Return next Y position
            };
            
            // Add method to create HTML buttons after drawing
            nodeType.prototype.createHTMLButtons = function() {
                if (!this.htmlButtonAreas || this.htmlButtonAreas.length === 0) return;
                
                const nodeRect = this.getBounding();
                const scale = app.canvas.ds.scale;
                const offsetX = app.canvas.ds.offset[0];
                const offsetY = app.canvas.ds.offset[1];
                
                this.htmlButtonAreas.forEach((buttonInfo, index) => {
                    const button = document.createElement('button');
                    button.className = `lora-viz-button lora-viz-${buttonInfo.type}-button`;
                    button.style.cssText = `
                position: fixed;
                        width: 16px;
                        height: 16px;
                        border: 1px solid ${buttonInfo.type === 'copy' ? '#777' : '#1976D2'};
                        background: ${buttonInfo.type === 'copy' ? '#555' : '#2196F3'};
                        color: white;
                        font-size: 10px;
                        padding: 0;
                        border-radius: 2px;
                        cursor: pointer;
                        z-index: 1000;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-family: Arial, sans-serif;
                    `;
                    button.textContent = buttonInfo.type === 'copy' ? '📋' : '🔗';
                    button.title = buttonInfo.type === 'copy' ? 'Copy trigger words' : 'Open on Civitai';
                    
                    // Calculate screen position
                    const canvasRect = app.canvas.canvas.getBoundingClientRect();
                    const screenX = canvasRect.left + ((buttonInfo.x + nodeRect[0] + offsetX) * scale);
                    const screenY = canvasRect.top + ((buttonInfo.y + nodeRect[1] + offsetY) * scale);
                    
                    button.style.left = screenX + 'px';
                    button.style.top = screenY + 'px';
                    
                    // Add click handler
                    if (buttonInfo.type === 'copy') {
                        button.onclick = () => this.copyTriggerWords(buttonInfo.triggerWords);
                    } else if (buttonInfo.type === 'link') {
                        button.onclick = () => window.open(buttonInfo.civitaiUrl, '_blank');
                    }
                    
                    document.body.appendChild(button);
                    
                    // Store reference for cleanup
                    if (!this.htmlButtons) {
                        this.htmlButtons = [];
                    }
                    this.htmlButtons.push(button);
                });
            };
            
            // Add method to remove HTML buttons
            nodeType.prototype.removeHTMLButtons = function() {
                if (this.htmlButtons) {
                    this.htmlButtons.forEach(button => {
                        if (document.body.contains(button)) {
                            document.body.removeChild(button);
                        }
                    });
                    this.htmlButtons = [];
                }
            };
            
            // Add method to update HTML button positions
            nodeType.prototype.updateHTMLButtonPositions = function() {
                if (!this.htmlButtons || !this.htmlButtonAreas) return;
                
                const nodeRect = this.getBounding();
                const scale = app.canvas.ds.scale;
                const offsetX = app.canvas.ds.offset[0];
                const offsetY = app.canvas.ds.offset[1];
                const canvasRect = app.canvas.canvas.getBoundingClientRect();
                
                this.htmlButtons.forEach((button, index) => {
                    if (index < this.htmlButtonAreas.length) {
                        const buttonInfo = this.htmlButtonAreas[index];
                        const screenX = canvasRect.left + ((buttonInfo.x + nodeRect[0] + offsetX) * scale);
                        const screenY = canvasRect.top + ((buttonInfo.y + nodeRect[1] + offsetY) * scale);
                        
                        button.style.left = screenX + 'px';
                        button.style.top = screenY + 'px';
                    }
                });
            };
            
            // Add method to copy trigger words
            nodeType.prototype.copyTriggerWords = function(triggerWords) {
                const triggerText = triggerWords.join(", ");
                
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(triggerText).then(() => {
                        console.debug("Trigger words copied to clipboard:", triggerText);
                        this.showCopyFeedback("Copied to clipboard!");
                    }).catch(err => {
                        console.error("Failed to copy to clipboard:", err);
                        this.showCopyFeedback("Copy failed!", true);
                    });
                } else {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = triggerText;
                    document.body.appendChild(textArea);
                    textArea.select();
                    try {
                        document.execCommand('copy');
                        console.debug("Trigger words copied to clipboard (fallback):", triggerText);
                        this.showCopyFeedback("Copied to clipboard!");
                    } catch (err) {
                        console.error("Failed to copy to clipboard (fallback):", err);
                        this.showCopyFeedback("Copy failed!", true);
                    }
                    document.body.removeChild(textArea);
                }
            };
            
            // Add method to show simple feedback
            nodeType.prototype.showCopyFeedback = function(message, isError = false) {
                // Create temporary feedback overlay
                const feedback = document.createElement('div');
                feedback.style.cssText = `
                    position: fixed;
                    background: ${isError ? '#ff4444' : '#4CAF50'};
                    color: white;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    z-index: 10000;
                    pointer-events: none;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                    top: 20px;
                    right: 20px;
                `;
                feedback.textContent = message;
                
                document.body.appendChild(feedback);
                
                // Remove after 2 seconds
                setTimeout(() => {
                    if (document.body.contains(feedback)) {
                        document.body.removeChild(feedback);
                    }
                }, 2000);
            };
            
                        // Note: LoRA parsing is now done in Python backend
            // JavaScript just displays the data received from backend
            
            // Add method to load and draw thumbnails
            nodeType.prototype.loadAndDrawThumbnail = function(ctx, lora, x, y, size) {
                // Create unique key for this LoRA
                const loraKey = `${lora.name}_${lora.type}`;
                
                // Check if we already have metadata for this LoRA
                if (!this.loraMetadataCache) {
                    this.loraMetadataCache = {};
                }
                
                if (!this.loraMetadataCache[loraKey]) {
                    // Load metadata asynchronously
                    this.loadLoRAMetadata(lora, loraKey);
                    
                    // Draw loading placeholder
                    if (lora.type === 'wanlora') {
                        ctx.fillStyle = "#ff9a4a";
    } else {
                        ctx.fillStyle = "#4a9eff";
                    }
                    ctx.fillRect(x + 5, y + 5, size - 10, size - 10);
                    
                    ctx.fillStyle = "#fff";
                    ctx.font = "8px Arial";
                    ctx.fillText("Loading...", x + 8, y + size/2);
                } else {
                    const metadata = this.loraMetadataCache[loraKey];
                    
                    if (metadata.thumbnailImage) {
                        // Draw the loaded image
                        try {
                            ctx.drawImage(metadata.thumbnailImage, x, y, size, size);
                            
                            // Add video indicator if this is a video thumbnail
                            if (metadata.isVideoThumbnail) {
                                // Draw video label overlay
                                ctx.fillStyle = "rgba(0,0,0,0.7)";
                                ctx.fillRect(x + size - 25, y + size - 12, 23, 10);
                                
                                ctx.fillStyle = "#fff";
                                ctx.font = "6px Arial";
                                ctx.fillText("VIDEO", x + size - 23, y + size - 4);
                            }
                        } catch (e) {
                            // Fallback to placeholder
                            this.drawPlaceholder(ctx, lora, x, y, size);
                        }
                    } else {
                        // Draw "No Image" placeholder
                        this.drawPlaceholder(ctx, lora, x, y, size);
                    }
                }
            };
            
            // Add method to draw placeholder thumbnails
            nodeType.prototype.drawPlaceholder = function(ctx, lora, x, y, size) {
                if (lora.type === 'wanlora') {
                    ctx.fillStyle = "#ff9a4a";
        } else {
                    ctx.fillStyle = "#4a9eff";
                }
                ctx.fillRect(x + 5, y + 5, size - 10, size - 10);
                
                ctx.fillStyle = "#fff";
                ctx.font = "8px Arial";
                ctx.fillText("No Image", x + 8, y + size/2);
            };
            
            // Add method to load LoRA metadata
            nodeType.prototype.loadLoRAMetadata = async function(lora, loraKey) {
                try {
                    console.debug(`Loading metadata for LoRA: ${lora.name}`);
                    
                    // Try to fetch metadata from backend
                    const response = await fetch(`/lora_metadata/${lora.name}`);
                    
                    if (response.ok) {
                        const metadata = await response.json();
                        console.debug(`Loaded metadata for ${lora.name}`);
                        
                        // Process metadata
                        const processedMetadata = {
                            triggerWords: metadata.civitai?.trainedWords || [],
                            previewUrl: metadata.preview_url,
                            exampleImages: metadata.civitai?.images || [],
                            baseModel: metadata.base_model,
                            civitaiUrl: metadata.civitai?.url || metadata.url || null,
                            thumbnailImage: null
                        };
                        
                        // Try to load thumbnail (prefer images, but allow videos for WanLoRAs)
                        if (metadata.civitai?.images && metadata.civitai.images.length > 0) {
                            // First try to find an image (not video)
                            let thumbnailEntry = metadata.civitai.images.find(img => 
                                img.type !== 'video' && !img.url.endsWith('.mp4')
                            );
                            
                            // If no image found and this is a WanLoRA, use the first video as thumbnail
                            if (!thumbnailEntry && lora.type === 'wanlora') {
                                thumbnailEntry = metadata.civitai.images.find(img => 
                                    img.type === 'video' || img.url.endsWith('.mp4')
                                );
                                console.debug(`Using video thumbnail for WanLoRA ${lora.name}`);
                            }
                            
                            if (thumbnailEntry) {
                                // Check if it's a video
                                const isVideo = thumbnailEntry.type === 'video' || thumbnailEntry.url.endsWith('.mp4');
                                
                                if (isVideo) {
                                    // Create video element for thumbnail
                                    const video = document.createElement('video');
                                    video.crossOrigin = "anonymous";
                                    video.muted = true;
                                    video.playsInline = true;
                                    video.autoplay = true;
                                    video.style.cssText = `
                                        position: fixed;
                                        top: -1000px;
                                        left: -1000px;
                                        width: 100px;
                                        height: 100px;
                                        opacity: 0;
                                        pointer-events: none;
                                    `;
                                    
                                    // Add to DOM temporarily for autoplay to work
                                    document.body.appendChild(video);
                                    
                                    // Wait for video to start playing, then capture frame
                                    video.onplaying = () => {
                                        console.debug(`Video started playing for ${lora.name}, capturing frame`);
                                        
                                        // Small delay to ensure we have a good frame
                                        setTimeout(() => {
                                            // Create canvas to capture frame
                                            const canvas = document.createElement('canvas');
                                            const ctx = canvas.getContext('2d');
                                            canvas.width = video.videoWidth || 512;
                                            canvas.height = video.videoHeight || 512;
                                            
                                            // Draw current frame to canvas
                                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                            
                                            // Convert to image
                                            const img = new Image();
                                            img.onload = () => {
                                                processedMetadata.thumbnailImage = img;
                                                processedMetadata.isVideoThumbnail = true;
                                                this.loraMetadataCache[loraKey] = processedMetadata;
                                                
                                                // Update trigger words in the parsed lora data
                                                lora.triggerWords = processedMetadata.triggerWords;
                                                
                                                // Force redraw
                                                this.setDirtyCanvas(true, true);
                                                
                                                // Clean up the video element
                                                if (document.body.contains(video)) {
                                                    document.body.removeChild(video);
                                                }
                                            };
                                            img.src = canvas.toDataURL();
                                        }, 100); // 100ms delay to get a good frame
                                    };
                                    
                                    video.onerror = () => {
                                        console.warn(`Failed to load video thumbnail for ${lora.name}`);
                                        this.loraMetadataCache[loraKey] = processedMetadata;
                                        this.setDirtyCanvas(true, true);
                                        
                                        // Clean up the video element
                                        if (document.body.contains(video)) {
                                            document.body.removeChild(video);
                                        }
                                    };
                                    
                                    // Add a timeout in case video doesn't play
                                    setTimeout(() => {
                                        if (document.body.contains(video)) {
                                            console.warn(`Video thumbnail timeout for ${lora.name}, removing video element`);
                                            document.body.removeChild(video);
                                            // Still cache the metadata even without thumbnail
                                            this.loraMetadataCache[loraKey] = processedMetadata;
                                            this.setDirtyCanvas(true, true);
                                        }
                                    }, 5000); // 5 second timeout
                                    
                                    video.src = thumbnailEntry.url;
                                } else {
                                    // Regular image thumbnail
                                    const img = new Image();
                                    img.crossOrigin = "anonymous";
                                    
                                    img.onload = () => {
                                        console.debug(`Loaded image thumbnail for ${lora.name}`);
                                        processedMetadata.thumbnailImage = img;
                                        this.loraMetadataCache[loraKey] = processedMetadata;
                                        
                                        // Update trigger words in the parsed lora data
                                        lora.triggerWords = processedMetadata.triggerWords;
                                        
                                        // Force redraw
                                        this.setDirtyCanvas(true, true);
                                    };
                                    
                                    img.onerror = () => {
                                        console.warn(`Failed to load image thumbnail for ${lora.name}`);
                                        this.loraMetadataCache[loraKey] = processedMetadata;
                                        this.setDirtyCanvas(true, true);
                                    };
                                    
                                    img.src = thumbnailEntry.url;
                                }
                            } else {
                                console.debug(`No suitable thumbnails found for ${lora.name}`);
                                this.loraMetadataCache[loraKey] = processedMetadata;
                                lora.triggerWords = processedMetadata.triggerWords;
                                this.setDirtyCanvas(true, true);
                            }
    } else {
                            this.loraMetadataCache[loraKey] = processedMetadata;
                            lora.triggerWords = processedMetadata.triggerWords;
                            this.setDirtyCanvas(true, true);
                        }
                    } else {
                        console.warn(`No metadata found for LoRA: ${lora.name}`);
                        // No metadata available
                        this.loraMetadataCache[loraKey] = {
                            triggerWords: [],
                            thumbnailImage: null
                        };
                        this.setDirtyCanvas(true, true);
                    }
    } catch (error) {
                    console.warn("Failed to load LoRA metadata:", error);
                    this.loraMetadataCache[loraKey] = {
                        triggerWords: [],
                        thumbnailImage: null
                    };
                    this.setDirtyCanvas(true, true);
                }
            };
            
            // Override widget value change to trigger updates
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value, old_value) {
                const result = onWidgetChanged?.apply(this, arguments);
                
                // If prompt text changed, force redraw
                if (name === "prompt_text" && value !== old_value) {
                    console.debug("Prompt text changed, redrawing visualization");
                    this.setDirtyCanvas(true, true);
                }
                
                return result;
            };
        }
    },
    
    async setup() {
        // Listen for LoRA visualization data from backend
        api.addEventListener("lora_visualization_data", (event) => {
            console.debug("Received LoRA visualization data");
            
            const data = event.detail;
            
            // Find all LoRAVisualizer nodes and update them
            let updatedNodes = 0;
            app.graph._nodes.forEach(node => {
                if (node.comfyClass === "LoRAVisualizer") {
                    node.loraData = data.data;
                    node.setDirtyCanvas(true, true);
                    updatedNodes++;
                }
            });
            
            if (updatedNodes === 0) {
                console.warn("No LoRAVisualizer nodes found to update");
            } else {
                console.debug(`Updated ${updatedNodes} LoRAVisualizer node(s)`);
            }
        });
    }
});