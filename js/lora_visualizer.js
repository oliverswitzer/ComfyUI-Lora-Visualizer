import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

// ============================================================================
// UTILITY FUNCTIONS (hoisted for readability)
// ============================================================================

function transformCanvasCoordinates(e, node) {
  const rect = app.canvas.canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const canvasX = x / app.canvas.ds.scale - app.canvas.ds.offset[0];
  const canvasY = y / app.canvas.ds.scale - app.canvas.ds.offset[1];

  const nodePos = node.pos;
  return {
    x: canvasX - nodePos[0],
    y: canvasY - nodePos[1],
  };
}

function createButtonArea(type, x, y, width, height, data) {
  return {
    type,
    x,
    y,
    width,
    height,
    ...data,
  };
}

function drawButton(
  ctx,
  x,
  y,
  width,
  height,
  text,
  backgroundColor,
  borderColor
) {
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = borderColor;
  ctx.strokeRect(x, y, width, height);

  ctx.fillStyle = "#fff";
  ctx.font = "10px Arial";
  ctx.fillText(text, x + 4, y + 12);
}

// ============================================================================
// METADATA PROCESSING
// ============================================================================

function loadThumbnailImage(lora, node) {
  // Skip if already loaded or loading
  if (lora.thumbnailImage || lora.thumbnailLoading) {
    return;
  }

  // Load thumbnail from the first available example image
  if (!lora.example_images || lora.example_images.length === 0) {
    return;
  }

  let thumbnailEntry = lora.example_images.find(
    (img) => !img.url.endsWith(".mp4")
  );

  if (!thumbnailEntry && lora.type === "wanlora") {
    thumbnailEntry = lora.example_images.find(
      (img) => img.url.endsWith(".mp4")
    );
  }

  if (!thumbnailEntry) return;

  // Mark as loading to prevent duplicate requests
  lora.thumbnailLoading = true;

  const isVideo = thumbnailEntry.url.endsWith(".mp4");

  if (isVideo) {
    loadVideoThumbnail(lora, thumbnailEntry, node);
  } else {
    loadImageThumbnail(lora, thumbnailEntry, node);
  }
}

function loadVideoThumbnail(lora, thumbnailEntry, node) {
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.loop = false;

  video.addEventListener("loadeddata", () => {
    video.currentTime = 1.0;
  });

  video.addEventListener("seeked", () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 512;
    canvas.height = video.videoHeight || 512;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const img = new Image();
    img.onload = () => {
      lora.thumbnailImage = img;
      lora.isVideoThumbnail = true;
      lora.thumbnailLoading = false;
      node.setDirtyCanvas(true, true);
    };
    img.src = canvas.toDataURL();
  });

  video.addEventListener("error", () => {
    console.warn(`Failed to load video thumbnail for ${lora.name}`);
    lora.thumbnailLoading = false;
    node.setDirtyCanvas(true, true);
  });

  video.src = thumbnailEntry.url;
}

function loadImageThumbnail(lora, thumbnailEntry, node) {
  const img = new Image();
  img.crossOrigin = "anonymous";

  img.onload = () => {
    lora.thumbnailImage = img;
    lora.thumbnailLoading = false;
    node.setDirtyCanvas(true, true);
  };

  img.onerror = () => {
    console.warn(`Failed to load image thumbnail for ${lora.name}`);
    lora.thumbnailLoading = false;
    node.setDirtyCanvas(true, true);
  };

  img.src = thumbnailEntry.url;
}

// ============================================================================
// DRAWING FUNCTIONS
// ============================================================================

function calculateItemDimensions(width) {
  // Conservative space allocation to ensure everything fits within bounds
  // Account for: left margin (20px) + thumbnail margin (10px) + text space (min 150px) + right margin (10px)
  const reservedSpace = 190; // 20 + 10 + 150 + 10
  const availableForThumbnail = Math.max(width - reservedSpace, 60);
  
  // Cap thumbnail size to reasonable maximum and ensure it fits
  const maxThumbSize = Math.min(width * 0.2, 120); // More conservative 20% with lower max
  const thumbSize = Math.min(maxThumbSize, availableForThumbnail);
  
  // Ensure minimum viable thumbnail size
  const actualThumbSize = Math.max(thumbSize, 60);
  
  // Item height should accommodate thumbnail + padding
  const itemHeight = Math.max(actualThumbSize + 20, 130);
  
  return { thumbSize: actualThumbSize, itemHeight };
}

function drawLoRAItem(ctx, lora, x, y, width, accentColor, node) {
  const { thumbSize, itemHeight } = calculateItemDimensions(width);

  // Background and border
  ctx.fillStyle = "#333";
  ctx.fillRect(x, y, width, itemHeight);
  ctx.fillStyle = accentColor;
  ctx.fillRect(x, y, 3, itemHeight);

  // Components - ensure thumbnail fits within bounds
  const thumbX = x + 10;
  const thumbY = y + 10;
  
  // Ensure thumbnail doesn't exceed available width
  const availableWidth = width - 20; // Account for left and right margins
  const maxThumbnailWidth = Math.min(thumbSize, availableWidth - 150); // Reserve 150px minimum for text
  const actualThumbSize = Math.max(maxThumbnailWidth, 60); // Minimum 60px thumbnail
  
  const textX = thumbX + actualThumbSize + 10;
  const textStartY = y + 25;
  
  // Ensure text doesn't exceed right boundary
  const textAreaWidth = width - textX - 10;

  drawThumbnail(ctx, lora, thumbX, thumbY, actualThumbSize, node);
  const triggerY = drawLoRAText(ctx, lora, textX, textStartY, textAreaWidth);
  drawLoRAButtons(ctx, lora, textX, triggerY, node, textAreaWidth);

  return y + itemHeight + 5;
}

function drawThumbnail(ctx, lora, x, y, size, node) {
  if (lora.thumbnailImage) {
    ctx.drawImage(lora.thumbnailImage, x, y, size, size);

    if (lora.isVideoThumbnail) {
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(x + size - 45, y + size - 20, 40, 15);
      ctx.fillStyle = "#fff";
      ctx.font = "10px Arial";
      ctx.fillText("VIDEO", x + size - 42, y + size - 10);
    }
  } else {
    ctx.fillStyle = "#555";
    ctx.fillRect(x, y, size, size);
    ctx.fillStyle = "#999";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.fillText("No Image", x + size / 2, y + size / 2);
    ctx.textAlign = "left";
  }

  storeThumbnailArea(lora, x, y, size, node);
}

function drawLoRAText(ctx, lora, x, startY, maxWidth = null) {
  // LoRA name
  ctx.fillStyle = "#fff";
  ctx.font = "bold 16px Arial";
  let nameText = lora.name;
  if (maxWidth) {
    // Use canvas measureText to ensure text fits within bounds
    while (ctx.measureText(nameText).width > maxWidth && nameText.length > 3) {
      nameText = nameText.substring(0, nameText.length - 4) + "...";
    }
  } else {
    nameText = lora.name.length > 25 ? lora.name.substring(0, 25) + "..." : lora.name;
  }
  ctx.fillText(nameText, x, startY);

  // Strength
  ctx.fillStyle = "#ccc";
  ctx.font = "14px Arial";
  ctx.fillText(`Strength: ${lora.strength}`, x, startY + 22);

  // Tag
  ctx.fillStyle = "#999";
  ctx.font = "12px monospace";
  let tagText = lora.tag;
  if (maxWidth) {
    while (ctx.measureText(tagText).width > maxWidth && tagText.length > 3) {
      tagText = tagText.substring(0, tagText.length - 4) + "...";
    }
  } else {
    tagText = lora.tag.length > 35 ? lora.tag.substring(0, 35) + "..." : lora.tag;
  }
  ctx.fillText(tagText, x, startY + 40);

  // Base model
  if (lora.base_model) {
    ctx.fillStyle = "#4CAF50";
    ctx.font = "12px Arial";
    ctx.fillText(`Base: ${lora.base_model}`, x, startY + 56);
  }

  // Trigger words
  const triggerY = lora.base_model ? startY + 74 : startY + 56;
  if (lora.trigger_words?.length > 0) {
    ctx.fillStyle = "#aaa";
    ctx.font = "12px Arial";
    let triggerText = `Triggers: ${lora.trigger_words.join(", ")}`;
    if (maxWidth) {
      while (ctx.measureText(triggerText).width > maxWidth && triggerText.length > 12) {
        // Keep at least "Triggers: ..."
        const words = triggerText.split(": ")[1];
        if (words && words.length > 3) {
          triggerText = `Triggers: ${words.substring(0, words.length - 4)}...`;
        } else {
          break;
        }
      }
    } else {
      triggerText = triggerText.length > 50 ? triggerText.substring(0, 50) + "..." : triggerText;
    }
    ctx.fillText(triggerText, x, triggerY);
  }

  return triggerY;
}

function drawLoRAButtons(ctx, lora, textX, triggerY, node, maxWidth = null) {
  const copyButtonY = triggerY + 16;

  // Calculate available width for buttons
  const availableButtonWidth = maxWidth || 300; // Fallback to reasonable default
  
  // If no trigger words and no civitai URL, don't show any buttons
  if (!lora.trigger_words?.length && !lora.civitai_url) return;
  
  let currentX = textX;
  
  // Copy button (only if there are trigger words)
  if (lora.trigger_words?.length) {
    const copyButtonWidth = Math.min(120, availableButtonWidth - 10);
    
    drawButton(
      ctx,
      currentX,
      copyButtonY - 10,
      copyButtonWidth,
      16,
      "Copy Trigger Words",
      "#555",
      "#777"
    );
    storeButtonArea(node, "copy", currentX, copyButtonY - 10, copyButtonWidth, 16, {
      triggerWords: lora.trigger_words,
      lora: lora,
    });
    
    currentX += copyButtonWidth + 5;
  }

  // Civitai button (if available)
  if (lora.civitai_url) {
    const remainingWidth = availableButtonWidth - (currentX - textX);
    const linkButtonWidth = Math.min(80, remainingWidth);
    
    // Only show if there's enough space
    if (linkButtonWidth > 40) {
      drawButton(
        ctx,
        currentX,
        copyButtonY - 10,
        linkButtonWidth,
        16,
        "Open Civitai",
        "#2196F3",
        "#1976D2"
      );
      storeButtonArea(node, "link", currentX, copyButtonY - 10, linkButtonWidth, 16, {
        civitaiUrl: lora.civitai_url,
        lora: lora,
      });
    }
  }
}

function storeThumbnailArea(lora, x, y, size, node) {
  if (!node.thumbnailAreas) node.thumbnailAreas = [];
  node.thumbnailAreas.push({
    lora: lora,
    x: x,
    y: y,
    width: size,
    height: size,
  });
}

function storeButtonArea(node, type, x, y, width, height, data) {
  if (!node.buttonAreas) node.buttonAreas = [];
  node.buttonAreas.push(createButtonArea(type, x, y, width, height, data));
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

function handleCanvasClick(e, node) {
  if (!node.buttonAreas?.length) return;

  const coords = transformCanvasCoordinates(e, node);

  for (const buttonArea of node.buttonAreas) {
    if (isPointInArea(coords, buttonArea)) {
      executeButtonAction(buttonArea, node);
      return;
    }
  }
}

function handleMouseMove(e, node) {
  if (!node.thumbnailAreas) return;

  const coords = transformCanvasCoordinates(e, node);
  const hoveredThumbnail = findHoveredThumbnail(coords, node.thumbnailAreas);

  if (hoveredThumbnail && hoveredThumbnail !== node.currentlyHovered) {
    node.currentlyHovered = hoveredThumbnail;
    const thumbSize = Math.min(Math.max(node.size[0] * 0.2, 60), 150);
    showHoverGallery(
      hoveredThumbnail.lora,
      e.clientX,
      e.clientY,
      thumbSize,
      node
    );
  } else if (!hoveredThumbnail && node.currentlyHovered) {
    scheduleHideGallery(node);
  }
}

function isPointInArea(point, area) {
  return (
    point.x >= area.x &&
    point.x <= area.x + area.width &&
    point.y >= area.y &&
    point.y <= area.y + area.height
  );
}

function findHoveredThumbnail(coords, thumbnailAreas) {
  for (const thumbArea of thumbnailAreas) {
    if (isPointInArea(coords, thumbArea)) {
      return thumbArea;
    }
  }
  return null;
}

function executeButtonAction(buttonArea, node) {
  if (buttonArea.type === "copy") {
    copyTriggerWords(buttonArea.triggerWords, node);
  } else if (buttonArea.type === "link") {
    window.open(buttonArea.civitaiUrl, "_blank");
  }
}

function scheduleHideGallery(node) {
  if (node.hideTimeout) clearTimeout(node.hideTimeout);
  node.hideTimeout = setTimeout(() => {
    if (!node.galleryHovered) {
      hideHoverGallery(node);
      node.currentlyHovered = null;
    }
  }, 300);
}

// ============================================================================
// CLIPBOARD FUNCTIONALITY
// ============================================================================

function copyTriggerWords(triggerWords, node) {
  const text = triggerWords.join(", ");

  if (navigator.clipboard?.writeText) {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        showCopyFeedback("Copied to clipboard!", false);
      })
      .catch(() => {
        fallbackCopyToClipboard(text);
      });
  } else {
    fallbackCopyToClipboard(text);
  }
}

function fallbackCopyToClipboard(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.cssText = "position: fixed; left: -999999px; top: -999999px;";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    document.execCommand("copy");
    showCopyFeedback("Copied to clipboard!", false);
  } catch (err) {
    showCopyFeedback("Failed to copy", true);
  }

  document.body.removeChild(textArea);
}

function showCopyFeedback(message, isError) {
  const feedback = document.createElement("div");
  feedback.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: ${isError ? "#ff4444" : "#4CAF50"};
        color: white;
        padding: 12px 20px;
        border-radius: 6px;
        font-size: 14px;
        font-family: Arial, sans-serif;
        z-index: 10001;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
  feedback.textContent = message;

  document.body.appendChild(feedback);

  setTimeout(() => {
    if (document.body.contains(feedback)) {
      document.body.removeChild(feedback);
    }
  }, 2000);
}

// ============================================================================
// HOVER GALLERY
// ============================================================================

function showHoverGallery(lora, x, y, thumbnailSize, node) {
  hideHoverGallery(node);

  if (!lora.example_images || lora.example_images.length === 0) return;

  const gallery = createGalleryElement(lora, x, y);

  addGalleryTitle(gallery, lora.name);
  addGalleryImages(gallery, lora, thumbnailSize);
  addGalleryCivitaiLink(gallery, lora);
  setupGalleryEventHandlers(gallery, node);

  document.body.appendChild(gallery);
  node.hoverGallery = gallery;
  node.galleryHovered = false;
}

function createGalleryElement(lora, x, y) {
  const gallery = document.createElement("div");
  gallery.className = "lora-hover-gallery";
  gallery.style.cssText = `
        position: fixed;
        left: ${x + 10}px;
        top: ${y + 10}px;
        background: rgba(0, 0, 0, 0.95);
        border: 2px solid ${lora.type === "wanlora" ? "#ff9a4a" : "#4a9eff"};
        border-radius: 8px;
        padding: 15px;
        max-width: 500px;
        max-height: 600px;
        overflow-y: auto;
        z-index: 10000;
        color: white;
        font-family: Arial, sans-serif;
    `;
  return gallery;
}

function addGalleryTitle(gallery, loraName) {
  const title = document.createElement("div");
  title.style.cssText =
    "font-size: 14px; font-weight: bold; margin-bottom: 10px;";
  title.textContent = `${loraName} Examples`;
  gallery.appendChild(title);
}

function addGalleryImages(gallery, lora, thumbnailSize) {
  if (!lora.example_images?.length) {
    const noImagesDiv = document.createElement("div");
    noImagesDiv.style.cssText =
      "margin-top: 10px; font-size: 12px; color: #999;";
    noImagesDiv.textContent = "No example images available";
    gallery.appendChild(noImagesDiv);
    return;
  }

  const imagesDiv = document.createElement("div");
  imagesDiv.style.cssText = `
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(${thumbnailSize}px, 1fr));
        gap: 12px;
        margin-top: 10px;
    `;

  lora.example_images.slice(0, 6).forEach((imageData) => {
    const element = createGalleryImageElement(imageData, thumbnailSize);
    imagesDiv.appendChild(element);
  });

  gallery.appendChild(imagesDiv);
}

function createGalleryImageElement(imageData, thumbnailSize) {
  const isVideo = imageData.type === "video" || imageData.url.endsWith(".mp4");

  if (isVideo) {
    return createGalleryVideoElement(imageData, thumbnailSize);
  } else {
    return createGalleryImageDiv(imageData, thumbnailSize);
  }
}

function createGalleryVideoElement(imageData, thumbnailSize) {
  const container = document.createElement("div");
  container.style.cssText = `
        width: ${thumbnailSize}px;
        background: rgba(40, 40, 40, 0.9);
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    `;

  const videoContainer = document.createElement("div");
  videoContainer.style.cssText = `
        width: 100%;
        height: ${thumbnailSize}px;
        position: relative;
        cursor: pointer;
    `;

  const video = document.createElement("video");
  video.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    `;
  video.src = imageData.url;
  video.autoplay = true;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;

  const label = document.createElement("div");
  label.style.cssText = `
        position: absolute;
        bottom: 2px;
        right: 2px;
        background: rgba(0,0,0,0.7);
        color: white;
        font-size: 8px;
        padding: 1px 3px;
        border-radius: 2px;
    `;
  label.textContent = "VIDEO";

  videoContainer.onclick = () => window.open(imageData.url, "_blank");
  videoContainer.appendChild(video);
  videoContainer.appendChild(label);
  
  container.appendChild(videoContainer);

  // Add prompt section if available
  if (imageData.meta?.prompt) {
    const promptSection = createPromptSection(imageData.meta.prompt);
    container.appendChild(promptSection);
  }

  return container;
}

function createGalleryImageDiv(imageData, thumbnailSize) {
  const container = document.createElement("div");
  container.style.cssText = `
        width: ${thumbnailSize}px;
        background: rgba(40, 40, 40, 0.9);
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    `;

  const img = document.createElement("img");
  img.style.cssText = `
        width: 100%;
        height: ${thumbnailSize}px;
        object-fit: cover;
        cursor: pointer;
        display: block;
    `;
  img.src = imageData.url;
  img.onclick = () => window.open(imageData.url, "_blank");

  container.appendChild(img);

  // Add prompt section if available
  if (imageData.meta?.prompt) {
    const promptSection = createPromptSection(imageData.meta.prompt);
    container.appendChild(promptSection);
  }

  return container;
}

function createPromptSection(prompt) {
  const promptSection = document.createElement("div");
  promptSection.style.cssText = `
        padding: 8px;
        background: rgba(20, 20, 20, 0.95);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    `;

  // Truncate prompt for display
  const maxLength = 80;
  const displayPrompt = prompt.length > maxLength ? 
    prompt.substring(0, maxLength) + "..." : prompt;

  const promptText = document.createElement("div");
  promptText.style.cssText = `
        font-size: 10px;
        color: #ccc;
        margin-bottom: 6px;
        line-height: 1.3;
        word-wrap: break-word;
        white-space: pre-wrap;
    `;
  promptText.textContent = displayPrompt;

  const copyButton = document.createElement("button");
  copyButton.style.cssText = `
        background: rgba(74, 158, 255, 0.8);
        border: none;
        color: white;
        padding: 4px 8px;
        border-radius: 3px;
        font-size: 9px;
        cursor: pointer;
        transition: background 0.2s;
        width: 100%;
    `;
  copyButton.textContent = "Copy Prompt";
  
  copyButton.onmouseover = () => {
    copyButton.style.background = "rgba(74, 158, 255, 1)";
  };
  
  copyButton.onmouseout = () => {
    copyButton.style.background = "rgba(74, 158, 255, 0.8)";
  };

  copyButton.onclick = (e) => {
    e.stopPropagation(); // Prevent triggering image click
    
    // Use the existing clipboard functionality
    if (navigator.clipboard?.writeText) {
      navigator.clipboard
        .writeText(prompt)
        .then(() => {
          showCopyFeedback("Copied to clipboard!", false);
        })
        .catch(() => {
          fallbackCopyToClipboard(prompt);
        });
    } else {
      fallbackCopyToClipboard(prompt);
    }
  };

  promptSection.appendChild(promptText);
  promptSection.appendChild(copyButton);

  return promptSection;
}



function addGalleryCivitaiLink(gallery, lora) {
  if (!lora.civitai_url) return;

  const link = document.createElement("a");
  link.style.cssText = `
        display: inline-block;
        margin-top: 10px;
        padding: 6px 12px;
        background: #2196F3;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        font-size: 12px;
    `;
  link.href = lora.civitai_url;
  link.target = "_blank";
  link.textContent = "View on Civitai";
  gallery.appendChild(link);
}

function setupGalleryEventHandlers(gallery, node) {
  gallery.addEventListener("mouseenter", () => (node.galleryHovered = true));
  gallery.addEventListener("mouseleave", () => {
    node.galleryHovered = false;
    setTimeout(() => {
      if (!node.galleryHovered) {
        hideHoverGallery(node);
      }
    }, 100);
  });
}

function hideHoverGallery(node) {
  if (node.hoverGallery) {
    document.body.removeChild(node.hoverGallery);
    node.hoverGallery = null;
  }
}

// ============================================================================
// MAIN DRAWING ORCHESTRATION
// ============================================================================

function drawLoRAVisualization(ctx, width, y, height, node) {
  clearDrawingState(node);

  const standardLoras = node.loraData?.standard_loras || [];
  const wanloras = node.loraData?.wanloras || [];

  if (standardLoras.length === 0 && wanloras.length === 0) {
    drawEmptyState(ctx, y);
    return;
  }

  let currentY = y + 15;

  if (standardLoras.length > 0) {
    currentY = drawLoRASection(
      ctx,
      "Standard LoRAs",
      standardLoras,
      currentY,
      width,
      "#4a9eff",
      node
    );
  }

  if (wanloras.length > 0) {
    currentY = drawLoRASection(
      ctx,
      "WanLoRAs",
      wanloras,
      currentY,
      width,
      "#ff9a4a",
      node
    );
  }

  updateNodeSize(node);
}

function clearDrawingState(node) {
  node.thumbnailAreas = [];
  node.buttonAreas = [];
  node.currentThumbnailSize = null;
}

function drawEmptyState(ctx, y) {
  ctx.fillStyle = "#666";
  ctx.font = "12px Arial";
  ctx.fillText("Execute node to see LoRA visualization...", 10, y + 20);
}

function drawLoRASection(ctx, title, loras, startY, width, accentColor, node) {
  // Section header
  ctx.fillStyle = accentColor;
  ctx.font = "bold 14px Arial";
  ctx.fillText(`${title} (${loras.length})`, 10, startY);

  let currentY = startY + 20;

  // Draw each LoRA
  loras.forEach((lora) => {
    loadThumbnailImage(lora, node);
    currentY = drawLoRAItem(
      ctx,
      lora,
      20,
      currentY,
      width - 30,
      accentColor,
      node
    );
  });

  return currentY + 10;
}

function updateNodeSize(node) {
  if (!node.manuallyResized) {
    // Force recalculation of widget size based on current content
    if (node.loraVisualizationWidget) {
      const currentSize = node.size;
      const newSize = node.loraVisualizationWidget.computeSize(currentSize[0]);
      node.setSize([currentSize[0], newSize[1]]);
    } else {
      node.setSize(node.computeSize());
    }
  }
}

// ============================================================================
// NODE SETUP AND INITIALIZATION
// ============================================================================

function setupNodeEventHandlers(node) {
  const canvas = app.canvas.canvas;

  // Throttled mouse movement
  let mouseTimeout;
  canvas.addEventListener("mousemove", (e) => {
    clearTimeout(mouseTimeout);
    mouseTimeout = setTimeout(() => {
      handleMouseMove(e, node);
    }, 16);
  });

  // Mouse leave
  canvas.addEventListener("mouseleave", (e) => {
    clearTimeout(mouseTimeout);
    if (node.hideTimeout) clearTimeout(node.hideTimeout);
    node.hideTimeout = setTimeout(() => {
      if (!node.galleryHovered) {
        hideHoverGallery(node);
        node.currentlyHovered = null;
      }
    }, 300);
  });

  // Click handling
  canvas.addEventListener("click", (e) => {
    handleCanvasClick(e, node);
  });
}

function setupNodeState(node) {
  node.loraData = { standard_loras: [], wanloras: [] };
  node.thumbnailAreas = [];
  node.buttonAreas = [];
  node.currentlyHovered = null;
  node.hoverGallery = null;
  node.galleryHovered = false;
  node.manuallyResized = false;
}

function createVisualizationWidget(node) {
  const widget = node.addWidget(
    "LORA_VIZ",
    "lora_visualization",
    "",
    () => {},
    {
      serialize: false,
    }
  );

  widget.computeSize = function (width) {
    // Calculate dynamic height based on content
    const standardLoras = node.loraData?.standard_loras || [];
    const wanloras = node.loraData?.wanloras || [];
    
    if (standardLoras.length === 0 && wanloras.length === 0) {
      return [width, 60]; // Minimal height for empty state
    }
    
    let totalHeight = 30; // Initial padding
    
    // Calculate height for standard LoRAs section
    if (standardLoras.length > 0) {
      totalHeight += 40; // Section header height
      standardLoras.forEach(() => {
        const { itemHeight } = calculateItemDimensions(width);
        totalHeight += itemHeight + 5; // Item height + spacing
      });
    }
    
    // Calculate height for WanLoRAs section  
    if (wanloras.length > 0) {
      totalHeight += 40; // Section header height
      wanloras.forEach(() => {
        const { itemHeight } = calculateItemDimensions(width);
        totalHeight += itemHeight + 5; // Item height + spacing
      });
    }
    
    totalHeight += 20; // Bottom padding
    
    return [width, Math.max(totalHeight, 150)]; // Minimum 150px height
  };

  widget.draw = function (ctx, node, widgetWidth, y, height) {
    drawLoRAVisualization(ctx, widgetWidth, y, height, node);
  };

  node.loraVisualizationWidget = widget;
  node.setSize(node.computeSize());
}

function setupResizeHandling(node) {
  const originalOnResize = node.constructor.prototype.onResize;

  node.onResize = function (size) {
    if (node.loraVisualizationWidget) {
      const autoSize = node.loraVisualizationWidget.computeSize(size[0]);
      if (Math.abs(size[1] - autoSize[1]) > 10) {
        node.manuallyResized = true;
      }
    }

    if (originalOnResize) {
      return originalOnResize.apply(this, arguments);
    }
  };
}

function setupContextMenu(node) {
  const originalGetExtraMenuOptions = node.getExtraMenuOptions;

  node.getExtraMenuOptions = function (_, options) {
    if (originalGetExtraMenuOptions) {
      originalGetExtraMenuOptions.apply(this, arguments);
    }

    options.push({
      content: "Enable Auto-Sizing",
      callback: () => {
        node.manuallyResized = false;
        if (node.loraVisualizationWidget) {
          node.setSize(node.computeSize());
        }
      },
      disabled: !node.manuallyResized,
    });
  };
}

function setupWebSocketHandler() {
  api.addEventListener("lora_visualization_data", (event) => {
    const messageData = event.detail;

    // Find the specific node by ID first, fallback to all LoRAVisualizer nodes
    let targetNodes = [];
    if (messageData.node_id) {
      const targetNode = app.graph._nodes.find(
        (n) => n.id.toString() === messageData.node_id
      );
      if (targetNode && targetNode.type === "LoRAVisualizer") {
        targetNodes = [targetNode];
      }
    }

    // Fallback to all LoRAVisualizer nodes if no specific node found
    if (targetNodes.length === 0) {
      targetNodes = app.graph._nodes.filter((n) => n.type === "LoRAVisualizer");
    }

    // Update the target nodes with new data
    targetNodes.forEach((node) => {
      node.loraData = messageData.data;
      updateNodeSize(node); // Recalculate size based on new content
      node.setDirtyCanvas(true, true);
    });
  });
}

// ============================================================================
// MAIN EXTENSION REGISTRATION
// ============================================================================

app.registerExtension({
  name: "LoRAVisualizer",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "LoRAVisualizer") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      setupNodeState(this);
      createVisualizationWidget(this);
      setupNodeEventHandlers(this);
      setupResizeHandling(this);
      setupContextMenu(this);

      return result;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      hideHoverGallery(this);
      if (onRemoved) {
        return onRemoved.apply(this, arguments);
      }
    };

    setupWebSocketHandler();
  },
});
