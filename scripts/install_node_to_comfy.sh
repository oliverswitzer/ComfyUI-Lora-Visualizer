#!/usr/bin/env bash

# Set paths
COMFYUI_DIR="../ComfyUI"
NODE_NAME="comfyui-lora-visualizer"
TARGET="$COMFYUI_DIR/custom_nodes/$NODE_NAME"
SOURCE="$(pwd)"

# Remove existing directory or symlink
if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
    echo "Removing existing $TARGET"
    rm -rf "$TARGET"
fi

# Create symlink
echo "Creating symlink: $TARGET -> $SOURCE"
ln -s "$SOURCE" "$TARGET"

# Verify
ls -l "$COMFYUI_DIR/custom_nodes/" | grep "$NODE_NAME"