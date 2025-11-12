#!/bin/bash
#
# Icon creation script for Data Points AI RSS Reader
#
# This script creates macOS icons from a source PNG image.
# Requires ImageMagick or sips (built into macOS)
#

set -e

SOURCE_IMAGE="${1:-icon-source.png}"
ICONSET_DIR="assets/icon.iconset"

echo "🎨 Creating macOS icons from ${SOURCE_IMAGE}..."

# Check if source image exists
if [ ! -f "$SOURCE_IMAGE" ]; then
    echo "Error: Source image '${SOURCE_IMAGE}' not found"
    echo ""
    echo "Usage: $0 [source-image.png]"
    echo ""
    echo "Please provide a 1024x1024px PNG image"
    echo "Creating placeholder icon instead..."

    # Create placeholder using built-in macOS tools
    mkdir -p assets

    # Create a simple colored square as placeholder
    # This requires ImageMagick, but we'll provide a fallback
    if command -v convert &> /dev/null; then
        convert -size 1024x1024 xc:'#3b82f6' \
            -gravity center \
            -pointsize 200 \
            -fill white \
            -font "Helvetica-Bold" \
            -annotate +0+0 "DP" \
            "assets/placeholder.png"
        SOURCE_IMAGE="assets/placeholder.png"
    else
        echo "Note: ImageMagick not found. Skipping icon generation."
        echo "Install ImageMagick with: brew install imagemagick"
        echo ""
        echo "Or provide a custom icon at: electron/assets/icon.icns"
        exit 0
    fi
fi

# Create iconset directory
mkdir -p "$ICONSET_DIR"

# Function to resize image
resize_image() {
    local size=$1
    local name=$2

    if command -v sips &> /dev/null; then
        # Use sips (built into macOS)
        sips -z $size $size "$SOURCE_IMAGE" --out "${ICONSET_DIR}/${name}" > /dev/null 2>&1
    elif command -v convert &> /dev/null; then
        # Use ImageMagick
        convert "$SOURCE_IMAGE" -resize ${size}x${size} "${ICONSET_DIR}/${name}"
    else
        echo "Error: Neither sips nor ImageMagick found"
        exit 1
    fi
}

# Generate all required icon sizes
echo "Generating icon sizes..."
resize_image 16 "icon_16x16.png"
resize_image 32 "icon_16x16@2x.png"
resize_image 32 "icon_32x32.png"
resize_image 64 "icon_32x32@2x.png"
resize_image 128 "icon_128x128.png"
resize_image 256 "icon_128x128@2x.png"
resize_image 256 "icon_256x256.png"
resize_image 512 "icon_256x256@2x.png"
resize_image 512 "icon_512x512.png"
resize_image 1024 "icon_512x512@2x.png"

# Convert iconset to icns
echo "Creating .icns file..."
iconutil -c icns "$ICONSET_DIR" -o "assets/icon.icns"

# Clean up
rm -rf "$ICONSET_DIR"

echo "✓ Icon created successfully at assets/icon.icns"

# Also create a PNG icon for other uses
if [ -f "assets/icon.icns" ]; then
    echo "✓ Icon ready for Electron build"
fi
