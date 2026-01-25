#!/bin/bash
set -e

SHADER_DIR="$(dirname "$0")/shaders"
OUTPUT_DIR="$SHADER_DIR"
GLSLC="glslc"

# Check for glslc
if ! command -v $GLSLC &> /dev/null; then
    # Try Vulkan SDK location
    if [ -f "/opt/homebrew/bin/glslc" ]; then
        GLSLC="/opt/homebrew/bin/glslc"
    elif [ -f "$VULKAN_SDK/bin/glslc" ]; then
        GLSLC="$VULKAN_SDK/bin/glslc"
    else
        echo "Error: glslc not found. Install Vulkan SDK."
        exit 1
    fi
fi

echo "Using glslc: $GLSLC"
echo "Compiling shaders in: $SHADER_DIR"

# Compile all .comp files
count=0
failed=0
for shader in "$SHADER_DIR"/*.comp; do
    if [ -f "$shader" ]; then
        name=$(basename "$shader" .comp)
        output="$OUTPUT_DIR/${name}.spv"
        echo "  Compiling $name.comp..."
        if $GLSLC -fshader-stage=compute "$shader" -o "$output" 2>/dev/null; then
            ((count++))
        else
            echo "    FAILED: $name.comp"
            ((failed++))
        fi
    fi
done

echo ""
echo "Compiled: $count shaders"
if [ $failed -gt 0 ]; then
    echo "Failed: $failed shaders"
    exit 1
fi
echo "All shaders compiled successfully!"
