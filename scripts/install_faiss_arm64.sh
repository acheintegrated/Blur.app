#!/bin/bash
# install_faiss_arm64.sh - Install pre-built FAISS for ARM64

set -e

echo "🔧 Installing FAISS for ARM64..."

# Download pre-built FAISS wheel for ARM64
# Using the conda-forge channel which has good ARM64 support
pip download faiss-cpu --platform macosx_11_0_arm64 --no-deps -d /tmp/

if [ $? -eq 0 ]; then
    # If download succeeds, install it
    WHEEL=$(ls /tmp/faiss_cpu-*-cp39-cp39-macosx_11_0_arm64.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "Installing pre-built wheel: $WHEEL"
        pip install "$WHEEL"
        echo "✅ FAISS installed via pre-built wheel"
        exit 0
    fi
fi

# Fallback: Try official pip with specific version
echo "Trying official FAISS with specific version..."
pip install faiss-cpu==1.7.4 --no-binary=faiss-cpu --no-cache-dir || \

# Last resort: Use older compatible version
echo "Trying older FAISS version..."
pip install faiss-cpu==1.7.2 --no-cache-dir