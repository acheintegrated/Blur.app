#!/bin/bash
# build_venv_core.sh - ARM64 with proper FAISS build

set -e

ARCH=${1:-arm64}
VENV_NAME="blur_env-darwin-${ARCH}"
PYTHON_BIN="/usr/bin/python3"

echo "Building venv: $VENV_NAME for $ARCH"

# Clean existing
rm -rf "$VENV_NAME"

# Create venv
"$PYTHON_BIN" -m venv "$VENV_NAME"

# Activate
source "$VENV_NAME/bin/activate"

# Upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install base requirements
echo "Installing base packages..."
pip install \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    orjson==3.9.10 \
    PyYAML==6.0.1 \
    langdetect==1.0.9 \
    numpy==1.26.0 \
    PyMuPDF==1.23.7

# Install FAISS for ARM64
if [ "$ARCH" = "arm64" ]; then
    echo "Installing FAISS for ARM64..."
    
    # Set OpenMP paths
    export OMP_PREFIX=$(brew --prefix libomp 2>/dev/null || echo "")
    if [ -n "$OMP_PREFIX" ]; then
        export CMAKE_PREFIX_PATH="$OMP_PREFIX:$CMAKE_PREFIX_PATH"
    fi
    
    # Try with OpenMP first
    bash scripts/build_faiss_arm64_fixed.sh || \
    
    # Fallback: without OpenMP
    (echo "OpenMP build failed, trying without OpenMP..." && \
     bash scripts/build_faiss_arm64_no_openmp.sh) || \
    
    # Final fallback: specific version
    (echo "Manual builds failed, trying pip version..." && \
     pip install faiss-cpu==1.7.2 --no-binary=faiss-cpu --no-cache-dir)
fi

# Install llama-cpp-python
echo "Installing llama-cpp-python..."
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo "✅ Venv built: $VENV_NAME"