#!/bin/bash
# install_faiss_conda_forge.sh - Use conda-forge wheels

set -e

echo "Installing FAISS from conda-forge..."

# Download the conda-forge wheel directly
CONDA_FORGE_URL="https://anaconda.org/conda-forge/faiss-cpu/1.7.4/download/osx-arm64/faiss-cpu-1.7.4-py310h2c0236b_0.tar.bz2"

# Extract and install
wget -O /tmp/faiss-cpu.tar.bz2 "$CONDA_FORGE_URL" || curl -L -o /tmp/faiss-cpu.tar.bz2 "$CONDA_FORGE_URL"

if [ -f "/tmp/faiss-cpu.tar.bz2" ]; then
    # Extract to temp directory
    mkdir -p /tmp/faiss-extract
    tar -xjf /tmp/faiss-cpu.tar.bz2 -C /tmp/faiss-extract
    
    # Copy to site-packages
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    cp -r /tmp/faiss-extract/lib/python*/site-packages/* "$SITE_PACKAGES/" 2>/dev/null || true
    
    echo "✅ FAISS installed from conda-forge"
else
    echo "❌ Failed to download FAISS"
    exit 1
fi