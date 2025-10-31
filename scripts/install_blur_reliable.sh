#!/bin/bash
set -e

echo "🔧 Setting up Blur AI environment (avoiding GitHub downloads)..."

# Clean up
deactivate 2>/dev/null || true
rm -rf blur_env-darwin-arm64

# Create fresh environment
python3.11 -m venv blur_env-darwin-arm64
source blur_env-darwin-arm64/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "📦 Installing base dependencies..."
pip install \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    orjson==3.9.10 \
    PyYAML==6.0.1 \
    langdetect==1.0.9 \
    numpy==1.26.0 \
    PyMuPDF==1.23.7 \
    faiss-cpu \
    "diskcache>=5.6.1" \
    "jinja2>=2.11.3" \
    "MarkupSafe>=2.0" \
    --no-cache-dir

echo "🔧 Installing llama-cpp-python..."

# Try PyPI first (most reliable)
if pip install llama-cpp-python --no-cache-dir; then
    echo "✅ llama-cpp-python installed from PyPI"
else
    echo "⚠️  PyPI failed, building from source with Metal..."
    brew install cmake pkg-config 2>/dev/null || true
    export CMAKE_ARGS="-DGGML_METAL=ON"
    export FORCE_CMAKE=1
    pip install llama-cpp-python --no-cache-dir --force-reinstall --no-binary=llama-cpp-python
    echo "✅ llama-cpp-python built from source with Metal"
fi

echo "🎉 Blur AI environment ready!"