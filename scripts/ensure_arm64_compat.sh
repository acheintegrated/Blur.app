#!/bin/bash
# ensure_arm64_compat.sh

echo "🔧 Ensuring ARM64 compatibility..."

# Check architecture
ARCH=$(uname -m)
echo "Building for: $ARCH"

# Ensure proper Python architecture
PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())")
echo "Python arch: $PYTHON_ARCH"

if [ "$ARCH" = "arm64" ] && [ "$PYTHON_ARCH" != "arm64" ]; then
    echo "❌ Architecture mismatch!"
    echo "Current Python is running as: $PYTHON_ARCH"
    echo "Install Python for ARM64:"
    echo "  brew install python@3.11"
    exit 1
fi

# Check for required ARM64 libraries
echo "📦 Checking ARM64 dependencies..."

# FAISS ARM64 check
python3 -c "
try:
    import faiss
    print('✅ FAISS: OK')
except ImportError as e:
    print('❌ FAISS: Missing - try: pip install faiss-cpu-noavx2')
    print(f'   Error: {e}')
"

# llama-cpp-python ARM64 check  
python3 -c "
try:
    from llama_cpp import Llama
    print('✅ llama-cpp-python: OK')
except ImportError as e:
    print('❌ llama-cpp-python: Missing ARM64 wheel')
    print('   Try: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu')
"