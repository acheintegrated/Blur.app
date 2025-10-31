#!/bin/bash
# build_faiss_complete.sh - Complete FAISS rebuild with proper NumPy

set -e

VENV_PATH="${1:-$(pwd)/blur_env-darwin-arm64}"
echo "Building FAISS for venv: $VENV_PATH"

# Activate venv
source "$VENV_PATH/bin/activate"

# Ensure consistent NumPy version
echo "Installing consistent NumPy version..."
pip uninstall numpy -y
pip install numpy==1.26.0

# Install build dependencies
brew install libomp swig cmake pkg-config

export OMP_PREFIX=$(brew --prefix libomp)
export CMAKE_PREFIX_PATH="$OMP_PREFIX:$CMAKE_PREFIX_PATH"

BUILD_DIR="/tmp/faiss-rebuild-$$"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Configure with explicit NumPy paths
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
echo "Using NumPy include path: $NUMPY_INCLUDE"

cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP_PREFIX/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="$OMP_PREFIX/lib/libomp.dylib" \
    -DPython_EXECUTABLE="$(which python)" \
    -DNumPy_INCLUDE_DIR="$NUMPY_INCLUDE" \
    .

# Build and install
cmake --build build -- -j$(sysctl -n hw.ncpu)
cd build/faiss/python
pip install .

# Cleanup
rm -rf "$BUILD_DIR"

echo "✅ FAISS rebuilt with consistent NumPy"