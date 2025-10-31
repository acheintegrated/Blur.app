#!/bin/bash
# build_faiss_arm64_fixed.sh - FAISS with OpenMP support

set -e

echo "🔧 Building FAISS with OpenMP support for ARM64..."

# Install dependencies
brew install libomp swig cmake pkg-config

# Set OpenMP paths for ARM64
export OMP_PREFIX=$(brew --prefix libomp)
export OPENMP_ROOT="$OMP_PREFIX"
export CMAKE_PREFIX_PATH="$OMP_PREFIX:$CMAKE_PREFIX_PATH"

echo "OpenMP installed at: $OMP_PREFIX"

# Create build directory
BUILD_DIR="/tmp/faiss-build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone FAISS
echo "Cloning FAISS..."
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Configure with OpenMP support
echo "Configuring FAISS with OpenMP..."
cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP_PREFIX/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="$OMP_PREFIX/lib/libomp.dylib" \
    -DPython_EXECUTABLE=$(which python) \
    .

# Build
echo "Building FAISS..."
cmake --build build -- -j$(sysctl -n hw.ncpu)

# Install
echo "Installing FAISS..."
cd build/faiss/python
pip install .

echo "✅ FAISS built with OpenMP support"