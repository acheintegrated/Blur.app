#!/usr/bin/env bash
set -euo pipefail

# --- Target architecture passed as the first argument ---
TARGET_ARCH=${1:-arm64}

# --- locate repo root (parent of scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_CORE="$ROOT_DIR/electron/backend/requirements-core.txt"

# --- Dynamically name VENV and WHEEL dirs based on architecture ---
VENV_DIR="$ROOT_DIR/blur_env-darwin-${TARGET_ARCH}"
WHEEL_DIR="$ROOT_DIR/vendor_wheels-${TARGET_ARCH}"

echo "Building venv for ${TARGET_ARCH} in ${VENV_DIR}"

# --- Determine if we're cross-compiling ---
CURRENT_ARCH=$(uname -m)
IS_CROSS_COMPILE=false
if [[ "$CURRENT_ARCH" == "arm64" && "$TARGET_ARCH" == "x64" ]]; then
  IS_CROSS_COMPILE=true
  echo "⚠️  Cross-compiling from ARM64 to x64"
elif [[ "$CURRENT_ARCH" == "x86_64" && "$TARGET_ARCH" == "arm64" ]]; then
  IS_CROSS_COMPILE=true
  echo "⚠️  Cross-compiling from x64 to ARM64"
fi

# --- pick Python 3.11 explicitly ---
if [[ "$TARGET_ARCH" == "x64" ]] && [[ "$IS_CROSS_COMPILE" == true ]]; then
  # For cross-compilation to x64, we need x64 Python
  # Try to find it via arch command or specific path
  if command -v arch &> /dev/null; then
    # Use arch to run x64 Python under Rosetta 2
    PY="arch -x86_64 python3.11"
    echo "Using Rosetta 2 for x64 Python"
  elif [[ -f "/usr/local/bin/python3.11" ]]; then
    # Homebrew x64 install location
    PY="/usr/local/bin/python3.11"
  else
    echo "❌ Cannot find x64 Python 3.11 for cross-compilation"
    echo "Install with: arch -x86_64 /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "Then: arch -x86_64 brew install python@3.11"
    exit 1
  fi
else
  # Native build or ARM target on ARM host
  if brew --prefix python@3.11 >/dev/null 2>&1; then
    PY="$(brew --prefix python@3.11)/bin/python3.11"
  else
    PY="python3.11"
  fi
fi

# sanity
$PY -V
python_arch=$($PY -c "import platform; print(platform.machine())")
echo "Python architecture: $python_arch"

# Verify Python matches target
if [[ "$TARGET_ARCH" == "x64" && "$python_arch" != "x86_64" ]]; then
  echo "❌ Python architecture ($python_arch) doesn't match target (x64)"
  exit 1
elif [[ "$TARGET_ARCH" == "arm64" && "$python_arch" != "arm64" ]]; then
  echo "❌ Python architecture ($python_arch) doesn't match target (arm64)"
  exit 1
fi

# reset
deactivate 2>/dev/null || true
rm -rf "$VENV_DIR" "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"

# venv
$PY -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel

# --- For cross-compilation, download pre-built wheels instead of building ---
if [[ "$IS_CROSS_COMPILE" == true ]]; then
  echo "📦 Downloading pre-built wheels for ${TARGET_ARCH}..."
  
  # Map our arch to pip's platform tags
  if [[ "$TARGET_ARCH" == "x64" ]]; then
    PLATFORM="macosx_10_9_x86_64"
  else
    PLATFORM="macosx_11_0_arm64"
  fi
  
  # Download wheels with explicit platform tag
  pip download \
    --platform "$PLATFORM" \
    --python-version 311 \
    --only-binary=:all: \
    --dest "$WHEEL_DIR" \
    -r "$REQ_CORE" || {
      echo "⚠️  Some packages may not have pre-built wheels"
      # Try without --only-binary for pure Python packages
      pip download \
        --platform "$PLATFORM" \
        --python-version 311 \
        --dest "$WHEEL_DIR" \
        -r "$REQ_CORE"
    }
  
  # Install from downloaded wheels
  pip install --no-index --find-links "$WHEEL_DIR" -r "$REQ_CORE"
else
  # Native build - can compile from source
  # Map our arch names to Apple's arch names
  if [[ "$TARGET_ARCH" == "x64" ]]; then
    APPLE_ARCH="x86_64"
  else
    APPLE_ARCH="arm64"
  fi
  export ARCHFLAGS="-arch ${APPLE_ARCH}"
  export CMAKE_OSX_ARCHITECTURES="${APPLE_ARCH}"
  export _PYTHON_HOST_PLATFORM="macosx-10.9-${APPLE_ARCH}"
  
  # Build wheelhouse online
  if ! pip wheel -r "$REQ_CORE" -w "$WHEEL_DIR"; then
    echo "⚠️  wheel build had issues; will install from PyPI"
  fi
  
  # Special handling for llama-cpp-python - install separately with build flags
  echo "📦 Installing llama-cpp-python with Metal support..."
  CMAKE_ARGS="-DGGML_METAL=on -DGGML_METAL_EMBED_LIBRARY=on" \
    pip install --upgrade --force-reinstall --no-cache-dir \
    "llama-cpp-python>=0.2.87" || {
      echo "⚠️  llama-cpp-python build failed, trying without Metal..."
      pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python>=0.2.87"
    }
  
  # Install: prefer wheelhouse if it has content, else PyPI
  if [ -n "$(ls -A "$WHEEL_DIR" 2>/dev/null)" ]; then
    echo "Using local wheelhouse…"
    # Create temp requirements file without llama-cpp-python
    TEMP_REQ=$(mktemp)
    grep -v "llama-cpp-python" "$REQ_CORE" | grep -v "^#" | grep -v "^$" | sed '/^$/d' > "$TEMP_REQ"
    pip install --no-index -f "$WHEEL_DIR" -r "$TEMP_REQ"
    rm "$TEMP_REQ"
  else
    echo "Installing from PyPI…"
    TEMP_REQ=$(mktemp)
    grep -v "llama-cpp-python" "$REQ_CORE" | grep -v "^#" | grep -v "^$" | sed '/^$/d' > "$TEMP_REQ"
    pip install -r "$TEMP_REQ"
    rm "$TEMP_REQ"
    pip download -r "$REQ_CORE" -d "$WHEEL_DIR" || true
  fi
  
  unset ARCHFLAGS
fi

# trim + perms
find "$VENV_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$VENV_DIR" -type f -name "*.pyc" -delete
chmod -R u+rwX "$VENV_DIR"

# Verify architecture of compiled extensions
echo "🔍 Verifying binary architecture..."
for so_file in $(find "$VENV_DIR" -name "*.so" | head -5); do
  echo "Checking: $so_file"
  if command -v lipo &> /dev/null; then
    lipo -info "$so_file" 2>/dev/null || file "$so_file"
  else
    file "$so_file"
  fi
done

# versions
python - <<'PY'
import sys, importlib
mods=["numpy","faiss","llama_cpp","fastapi","uvicorn","orjson","langdetect"]
print("python:", sys.version)
for m in mods:
  try:
    mod = importlib.import_module(m.replace("-","_"))
    v = getattr(mod, "__version__", "?")
    print(f"{m}: {v}")
  except Exception as e:
    print(f"{m}: FAILED - {e}")
PY

echo "✅ Venv built successfully for ${TARGET_ARCH}"