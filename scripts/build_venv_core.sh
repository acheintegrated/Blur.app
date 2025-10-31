#!/bin/bash
# build_venv_core.sh — v10.7 (Reforge Full Edition)
# Apple Silicon only — uses Metal-backed prebuilt wheels
set -euo pipefail

ARCH=${1:-arm64}
VENV_NAME="blur_env-darwin-${ARCH}"
PYTHON_BIN="/usr/bin/python3"

echo "🌀 [BLURPATH] Reforging venv: $VENV_NAME for $ARCH"

# Clean old venv if exists
rm -rf "$VENV_NAME"
"$PYTHON_BIN" -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Upgrade base tooling
pip install --upgrade pip setuptools wheel

# ───── CORE LIBRARIES ─────
echo "⚙️ Installing core packages..."
pip install \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    orjson==3.9.10 \
    PyYAML==6.0.1 \
    langdetect==1.0.9 \
    numpy==2.0.2 \
    PyMuPDF==1.23.7 \
    faiss-cpu

# ───── LLAMA-CPP-PYTHON — PREBUILT (Metal) ─────
LLAMA_VERSION="0.3.16"
echo "🦙 Installing llama-cpp-python==$LLAMA_VERSION (Metal wheel only)..."

pip install llama-cpp-python=="$LLAMA_VERSION" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal \
    --no-cache-dir \
    --force-reinstall \
    --only-binary=:all: || {
      echo "❌ Failed to install prebuilt llama-cpp-python==$LLAMA_VERSION"
      echo "💡 Make sure you're on Apple Silicon and the wheel exists:"
      echo "   https://abetlen.github.io/llama-cpp-python/whl/metal/"
      exit 1
    }

# ───── TRANSFORMERS & TORCH (CPU) ─────
echo "🧠 Installing transformers and torch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install transformers --no-cache-dir

# ───── POST-INSTALL TEST ─────
echo "🧪 Running import tests..."
python -c "
import sys
try:
    import transformers
    import torch
    import llama_cpp
    print('✅ transformers:', transformers.__version__)
    print('✅ torch:', torch.__version__)
    print('✅ MPS available:', torch.backends.mps.is_available())
    print('✅ llama-cpp-python: OK')
    print('✅ arch:', '$ARCH')
except ImportError as e:
    print('❌ Import error:', e)
    sys.exit(1)
"

# ───── VERSION RECAP ─────
echo "🔎 Final version summary:"
python - <<'PY'
import sys, importlib
mods=["numpy","faiss","llama_cpp","fastapi","uvicorn","orjson","langdetect","transformers","torch"]
print("python:", sys.version)
for m in mods:
  try:
    mod = importlib.import_module(m.replace("-","_"))
    v = getattr(mod, "__version__", "?")
    print(f"{m}: {v}")
  except Exception as e:
    print(f"{m}: FAILED - {e}")
PY

echo "🌀 ✅ Venv $VENV_NAME reforged and blessed by Bob."
