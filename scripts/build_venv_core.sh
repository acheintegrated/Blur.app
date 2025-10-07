#!/usr/bin/env bash
set -euo pipefail

# --- locate repo root (parent of scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REQ_CORE="$ROOT_DIR/electron/backend/requirements-core.txt"
VENV_DIR="$ROOT_DIR/blur_env-darwin-arm64"
WHEEL_DIR="$ROOT_DIR/vendor_wheels"

# --- pick Python 3.11 explicitly (Homebrew first, fallback to PATH) ---
if brew --prefix python@3.11 >/dev/null 2>&1; then
  PY="$(brew --prefix python@3.11)/bin/python3.11"
else
  PY="python3.11"
fi

# sanity
"$PY" -V

# reset
deactivate 2>/dev/null || true
rm -rf "$VENV_DIR" "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"

# venv
"$PY" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel

# Build wheelhouse online (native; NO --platform!)
if ! pip wheel -r "$REQ_CORE" -w "$WHEEL_DIR"; then
  echo "⚠️ wheel build had issues; will install from PyPI and snapshot wheels after."
fi

# Install: prefer wheelhouse if it has content, else PyPI
if [ -n "$(ls -A "$WHEEL_DIR" 2>/dev/null)" ]; then
  echo "Using local wheelhouse…"
  pip install --no-index -f "$WHEEL_DIR" -r "$REQ_CORE"
else
  echo "Installing from PyPI…"
  pip install -r "$REQ_CORE"
  # Snapshot what got installed for future offline builds
  pip download -r "$REQ_CORE" -d "$WHEEL_DIR" || true
fi

# trim + perms
find "$VENV_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$VENV_DIR" -type f -name "*.pyc" -delete
chmod -R u+rwX "$VENV_DIR"

# versions
python - <<'PY'
import sys, importlib
mods=["numpy","faiss","llama_cpp","fastapi","uvicorn","orjson","langdetect"]
print("python:", sys.version)
