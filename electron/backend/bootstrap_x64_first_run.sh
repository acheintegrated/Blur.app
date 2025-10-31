#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/blur_env-darwin-x64"
WHEEL_DIR="$ROOT_DIR/vendor_wheels-x64"
REQ_FILE="$ROOT_DIR/electron/backend/requirements-core-x64.txt"

# Must be Intel at runtime
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap must run on an Intel (x86_64) Mac."
  exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install strictly from bundled wheelhouse
python -m pip install --no-index --find-links "$WHEEL_DIR" -r "$REQ_FILE"

touch "$WHEEL_DIR/.installed_on_this_machine"
echo "✅ core deps installed for x64"
