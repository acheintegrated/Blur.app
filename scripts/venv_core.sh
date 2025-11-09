#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ! -x "blur_env-darwin-arm64/bin/python3" ]]; then
  bash scripts/build_venv_core.sh arm64
fi

./blur_env-darwin-arm64/bin/python3 - <<'PY'
import importlib, sys
try:
    importlib.import_module("llama_cpp")
except Exception as e:
    print("llama_cpp missing/broken:", e)
    sys.exit(1)
PY
