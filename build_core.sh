#!/bin/bash
set -e

# --- CONFIGURATION ---
# Please double-check this path is correct from the 'find' command you ran before.
LLAMA_LIB_PATH="/opt/blur_env/lib/python3.9/site-packages/llama_cpp/lib/libllama.dylib"

# (Optional but Recommended) Fix for the ffmpeg warning.
FFMPEG_PATH="/opt/homebrew/bin/ffmpeg"


# --- SCRIPT ---
echo "--- Cleaning up previous builds ---"
rm -rf ./dist_core
rm -rf ./build
rm -rf ./blur_core.spec

echo "--- Building blur_core with PyInstaller ---"
pyinstaller --noconfirm --onedir --windowed \
  --name blur_core \
  --add-binary="${LLAMA_LIB_PATH}:llama_cpp/lib" \
  --add-binary="${FFMPEG_PATH}:." \
  electron/backend/convo_chat_core.py

echo "--- Moving build output to dist_core ---"
mkdir -p ./dist_core
# The output is a directory named blur_core, we move the whole thing
mv ./dist/blur_core ./dist_core/

echo "--- Cleaning up ---"
rm -rf ./build
rm -rf ./blur_core.spec

echo "✅ Python core build complete."