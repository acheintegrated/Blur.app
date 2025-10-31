#!/bin/bash
set -e

echo "🔧 Setting up reliable Blur AI environment..."

# Clean up
deactivate 2>/dev/null || true
rm -rf ~/blur_env

# Create fresh environment
echo "🐍 Creating Python environment..."
/usr/bin/python3 -m venv ~/blur_env
source ~/blur_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install \
    fastapi \
    uvicorn \
    pydantic \
    orjson \
    PyYAML \
    langdetect \
    numpy \
    PyMuPDF \
    faiss-cpu \
    --no-cache-dir

# Install LLM backend
echo "🔧 Installing LLM backend..."
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

echo "🎉 Setup complete! Testing environment..."

# Test everything
python -c "
import fastapi
import uvicorn  
import pydantic
import orjson
import yaml
from langdetect import detect
import numpy as np
import faiss
import fitz
import transformers
import torch

print('✅ All core dependencies working')
print('✅ Transformers LLM backend ready')
print(f'✅ PyTorch device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')

print('🚀 BLUR AI ENVIRONMENT READY!')
"