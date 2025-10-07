import sys
from llama_cpp import Llama

# --- CONFIGURATION ---
# Set the full path to your model file here
MODEL_PATH = "/Users/blur/blur/resources/models/Qwen3-4B-Instruct-2507-Q8_0.gguf"

# Set n_gpu_layers to 0 to test CPU-only loading
# Set to -1 or a high number (e.g., 35) to test GPU loading
N_GPU_LAYERS = -1
# ---------------------

if not MODEL_PATH:
    print("ERROR: Please set the MODEL_PATH variable in the script.")
    sys.exit(1)

print(f"--- Attempting to load model ---")
print(f"Path: {MODEL_PATH}")
print(f"GPU Layers: {N_GPU_LAYERS}")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=True
    )
    print("\n✅✅✅ SUCCESS: Model loaded successfully! ✅✅✅")

except Exception as e:
    print(f"\n❌❌❌ FAILURE: Model failed to load. ❌❌❌")
    print(f"Error: {e}")
    sys.exit(1)