import sys, subprocess, shutil

# Check torch CUDA
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, VRAM: {props.total_memory / 1024**3:.1f} GB")
except ImportError:
    print("torch: not installed")

# Check ultralytics
try:
    import ultralytics
    print(f"ultralytics: {ultralytics.__version__}")
except ImportError:
    print("ultralytics: not installed")

# Check transformers
try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except ImportError:
    print("transformers: not installed")

# Check google-genai
try:
    import google.genai
    print(f"google-genai: available")
except ImportError:
    print("google-genai: not installed")

# Check ollama
ollama_path = shutil.which("ollama")
print(f"ollama: {'found at ' + ollama_path if ollama_path else 'not found'}")

# Check API key from env
import os
gk = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
print(f"Google API key in env: {'yes' if gk else 'no'}")
