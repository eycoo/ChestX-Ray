# ==============================================================
# ChestPrior — Google Colab GPU Backend + Ngrok Tunnel
# ==============================================================
# 1. Set your NGROK_AUTH_TOKEN (get it from https://dashboard.ngrok.com)
# 2. Upload your backend/ folder and research/ to Colab
# 3. Run this single cell — it prints the public URL at the end
# ==============================================================

import os
# Prefer env var to avoid leaking the token via git. Fall back to inline string.
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "PASTE_YOUR_NGROK_TOKEN_HERE")

# ── 1. Install dependencies ─────────────────────────────────
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "pyngrok", "nest-asyncio",
    "fastapi==0.111.0", "uvicorn==0.29.0", "python-multipart==0.0.9",
    "diffusers>=0.25.0", "transformers>=4.36.0",
    "accelerate>=0.25.0", "safetensors>=0.4.0",
    "torch", "numpy", "pillow",
])

# ── 2. Patch event loop for Jupyter (only if running inside IPython/Jupyter) ──
try:
    import IPython
    _in_jupyter = IPython.get_ipython() is not None
except Exception:
    _in_jupyter = False

if _in_jupyter:
    import nest_asyncio
    nest_asyncio.apply()

# ── 3. Add backend/ to Python path ──────────────────────────
import sys
BACKEND_DIR = "/content/backend"  # adjust if your layout differs
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# ── 4. Set env vars BEFORE importing app ─────────────────────
# Adjust these paths to match your Colab file layout
os.environ["PT_PATH"] = "/content/research/fa_best_med_balanced_1.pt"
os.environ["MLP_PATH"] = "/content/research/best_overall_weights.pt"
# os.environ["LORA_MODEL_ID"] = "Osama03/Medical-X-ray-image-generation-stable-diffusion"
# os.environ["LORA_WEIGHT"] = "pytorch_lora_weights.safetensors"

# ── 5. Open Ngrok tunnel ────────────────────────────────────
from pyngrok import ngrok, conf

conf.get_default().auth_token = NGROK_AUTH_TOKEN
public_url = ngrok.connect(8000, "http").public_url

print("=" * 60)
print(f"  NGROK PUBLIC URL:  {public_url}")
print("=" * 60)
print()
print("  Copy this URL into your .env.local:")
print(f'  NEXT_PUBLIC_API_URL={public_url}')
print()
print("  Then restart your Next.js dev server (npm run dev)")
print("=" * 60)

# ── 6. Start FastAPI via uvicorn ─────────────────────────────
import uvicorn
from app import app  # triggers @app.on_event("startup") → loads models

uvicorn.run(app, host="0.0.0.0", port=8000)
