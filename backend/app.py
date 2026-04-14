import io
import os
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from PIL import Image

from inference import MedicalXRayPipeline

app = FastAPI(
    title="Medical Chest X-Ray Inference API",
    description="Hybrid SD Feature Extractor + FA Module + MLP",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ────────────────────────────────────────────
pipeline: Optional[MedicalXRayPipeline] = None
_warmup_done: bool = False

# ── Config (env-overridable) ────────────────────────────────
SD_MODEL_ID = os.getenv("SD_MODEL_ID", "Osama03/Medical-X-ray-image-generation-stable-diffusion")
LORA_WEIGHT = os.getenv("LORA_WEIGHT", "pytorch_lora_weights.safetensors")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_PATH  = os.getenv("PT_PATH", os.path.join(BASE_DIR, "research_file", "fa_best_med_balanced.pt"))


# ── Lifecycle ───────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global pipeline

    pt_path = PT_PATH
    if not os.path.exists(pt_path):
        alts = [
            os.path.join(BASE_DIR, "research_file", "fa_best_med_balanced_1.pt"),
            os.path.join(BASE_DIR, "research_file", "fa_best_med_imbalanced.pt"),
        ]
        for alt in alts:
            if os.path.exists(alt):
                pt_path = alt
                break

    if not os.path.exists(pt_path):
        print(f"WARNING: Checkpoint {pt_path} not found. Ensure it exists before running predictions.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[startup] Using device: {device}")

    pipeline = MedicalXRayPipeline(
        sd_model_id=SD_MODEL_ID,
        lora_weight=LORA_WEIGHT,
        pt_path=pt_path,
        device=device,
    )
    pipeline.load_models()


@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    if pipeline:
        pipeline.cleanup()
        del pipeline


# ── Pydantic schemas ────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


class WarmupResponse(BaseModel):
    status: str
    device: str
    vram_allocated_mb: float
    warmup_latency_s: float


class HealthResponse(BaseModel):
    status: str
    device: str
    pipeline_loaded: bool
    warmup_done: bool


# ── Endpoints ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Basic health-check — call this to verify the server is alive."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok",
        device=device,
        pipeline_loaded=pipeline is not None,
        warmup_done=_warmup_done,
    )


@app.get("/warmup", response_model=WarmupResponse)
async def warmup():
    """
    Force a full dummy forward pass through VAE → U-Net → FA → MLP
    so that every CUDA kernel is compiled, all weights are in VRAM,
    and the first real prediction has zero cold-start delay.
    Call this 1–2 minutes before the live demo starts.
    """
    global _warmup_done
    import torch

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    if _warmup_done:
        vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        return WarmupResponse(
            status="already_warm",
            device=pipeline.device,
            vram_allocated_mb=round(vram, 1),
            warmup_latency_s=0.0,
        )

    t0 = time.time()

    # Create a dummy 512x512 black image
    dummy_img = Image.new("RGB", (512, 512), color=(0, 0, 0))

    # Run full pipeline (VAE encode → noise → U-Net forward → FA → MLP)
    try:
        _ = pipeline.predict(dummy_img, prompt="A chest X-ray", timestep=10)
    except Exception as e:
        # Even if prediction fails (e.g. bad checkpoint), the CUDA kernels
        # and weights are now loaded — that's the goal.
        print(f"[warmup] Dummy pass raised (non-fatal): {e}")

    elapsed = time.time() - t0
    _warmup_done = True

    vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"[warmup] Done in {elapsed:.1f}s — VRAM: {vram:.0f} MB")

    return WarmupResponse(
        status="warm",
        device=pipeline.device,
        vram_allocated_mb=round(vram, 1),
        warmup_latency_s=round(elapsed, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), prompt: str = Form("A chest X-ray")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format. {e}")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized.")

    try:
        result = pipeline.predict(image, prompt=prompt)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(files: List[UploadFile] = File(...), prompt: str = Form("A chest X-ray")):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized.")

    results = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append({"error": f"File {file.filename} is not an image."})
            continue
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            res = pipeline.predict(image, prompt=prompt)
            results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"error": f"Failed processing {file.filename}: {e}"})

    return JSONResponse(status_code=200, content=results)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
