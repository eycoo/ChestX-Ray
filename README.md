# ChestPrior — Generative Priors for Robust Chest X-ray Classification

> **KCV Final Project 2026 · ITS Surabaya**
>
> Evaluasi komparatif generative prior berbasis diffusion model dan modul Dual Feature Aggregation untuk klasifikasi chest X-ray.

---

## Project Structure

```text
ChestX-Ray/
├── app/                        # Next.js pages (frontend website)
│   ├── page.tsx                # Homepage / Overview
│   ├── methodology/            # Pipeline & experimental design
│   ├── model/                  # Feature extractor architectures
│   ├── results/                # Experimental results (TBD)
│   ├── inference/              # Interactive demo UI
│   ├── references/             # Citations
│   └── team/                   # Team members
├── components/                 # Shared React components
│   ├── Navbar.tsx
│   ├── Footer.tsx
│   └── SectionHeader.tsx
├── backend/                    # FastAPI inference server (Python)
│   ├── app.py                  # API entrypoint (uvicorn)
│   ├── inference.py            # Pipeline orchestration
│   ├── models.py               # FA encoder, MLP, hooks, utils
│   └── requirements.txt        # Python dependencies
├── research_file/              # Research pipeline scripts
│   ├── Medical X-ray Stable Diffusion_feature_map_extractor.py
│   ├── fp-medical-banun.py     # FA training pipeline
│   ├── fp-mlp-classifier.py    # MLP classifier (4 scenarios)
│   └── *.pt                    # Trained checkpoints (gitignored)
└── package.json                # Next.js config
```

## Quick Start

### Frontend (Documentation Website)

```bash
npm install
npm run dev
# opens at http://localhost:3000
```

### Backend (Inference API)

> Requires a CUDA GPU and the trained `.pt` checkpoint in `research_file/`.

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API loads the full SD U-Net + FA model + MLP head into VRAM **once at startup**.
Frontend calls `POST /predict` from the `/inference` page.

### Colab GPU Runtime

The backend supports remote GPU execution via Google Colab.
Place the checkpoint `.pt` file in your Colab environment and launch:

```python
!pip install fastapi uvicorn python-multipart diffusers transformers accelerate
# upload backend/ files, then:
!uvicorn app:app --host 0.0.0.0 --port 8000
```

Use an ngrok tunnel to expose the Colab API to the frontend.

---

## Pipeline Overview

```
X-Ray (512×512) → VAE Encoder (frozen)
                → Gaussian Noise (t=10)
                → U-Net Forward (frozen SD v1.4 + LoRA)
                    ├── Feature Maps [B, C, H, W] × 4 scales
                    └── Attention Maps [B, 1, H, W] saliency
                → DFATB (Spatial + Channel Attention)
                → FAFN  (Split-gate MLP)
                → Differential Denoising (λ·A₁ − A₂)
                → GAP + Bottleneck → z₁₂₈ [B, 128]
                → MLP Head → 6 class logits
```

## Four Experimental Scenarios

| # | Distribution | Augmentation | Description |
|---|-------------|-------------|-------------|
| 1 | Balanced (~416/class) | None | Baseline — pure feature quality |
| 2 | Balanced (~416/class) | +FSA | Gaussian noise + feature dropout + Mixup |
| 3 | Imbalanced (10:1) | None | Real-world class skew simulation |
| 4 | Imbalanced (10:1) | +FSA | Worst-case — imbalance + augmentation |

## Feature Extractors

| Abbr | Name | Type |
|------|------|------|
| **MedSD** | Medical X-ray Stable Diffusion + FE+FA | Generative Prior (Proposed) |
| CNX | ConvNeXtV2 | CNN |
| DINO | DINOv2 | Vision Transformer |
| MXVT | MaxViT | Hybrid |

## Deployment

### Vercel (Frontend)

```bash
npm install -g vercel
vercel
```

Or connect your GitHub repo at [vercel.com](https://vercel.com).

---

ITS Surabaya · KCV · 2026
