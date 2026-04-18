
# Optimalisasi Klasifikasi X-Ray Menggunakan Medical Stable Diffusion dan Dual Feature Aggregation

> **KCV Lab Selection Project — Institut Teknologi Sepuluh Nopember, 2026**

---

## Overview

Proyek ini mengklasifikasikan citra chest X-ray ke dalam **6 kelas** menggunakan pipeline hybrid yang memanfaatkan **frozen Stable Diffusion U-Net** sebagai feature extractor, diikuti oleh modul **Dual Feature Aggregation (DFATB + FAFN + Differential Denoising)** dan **MLP classification head**.

### Kelas Target
| # | Kelas | Deskripsi |
|---|-------|-----------|
| 0 | Atelectasis | Kolaps paru parsial |
| 1 | Effusion | Cairan di rongga pleura |
| 2 | Infiltration | Substansi abnormal di jaringan paru |
| 3 | No Finding | Normal / tidak ada temuan klinis |
| 4 | Nodule | Massa kecil di paru |
| 5 | Pneumothorax | Udara di rongga pleura |

---

##  End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Chest X-Ray (512×512)                │
│                        + Text Prompt ("A chest X-ray")             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 1: LATENT ENCODING (Frozen VAE)                            │
│  ─────────────────────────────────────                            │
│  • Image → VAE Encoder → latent z ∈ ℝ^(4×64×64)                 │
│  • Gaussian noise injection at timestep t=10                      │
│  • noisy_latent = scheduler.add_noise(z, noise, t)               │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 2: U-NET FEATURE EXTRACTION (Frozen SD v1.4 + LoRA)       │
│  ─────────────────────────────────────────────────────────        │
│  • Single forward pass through frozen U-Net                       │
│  • BlockFeatureCollector: hooks on up_blocks → 4D feature maps   │
│    [B, C, H, W] for each resolution level                        │
│  • AttentionCollector: custom cross-attention processor captures  │
│    spatial attention maps from text-image alignment               │
└──────────┬────────────────────────────────┬───────────────────────┘
           │ Feature Maps                   │ Attention Maps
           ▼                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 3: DUAL FEATURE AGGREGATION (FA Module — Trainable)        │
│  ─────────────────────────────────────────────────────────        │
│                                                                   │
│  ┌─────────────┐   ┌──────────┐   ┌──────────────────────┐       │
│  │   DFATB      │ → │  FAFN    │ → │ Differential         │       │
│  │ (Dual-Focus  │   │ (Feed-   │   │ Denoising            │       │
│  │  Attention   │   │  forward │   │ (attention-guided     │       │
│  │  Block)      │   │  Gating) │   │  noise suppression)  │       │
│  └─────────────┘   └──────────┘   └──────────┬───────────┘       │
│  • SpatialAttn                                │                   │
│  • ChannelAttn                    ┌───────────▼───────────┐       │
│  • BatchNorm                      │  GAP + Concat         │       │
│                                   │  → BottleneckProj     │       │
│                                   │  → z ∈ ℝ^128          │       │
│                                   └───────────────────────┘       │
└───────────────────────────────┬───────────────────────────────────┘
                                │ z (128-dim)
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 4: MLP CLASSIFICATION HEAD                                 │
│  ────────────────────────────────                                 │
│  LayerNorm(128) → Linear(128, 512) → GELU → Dropout(0.3)        │
│  → Linear(512, 6) → softmax → prediction                        │
└───────────────────────────────────────────────────────────────────┘
```

---

##  Experimental Scenarios

Pipeline dievaluasi dalam **4 skenario** yang merupakan kombinasi dari dua dimensi:

| Scenario | Dataset | Augmentation | Deskripsi |
|----------|---------|-------------|-----------|
| **1** | Balanced (~416/kelas) | ❌ None | Baseline ideal — distribusi seimbang, tanpa augmentasi |
| **2** | Balanced (~416/kelas) | ✅ FSA | Uji apakah FSA meningkatkan generalisasi pada data seimbang |
| **3** | Imbalanced (10:1 skew) | ❌ None | Simulasi kondisi medis nyata — class weights menangani skew |
| **4** | Imbalanced (10:1 skew) | ✅ FSA | Worst-case scenario — imbalance + augmentasi |

Setiap skenario dijalankan untuk **4 feature extractor** yang dibandingkan:

| Feature Extractor | Tipe | Deskripsi |
|-------------------|------|-----------|
| **MedSD (FE+FA)** | Generative Prior | Medical SD v1.4 + LoRA + Dual Feature Aggregation *(proposed)* |
| **DINOv2** | Vision Transformer | Self-supervised ViT dari Meta AI |
| **ConvNeXtV2** | CNN | Modern ConvNet dari Facebook/Meta |
| **MaxViT** | Hybrid CNN+ViT | Multi-axis attention dari Google |

### Feature Space Augmentation (FSA) — 3-Stage Pipeline

Diterapkan secara **berurutan** di feature space saat training (Scenario 2 & 4):

1. **Feature Space SMOTE** — Oversample kelas minoritas via interpolasi k-NN di ruang fitur
2. **Gaussian Noise Injection** — Perturbasi `N(0, 0.01²)` ke seluruh batch (asli + sintetis)
3. **Mixup** — Interpolasi konveks antar sampel `λ ~ Beta(0.2, 0.2)` → soft label

---

##  Project Structure

```
ChestX-Ray/
├── app/                          # Next.js frontend
│   ├── page.tsx                  #   Homepage / overview
│   ├── methodology/page.tsx      #   Pipeline & scenario descriptions
│   ├── results/page.tsx          #   Tables, charts, confusion matrices
│   ├── model/page.tsx            #   Architecture documentation
│   ├── inference/page.tsx        #   Live demo (upload X-ray → predict)
│   ├── team/page.tsx             #   Research team
│   ├── references/page.tsx       #   Bibliography
│   ├── globals.css               #   Design system
│   └── layout.tsx                #   Root layout + metadata
│
├── backend/                      # FastAPI inference server
│   ├── app.py                    #   Server entry + /health, /warmup, /predict
│   ├── inference.py              #   MedicalXRayPipeline orchestrator
│   ├── models.py                 #   FA module, MLP head, hooks, utilities
│   ├── colab_run.py              #   Google Colab deployment script (pyngrok)
│   ├── Dockerfile                #   Backend container
│   └── requirements.txt          #   Python dependencies
│
├── research/                     # Experiment notebooks & trained weights
│   ├── *.ipynb                   #   Jupyter notebooks (feature extraction + classifier)
│   ├── fa_best_med_balanced_1.pt #   FA encoder weights (trained)
│   └── best_overall_weights.pt   #   MLP classifier weights
│
├── components/                   # Shared React components
│   ├── Navbar.tsx
│   ├── Footer.tsx
│   └── SectionHeader.tsx
│
├── public/                       # Static assets
│   ├── charts/                   #   Result visualizations (barchart, radar, CM)
│   └── team/                     #   Team member photos
│
├── docker-compose.yml            # Full-stack orchestration
├── Dockerfile.frontend           # Frontend container
├── .env.local                    # API URL config (local/ngrok)
├── .gitignore
└── README.md                     # ← You are here
```

---

##  Running the Experiments

Folder `research/` berisi **dua Jupyter notebook** yang merepresentasikan dua tahap eksperimen. Kedua notebook didesain untuk dijalankan di **Google Colab dengan GPU (T4 / A100)** karena membutuhkan VRAM ≥ 12 GB untuk inference Stable Diffusion.

### Prasyarat

1. **Dataset** — Chest X-ray dataset dengan 6 kelas target (Atelectasis, Effusion, Infiltration, No Finding, Nodule, Pneumothorax) disusun dalam dua skenario:
   - **Balanced** — ~416 sampel/kelas
   - **Imbalanced** — rasio skew 10:1 (simulasi distribusi klinis nyata)
2. **Google Drive mount** — notebook mengakses dataset & menyimpan checkpoint via Drive
3. **Hugging Face access** — model `Osama03/Medical-X-ray-image-generation-stable-diffusion` (LoRA) dan `CompVis/stable-diffusion-v1-4` (base U-Net)
4. Dependencies utama: `torch`, `diffusers`, `transformers`, `peft`, `scikit-learn`, `imbalanced-learn`, `pandas`

---

### Notebook 1 — Feature Extraction + FA Training

**Tujuan:** Mengekstrak feature vector 128-dim dari setiap citra X-ray menggunakan frozen Medical SD v1.4 + LoRA, lalu melatih modul Dual Feature Aggregation (DFATB + FAFN + Differential Denoising).

**Alur eksekusi (per cell):**
1. Mount Google Drive & install dependencies
2. Load `CONFIG` — skenario (`balanced` / `imbalanced`), timestep noise `t=10`, hyperparameter FA (`z_dim=128`, `num_heads=4`, `lambda_init=0.5`)
3. Path discovery — scan dataset pada Google Drive
4. **Training FA module** — 20 epoch, `lr=1e-4`, `batch=1 × accum_steps=4`, AMP (fp16). Output: checkpoint `fa_best_med_{scenario}.pt`
5. **Inference & feature export** — forward pass seluruh dataset → simpan feature vector ke CSV (`vektor_timestep_10.csv`)

**Output:**
- `fa_best_med_balanced_1.pt` — bobot FA encoder terbaik (dipakai oleh backend)
- CSV feature vector per skenario (dipakai oleh Notebook 2)

---

### Notebook 2 — MLP Classifier & Skenario Evaluasi

**Tujuan:** Melatih MLP classification head di atas feature vector hasil Notebook 1, dan membandingkan performa FE+FA (Medical SD) dengan **4 baseline feature extractor** (DINOv2, Swin Transformer, MaxViT, ConvNeXtV2).

**Alur eksekusi:**
1. Baca feature CSV dari Google Drive (baseline FE + Medical SD)
2. Jalankan **4 skenario evaluasi**:

   | Scenario | Data | FSA |
   |----------|------|-----|
   | 1 | Balanced | ❌ |
   | 2 | Balanced | ✅ |
   | 3 | Imbalanced | ❌ |
   | 4 | Imbalanced | ✅ |

3. Untuk setiap kombinasi `(feature_extractor × scenario)`:
   - **FSA Pipeline** (jika aktif): FS-SMOTE → Gaussian Noise `N(0, 0.01²)` → Mixup `λ ~ Beta(0.2, 0.2)`
   - **MLP Training** — 50 epoch, `lr=1e-3`, `batch_size=64`, `hidden=512`, `dropout=0.3`, `weight_decay=1e-4`, class weights (inverse-frequency) untuk skenario imbalanced
   - **Evaluasi** — Accuracy, F1-macro, AUC (one-vs-rest), confusion matrix, per-class precision/recall
4. Simpan bobot terbaik → `best_overall_weights.pt`

**Output:**
- `best_overall_weights.pt` — bobot MLP classifier terbaik (dipakai oleh backend)
- Tabel metrik per skenario & feature extractor
- Confusion matrix & classification report

---

### Reproduksi Hasil

Untuk mereplikasi hasil penelitian dari awal:

```text
Notebook 1 (Balanced)   →  fa_best_med_balanced.pt  +  vektor_balanced.csv
Notebook 1 (Imbalanced) →  fa_best_med_imbalanced.pt + vektor_imbalanced.csv
Notebook 2              →  best_overall_weights.pt + metric tables
```

Gunakan kombinasi `fa_best_med_balanced_1.pt` + `best_overall_weights.pt` di folder `research/` untuk inference di web app (backend FastAPI akan auto-load keduanya).

---

##  Checkpoint Reference

| File | Deskripsi |
|------|-----------|
| **`fa_best_med_balanced_1.pt`** | Trained Feature Aggregation model (~315 MB). Berisi: `fa_model_state`, hyperparameters (`feature_channels`, `attn_channels`, `z_dim=128`), `class_names`, dan metadata training. |
| **`best_overall_weights.pt`** | Trained MLP classifier (~274 KB). State dict berisi `net.0` (LayerNorm 128), `net.1` (Linear 128→512), `net.4` (Linear 512→6). |

---

##  Deployment

### Local Development
```bash
# Frontend
npm install && npm run dev       # http://localhost:3000

# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Google Colab (GPU)
1. Upload `backend/` dan `research/` ke Colab
2. Copy isi `backend/colab_run.py` ke cell Colab
3. Set `NGROK_AUTH_TOKEN` dari [ngrok dashboard](https://dashboard.ngrok.com)
4. Jalankan cell — salin URL ngrok ke `.env.local`
5. Restart `npm run dev`

### Docker
```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## 👥 Team

| Nama | Role | Kontribusi |
|------|------|------------|
| Syahribanun | Lead Researcher | Research design · pipeline development · FE+FA implementation · writeup |
| Ahmad Naufal Farras | Researcher | Classification model development · Feature Extraction module implementation · Model & web deployment |

**Institution:** Departemen Teknik Informatika, Lab Komputasi Cerdas dan Visi (KCV), ITS Surabaya
