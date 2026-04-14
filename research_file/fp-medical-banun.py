# ============================================================
# CELL 1 → Mount Google Drive
# ============================================================

import sys, os
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
DRIVE_MOUNT = '/content/drive'

if IN_COLAB:
    from google.colab import drive
    if not Path(DRIVE_MOUNT + '/MyDrive').exists():
        drive.mount(DRIVE_MOUNT)
    print(f'Drive mounted: {Path(DRIVE_MOUNT + "/MyDrive").exists()}')
else:
    print('Bukan di Colab - skip mount drive (jalan lokal).')


# ============================================================
# CELL 2 → Install dependencies
# ============================================================

import importlib, subprocess

def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f'Installing {pkg} ...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_ensure('diffusers>=0.25.0', 'diffusers')
_ensure('transformers>=4.36.0', 'transformers')
_ensure('accelerate>=0.25.0', 'accelerate')
_ensure('safetensors>=0.4.0', 'safetensors')
print('Dependencies OK.')

# ============================================================
# CELL 3 → Imports + CONFIG
# ============================================================

import gc
import random
import shutil
import time
import platform
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# === SEED LOCKING ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IS_LINUX = platform.system() == 'Linux'

# ── HuggingFace Token (isi agar download model lebih cepat & tidak rate-limit) ──
HF_TOKEN = ""  # ← GANTI DENGAN TOKEN ANDA, contoh: "hf_xxxxxxxxxxxxxxxxxxxx"

if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print(f'HuggingFace: logged in successfully.')
    except Exception as e:
        print(f'HuggingFace login gagal: {e} — lanjut tanpa token.')
else:
    print('HuggingFace: no token set (download mungkin lambat / rate-limited).')

print(f'IN_COLAB  : {IN_COLAB}')
print(f'IS_LINUX  : {IS_LINUX}')
print(f'Python exe: {sys.executable}')
print(f'CWD       : {Path.cwd()}')
if torch.cuda.is_available():
    print(f'GPU       : {torch.cuda.get_device_name(0)}')
    print(f'VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ==========================================================
# HARDCODED CLASS MAPPING — konsisten lintas SEMUA eksperimen
# ==========================================================
CLASS_NAMES = sorted(['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax'])
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')


def normalize_scenarios(values, fallback=None):
    fallback = fallback or ['balanced', 'imbalanced']
    valid = {'balanced', 'imbalanced'}
    out = []
    for v in values or []:
        s = str(v).strip().lower()
        if s in valid and s not in out:
            out.append(s)
    return out or fallback


# ==========================================================
# CONFIG UTAMA — Colab format, Medical extraction
# ==========================================================
CONFIG = {
    # Skenario extraction
    'SCENARIOS_TO_RUN': ['balanced', 'imbalanced'],

    # Skenario training FA (checkpoint terpisah per skenario)
    'FA_TRAIN_SCENARIOS': ['balanced', 'imbalanced'],

    # === Model Selection ===
    # 'medical': Medical X-ray Stable Diffusion (SD v1.4 + LoRA, 512x512)
    # 'sdxl':    Stable Diffusion XL (SDXL base 1.0, 1024x1024) — gunakan fp-sdxl.py
    'ACTIVE_MODELS': ['medical'],

    # Sampling dan timestep
    'TIMESTEPS': [10],
    'MAX_SAMPLES_PER_SCENARIO': None,  # None = semua

    # Runtime
    'SKIP_IF_OUTPUT_EXISTS': True,
    'SUBPROCESS_TIMEOUT': 999999,      # ~11.5 hari — biarkan proses jalan sampai selesai

    # Device
    'MED_DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SDXL_DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'MED_DTYPE': 'fp16' if torch.cuda.is_available() else 'fp32',
    'SDXL_DTYPE': 'fp16' if torch.cuda.is_available() else 'fp32',

    # Resolusi
    'MED_WIDTH': 512, 'MED_HEIGHT': 512,
    'SDXL_WIDTH': 1024, 'SDXL_HEIGHT': 1024,

    # FA
    'FA_Z_DIM': 128,

    # Model IDs
    'MED_MODEL_ID': 'Osama03/Medical-X-ray-image-generation-stable-diffusion',
    'MED_BASE_MODEL_ID': 'CompVis/stable-diffusion-v1-4',
    'MED_LORA_WEIGHT': 'pytorch_lora_weights.safetensors',
    'SDXL_MODEL_ID': 'stabilityai/stable-diffusion-xl-base-1.0',

    # Training FA
    'RUN_FA_TRAINING': False,
    'TRAIN_IF_CHECKPOINT_MISSING': True,
    'TRAIN_EPOCHS': 20,
    'TRAIN_BATCH_SIZE': 1,             # Effective batch = 1 × ACCUM_STEPS = 4
    'TRAIN_ACCUM_STEPS': 4,
    'TRAIN_LR': 1e-4,
    'TRAIN_WEIGHT_DECAY': 1e-4,
    'TRAIN_TIMESTEPS': [10, 20],
    'TRAIN_NUM_WORKERS': 0,
    'TRAIN_LOCAL_FILES_ONLY': False,
    'TRAIN_VAL_SPLIT': 0.2,
    'MIN_ACCEPTABLE_F1': 0.10,
    'GRAD_CLIP_MAX_NORM': 1.0,

    # Random horizontal flip sebelum VAE encoding (augmentasi training)
    'USE_RANDOM_FLIP': True,

    # Konfigurasi sumber data training per skenario
    'TRAIN_SCENARIO_TO_CSV': {
        'balanced': ['balanced_2500.csv', 'balanced_prompts.csv'],
        'imbalanced': ['imbalanced_2500.csv', 'imbalanced_prompts.csv'],
    },
    'TRAIN_SCENARIO_TO_IMAGE_DIR': {
        'balanced': ['balanced'],
        'imbalanced': ['imbalanced'],
    },

    # === Colab paths (referensi dari fp-baseline.py) ===
    'COLAB_DRIVE_DATA_PATH':   '/content/drive/MyDrive/FP_KCV/Dataset_fix',
    'COLAB_DRIVE_OUTPUT_PATH': '/content/drive/MyDrive/FP_KCV',

    # Speed optimization: copy data ke local disk
    'COPY_DATA_TO_LOCAL': True,
    'LOCAL_DATA_DIR':     '/content/data/fp',

    # Override — set None untuk auto-discovery
    'DATA_ROOT_OVERRIDE':   None,
    'OUTPUT_ROOT_OVERRIDE': None,

    # Optional override checkpoint (per skenario)
    'FA_CHECKPOINT_MED_BY_SCENARIO': {
        'balanced': None,
        'imbalanced': None,
    },
    'FA_CHECKPOINT_SDXL_OVERRIDE': None,
}

# Derive legacy toggles dari ACTIVE_MODELS
CONFIG['RUN_MEDICAL'] = 'medical' in CONFIG['ACTIVE_MODELS']
CONFIG['RUN_SDXL'] = 'sdxl' in CONFIG['ACTIVE_MODELS']

CONFIG['SCENARIOS_TO_RUN'] = normalize_scenarios(CONFIG.get('SCENARIOS_TO_RUN'))
CONFIG['FA_TRAIN_SCENARIOS'] = normalize_scenarios(CONFIG.get('FA_TRAIN_SCENARIOS'))

print('CONFIG loaded.')
print(f"ACTIVE_MODELS      : {CONFIG['ACTIVE_MODELS']}")
print(f"SCENARIOS_TO_RUN   : {CONFIG['SCENARIOS_TO_RUN']}")
print(f"FA_TRAIN_SCENARIOS : {CONFIG['FA_TRAIN_SCENARIOS']}")

# ============================================================
# CELL 4 → Path Discovery (Colab-style dari fp-baseline.py)
# ============================================================

MED_SCRIPT_NAME  = 'Medical X-ray Stable Diffusion_feature_map_extractor.py'
SDXL_SCRIPT_NAME = 'sdxl_feature_map_extractor.py'


def first_existing(paths):
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def _find_local_data_root() -> Path:
    candidates = [
        Path.cwd() / 'fp',
        Path.cwd().parent / 'fp',
        Path.cwd() / 'data' / 'sample_baru',
        Path.cwd().parent / 'data' / 'sample_baru',
        Path(r'D:\Main Storage\Vscode\FP_Admin_KCV\data\sample_baru'),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('Tidak menemukan folder data. Set CONFIG["DATA_ROOT_OVERRIDE"].')


def _copy_drive_to_local(src: Path, dst: Path) -> Path:
    dst = Path(dst)
    if dst.exists():
        n_files = sum(1 for _ in dst.rglob('*') if _.is_file())
        print(f'Local data sudah ada: {dst} ({n_files} file). Skip copy.')
        return dst
    print(f'Copying {src} -> {dst} ...')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    n_files = sum(1 for _ in dst.rglob('*') if _.is_file())
    print(f'  Done. {n_files} file copied.')
    return dst


def _default_med_ckpt_by_scenario(base_dir):
    return {
        'balanced': base_dir / 'fa_best_med_balanced.pt',
        'imbalanced': base_dir / 'fa_best_med_imbalanced.pt',
    }


def _resolve_med_ckpt_by_scenario(base_dir):
    defaults = _default_med_ckpt_by_scenario(base_dir)
    resolved = {}
    overrides = CONFIG.get('FA_CHECKPOINT_MED_BY_SCENARIO') or {}

    for sc in ['balanced', 'imbalanced']:
        ov = overrides.get(sc)
        resolved[sc] = Path(ov) if ov else defaults[sc]

    # fallback legacy supaya backward compatible
    legacy = base_dir / 'fa_best.pt'
    if legacy.exists() and not resolved['balanced'].exists() and not resolved['imbalanced'].exists():
        print(f'INFO: Menggunakan legacy checkpoint untuk balanced+imbalanced: {legacy}')
        resolved['balanced'] = legacy
        resolved['imbalanced'] = legacy

    return resolved


def _resolve_paths():
    # === Data root ===
    if CONFIG['DATA_ROOT_OVERRIDE']:
        data_root = Path(CONFIG['DATA_ROOT_OVERRIDE'])
    elif IN_COLAB:
        drive_data = Path(CONFIG['COLAB_DRIVE_DATA_PATH'])
        if not drive_data.exists():
            raise FileNotFoundError(
                f'Drive data path tidak ada: {drive_data}\n'
                f'Pastikan dataset sudah di-upload ke Google Drive dan path sesuai.'
            )
        if CONFIG['COPY_DATA_TO_LOCAL']:
            data_root = _copy_drive_to_local(drive_data, Path(CONFIG['LOCAL_DATA_DIR']))
        else:
            data_root = drive_data
    else:
        data_root = _find_local_data_root()

    # === Output root ===
    if CONFIG['OUTPUT_ROOT_OVERRIDE']:
        output_root = Path(CONFIG['OUTPUT_ROOT_OVERRIDE'])
    elif IN_COLAB:
        output_root = Path(CONFIG['COLAB_DRIVE_OUTPUT_PATH'])
    else:
        output_root = data_root.parent / 'raw_features' / 'kaggle_dual_scenario'

    output_root.mkdir(parents=True, exist_ok=True)

    # === Project root (untuk menemukan scripts) ===
    if IN_COLAB:
        drive_data = Path(CONFIG['COLAB_DRIVE_DATA_PATH'])
        project_root = drive_data.parent
        fe_dir_candidates = [
            project_root / 'FE_with_FA',
            drive_data,
            Path('/content'),
        ]
    else:
        candidates = [
            Path.cwd(),
            Path.cwd().parent,
            Path(r'D:\Main Storage\Vscode\FP_Admin_KCV'),
        ]
        project_root = None
        for c in candidates:
            fe_dir = c / 'FE_with_FA'
            if (fe_dir / MED_SCRIPT_NAME).exists():
                project_root = c
                break
        if project_root is None:
            raise FileNotFoundError(f'Project root tidak ditemukan (butuh FE_with_FA/{MED_SCRIPT_NAME}).')
        fe_dir_candidates = [project_root / 'FE_with_FA']

    # === Medical script ===
    med_script = None
    for d in fe_dir_candidates:
        candidate = d / MED_SCRIPT_NAME
        if candidate.exists():
            med_script = candidate
            break
    if med_script is None:
        raise FileNotFoundError(f'Medical script tidak ditemukan: {MED_SCRIPT_NAME}')

    # === SDXL script (opsional) ===
    sdxl_script = None
    for d in fe_dir_candidates:
        candidate = d / SDXL_SCRIPT_NAME
        if candidate.exists():
            sdxl_script = candidate
            break

    # Jika di Colab, copy scripts ke /content agar runnable
    if IN_COLAB:
        dst_med = Path('/content') / MED_SCRIPT_NAME
        shutil.copy2(med_script, dst_med)
        med_script = dst_med
        if sdxl_script is not None:
            dst_sdxl = Path('/content') / SDXL_SCRIPT_NAME
            shutil.copy2(sdxl_script, dst_sdxl)
            sdxl_script = dst_sdxl
        project_root = Path('/content')

    # === FA Checkpoints ===
    if IN_COLAB:
        ckpt_base = Path(CONFIG['COLAB_DRIVE_DATA_PATH']).parent / 'FE_with_FA'
    else:
        ckpt_base = project_root / 'FE_with_FA' if project_root else Path('.')

    fa_ckpt_med_by_scenario = _resolve_med_ckpt_by_scenario(ckpt_base)
    fa_ckpt_sdxl = (
        Path(CONFIG['FA_CHECKPOINT_SDXL_OVERRIDE'])
        if CONFIG['FA_CHECKPOINT_SDXL_OVERRIDE']
        else ckpt_base / 'fa_best_sdxl.pt'
    )

    return data_root, output_root, med_script, sdxl_script, project_root, fa_ckpt_med_by_scenario, fa_ckpt_sdxl


def validate_checkpoint(ckpt_path, expected_num_classes=None):
    if not ckpt_path.exists():
        return False, 'File tidak ada'
    if ckpt_path.stat().st_size == 0:
        return False, 'File 0 byte (corrupt)'
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if not isinstance(state, dict):
            return False, 'Format bukan dict'
        if expected_num_classes and 'num_classes' in state:
            if state['num_classes'] != expected_num_classes:
                return False, f"num_classes={state['num_classes']} expected={expected_num_classes}"
        return True, 'OK'
    except Exception as e:
        return False, f'Load error: {e}'


# === RESOLVE ===
DATA_ROOT, OUTPUT_ROOT, MED_SCRIPT, SDXL_SCRIPT, PROJECT_ROOT, FA_CHECKPOINT_MED_BY_SCENARIO, FA_CHECKPOINT_SDXL = _resolve_paths()
FA_CHECKPOINT_MED = FA_CHECKPOINT_MED_BY_SCENARIO.get('balanced')

print(f'\n{"="*60}')
print(f'IN_COLAB         : {IN_COLAB}')
print(f'DATA_ROOT        : {DATA_ROOT}')
print(f'OUTPUT_ROOT      : {OUTPUT_ROOT}')
print(f'MED_SCRIPT       : {MED_SCRIPT}')
print(f'SDXL_SCRIPT      : {SDXL_SCRIPT}')
print(f'PROJECT_ROOT     : {PROJECT_ROOT}')
print(f'FA_CKPT_MED_MAP  :')
for sc in ['balanced', 'imbalanced']:
    print(f'  - {sc:<10}: {FA_CHECKPOINT_MED_BY_SCENARIO[sc]}')
print(f'FA_CKPT_SDXL     : {FA_CHECKPOINT_SDXL}')
print(f'{"="*60}')

assert MED_SCRIPT.exists(), f'Medical script TIDAK ADA: {MED_SCRIPT}'
assert DATA_ROOT.exists(), f'Data root TIDAK ADA: {DATA_ROOT}'

# Checkpoint validation (Medical per skenario)
for sc in CONFIG['FA_TRAIN_SCENARIOS']:
    ckpt = FA_CHECKPOINT_MED_BY_SCENARIO[sc]
    valid, msg = validate_checkpoint(Path(ckpt), NUM_CLASSES)
    status = 'VALID' if valid else f'INVALID ({msg})'
    print(f'FA Checkpoint Medical [{sc}]: {status}')

# SDXL checkpoint (optional)
valid_sdxl, msg_sdxl = validate_checkpoint(FA_CHECKPOINT_SDXL, NUM_CLASSES)
print(f"FA Checkpoint SDXL: {'VALID' if valid_sdxl else f'INVALID ({msg_sdxl})'}")

# FAIL-FAST
if not CONFIG['TRAIN_IF_CHECKPOINT_MISSING'] and not CONFIG['RUN_FA_TRAINING']:
    if CONFIG['RUN_MEDICAL']:
        missing = [sc for sc in CONFIG['SCENARIOS_TO_RUN'] if not FA_CHECKPOINT_MED_BY_SCENARIO[sc].exists()]
        if missing:
            raise SystemExit(f'ABORT: FA checkpoint Medical tidak ada untuk skenario: {missing}')
    if CONFIG['RUN_SDXL'] and not FA_CHECKPOINT_SDXL.exists():
        raise SystemExit('ABORT: FA checkpoint SDXL tidak ada.')

# List data files
print(f'\nData files:')
for f in sorted(DATA_ROOT.glob('*.csv')):
    print(f'  {f.name} ({f.stat().st_size:,} bytes)')
for d in sorted(DATA_ROOT.iterdir()):
    if d.is_dir():
        n = len(list(d.glob('*')))
        print(f'  {d.name}/ ({n} files)')

# ============================================================
# CELL 5 → FA TRAINING
# ============================================================
# ==========================================================
# FA TRAINING — SINGLE-CLASS, VALIDATION SPLIT, FEATURE CACHE
# ==========================================================
from diffusers import StableDiffusionPipeline


def _load_med_fa_module(module_path):
    spec = importlib.util.spec_from_file_location('med_fa_module', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Gagal load module: {module_path}')
    module = importlib.util.module_from_spec(spec)
    # Penting untuk Python 3.12: dataclass butuh modul ada di sys.modules
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SingleClassFADataset(Dataset):
    def __init__(self, csv_path, image_roots, image_col='Image Index', labels_col='Finding Labels', prompt_col='Prompt'):
        df = pd.read_csv(csv_path)
        required = {image_col, labels_col}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f'Kolom wajib tidak ada: {miss}')

        df[image_col] = df[image_col].astype(str).str.strip()
        df[labels_col] = df[labels_col].fillna('').astype(str).str.strip()

        if prompt_col not in df.columns:
            def make_prompt(label):
                if label == 'No Finding':
                    return 'A normal healthy chest X-ray with clear lungs'
                return f'A chest X-ray showing {label}' if label else 'A chest X-ray'
            df[prompt_col] = df[labels_col].apply(make_prompt)
        df[prompt_col] = df[prompt_col].fillna('').astype(str)

        valid_rows = []
        skipped = 0
        for _, row in df.iterrows():
            label = row[labels_col]
            if label not in CLASS_TO_IDX:
                skipped += 1
                continue
            img_path = self._resolve(row[image_col], image_roots)
            if img_path is None:
                skipped += 1
                continue
            valid_rows.append({
                'image_path': str(img_path),
                'prompt': row[prompt_col],
                'label': CLASS_TO_IDX[label],
            })

        if skipped > 0:
            pct = skipped / len(df) * 100
            print(f'WARNING: {skipped}/{len(df)} baris dilewati ({pct:.1f}%)')
            if pct > 50:
                raise ValueError('>50% data hilang. Cek path gambar.')

        if not valid_rows:
            raise ValueError('Tidak ada data valid.')

        self.rows = valid_rows
        self.labels = [r['label'] for r in valid_rows]
        print(f'Dataset: {len(valid_rows)} valid dari {len(df)} baris')

    def _resolve(self, img_name, roots):
        raw = str(img_name).strip()
        for base in roots:
            p = base / raw
            if p.exists() and p.stat().st_size > 0:
                return p
        stem = Path(raw).stem
        for base in roots:
            for ext in ['.png', '.jpg', '.jpeg']:
                p = base / f'{stem}{ext}'
                if p.exists() and p.stat().st_size > 0:
                    return p
        return None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {'idx': idx, 'image_path': r['image_path'], 'prompt': r['prompt'], 'label': r['label']}


def _scenario_csv_candidates(scenario):
    names = (CONFIG.get('TRAIN_SCENARIO_TO_CSV') or {}).get(scenario, [])
    return [DATA_ROOT / n for n in names]


def _scenario_image_roots(scenario):
    names = (CONFIG.get('TRAIN_SCENARIO_TO_IMAGE_DIR') or {}).get(scenario, [])
    roots = [DATA_ROOT / n for n in names]
    # fallback umum jika struktur berbeda
    roots.extend([DATA_ROOT / scenario, DATA_ROOT / 'images', DATA_ROOT])
    uniq = []
    for r in roots:
        if r.exists() and r not in uniq:
            uniq.append(r)
    return uniq


def resolve_training_sources(scenario):
    csv_path = first_existing(_scenario_csv_candidates(scenario))
    if csv_path is None:
        raise FileNotFoundError(f'CSV training untuk {scenario} tidak ditemukan di {DATA_ROOT}')

    image_roots = _scenario_image_roots(scenario)
    if not image_roots:
        raise FileNotFoundError(f'Image root untuk {scenario} tidak ditemukan di {DATA_ROOT}')

    return csv_path, image_roots


def train_fa_checkpoint(ready_csv, image_roots, output_ckpt, scenario_name='unknown'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    amp_enabled = device == 'cuda'

    dataset = SingleClassFADataset(csv_path=ready_csv, image_roots=image_roots)

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=CONFIG['TRAIN_VAL_SPLIT'],
        stratify=dataset.labels, random_state=SEED,
    )
    print(f'[{scenario_name}] Train: {len(train_idx)} | Val: {len(val_idx)}')

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=int(CONFIG['TRAIN_BATCH_SIZE']),
                              shuffle=True, num_workers=int(CONFIG['TRAIN_NUM_WORKERS']),
                              pin_memory=(device == 'cuda'))
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False,
                            num_workers=int(CONFIG['TRAIN_NUM_WORKERS']))

    # Load extractor module
    med_module = _load_med_fa_module(MED_SCRIPT)
    FAEncoder = med_module.FeatureAggregationEncoder
    BlockFC = med_module.BlockFeatureCollector
    AttnC = med_module.AttentionCollector
    _preprocess = med_module._preprocess_image
    _encode = med_module._encode_latent
    _text_emb = med_module._build_text_embeddings
    _noise = med_module._add_noise_at_timestep
    _sorted_maps = med_module._prepare_sorted_maps
    _model_fc = med_module._model_feature_channels
    _model_ac = med_module._model_attention_channels
    _align_f = med_module._align_features_to_model
    _build_ap = med_module._build_attention_pairs
    _adapt_ac = med_module._adapt_attention_channels

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        CONFIG['MED_BASE_MODEL_ID'], torch_dtype=dtype,
        local_files_only=bool(CONFIG['TRAIN_LOCAL_FILES_ONLY']),
    ).to(device)
    try:
        pipe.load_lora_weights(CONFIG['MED_MODEL_ID'],
            weight_name=CONFIG['MED_LORA_WEIGHT'],
            local_files_only=bool(CONFIG['TRAIN_LOCAL_FILES_ONLY']))
    except TypeError:
        pipe.load_lora_weights(CONFIG['MED_MODEL_ID'], weight_name=CONFIG['MED_LORA_WEIGHT'])
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False

    # Pre-cache text embeddings + VAE latents
    print(f'[{scenario_name}] Pre-computing text embeddings dan VAE latents...')
    text_cache, latent_cache = {}, {}
    use_flip = bool(CONFIG.get('USE_RANDOM_FLIP', False))
    for i in tqdm(range(len(dataset)), desc=f'Caching {scenario_name}'):
        sample = dataset[i]
        with torch.no_grad():
            text_cache[i] = _text_emb(pipe, prompt=sample['prompt']).cpu()
            img_t = _preprocess(Path(sample['image_path']), CONFIG['MED_WIDTH'], CONFIG['MED_HEIGHT'])
            # Cache laten asli tanpa augmentasi — flip dilakukan dinamis di training loop
            latent_cache[i] = _encode(pipe, image_tensor=img_t, dtype=dtype).cpu()

    # Offload VAE + text encoder
    pipe.vae.cpu(); pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'[{scenario_name}] VAE + Text Encoder offloaded ke CPU.')

    # FA model + cls head
    feat_ch = [320, 640, 1280, 1280]
    attn_ch = [320, 640, 1280, 1280]
    fa_model = FAEncoder(feature_channels=feat_ch, attn_channels=attn_ch,
                         z_dim=int(CONFIG['FA_Z_DIM']), num_heads=4,
                         fafn_expansion=2, lambda_init=0.5, dropout=0.1,
                         spatial_max_hw=32).to(device)
    cls_head = nn.Linear(int(CONFIG['FA_Z_DIM']), NUM_CLASSES).to(device)

    params = list(fa_model.parameters()) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(CONFIG['TRAIN_LR']),
                                  weight_decay=float(CONFIG['TRAIN_WEIGHT_DECAY']))

    class_counts = np.zeros(NUM_CLASSES)
    for lbl in dataset.labels:
        class_counts[lbl] += 1
    total = len(dataset)
    class_weights = total / (NUM_CLASSES * np.maximum(class_counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    accum_steps = max(1, int(CONFIG['TRAIN_ACCUM_STEPS']))

    best_val_f1 = -1.0
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, int(CONFIG['TRAIN_EPOCHS']) + 1):
            fa_model.train(); cls_head.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Train {scenario_name} E{epoch}'), 1):
                oidx = batch['idx'].item()
                lbl = batch['label'].item()
                ts = random.choice(CONFIG['TRAIN_TIMESTEPS'])
                lat = latent_cache[oidx].to(device, dtype=dtype)
                # Dynamic random flip pada latent (berubah setiap iterasi, bukan terkunci saat caching)
                if use_flip and random.random() > 0.5:
                    lat = torch.flip(lat, dims=[-1])
                te = text_cache[oidx].to(device)

                fc = BlockFC(pipe.unet, blocks='up_mid')
                ac = AttnC(pipe.unet)
                fc.register(); ac.install()
                try:
                    with torch.no_grad():
                        noisy, tt = _noise(pipe, latents=lat, timestep=ts)
                        _ = pipe.unet(noisy, tt, encoder_hidden_states=te, return_dict=True)
                    fmaps = {k: v for k, v in fc.features.items() if v.dim() == 4}
                    amaps = {k: v for k, v in ac.maps.items() if v.dim() == 4}
                    if not fmaps: continue
                    fk, fl, ak, al = _sorted_maps(fmaps, amaps)
                    del fmaps, amaps
                    _, sel_fl = _align_f(fk, fl, _model_fc(fa_model))
                    del fk, fl
                    ap, _, _ = _build_ap(sel_fl, ak, al)
                    del ak, al
                    aligned_ap = [(_adapt_ac(a1, tc), _adapt_ac(a2, tc)) for (a1, a2), tc in zip(ap, _model_ac(fa_model))]
                    del ap
                    f_dev = [f.to(device) for f in sel_fl]
                    a_dev = [(a1.to(device), a2.to(device)) for a1, a2 in aligned_ap]
                    del sel_fl, aligned_ap

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                        fa_out = fa_model(f_dev, a_dev)
                        logits = cls_head(fa_out['z'])
                        target = torch.tensor([lbl], dtype=torch.long, device=device)
                        loss = criterion(logits, target) / accum_steps
                    scaler.scale(loss).backward()
                    epoch_losses.append(loss.item() * accum_steps)
                    del f_dev, a_dev, fa_out, logits, target, loss
                finally:
                    fc.remove(); ac.restore()
                    del noisy, tt
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if batch_idx % accum_steps == 0 or batch_idx == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(CONFIG['GRAD_CLIP_MAX_NORM']))
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            # === VALIDATION ===
            fa_model.eval(); cls_head.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for vb in tqdm(val_loader, desc=f'Val {scenario_name} E{epoch}'):
                    vi = vb['idx'].item(); vl = vb['label'].item()
                    vlat = latent_cache[vi].to(device, dtype=dtype)
                    vte = text_cache[vi].to(device)

                    # Rata-rata logits dari SEMUA timestep (konsisten dgn training)
                    avg_logits = None
                    for v_ts in CONFIG['TRAIN_TIMESTEPS']:
                        vfc = BlockFC(pipe.unet, blocks='up_mid'); vac = AttnC(pipe.unet)
                        vfc.register(); vac.install()
                        try:
                            vn, vtt = _noise(pipe, latents=vlat, timestep=v_ts)
                            _ = pipe.unet(vn, vtt, encoder_hidden_states=vte, return_dict=True)
                            fm = {k: v for k, v in vfc.features.items() if v.dim() == 4}
                            am = {k: v for k, v in vac.maps.items() if v.dim() == 4}
                            if not fm: continue
                            fk, fl, ak, al = _sorted_maps(fm, am)
                            del fm, am
                            _, sf = _align_f(fk, fl, _model_fc(fa_model))
                            del fk, fl
                            ap, _, _ = _build_ap(sf, ak, al)
                            del ak, al
                            aa = [(_adapt_ac(a1, tc), _adapt_ac(a2, tc)) for (a1, a2), tc in zip(ap, _model_ac(fa_model))]
                            del ap
                            vf = [f.to(device) for f in sf]
                            va = [(a1.to(device), a2.to(device)) for a1, a2 in aa]
                            del sf, aa
                            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                                vo = fa_model(vf, va); vlog = cls_head(vo['z'])
                            del vf, va, vo
                            if avg_logits is None:
                                avg_logits = vlog
                            else:
                                avg_logits = avg_logits + vlog
                        finally:
                            vfc.remove(); vac.restore()
                            del vn, vtt
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    if avg_logits is not None:
                        avg_logits = avg_logits / len(CONFIG['TRAIN_TIMESTEPS'])
                        val_preds.append(torch.argmax(avg_logits, dim=1).cpu().item())
                        val_targets.append(vl)

            t_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            v_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) if val_preds else 0.0
            v_acc = accuracy_score(val_targets, val_preds) if val_preds else 0.0
            print(f'[{scenario_name}] E{epoch:02d} | Loss={t_loss:.6f} | Val F1={v_f1:.4f} | Val Acc={v_acc:.4f}')

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save({
                    'fa_model_state': fa_model.state_dict(),
                    'cls_head_state': cls_head.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'epoch': epoch, 'best_val_f1': best_val_f1,
                    'class_names': CLASS_NAMES, 'num_classes': NUM_CLASSES,
                    'feature_channels': feat_ch, 'attn_channels': attn_ch,
                    'fa_z_dim': int(CONFIG['FA_Z_DIM']), 'model_arch': 'medical',
                    'scenario': scenario_name,
                }, output_ckpt)
                print(f'  -> [{scenario_name}] Best checkpoint saved (F1={best_val_f1:.4f})')

        if best_val_f1 < CONFIG['MIN_ACCEPTABLE_F1']:
            raise RuntimeError(f'Training gagal [{scenario_name}]: F1={best_val_f1:.4f} < {CONFIG["MIN_ACCEPTABLE_F1"]}')

        return {'scenario': scenario_name, 'best_f1': best_val_f1, 'checkpoint': str(output_ckpt)}
    finally:
        del pipe; text_cache.clear(); latent_cache.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def should_train_scenario(ckpt_path):
    if bool(CONFIG['RUN_FA_TRAINING']):
        return True
    if bool(CONFIG['TRAIN_IF_CHECKPOINT_MISSING']) and not ckpt_path.exists():
        return True
    return False


# === EKSEKUSI TRAINING PER SKENARIO ===
if not CONFIG['RUN_MEDICAL']:
    print('RUN_MEDICAL=False, training FA Medical dilewati.')
else:
    train_scenarios = [s for s in CONFIG['FA_TRAIN_SCENARIOS'] if s in {'balanced', 'imbalanced'}]
    if not train_scenarios:
        print('Tidak ada FA_TRAIN_SCENARIOS valid, training dilewati.')
    else:
        train_results = []
        for sc in train_scenarios:
            ckpt_path = FA_CHECKPOINT_MED_BY_SCENARIO[sc]
            need_train = should_train_scenario(ckpt_path)

            if not need_train:
                print(f'[{sc}] Training FA dilewati (checkpoint sudah ada): {ckpt_path}')
                continue

            train_csv, train_roots = resolve_training_sources(sc)
            print(f'[{sc}] Training CSV : {train_csv}')
            print(f'[{sc}] Image roots  : {train_roots}')
            print(f'[{sc}] Output CKPT  : {ckpt_path}')

            result = train_fa_checkpoint(train_csv, train_roots, ckpt_path, scenario_name=sc)
            train_results.append(result)
            print(f'[{sc}] Training selesai: {result}')

        if train_results:
            print('\nRingkasan training per skenario:')
            print(pd.DataFrame(train_results).to_string(index=False))

if CONFIG['RUN_SDXL']:
    print('WARNING: SDXL FA training belum diimplementasi di fp-medical.py. Gunakan fp-sdxl.py.')

# ============================================================
# CELL 6 → EXTRACTION via subprocess
# ============================================================

SCENARIO_LAYOUT = {
    'balanced': {
        'prompt_csv': ['balanced_prompts.csv'],
        'image_dirs': ['balanced', 'images'],
    },
    'imbalanced': {
        'prompt_csv': ['imbalanced_prompts.csv'],
        'image_dirs': ['imbalanced', 'images'],
    },
}


def resolve_scenario(name):
    layout = SCENARIO_LAYOUT[name]
    csv = first_existing([DATA_ROOT / c for c in layout['prompt_csv']])
    img_dir = first_existing([DATA_ROOT / d for d in layout['image_dirs']])
    if csv is None:
        raise FileNotFoundError(f'CSV untuk {name} tidak ditemukan di {DATA_ROOT}')
    if img_dir is None:
        raise FileNotFoundError(f'Folder gambar untuk {name} tidak ditemukan di {DATA_ROOT}')
    return csv, img_dir


OOM_CODES = {-9, 137} if IS_LINUX else {3221225477}

active_models = []
if CONFIG['RUN_MEDICAL']:
    active_models.append(('medical', MED_SCRIPT, {  # ckpt dipilih per skenario
        '--model-id': CONFIG['MED_MODEL_ID'],
        '--base-model-id': CONFIG['MED_BASE_MODEL_ID'],
        '--lora-weight-name': CONFIG['MED_LORA_WEIGHT'],
        '--dtype': CONFIG['MED_DTYPE'],
        '--device': CONFIG['MED_DEVICE'],
        '--width': str(CONFIG['MED_WIDTH']),
        '--height': str(CONFIG['MED_HEIGHT']),
    }))
if CONFIG['RUN_SDXL'] and SDXL_SCRIPT is not None:
    valid, msg = validate_checkpoint(FA_CHECKPOINT_SDXL, NUM_CLASSES)
    if valid:
        active_models.append(('sdxl', SDXL_SCRIPT, {
            '--model-id': CONFIG['SDXL_MODEL_ID'],
            '--dtype': CONFIG['SDXL_DTYPE'],
            '--device': CONFIG['SDXL_DEVICE'],
            '--width': str(CONFIG['SDXL_WIDTH']),
            '--height': str(CONFIG['SDXL_HEIGHT']),
        }))
    else:
        print(f'WARNING: SDXL checkpoint tidak valid ({msg}), SDXL extraction dilewati.')

if not active_models:
    raise ValueError('Tidak ada model aktif.')

results = []

for scenario_name in CONFIG['SCENARIOS_TO_RUN']:
    prompt_csv, image_dir = resolve_scenario(scenario_name)
    print(f'\n=== Skenario: {scenario_name} ===')
    print(f'  CSV : {prompt_csv}')
    print(f'  Dir : {image_dir}')

    for timestep in CONFIG['TIMESTEPS']:
        for model_tag, script, extra_args in active_models:
            out_dir = OUTPUT_ROOT / scenario_name / model_tag / f'timestep_{timestep}'
            out_dir.mkdir(parents=True, exist_ok=True)

            if model_tag == 'medical':
                fa_ckpt = FA_CHECKPOINT_MED_BY_SCENARIO[scenario_name]
                valid, msg = validate_checkpoint(fa_ckpt, NUM_CLASSES)
                if not valid:
                    raise RuntimeError(f'Checkpoint Medical [{scenario_name}] invalid: {msg}')
            else:
                fa_ckpt = FA_CHECKPOINT_SDXL

            cmd = [
                sys.executable, "-u", str(script),
                '--prompts-csv', str(prompt_csv),
                '--image-dir', str(image_dir),
                '--output-dir', str(out_dir),
                '--timestep', str(timestep),
                '--blocks', 'up_mid',
                '--fa-z-dim', str(CONFIG['FA_Z_DIM']),
                '--fa-checkpoint', str(fa_ckpt),
                '--fa-attn-channels', '320,640,1280,1280', # <--- TAMBAHKAN BARIS INI MUTLAK!
            ]
            for k, v in extra_args.items():
                cmd.extend([k, v])

            max_n = CONFIG['MAX_SAMPLES_PER_SCENARIO']
            if isinstance(max_n, int) and max_n > 0:
                cmd.extend(['--max-samples', str(max_n)])

            desc = f'{scenario_name}/{model_tag}/t{timestep}'
            print(f'\n  [{desc}] Starting...')
            print(f'  Checkpoint: {fa_ckpt}')
            print(f'  (logs ditampilkan langsung di cell output)')

            # Hitung total sampel dari CSV untuk progress bar
            try:
                total_csv = len(pd.read_csv(prompt_csv))
            except Exception:
                total_csv = None

            status = 'running'
            
            try:
                try:
                    total_csv = len(pd.read_csv(prompt_csv))
                except Exception:
                    total_csv = 2500

                print(f"  Memulai Arsitektur Respawner. Target: {total_csv} gambar")
                
                pre_existing = len(list(out_dir.glob('*.npy')))
                pbar = tqdm(total=total_csv, initial=pre_existing, desc=desc, unit='img')
                
                # LOOP KEBANGKITAN ABADI (MANDOR)
                while True:
                    current_files = len(list(out_dir.glob('*.npy')))
                    if current_files >= total_csv:
                        break # Target tercapai, Mandor berhenti
                    
                    log_file = out_dir / "extraction.log"
                    # BUNGKAM LAYAR KE FILE -> INI YANG MENCEGAH ERROR FILENO & BROWSER HANG!
                    with open(log_file, 'a', encoding='utf-8') as lf:
                        proc = subprocess.Popen(
                            cmd, stdout=lf, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace',
                            cwd=str(PROJECT_ROOT),
                        )
                        
                        prev_count = current_files
                        # Pantau selama pekerja masih hidup
                        while proc.poll() is None:
                            current = len(list(out_dir.glob('*.npy')))
                            if current > prev_count:
                                pbar.update(current - prev_count)
                                prev_count = current
                            time.sleep(3)
                            
                        # Update progress saat pekerja mati (untuk persiapan restart)
                        current = len(list(out_dir.glob('*.npy')))
                        if current > prev_count:
                            pbar.update(current - prev_count)
                
                pbar.close()
                status = 'ok'
                
            except Exception as _e:
                status = 'error'
                proc = type('P', (), {'returncode': -1})()
                print(f'  Error: {_e}')
                if 'pbar' in locals() and pbar is not None:
                    pbar.close()    

            npy_files = list(out_dir.glob('*.npy'))
            valid_files = [f for f in npy_files if f.stat().st_size > 0]
            corrupt = [f for f in npy_files if f.stat().st_size == 0]
            for cf in corrupt:
                cf.unlink()

            results.append({
                'scenario': scenario_name, 'model': model_tag,
                'timestep': timestep, 'status': status,
                'returncode': proc.returncode, 'valid_outputs': len(valid_files),
                'fa_checkpoint': str(fa_ckpt),
            })
            print(f'  Status: {status} | Outputs: {len(valid_files)}')

results_df = pd.DataFrame(results)
print(f'\nTotal runs: {len(results_df)}')

# ============================================================
# CELL 7 → REKAP HASIL
# ============================================================

if 'results_df' not in globals() or results_df.empty:
    print('Belum ada hasil.')
else:
    print(results_df.to_string(index=False))
    summary_csv = OUTPUT_ROOT / 'extraction_summary.csv'
    results_df.to_csv(summary_csv, index=False)
    print(f'\nSummary: {summary_csv}')

    print('\n--- Validasi sample ---')
    for _, row in results_df.iterrows():
        if row['status'] != 'ok': continue
        out_dir = OUTPUT_ROOT / row['scenario'] / row['model'] / f"timestep_{row['timestep']}"
        for s in list(out_dir.glob('*.npy'))[:3]:
            try:
                arr = np.load(s)
                ok = not np.isnan(arr).any() and not np.isinf(arr).any()
                print(f'  {s.name}: shape={arr.shape} | {"OK" if ok else "CORRUPT"}')
            except Exception as e:
                print(f'  {s.name}: ERROR ({e})')

    failed = results_df[results_df['status'] != 'ok']
    if failed.empty:
        print('\nSemua extraction berhasil!')
    else:
        print(f'\n{len(failed)} gagal.')
        

# ============================================================
# CELL 8 → GENERATE CSV DARI FILE NPY
# ============================================================
# Format tiap CSV : image_name, v0, v1, ..., v127, label
# Output: {model}_{scenario}_vektor_timestep_{X}.csv di OUTPUT_ROOT

import shutil

METADATA_CSV = {
    'balanced': DATA_ROOT / 'balanced_2500.csv',
    'imbalanced': DATA_ROOT / 'imbalanced_2500.csv',
}

generated_csvs = []

for scenario_name in CONFIG['SCENARIOS_TO_RUN']:
    meta_path = METADATA_CSV.get(scenario_name)
    if meta_path is None or not meta_path.exists():
        print(f'WARNING: Metadata CSV untuk {scenario_name} tidak ditemukan: {meta_path}')
        continue

    meta_df = pd.read_csv(meta_path)
    stem_to_label = {}
    for _, row in meta_df.iterrows():
        stem = Path(str(row['Image Index']).strip()).stem
        stem_to_label[stem] = str(row['Finding Labels']).strip()

    for model_tag in [m[0] for m in active_models]:
        for timestep in CONFIG['TIMESTEPS']:
            out_dir = OUTPUT_ROOT / scenario_name / model_tag / f'timestep_{timestep}'
            if not out_dir.exists():
                continue

            rows = []
            skipped = 0
            for npy_file in sorted(out_dir.glob('*.npy')):
                if npy_file.stat().st_size == 0:
                    continue

                # Parse image_stem: {stem}_t{ts}_fa_z{dim}.npy
                fname = npy_file.stem
                fa_suffix = f'_fa_z{CONFIG["FA_Z_DIM"]}'
                if fname.endswith(fa_suffix):
                    fname = fname[:-len(fa_suffix)]
                t_suffix = f'_t{timestep}'
                if fname.endswith(t_suffix):
                    fname = fname[:-len(t_suffix)]
                image_stem = fname

                label = stem_to_label.get(image_stem)
                if label is None or CLASS_TO_IDX.get(label) is None:
                    skipped += 1
                    continue

                vec = np.load(npy_file).flatten()
                row_dict = {'image_name': npy_file.name}
                for i, val in enumerate(vec):
                    row_dict[f'v{i}'] = val
                row_dict['label'] = label
                rows.append(row_dict)

            if not rows:
                print(f'[{model_tag}][{scenario_name}][t={timestep}] Tidak ada data.')
                continue

            csv_name = f'{model_tag}_{scenario_name}_vektor_timestep_{timestep}.csv'
            csv_out = OUTPUT_ROOT / csv_name
            pd.DataFrame(rows).to_csv(csv_out, index=False)
            generated_csvs.append(csv_out)
            print(f'[{model_tag}][{scenario_name}][t={timestep}] {len(rows)} baris -> {csv_out.name}'
                  + (f'  ({skipped} dilewati)' if skipped else ''))

print('\nCSV generation selesai.')

# ============================================================
# COPY OUTPUT KE GOOGLE DRIVE (FP_KCV)
# ============================================================
drive_target_dir = Path('/content/drive/MyDrive/FP_KCV')
drive_target_dir.mkdir(parents=True, exist_ok=True)

print('\nMenyimpan hasil ke Google Drive (FP_KCV)...')
for csv_path in generated_csvs:
    target_path = drive_target_dir / csv_path.name
    # Hindari copy jika asal dan tujuan adalah file yang sama
    if csv_path.resolve() != target_path.resolve():
        shutil.copy2(csv_path, target_path)
        print(f'  -> Disalin ke Drive: {target_path}')
    else:
        print(f'  -> Sudah berada di Drive: {csv_path}')

print('Semua output berhasil diamankan ke Drive!')