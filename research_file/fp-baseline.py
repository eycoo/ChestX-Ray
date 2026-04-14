# ============================================================
# CELL 1 → Mount Google Drive + Install Dependencies
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

_ensure('timm>=0.9.0', 'timm')
_ensure('opencv-python', 'cv2')



# ============================================================
# CELL 3 → Imports + CONFIG
# ============================================================

import gc
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

import timm

# ViG tidak ada di timm (baik PyPI maupun dev branch).
# Model ViG akan diload dari repo aslinya (Huawei Noah) di load_vig().
print(f'timm version: {timm.__version__}')

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print('WARNING: opencv-python tidak tersedia. CLAHE akan fallback ke PIL.ImageOps.equalize.')

print(f'torch        : {torch.__version__}')
print(f'timm         : {timm.__version__}')
print(f'cuda         : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda device  : {torch.cuda.get_device_name(0)}')


CONFIG = {
    # === Scenarios & models ===
    'SCENARIOS_TO_RUN': ['imbalanced', 'balanced'],
    'MODELS_TO_RUN':    ['dinov2'],

    # === Preprocessing ===
    'IMAGE_SIZE': 224,
    'USE_CLAHE': True,
    'CLAHE_CLIP': 2.0,
    'CLAHE_TILE': 8,
    'USE_RANDOM_FLIP': False,

    # === Inference ===
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 2,
    'DTYPE': 'fp16',
    'SEED': 42,

    # === Colab paths ===
    'COLAB_DRIVE_DATA_PATH':   '/content/drive/MyDrive/FP_Admin_KCV/fp',
    'COLAB_DRIVE_OUTPUT_PATH': '/content/drive/MyDrive/FP_Admin_KCV/fp/raw_features/fe_vig_dinov2',

    # === Speed optimization ===
    'COPY_DATA_TO_LOCAL': True,
    'LOCAL_DATA_DIR':     '/content/data/fp',

    # === Override (None = pakai resolver default) ===
    'DATA_ROOT_OVERRIDE':   None,
    'OUTPUT_ROOT_OVERRIDE': None,

    # === Behaviour ===
    'SKIP_IF_OUTPUT_EXISTS': True,
    'MAX_SAMPLES_PER_SCENARIO': 0,    # 0 = semua, >0 = smoke test

    # === Model identifiers (timm) ===
    'DINOV2_MODEL': 'vit_base_patch14_dinov2.lvd142m',
    'SWIN_MODEL':   'swin_base_patch4_window7_224',
}


# ============================================================
# CELL 4 → Locked classes, seed, path resolution
# ============================================================

LOCKED_CLASSES = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax']
CLASS_TO_IDX = {c: i for i, c in enumerate(LOCKED_CLASSES)}
NUM_CLASSES = len(LOCKED_CLASSES)
print(f'Locked classes ({NUM_CLASSES}): {LOCKED_CLASSES}')

SEED = CONFIG['SEED']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

def _find_local_data_root() -> Path:
    candidates = [
        Path.cwd() / 'fp',
        Path.cwd().parent / 'fp',
        Path.cwd() / 'data' / 'sample_baru',
        Path.cwd().parent / 'data' / 'sample_baru',
        Path(r'D:\Main Storage\Vscode\FP_Admin_KCV\fp'),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('Tidak menemukan folder fp/. Set CONFIG["DATA_ROOT_OVERRIDE"].')

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

def _resolve_paths():
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

    if CONFIG['OUTPUT_ROOT_OVERRIDE']:
        output_root = Path(CONFIG['OUTPUT_ROOT_OVERRIDE'])
    elif IN_COLAB:
        output_root = Path(CONFIG['COLAB_DRIVE_OUTPUT_PATH'])
    else:
        output_root = data_root.parent / 'raw_features' / 'fe_vig_dinov2'

    output_root.mkdir(parents=True, exist_ok=True)
    return data_root, output_root

DATA_ROOT, OUTPUT_ROOT = _resolve_paths()
print(f'\nIN_COLAB    : {IN_COLAB}')
print(f'DATA_ROOT   : {DATA_ROOT}')
print(f'OUTPUT_ROOT : {OUTPUT_ROOT}')
assert DATA_ROOT.exists(), f'DATA_ROOT tidak ada: {DATA_ROOT}'
for f in sorted(DATA_ROOT.glob('*.csv')):
    print(f'  csv : {f.name}')
for d in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
    print(f'  dir : {d.name}')


SCENARIO_LAYOUT = {
    'balanced': {
        'csv_candidates':   ['balanced_2500.csv', 'balanced_prompts.csv'],
        'image_dir_candidates': ['balanced', 'images'],
    },
    'imbalanced': {
        'csv_candidates':   ['imbalanced_2500.csv', 'imbalanced_prompts.csv'],
        'image_dir_candidates': ['imbalanced', 'images'],
    },
}

def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def resolve_scenario(name: str):
    if name not in SCENARIO_LAYOUT:
        raise ValueError(f'Scenario tidak dikenal: {name}. Valid: {list(SCENARIO_LAYOUT)}')
    layout = SCENARIO_LAYOUT[name]
    csv_path = _first_existing([DATA_ROOT / c for c in layout['csv_candidates']])
    img_dir  = _first_existing([DATA_ROOT / d for d in layout['image_dir_candidates']])
    if csv_path is None:
        raise FileNotFoundError(f'CSV untuk {name} tidak ditemukan di {DATA_ROOT}')
    if img_dir is None:
        raise FileNotFoundError(f'Image dir untuk {name} tidak ditemukan di {DATA_ROOT}')
    return csv_path, img_dir

for sc in CONFIG['SCENARIOS_TO_RUN']:
    csv_p, img_p = resolve_scenario(sc)
    print(f'  [{sc}] csv={csv_p.name}  dir={img_p.name}')


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class CLAHEThenRGB:
    def __init__(self, clip_limit: float, tile_grid: int, use_clahe: bool):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self.use_clahe = use_clahe
        if use_clahe and _HAS_CV2:
            self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        else:
            self._clahe = None

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert('L')
        if self.use_clahe:
            if self._clahe is not None:
                arr = np.asarray(gray, dtype=np.uint8)
                arr = self._clahe.apply(arr)
                gray = Image.fromarray(arr)
            else:
                from PIL import ImageOps
                gray = ImageOps.equalize(gray)
        return gray.convert('RGB')

def build_nih_transform(image_size: int, use_clahe: bool, use_flip: bool = False):
    steps = [
        CLAHEThenRGB(
            clip_limit=CONFIG['CLAHE_CLIP'],
            tile_grid=CONFIG['CLAHE_TILE'],
            use_clahe=use_clahe,
        ),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if use_flip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)

# Quick visual sanity-check
_csv0, _imgd0 = resolve_scenario(CONFIG['SCENARIOS_TO_RUN'][0])
_df0 = pd.read_csv(_csv0)
_sample = _df0.iloc[0]['Image Index']
_t = build_nih_transform(CONFIG['IMAGE_SIZE'], CONFIG['USE_CLAHE'], CONFIG['USE_RANDOM_FLIP'])
_img = Image.open(_imgd0 / _sample)
_tensor = _t(_img)
print(f'Sample : {_sample}')
print(f'Input  : mode={_img.mode} size={_img.size}')
print(f'Output : shape={tuple(_tensor.shape)} dtype={_tensor.dtype} min={_tensor.min().item():.3f} max={_tensor.max().item():.3f}')
del _df0, _img, _tensor


# ============================================================
# CELL 5 → Dataset class
# ============================================================

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path: Path, image_dir: Path, transform, max_samples: int = 0):
        df = pd.read_csv(csv_path)
        if 'Image Index' not in df.columns:
            raise ValueError(f"CSV harus punya kolom 'Image Index': {csv_path}")
        if 'Finding Labels' not in df.columns:
            raise ValueError(f"CSV harus punya kolom 'Finding Labels': {csv_path}")

        df['Image Index']    = df['Image Index'].astype(str).str.strip()
        df['Finding Labels'] = df['Finding Labels'].astype(str).str.strip()

        before = len(df)
        df = df[df['Finding Labels'].isin(LOCKED_CLASSES)].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f'  Filter locked classes: drop {dropped} baris (multi-label / out-of-class)')

        exists_mask = df['Image Index'].apply(lambda n: (image_dir / n).exists())
        missing = (~exists_mask).sum()
        if missing > 0:
            print(f'  WARNING: {missing} gambar tidak ditemukan di {image_dir}, di-drop.')
        df = df[exists_mask].reset_index(drop=True)

        if max_samples > 0:
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.df = df
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['Image Index']
        img = Image.open(img_path)
        img_t = self.transform(img)
        stem = Path(row['Image Index']).stem
        label_idx = CLASS_TO_IDX[row['Finding Labels']]
        return img_t, stem, label_idx

    def label_distribution(self):
        return self.df['Finding Labels'].value_counts().to_dict()


# ============================================================
# CELL 6 → Model loaders
# ============================================================

def _freeze_eval(model: nn.Module):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def load_dinov2(model_name: str, device: str, dtype: torch.dtype, img_size: int = 224):
    print(f'  Loading DINOv2: {model_name} (img_size={img_size})')
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
    model = _freeze_eval(model).to(device, dtype=dtype)
    return model, model_name

def load_swin(model_name: str, device: str, dtype: torch.dtype, img_size: int = 224):
    print(f'  Loading Swin Transformer: {model_name} (img_size={img_size})')
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
    model = _freeze_eval(model).to(device, dtype=dtype)
    return model, model_name

def probe_feature_dim(model: nn.Module, device: str, dtype: torch.dtype, image_size: int) -> int:
    dummy = torch.zeros(1, 3, image_size, image_size, device=device, dtype=dtype)
    with torch.no_grad():
        out = model(dummy)
    return int(out.shape[-1])


# ============================================================
# CELL 7 → Extraction loop
# ============================================================

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    output_dir: Path,
    model_tag: str,
    dtype: torch.dtype,
    device: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = failed = 0
    last_dim = None

    use_amp = (dtype == torch.float16) and (device == 'cuda')

    for imgs, stems, _labels in tqdm(loader, desc=f'Extract {model_tag}'):
        try:
            imgs = imgs.to(device, dtype=dtype, non_blocking=True)
            if use_amp:
                with torch.autocast('cuda', dtype=torch.float16):
                    feats = model(imgs)
            else:
                feats = model(imgs)
            feats_np = feats.float().cpu().numpy()
            last_dim = int(feats_np.shape[-1])

            for stem, vec in zip(stems, feats_np):
                out_path = output_dir / f'{stem}_{model_tag}.npy'
                if CONFIG['SKIP_IF_OUTPUT_EXISTS'] and out_path.exists() and out_path.stat().st_size > 0:
                    skipped += 1
                    continue
                np.save(out_path, vec.astype(np.float32))
                saved += 1
        except Exception as exc:
            print(f'  Batch gagal ({type(exc).__name__}): {exc}')
            failed += len(stems)
        finally:
            del imgs
            if 'feats' in locals():
                del feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {'saved': saved, 'skipped': skipped, 'failed': failed, 'dim': last_dim}


# ============================================================
# CELL 8 → Main orchestration
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype  = torch.float16 if (CONFIG['DTYPE'] == 'fp16' and device == 'cuda') else torch.float32
    print(f'Device: {device}  |  dtype: {dtype}')

    transform = build_nih_transform(CONFIG['IMAGE_SIZE'], CONFIG['USE_CLAHE'], CONFIG['USE_RANDOM_FLIP'])

    summary = []

    for model_tag in CONFIG['MODELS_TO_RUN']:
        print(f'\n========== MODEL: {model_tag} ==========')
        try:
            if model_tag == 'dinov2':
                model, model_full_name = load_dinov2(CONFIG['DINOV2_MODEL'], device, dtype, img_size=CONFIG['IMAGE_SIZE'])
            elif model_tag == 'swin':
                model, model_full_name = load_swin(CONFIG['SWIN_MODEL'], device, dtype, img_size=CONFIG['IMAGE_SIZE'])
            else:
                print(f'  Skip: model_tag tidak dikenal ({model_tag})')
                continue
        except RuntimeError as e:
            print(f'  SKIP {model_tag}: {e}')
            continue

        feat_dim = probe_feature_dim(model, device, dtype, CONFIG['IMAGE_SIZE'])
        print(f'  Feature dim: {feat_dim}')

        for scenario in CONFIG['SCENARIOS_TO_RUN']:
            print(f'\n  --- Scenario: {scenario} ---')
            csv_path, image_dir = resolve_scenario(scenario)
            print(f'  csv={csv_path.name}  dir={image_dir}')

            ds = NIHChestXrayDataset(
                csv_path=csv_path,
                image_dir=image_dir,
                transform=transform,
                max_samples=CONFIG['MAX_SAMPLES_PER_SCENARIO'],
            )
            print(f'  Dataset size: {len(ds)}')
            print(f'  Class dist  : {ds.label_distribution()}')

            loader = DataLoader(
                ds,
                batch_size=CONFIG['BATCH_SIZE'],
                shuffle=False,
                num_workers=CONFIG['NUM_WORKERS'],
                pin_memory=(device == 'cuda'),
                drop_last=False,
            )

            out_dir = OUTPUT_ROOT / scenario / f'{model_tag}_base'

            # Cek apakah sudah ada output dari run sebelumnya
            pre_existing = len(list(out_dir.glob('*.npy'))) if out_dir.exists() else 0
            if pre_existing > 0 and CONFIG['SKIP_IF_OUTPUT_EXISTS']:
                print(f'  INFO: {pre_existing} file .npy sudah ada di {out_dir}')
                print(f'        SKIP_IF_OUTPUT_EXISTS=True -> hanya ekstrak yang belum ada.')

            stats = extract_features(model, loader, out_dir, model_tag, dtype, device)
            stats.update({
                'scenario':   scenario,
                'model':      model_tag,
                'model_name': model_full_name,
                'expected_dim': feat_dim,
                'output_dir': str(out_dir),
            })

            total_on_disk = len(list(out_dir.glob('*.npy')))
            if stats['skipped'] > 0 and stats['saved'] == 0:
                print(f'  Result: SEMUA SUDAH ADA (skipped={stats["skipped"]}). Total .npy di disk: {total_on_disk}')
            else:
                print(f'  Result: saved={stats["saved"]} skipped={stats["skipped"]} failed={stats["failed"]} dim={stats["dim"]}')
                print(f'  Total .npy di disk: {total_on_disk}')
            summary.append(stats)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(summary)

summary_df = main()
print('\n========== SUMMARY ==========')
if len(summary_df) > 0:
    print(summary_df.to_string(index=False))
else:
    print('Tidak ada model yang berhasil dijalankan.')


# ============================================================
# CELL 9 → Verification
# ============================================================

def verify_outputs():
    print(f'OUTPUT_ROOT: {OUTPUT_ROOT}\n')
    rows = []
    for scenario in CONFIG['SCENARIOS_TO_RUN']:
        csv_path, _ = resolve_scenario(scenario)
        n_csv = len(pd.read_csv(csv_path))
        for model_tag in CONFIG['MODELS_TO_RUN']:
            out_dir = OUTPUT_ROOT / scenario / f'{model_tag}_base'
            if not out_dir.exists():
                rows.append({'scenario': scenario, 'model': model_tag, 'status': 'MISSING DIR', 'n_files': 0, 'n_csv': n_csv})
                continue
            files = sorted(out_dir.glob('*.npy'))
            zero_byte = [f for f in files if f.stat().st_size == 0]
            sample_shapes = []
            sample_stats = []
            for f in files[:3]:
                v = np.load(f)
                sample_shapes.append(v.shape)
                sample_stats.append((float(v.mean()), float(v.std()), float(v.min()), float(v.max())))
            rows.append({
                'scenario':  scenario,
                'model':     model_tag,
                'n_files':   len(files),
                'n_csv':     n_csv,
                'zero_byte': len(zero_byte),
                'shape':     sample_shapes[0] if sample_shapes else None,
                'sample_mean_std_min_max': sample_stats[0] if sample_stats else None,
                'status':    'OK' if (len(files) > 0 and not zero_byte) else 'CHECK',
            })
    return pd.DataFrame(rows)

verify_df = verify_outputs()
print(verify_df.to_string(index=False))


# ============================================================
# CELL 10 → Sanity-check: load 1 npy per model dari skenario pertama
# ============================================================

_first_scenario = CONFIG['SCENARIOS_TO_RUN'][0]
for tag in CONFIG['MODELS_TO_RUN']:
    d = OUTPUT_ROOT / _first_scenario / f'{tag}_base'
    if not d.exists():
        print(f'[{_first_scenario}/{tag}] direktori tidak ada - SKIP')
        continue
    files = sorted(d.glob('*.npy'))
    if not files:
        print(f'[{_first_scenario}/{tag}] kosong')
        continue
    v = np.load(files[0])
    print(f'[{_first_scenario}/{tag}] {files[0].name}  shape={v.shape}  dtype={v.dtype}  '
          f'finite={np.isfinite(v).all()}  norm={np.linalg.norm(v):.3f}')


# ============================================================
# CELL 11 → Simpan fitur ke CSV (image_name, v0..vN, label)
# Output: {model_tag}_{scenario}_vektor.csv di OUTPUT_ROOT
# ============================================================

print('\n========== GENERATE CSV ==========')
print(f'OUTPUT_ROOT: {OUTPUT_ROOT}\n')

for scenario_name in CONFIG['SCENARIOS_TO_RUN']:
    csv_path, _ = resolve_scenario(scenario_name)
    meta_df = pd.read_csv(csv_path)
    meta_df['Image Index']    = meta_df['Image Index'].astype(str).str.strip()
    meta_df['Finding Labels'] = meta_df['Finding Labels'].astype(str).str.strip()
    stem_to_label = {}
    for _, row in meta_df.iterrows():
        lbl = row['Finding Labels']
        if lbl in CLASS_TO_IDX:
            stem = Path(row['Image Index']).stem
            stem_to_label[stem] = CLASS_TO_IDX[lbl]

    for model_tag in CONFIG['MODELS_TO_RUN']:
        out_dir = OUTPUT_ROOT / scenario_name / f'{model_tag}_base'
        if not out_dir.exists():
            print(f'  [{model_tag}/{scenario_name}] SKIP - direktori tidak ada: {out_dir}')
            continue

        npy_files = sorted(out_dir.glob('*.npy'))
        if not npy_files:
            print(f'  [{model_tag}/{scenario_name}] SKIP - tidak ada file .npy')
            continue

        rows = []
        skipped = 0
        for npy_file in npy_files:
            # Format nama file: {stem}_{model_tag}.npy
            raw_stem = npy_file.stem
            suffix = f'_{model_tag}'
            if raw_stem.endswith(suffix):
                image_stem = raw_stem[:-len(suffix)]
            else:
                image_stem = raw_stem

            if image_stem not in stem_to_label:
                skipped += 1
                continue

            vec = np.load(npy_file).flatten()
            row_dict = {'image_name': npy_file.name}
            for i, val in enumerate(vec):
                row_dict[f'v{i}'] = float(val)
            row_dict['label'] = stem_to_label[image_stem]
            rows.append(row_dict)

        if not rows:
            print(f'  [{model_tag}/{scenario_name}] SKIP - 0 vektor matched')
            continue

        csv_name = f'{model_tag}_{scenario_name}_vektor.csv'
        csv_out = OUTPUT_ROOT / csv_name
        df_out = pd.DataFrame(rows)
        df_out.to_csv(csv_out, index=False)
        n_dims = len([c for c in df_out.columns if c.startswith('v')])
        print(f'  [{model_tag}/{scenario_name}] Saved {len(rows)} rows ({n_dims}-dim) -> {csv_out}')
        if skipped > 0:
            print(f'    (skipped {skipped} - label tidak ditemukan di metadata)')

print('\nCSV generation selesai.')


# ============================================================
# CELL 12 → (Opsional) Bersihkan local cache /content/data/
# ============================================================

if IN_COLAB and CONFIG['COPY_DATA_TO_LOCAL']:
    local = Path(CONFIG['LOCAL_DATA_DIR'])
    if local.exists():
        ans = input(f'Hapus {local}? Ketik "y" untuk konfirmasi: ')
        if ans.strip().lower() == 'y':
            shutil.rmtree(local)
            print(f'Deleted: {local}')
        else:
            print('Skip cleanup.')
