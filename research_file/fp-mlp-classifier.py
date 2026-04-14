# ============================================================
# CELL 0 — Imports + Config
# ============================================================

import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────
# Root direktori feature CSVs
CSV_ROOT_BASELINE = Path('/content/drive/MyDrive/FP_Admin_KCV/fp/raw_features/fe_vig_dinov2')
CSV_ROOT_MEDICAL  = Path('/content/drive/MyDrive/FP_Admin_KCV/fp/raw_features/kaggle_dual_scenario')

# Checkpoint output
CKPT_DIR = Path('/content/drive/MyDrive/FP_Admin_KCV/fp/checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: (fe_tag, split_tag) → CSV path
CSV_MAP = {
    ('dinov2',      'balanced'):   CSV_ROOT_BASELINE / 'dinov2_balanced_vektor.csv',
    ('dinov2',      'imbalanced'): CSV_ROOT_BASELINE / 'dinov2_imbalanced_vektor.csv',
    ('swin',        'balanced'):   CSV_ROOT_BASELINE / 'swin_balanced_vektor.csv',
    ('swin',        'imbalanced'): CSV_ROOT_BASELINE / 'swin_imbalanced_vektor.csv',
    ('maxvit',      'balanced'):   CSV_ROOT_BASELINE / 'features_maxvit_balanced.csv',
    ('maxvit',      'imbalanced'): CSV_ROOT_BASELINE / 'features_maxvit_imbalanced.csv',
    ('convnextv2',  'balanced'):   CSV_ROOT_BASELINE / 'features_convnextv2_balanced.csv',
    ('convnextv2',  'imbalanced'): CSV_ROOT_BASELINE / 'features_convnextv2_imbalanced.csv',
    ('medical',     'balanced'):   CSV_ROOT_MEDICAL  / 'medical_balanced_vektor_timestep_10.csv',
    ('medical',     'imbalanced'): CSV_ROOT_MEDICAL  / 'medical_imbalanced_vektor_timestep_10.csv',
}

# Semua FE yang dianalisis
ALL_FE = ['dinov2', 'swin', 'maxvit', 'convnextv2', 'medical']

# ── Hyperparameters ───────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY= 1e-4
MLP_HIDDEN  = 512
DROPOUT     = 0.3
NUM_CLASSES = 6
VAL_RATIO   = 0.15   # 15% val
TEST_RATIO  = 0.15   # 15% test (dari keseluruhan)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


# ============================================================
# CELL 1 — Dataset & Data Utilities
# ============================================================

class FeatureDataset(Dataset):
    """Dataset dari pre-extracted feature vectors (CSV rows)."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_csv_splits(csv_path: Path):
    """Baca CSV dan split stratified 70/15/15 → (train, val, test) dataset + class_weights tensor."""
    df = pd.read_csv(csv_path)

    # Kolom fitur: semua kolom numerik selain 'label'
    # (handles berbagai nama kolom filename: 'image_name', 'filename', 'image', dst.)
    non_feat = {'label'}
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in non_feat]
    X = df[feat_cols].values.astype(np.float32)

    # Encode string labels → integer indices jika belum numerik
    raw_labels = df['label'].values
    if not np.issubdtype(raw_labels.dtype, np.integer):
        classes = sorted(set(raw_labels))
        label_map = {cls: i for i, cls in enumerate(classes)}
        print(f'  Label encoding: {label_map}')
        raw_labels = np.array([label_map[v] for v in raw_labels])
    y = raw_labels.astype(np.int64)

    # Split: 70 train | 15 val | 15 test (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
    )

    # Class weights dari training split saja
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)   # hindari div-by-zero
    weights = len(y_train) / (NUM_CLASSES * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    train_ds = FeatureDataset(X_train, y_train)
    val_ds   = FeatureDataset(X_val,   y_val)
    test_ds  = FeatureDataset(X_test,  y_test)

    print(f'  Split → train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}')
    print(f'  feat_dim={X.shape[1]}  |  class_weights={weights.round(3)}')

    return train_ds, val_ds, test_ds, class_weights, int(X.shape[1])


# ============================================================
# CELL 2 — MLP Classifier
# ============================================================

class MLPClassifier(nn.Module):
    """
    LayerNorm → Linear(feat_dim, MLP_HIDDEN) → GELU → Dropout → Linear(MLP_HIDDEN, num_classes)
    Mirip head dari XRayClassifier di notebook, tapi tanpa timm backbone
    karena fitur sudah di-ekstrak sebelumnya.
    """
    def __init__(self, feat_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# CELL 3 — Feature Space Augmentation (FSA)
# ============================================================

def apply_fsa(x: torch.Tensor, y: torch.Tensor, num_classes: int = NUM_CLASSES):
    """
    Feature Space Augmentation — diterapkan hanya saat training.
    Returns augmented (x, y_soft).

    Techniques:
      1. Gaussian noise injection
      2. Feature dropout mask (10% dims → 0)
      3. Mixup in feature space
    """
    B = x.size(0)

    # 1. Gaussian noise
    x = x + torch.randn_like(x) * 0.01

    # 2. Feature dropout mask
    mask = (torch.rand_like(x) > 0.10).float()
    x = x * mask

    # 3. Mixup
    alpha = 0.2
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(B, device=x.device)
    x     = lam * x + (1 - lam) * x[idx]

    # One-hot encode labels → soft labels setelah mixup
    y_onehot = torch.zeros(B, num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    y_mix_onehot = lam * y_onehot + (1 - lam) * y_onehot[idx]

    return x, y_mix_onehot   # y_mix_onehot untuk soft CrossEntropy


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropy dengan soft (mixup) targets."""
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ============================================================
# CELL 4 — Training & Evaluation Loop
# ============================================================

def train_epoch(model, loader, optimizer, criterion, use_aug: bool = False):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        if use_aug:
            x_aug, y_soft = apply_fsa(x, y)
            logits = model(x_aug)
            loss   = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss   = criterion(logits, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader):
    """Returns (acc, f1_macro, auc_ovr, all_preds, all_labels, all_probs)."""
    model.eval()
    all_logits, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(dim=1).numpy()

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    return acc, f1, auc, preds, labels, probs


# ============================================================
# CELL 5 — run_experiment: wrapper untuk 1 FE × 1 scenario
# ============================================================

def run_experiment(fe_tag: str, split_tag: str, scenario_num: int,
                   use_aug: bool = False) -> dict:
    """
    Load CSV → split → train MLP → evaluate test set.
    Simpan best checkpoint berdasarkan val F1-macro.

    Returns dict dengan kolom untuk summary table.
    """
    csv_path = CSV_MAP.get((fe_tag, split_tag))
    if csv_path is None or not csv_path.exists():
        print(f'  [SKIP] CSV tidak ditemukan: {csv_path}')
        return {}

    aug_label = '+aug' if use_aug else ''
    exp_tag   = f'sc{scenario_num}_{fe_tag}_{split_tag}{aug_label}'
    ckpt_path = CKPT_DIR / f'{exp_tag}_best.pt'

    print(f'\n{"=" * 55}')
    print(f'  Scenario {scenario_num} | FE={fe_tag}  |  split={split_tag}  |  aug={use_aug}')
    print(f'{"=" * 55}')

    train_ds, val_ds, test_ds, class_weights, feat_dim = load_csv_splits(csv_path)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model     = MLPClassifier(feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_epoch  = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, use_aug)
        scheduler.step()
        val_acc, val_f1, val_auc, _, _, _ = eval_epoch(model, val_loader)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f'  Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  '
                  f'val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  val_auc={val_auc:.4f}')

    # Load best checkpoint → evaluate on test
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_acc, test_f1, test_auc, _, _, _ = eval_epoch(model, test_loader)

    print(f'\n  >> Best epoch: {best_epoch}  |  '
          f'Test Acc={test_acc:.4f}  F1={test_f1:.4f}  AUC={test_auc:.4f}')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'Scenario':         scenario_num,
        'Model':            fe_tag,
        'Dataset':          f'{split_tag}{aug_label}',
        'Best Epoch':       best_epoch,
        'Test Acc':         round(test_acc, 4),
        'Test F1 (macro)':  round(test_f1,  4),
        'Test AUC (OvR)':   round(test_auc, 4),
    }


# ============================================================
# CELL 6 — Scenario 1: Balanced, semua FE, tanpa augmentasi
# ============================================================
# CSV balanced untuk: dinov2, swin, maxvit, convnextv2, medical
# Tidak ada augmentasi.

results = []

print('\n\n########## SCENARIO 1 — Balanced, No Augmentation ##########')
for fe in ALL_FE:
    r = run_experiment(fe_tag=fe, split_tag='balanced', scenario_num=1, use_aug=False)
    if r:
        results.append(r)


# ============================================================
# CELL 7 — Scenario 2: Balanced + Feature Space Augmentation
# ============================================================
# Semua FE pada dataset balanced, dengan FSA (Gaussian noise +
# feature dropout mask + Mixup) diterapkan saat training.

print('\n\n########## SCENARIO 2 — Balanced + Feature Space Augmentation ##########')
for fe in ALL_FE:
    r = run_experiment(fe_tag=fe, split_tag='balanced', scenario_num=2, use_aug=True)
    if r:
        results.append(r)


# ============================================================
# CELL 8 — Scenario 3: Imbalanced, semua FE, tanpa augmentasi
# ============================================================
# Semua FE pada dataset imbalanced.
# Class weights menangani skew No Finding:Lainnya ≈ 1500:200.

print('\n\n########## SCENARIO 3 — Imbalanced, No Augmentation ##########')
for fe in ALL_FE:
    r = run_experiment(fe_tag=fe, split_tag='imbalanced', scenario_num=3, use_aug=False)
    if r:
        results.append(r)


# ============================================================
# CELL 9 — Scenario 4: Imbalanced + Feature Space Augmentation
# ============================================================
# Semua FE pada dataset imbalanced, ditambah FSA saat training.

print('\n\n########## SCENARIO 4 — Imbalanced + Feature Space Augmentation ##########')
for fe in ALL_FE:
    r = run_experiment(fe_tag=fe, split_tag='imbalanced', scenario_num=4, use_aug=True)
    if r:
        results.append(r)


# ============================================================
# CELL 10 — Summary Table
# ============================================================

summary_df = pd.DataFrame(results, columns=[
    'Scenario', 'Model', 'Dataset', 'Best Epoch',
    'Test Acc', 'Test F1 (macro)', 'Test AUC (OvR)'
])

# Highlight: hijau = terbaik per kolom metrik, merah = terburuk
def highlight_minmax(s):
    styles = [''] * len(s)
    if s.dtype in [float, np.float64]:
        max_idx = s.idxmax()
        min_idx = s.idxmin()
        styles[max_idx] = 'background-color: #c6efce'  # hijau
        styles[min_idx] = 'background-color: #ffc7ce'  # merah
    return styles

print('\n\n========== SUMMARY ==========')
print(summary_df.to_string(index=True))

# Styled (untuk Jupyter display)
try:
    display(
        summary_df.style
        .apply(highlight_minmax, subset=['Test Acc', 'Test F1 (macro)', 'Test AUC (OvR)'])
        .format({'Test Acc': '{:.4f}', 'Test F1 (macro)': '{:.4f}', 'Test AUC (OvR)': '{:.4f}'})
    )
except NameError:
    pass  # bukan Jupyter, sudah print di atas

# Simpan ke CSV
out_summary = CKPT_DIR / 'classification_summary.csv'
summary_df.to_csv(out_summary, index=False)
print(f'\nSummary disimpan: {out_summary}')
