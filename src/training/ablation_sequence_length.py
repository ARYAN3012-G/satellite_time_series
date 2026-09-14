"""
src/training/ablation_sequence_length.py
=========================================
Runs the Transformer model for each sequence length in
config.SEQUENCE_LENGTH_ABLATION (7, 14, 30, 60, 90 days).

Produces: results/ablation_sequence_length.csv

Usage on HPC (SLURM):
    python src/training/ablation_sequence_length.py

This is designed to run AFTER preprocessing is done
(i.e., data/processed/weather/ CSV files exist).
It rebuilds sequences fresh for each window size.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config
from features.sequence_builder import build_sequences_for_field, build_sequences_all_fields
from utils.logger import get_logger
from models.ml_models import DroughtTransformer

logger = get_logger(__name__)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = config.RANDOM_SEED
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_labeled_frames():
    """Load all field processed CSVs."""
    import glob
    labeled_frames = {}
    for csv_path in glob.glob(str(config.PROCESSED_WEATHER_DIR / "*_with_spei.csv")):
        field_name = Path(csv_path).stem.replace("_with_spei", "")
        if field_name == "pooled":
            continue
        df = pd.read_csv(csv_path, parse_dates=[config.DATE_COLUMN])
        labeled_frames[field_name] = df
    return labeled_frames


def build_sequences_for_length(labeled_frames, seq_len):
    """Build (N, seq_len, 9) sequences for a given window size."""
    import importlib
    # Temporarily patch config.SEQUENCE_LENGTH_DAYS
    orig = config.SEQUENCE_LENGTH_DAYS
    config.SEQUENCE_LENGTH_DAYS = seq_len
    try:
        X, y, prev_label, meta = build_sequences_all_fields(labeled_frames)
    finally:
        config.SEQUENCE_LENGTH_DAYS = orig
    return X, y, prev_label, meta


def chronological_split(X, y, meta):
    train_idx, val_idx, test_idx = [], [], []
    for field in meta["field"].unique():
        mask = (meta["field"] == field).values
        idx = np.where(mask)[0]
        n = len(idx)
        n_train = int(n * config.TRAIN_FRACTION)
        n_val   = int(n * config.VAL_FRACTION)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train: n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def fit_scaler(X_train):
    mean = X_train.mean(axis=(0, 1))
    std  = X_train.std(axis=(0, 1))
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std


def train_transformer_quick(X_tr, y_tr, X_va, y_va, X_te, y_te,
                             class_weights, seq_len):
    """Train Transformer for up to 60 epochs with early stopping."""
    n_features = X_tr.shape[2]
    model = DroughtTransformer(
        input_size=n_features, d_model=128, nhead=4,
        num_layers=3, dim_feedforward=256, dropout=0.2,
        num_classes=3, seq_len=seq_len,
    ).to(DEVICE)

    weight_t  = torch.tensor([class_weights[i] for i in range(3)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=128, shuffle=shuffle,
        )

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va)
    test_loader  = make_loader(X_te, y_te)

    best_val_f1, best_state, no_impr = 0.0, None, 0
    patience = 12

    for epoch in range(1, 61):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                logits = model(Xb.to(DEVICE))
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(yb.numpy())
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_proba = [], []
    with torch.no_grad():
        for Xb, _ in test_loader:
            probs = torch.softmax(model(Xb.to(DEVICE)), dim=1).cpu().numpy()
            test_proba.extend(probs)
            test_preds.extend(probs.argmax(axis=1))

    test_preds = np.array(test_preds)
    test_proba = np.array(test_proba)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    acc      = accuracy_score(y_te, test_preds)
    bal      = balanced_accuracy_score(y_te, test_preds)
    mf1      = f1_score(y_te, test_preds, average="macro", zero_division=0)
    sev_f1   = f1_score(y_te, test_preds, labels=[2], average="macro", zero_division=0)
    try:
        y_bin   = label_binarize(y_te, classes=[0, 1, 2])
        roc_auc = roc_auc_score(y_bin, test_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None
    return acc, bal, mf1, sev_f1, roc_auc


def main():
    print("\n" + "=" * 60)
    print("  ABLATION: Sequence Length vs Accuracy (Transformer)")
    print(f"  Lengths to test: {config.SEQUENCE_LENGTH_ABLATION}")
    print("=" * 60)

    labeled_frames = load_labeled_frames()
    if not labeled_frames:
        print("ERROR: No processed CSVs found. Run preprocessing first.")
        sys.exit(1)

    # Compute class weights from first full build
    X_full, y_full, _, meta_full = build_sequences_for_length(
        labeled_frames, config.SEQUENCE_LENGTH_ABLATION[0])
    train_idx, _, _ = chronological_split(X_full, y_full, meta_full)
    y_tr0 = y_full[train_idx]
    classes, counts = np.unique(y_tr0, return_counts=True)
    total = len(y_tr0)
    class_weights = {int(c): float(total / (len(classes) * cnt))
                     for c, cnt in zip(classes, counts)}

    records = []
    for seq_len in config.SEQUENCE_LENGTH_ABLATION:
        logger.info(f"\n--- Testing seq_len={seq_len} days ---")
        try:
            X, y, prev_label, meta = build_sequences_for_length(labeled_frames, seq_len)
            tr, va, te = chronological_split(X, y, meta)

            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]
            X_te, y_te = X[te], y[te]

            mean, std = fit_scaler(X_tr)
            X_tr_s = (X_tr - mean) / std
            X_va_s = (X_va - mean) / std
            X_te_s = (X_te - mean) / std

            acc, bal, mf1, sev_f1, roc_auc = train_transformer_quick(
                X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te, class_weights, seq_len)

            records.append({
                "seq_len_days": seq_len,
                "n_sequences": len(y),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal, 4),
                "macro_f1": round(mf1, 4),
                "severe_f1": round(sev_f1, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc else None,
            })
            logger.info(f"  seq_len={seq_len}d -> acc={acc:.4f} macro_f1={mf1:.4f} severe_f1={sev_f1:.4f}")
        except Exception as e:
            logger.error(f"  seq_len={seq_len}d FAILED: {e}")
            records.append({"seq_len_days": seq_len, "error": str(e)})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "ablation_sequence_length.csv", index=False)
    print(f"\n--- Ablation Results ---")
    print(df.to_string(index=False))
    print(f"\nResults saved to results/ablation_sequence_length.csv")

    # Best window
    if "macro_f1" in df.columns:
        best = df.loc[df["macro_f1"].idxmax()]
        print(f"\n>>> BEST window: {int(best['seq_len_days'])} days "
              f"(Macro F1 = {best['macro_f1']:.4f}, Severe F1 = {best['severe_f1']:.4f})")


if __name__ == "__main__":
    main()
