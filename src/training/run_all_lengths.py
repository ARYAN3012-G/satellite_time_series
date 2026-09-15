"""
src/training/run_all_lengths.py  (v3 -- all bugs fixed)
=========================================================
Trains ALL 4 models for EACH sequence length [30, 60, 90] days.
Finds the single best (model + window) combination.

Bugs fixed vs v2:
  1. Sequence builder now receives seq_len directly (no config monkey-patch)
  2. Transformer positional encoding uses actual X.shape[1] (not seq_len param)
  3. Lag label feature added to LSTM and Transformer (as 10th input channel)

Results saved to:
  results/all_lengths_comparison.csv
  results/best_overall_summary.json
  results/30d/  results/60d/  results/90d/

Usage:
    python src/training/run_all_lengths.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config
from preprocessing.run_pipeline import run_preprocessing_pipeline
from features.sequence_builder import build_sequences_all_fields
from utils.logger import get_logger
from models.lstm_model import DroughtLSTM
from models.ml_models import DroughtTransformer, flatten_sequences

logger = get_logger(__name__)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = config.RANDOM_SEED
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU_XGB = torch.cuda.is_available()
logger.info(f"Device: {DEVICE}  |  XGBoost GPU: {USE_GPU_XGB}")

# ── Sequence lengths to evaluate ──────────────────────────────────────────────
SEQUENCE_LENGTHS = [30, 60, 90]


# ==============================================================================
# Utilities
# ==============================================================================

def ensure_preprocessing():
    csvs = list(config.PROCESSED_WEATHER_DIR.glob("*_with_spei.csv"))
    if not csvs:
        logger.info("Processed CSVs not found -- running preprocessing ...")
        run_preprocessing_pipeline(save=True)
    else:
        logger.info(f"Found {len(csvs)} processed CSV(s) -- skipping preprocessing.")


def load_labeled_frames() -> dict:
    frames = {}
    for p in config.PROCESSED_WEATHER_DIR.glob("*_with_spei.csv"):
        name = p.stem.replace("_with_spei", "")
        if name == "pooled":
            continue
        df = pd.read_csv(p, parse_dates=[config.DATE_COLUMN])
        frames[name] = df
    if not frames:
        raise RuntimeError("No processed CSVs found.")
    return frames


def chronological_split(y, meta):
    train_idx, val_idx, test_idx = [], [], []
    for field in meta["field"].unique():
        idx = np.where((meta["field"] == field).values)[0]
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


def apply_scaler(X, mean, std):
    return (X - mean) / std


def compute_class_weights(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    return {int(c): float(total / (len(classes) * cnt))
            for c, cnt in zip(classes, counts)}


def flatten_with_lag(X_flat, prev_label):
    """Append lag label as one extra column for tree models."""
    return np.concatenate(
        [X_flat, prev_label.reshape(-1, 1).astype(np.float32)], axis=1)


def add_lag_channel(X, prev_label):
    """
    Add lag label as a 10th constant channel across all timesteps for
    LSTM / Transformer.  prev_label values (0,1,2) are normalized to
    (0.0, 0.5, 1.0) so they sit in the same range as scaled weather features.
    Returns X_aug with shape (N, seq_len, 10).
    """
    n, seq_len, n_feat = X.shape
    lag_norm = (prev_label.astype(np.float32) / 2.0)           # (N,)
    lag_ch   = np.broadcast_to(
        lag_norm[:, np.newaxis, np.newaxis],
        (n, seq_len, 1)).copy()                                  # (N, T, 1)
    return np.concatenate([X, lag_ch], axis=2)                   # (N, T, 10)


def evaluate(y_true, y_pred, y_proba=None):
    acc      = accuracy_score(y_true, y_pred)
    bal      = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    sev_f1   = f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    hlt_f1   = f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    mod_f1   = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    roc_auc  = None
    if y_proba is not None:
        try:
            y_bin  = label_binarize(y_true, classes=[0, 1, 2])
            roc_auc = round(roc_auc_score(y_bin, y_proba,
                            multi_class="ovr", average="macro"), 4)
        except Exception:
            pass
    return {
        "accuracy":          round(acc,      4),
        "balanced_accuracy": round(bal,      4),
        "macro_f1":          round(macro_f1, 4),
        "roc_auc":           roc_auc,
        "healthy_f1":        round(hlt_f1,   4),
        "moderate_f1":       round(mod_f1,   4),
        "severe_f1":         round(sev_f1,   4),
    }


def print_result(model_name, seq_len, m):
    print(f"  [{seq_len:2d}d] {model_name:26s}  "
          f"Acc={m['accuracy']:.4f}  "
          f"MacroF1={m['macro_f1']:.4f}  "
          f"SevereF1={m['severe_f1']:.4f}  "
          f"ROC={m['roc_auc']}")


# ==============================================================================
# Train: Random Forest
# ==============================================================================
def train_rf(X_tr, y_tr, X_te, y_te, prev_tr, prev_te, cw):
    t0 = time.time()
    Xf_tr = flatten_with_lag(flatten_sequences(X_tr), prev_tr)
    Xf_te = flatten_with_lag(flatten_sequences(X_te), prev_te)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        max_features="sqrt", class_weight=cw, n_jobs=4, random_state=SEED,
    )
    rf.fit(Xf_tr, y_tr)
    y_pred  = rf.predict(Xf_te)
    y_proba = rf.predict_proba(Xf_te)
    m = evaluate(y_te, y_pred, y_proba)
    m["train_time_s"] = round(time.time() - t0, 1)
    return m, rf, y_proba


# ==============================================================================
# Train: XGBoost
# ==============================================================================
def train_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te,
              prev_tr, prev_va, prev_te, cw):
    t0  = time.time()
    dev = "cuda" if USE_GPU_XGB else "cpu"
    Xf_tr = flatten_with_lag(flatten_sequences(X_tr), prev_tr)
    Xf_va = flatten_with_lag(flatten_sequences(X_va), prev_va)
    Xf_te = flatten_with_lag(flatten_sequences(X_te), prev_te)
    sw = np.array([cw[int(c)] for c in y_tr])
    clf = XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=30,
        random_state=SEED, tree_method="hist", device=dev, verbosity=0,
    )
    clf.fit(Xf_tr, y_tr, sample_weight=sw,
            eval_set=[(Xf_va, y_va)], verbose=False)
    y_pred  = clf.predict(Xf_te)
    y_proba = clf.predict_proba(Xf_te)
    m = evaluate(y_te, y_pred, y_proba)
    m["train_time_s"] = round(time.time() - t0, 1)
    return m, clf, y_proba


# ==============================================================================
# Train: PyTorch (LSTM / Transformer)  -- lag label added as 10th channel
# ==============================================================================
def train_pytorch(model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                  prev_tr, prev_va, prev_te, cw,
                  lr=5e-4, batch_size=128, epochs=100, patience=12):
    # Add lag label as extra channel so neural nets also benefit from it
    X_tr_aug = add_lag_channel(X_tr, prev_tr)
    X_va_aug = add_lag_channel(X_va, prev_va)
    X_te_aug = add_lag_channel(X_te, prev_te)

    model = model.to(DEVICE)
    wt = torch.tensor([cw[i] for i in range(3)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=wt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=batch_size, shuffle=shuffle, pin_memory=True,
        )

    tr_loader = make_loader(X_tr_aug, y_tr, shuffle=True)
    va_loader = make_loader(X_va_aug, y_va)
    te_loader = make_loader(X_te_aug, y_te)

    best_f1, best_state, no_impr = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for Xb, yb in va_loader:
                vp.extend(model(Xb.to(DEVICE)).argmax(1).cpu().numpy())
                vt.extend(yb.numpy())
        val_f1 = f1_score(vt, vp, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_impr    = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                logger.info(f"    Early stop at epoch {epoch} (best val_f1={best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for Xb, _ in te_loader:
            p = torch.softmax(model(Xb.to(DEVICE)), dim=1).cpu().numpy()
            probas.extend(p)
            preds.extend(p.argmax(axis=1))

    m = evaluate(y_te, np.array(preds), np.array(probas))
    m["train_time_s"] = round(time.time() - t0, 1)
    return m, model, np.array(probas)


def run_ensemble(y_te, tf_proba, xgb_proba, lstm_proba):
    combined = 0.50 * tf_proba + 0.30 * xgb_proba + 0.20 * lstm_proba
    return evaluate(y_te, combined.argmax(axis=1), combined)


# ==============================================================================
# One full experiment for a given sequence length
# ==============================================================================
def run_for_length(labeled_frames: dict, seq_len: int,
                   save_dir: Path) -> list:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info(f"  SEQUENCE LENGTH: {seq_len} days")
    logger.info(f"{'='*60}")

    # Build sequences — pass seq_len DIRECTLY (no config patching)
    X, y, prev_label, meta = build_sequences_all_fields(
        labeled_frames, sequence_length=seq_len)
    actual_seq_len = X.shape[1]  # ground truth from data
    n_features     = X.shape[2]  # always 9 (weather features)
    logger.info(f"Sequences: X={X.shape}  y={y.shape}")

    # Verify seq_len matches what we asked for
    assert actual_seq_len == seq_len, \
        f"Expected {seq_len}-day sequences but got {actual_seq_len}-day!"

    # Split
    tr, va, te = chronological_split(y, meta)
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]
    pr_tr = prev_label[tr]
    pr_va = prev_label[va]
    pr_te = prev_label[te]

    # Scale
    mean, std = fit_scaler(X_tr)
    X_tr_s = apply_scaler(X_tr, mean, std)
    X_va_s = apply_scaler(X_va, mean, std)
    X_te_s = apply_scaler(X_te, mean, std)

    cw = compute_class_weights(y_tr)
    scaler_dict = {"mean": mean.tolist(), "std": std.tolist()}

    # Print class distribution
    for split_name, ys in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
        classes, counts = np.unique(ys, return_counts=True)
        dist = {config.DROUGHT_CLASSES[int(c)]: f"{cnt}({100*cnt/len(ys):.0f}%)"
                for c, cnt in zip(classes, counts)}
        logger.info(f"  {split_name}: {dist}")

    results = []

    def record(model_name, m):
        row = {"seq_len": seq_len, "model": model_name, **m}
        results.append(row)
        print_result(model_name, seq_len, m)
        return row

    def save_meta(model_name, fname, m, extra=None):
        d = {"model": model_name, "seq_len": seq_len,
             "scaler": scaler_dict, "metrics": m,
             "label_map": config.LABEL_MAP,
             "feature_names": config.MODEL_INPUT_FEATURES,
             "sequence_length": seq_len,
             "use_lag_label": True}
        if extra:
            d.update(extra)
        json.dump(d, open(save_dir / fname, "w"), indent=2)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"\n  Training Random Forest ({seq_len}d) ...")
    m_rf, rf_model, rf_proba = train_rf(
        X_tr_s, y_tr, X_te_s, y_te, pr_tr, pr_te, cw)
    record("Random Forest", m_rf)
    joblib.dump(rf_model, save_dir / "random_forest.joblib")
    save_meta("Random Forest", "random_forest_metadata.json", m_rf)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n  Training XGBoost ({seq_len}d) ...")
    m_xgb, xgb_model, xgb_proba = train_xgb(
        X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
        pr_tr, pr_va, pr_te, cw)
    record("XGBoost", m_xgb)
    joblib.dump(xgb_model, save_dir / "xgboost.joblib")
    save_meta("XGBoost", "xgboost_metadata.json", m_xgb)

    # ── LSTM  (input_size=10: 9 weather + 1 lag channel) ──────────────────────
    print(f"\n  Training LSTM ({seq_len}d) ...")
    lstm = DroughtLSTM(
        input_size=n_features + 1,          # 10 = 9 weather + lag
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        num_classes=3,
        dropout=config.LSTM_DROPOUT,
        bidirectional=True,
    )
    m_lstm, lstm_model, lstm_proba = train_pytorch(
        lstm, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
        pr_tr, pr_va, pr_te, cw,
        lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
    )
    record("LSTM", m_lstm)
    torch.save({k: v.cpu() for k, v in lstm_model.state_dict().items()},
               save_dir / "drought_lstm.pt")
    save_meta("LSTM", "drought_lstm_metadata.json", m_lstm,
              extra={"architecture": {
                  "input_size": n_features + 1,
                  "hidden_size": config.LSTM_HIDDEN_SIZE,
                  "num_layers": config.LSTM_NUM_LAYERS,
                  "num_classes": 3, "dropout": config.LSTM_DROPOUT,
                  "bidirectional": True}})

    # ── Transformer  (input_size=10, seq_len from actual data shape) ───────────
    print(f"\n  Training Transformer ({seq_len}d) ...")
    transformer = DroughtTransformer(
        input_size=n_features + 1,          # 10 = 9 weather + lag
        d_model=128, nhead=4,
        num_layers=3, dim_feedforward=256, dropout=0.2,
        num_classes=3,
        seq_len=actual_seq_len,             # uses real data shape -- no mismatch
    )
    m_tf, tf_model, tf_proba = train_pytorch(
        transformer, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
        pr_tr, pr_va, pr_te, cw,
        lr=5e-4, batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
    )
    record("Transformer", m_tf)
    torch.save({k: v.cpu() for k, v in tf_model.state_dict().items()},
               save_dir / "transformer.pt")
    save_meta("Transformer", "transformer_metadata.json", m_tf,
              extra={"architecture": {
                  "input_size": n_features + 1, "d_model": 128,
                  "nhead": 4, "num_layers": 3,
                  "dim_feedforward": 256, "num_classes": 3,
                  "seq_len": actual_seq_len}})

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print(f"\n  Ensemble ({seq_len}d): Transformer×50% + XGBoost×30% + LSTM×20% ...")
    m_ens = run_ensemble(y_te, tf_proba, xgb_proba, lstm_proba)
    record("Ensemble", m_ens)
    save_meta("Ensemble", "ensemble_metadata.json", m_ens,
              extra={"weights": {"Transformer": 0.50, "XGBoost": 0.30, "LSTM": 0.20}})

    return results


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "=" * 65)
    print("  DROUGHT DETECTION -- ALL SEQUENCE LENGTHS COMPARISON  (v3)")
    print(f"  Lengths: {SEQUENCE_LENGTHS} days  |  Device: {DEVICE}")
    print(f"  Lag label added to ALL models (weather → 10 features for NN)")
    print("=" * 65)

    ensure_preprocessing()
    labeled_frames = load_labeled_frames()
    logger.info(f"Loaded {len(labeled_frames)} fields: {sorted(labeled_frames.keys())}")

    all_results = []
    for seq_len in SEQUENCE_LENGTHS:
        save_dir = RESULTS_DIR / f"{seq_len}d"
        try:
            rows = run_for_length(labeled_frames, seq_len, save_dir)
            all_results.extend(rows)
        except Exception as e:
            logger.error(f"FAILED for seq_len={seq_len}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("No results produced. Check errors above.")
        sys.exit(1)

    # Full comparison table
    df = pd.DataFrame(all_results)
    col_order = ["seq_len", "model", "accuracy", "balanced_accuracy",
                 "macro_f1", "roc_auc", "healthy_f1", "moderate_f1",
                 "severe_f1", "train_time_s"]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(RESULTS_DIR / "all_lengths_comparison.csv", index=False)

    # Print per-length summaries
    print("\n" + "=" * 90)
    print("  FULL COMPARISON -- ALL LENGTHS x ALL MODELS")
    print("=" * 90)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    for seq_len in SEQUENCE_LENGTHS:
        subset = df[df["seq_len"] == seq_len].drop(columns=["seq_len"])
        print(f"\n--- {seq_len}-day window ---")
        print(subset.to_string(index=False))

    # Best per length
    print("\n" + "=" * 65)
    print("  BEST MODEL PER SEQUENCE LENGTH")
    print("=" * 65)
    for seq_len in SEQUENCE_LENGTHS:
        sub = df[df["seq_len"] == seq_len]
        best = sub.loc[sub["macro_f1"].idxmax()]
        print(f"  {seq_len:2d}d  Best: {best['model']:26s}"
              f"  MacroF1={best['macro_f1']:.4f}"
              f"  SevereF1={best['severe_f1']:.4f}"
              f"  Acc={best['accuracy']:.4f}")

    # Overall best
    best_row   = df.loc[df["macro_f1"].idxmax()]
    best_model = best_row["model"]
    best_len   = int(best_row["seq_len"])
    print(f"\n{'='*65}")
    print(f"  >>> OVERALL BEST: {best_model} with {best_len}-day window")
    print(f"  >>> Macro F1   = {best_row['macro_f1']:.4f}")
    print(f"  >>> Severe F1  = {best_row['severe_f1']:.4f}")
    print(f"  >>> Accuracy   = {best_row['accuracy']:.4f}")
    print(f"{'='*65}")

    summary = {
        "best_model":       best_model,
        "best_seq_len":     best_len,
        "best_macro_f1":    float(best_row["macro_f1"]),
        "best_severe_f1":   float(best_row["severe_f1"]),
        "best_accuracy":    float(best_row["accuracy"]),
        "all_lengths_tested": SEQUENCE_LENGTHS,
        "full_results":     df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "best_overall_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to:  results/all_lengths_comparison.csv")
    print(f"Best model summary:    results/best_overall_summary.json")
    for seq_len in SEQUENCE_LENGTHS:
        print(f"Models for {seq_len}d:         results/{seq_len}d/")


if __name__ == "__main__":
    main()
