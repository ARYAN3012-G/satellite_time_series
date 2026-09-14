"""
src/training/train_all.py  (v2 -- HPC-ready)
=============================================
Full pipeline:  preprocessing -> sequences -> train 4 models + ensemble
                -> LOFO -> ablation ready -> save all results

Key improvements over v1:
  1. 60-day window (monsoon context)
  2. Lag label feature (prev day's drought class -> captures 94.8% autocorrelation)
  3. LSTM: smaller hidden (64), higher dropout (0.45), lower LR (5e-4)
  4. Ensemble: Transformer 50% + XGBoost 30% + LSTM 20% (weighted softmax voting)
  5. Proper early stopping on val Macro F1 (not loss)
  6. XGBoost and LOFO use GPU when NVIDIA detected
  7. Full metrics: accuracy, balanced_acc, macro_f1, per-class f1, roc_auc

Usage (laptop):
    python src/training/train_all.py

Usage (HPC -- SLURM):
    sbatch hpc_train.sh        (see hpc_train.sh in repo root)
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
    classification_report, confusion_matrix,
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
from features.sequence_builder import build_sequences_all_fields, save_sequences
from utils.logger import get_logger
from models.lstm_model import DroughtLSTM
from models.ml_models import DroughtTransformer, flatten_sequences

logger = get_logger(__name__)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = config.RANDOM_SEED
torch.manual_seed(SEED)
np.random.seed(SEED)

# Auto-detect GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU_XGB = torch.cuda.is_available()
logger.info(f"Device: {DEVICE}  |  XGBoost GPU: {USE_GPU_XGB}")


# ==============================================================================
# 1. Preprocessing
# ==============================================================================
def step_preprocessing():
    seq_path = config.SEQUENCES_DIR / "pooled_sequences.npz"
    if seq_path.exists():
        logger.info("Sequences already exist -- skipping preprocessing.")
        return
    run_preprocessing_pipeline(save=True)


# ==============================================================================
# 2. Load sequences + lag label
# ==============================================================================
def step_load_sequences():
    seq_path = config.SEQUENCES_DIR / "pooled_sequences.npz"

    if not seq_path.exists():
        logger.info("Building sequences from processed CSVs ...")
        import glob
        labeled_frames = {}
        for csv_path in glob.glob(str(config.PROCESSED_WEATHER_DIR / "*_with_spei.csv")):
            field_name = Path(csv_path).stem.replace("_with_spei", "")
            if field_name == "pooled":
                continue
            df = pd.read_csv(csv_path, parse_dates=[config.DATE_COLUMN])
            labeled_frames[field_name] = df
        X, y, prev_label, meta = build_sequences_all_fields(labeled_frames)
        save_sequences(X, y, meta, seq_path, prev_label=prev_label)
    else:
        logger.info(f"Loading sequences from {seq_path} ...")

    data = np.load(seq_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    prev_label = data["prev_label"] if "prev_label" in data else None
    fields = data["field"]
    dates = data["target_date"]
    meta = pd.DataFrame({"field": fields, "target_date": pd.to_datetime(dates)})

    logger.info(f"X={X.shape}, y={y.shape}, lag_label={'yes' if prev_label is not None else 'no'}")
    return X, y, prev_label, meta


# ==============================================================================
# 3. Chronological per-field split
# ==============================================================================
def chronological_split(X, y, meta):
    train_idx, val_idx, test_idx = [], [], []
    for field in meta["field"].unique():
        mask = (meta["field"] == field).values
        idx = np.where(mask)[0]
        n = len(idx)
        n_train = int(n * config.TRAIN_FRACTION)
        n_val = int(n * config.VAL_FRACTION)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train: n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    logger.info(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# ==============================================================================
# 4. Scaler (train only)
# ==============================================================================
def fit_scaler(X_train):
    mean = X_train.mean(axis=(0, 1))
    std  = X_train.std(axis=(0, 1))
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std

def apply_scaler(X, mean, std):
    return (X - mean) / std


# ==============================================================================
# 5. Class weights
# ==============================================================================
def compute_class_weights(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = {int(c): float(total / (len(classes) * cnt))
               for c, cnt in zip(classes, counts)}
    logger.info(f"Class weights: {weights}")
    return weights


# ==============================================================================
# 6. Evaluation helper
# ==============================================================================
def evaluate(y_true, y_pred, y_proba=None, split_name="Test"):
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report   = classification_report(y_true, y_pred,
                   target_names=config.DROUGHT_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "split": split_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        **{f"{cls.lower()}_{k}": round(report[cls][k], 4)
           for cls in config.DROUGHT_CLASSES
           for k in ("precision", "recall", "f1-score")},
        "confusion_matrix": cm.tolist(),
    }
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            metrics["roc_auc"] = round(
                roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"), 4)
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def print_metrics(name, metrics):
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {name} -- {metrics['split']} Results")
    print(sep)
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:           {metrics.get('roc_auc', 'N/A')}")
    for cls in config.DROUGHT_CLASSES:
        key = cls.lower()
        print(f"  {cls:10s} F1:      {metrics[f'{key}_f1-score']:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    print("  Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"    {config.DROUGHT_CLASSES[i]:10s}: {row}")


# ==============================================================================
# 7. Flatten sequences + lag label for tree models
# ==============================================================================
def flatten_with_lag(X_flat, prev_label):
    """Append the lag label as a single extra column to the flat features."""
    return np.concatenate([X_flat, prev_label.reshape(-1, 1).astype(np.float32)], axis=1)


# ==============================================================================
# 8. Baselines
# ==============================================================================
def run_baselines(y_train, y_test, test_idx):
    # Majority class
    classes, counts = np.unique(y_train, return_counts=True)
    majority = classes[np.argmax(counts)]
    maj_pred = np.full_like(y_test, majority)
    maj_m = evaluate(y_test, maj_pred, split_name="Test")

    # Persistence (yesterday's label as today's prediction)
    valid = test_idx > 0
    pers_pred = np.where(valid, y_test[np.arange(len(test_idx)) - 1 + (test_idx > 0).astype(int) * 0], y_test)
    # Simpler: use test labels shifted by 1
    pers_pred = np.concatenate([[y_test[0]], y_test[:-1]])
    pers_m = evaluate(y_test, pers_pred, split_name="Test")

    print(f"\n--- Baselines ---")
    print(f"  Majority Class:  Acc={maj_m['accuracy']:.4f}  Macro F1={maj_m['macro_f1']:.4f}")
    print(f"  Persistence:     Acc={pers_m['accuracy']:.4f}  Macro F1={pers_m['macro_f1']:.4f}")
    return maj_m, pers_m


# ==============================================================================
# 9. Random Forest
# ==============================================================================
def train_random_forest(X_tr, y_tr, X_te, y_te, prev_tr, prev_te, class_weights, mean, std):
    logger.info("Training Random Forest ...")
    t0 = time.time()
    X_tr_f = flatten_sequences(X_tr)
    X_te_f = flatten_sequences(X_te)
    if config.USE_LAG_LABEL_FEATURE:
        X_tr_f = flatten_with_lag(X_tr_f, prev_tr)
        X_te_f = flatten_with_lag(X_te_f, prev_te)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        max_features="sqrt", class_weight=class_weights,
        n_jobs=4, random_state=SEED,
    )
    rf.fit(X_tr_f, y_tr)
    y_pred  = rf.predict(X_te_f)
    y_proba = rf.predict_proba(X_te_f)
    metrics = evaluate(y_te, y_pred, y_proba, "Test")
    metrics["train_time_s"] = round(time.time() - t0, 1)
    print_metrics("Random Forest", metrics)

    joblib.dump(rf, config.MODELS_SAVED_DIR / "random_forest.joblib")
    _save_meta("Random Forest", "random_forest_metadata.json",
               metrics, mean, std, extra={"n_features_flat": X_tr_f.shape[1]})
    return metrics, rf


# ==============================================================================
# 10. XGBoost
# ==============================================================================
def train_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te,
                  prev_tr, prev_va, prev_te, class_weights, mean, std):
    logger.info("Training XGBoost ...")
    t0 = time.time()
    X_tr_f = flatten_sequences(X_tr)
    X_va_f = flatten_sequences(X_va)
    X_te_f = flatten_sequences(X_te)
    if config.USE_LAG_LABEL_FEATURE:
        X_tr_f = flatten_with_lag(X_tr_f, prev_tr)
        X_va_f = flatten_with_lag(X_va_f, prev_va)
        X_te_f = flatten_with_lag(X_te_f, prev_te)

    sample_w = np.array([class_weights[int(c)] for c in y_tr])
    device   = "cuda" if USE_GPU_XGB else "cpu"

    xgb = XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=30,
        random_state=SEED, tree_method="hist", device=device, verbosity=0,
    )
    xgb.fit(X_tr_f, y_tr, sample_weight=sample_w,
            eval_set=[(X_va_f, y_va)], verbose=False)

    y_pred  = xgb.predict(X_te_f)
    y_proba = xgb.predict_proba(X_te_f)
    metrics = evaluate(y_te, y_pred, y_proba, "Test")
    metrics["train_time_s"] = round(time.time() - t0, 1)
    print_metrics("XGBoost", metrics)

    joblib.dump(xgb, config.MODELS_SAVED_DIR / "xgboost.joblib")
    _save_meta("XGBoost", "xgboost_metadata.json", metrics, mean, std)
    return metrics, xgb


# ==============================================================================
# 11. PyTorch training loop (shared: LSTM + Transformer)
# ==============================================================================
def train_pytorch_model(model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                        prev_tr, prev_va, prev_te,
                        class_weights, model_name, model_file, meta_file,
                        mean, std, lr=5e-4, batch_size=128,
                        epochs=100, patience=12):
    model = model.to(DEVICE)
    weight_t = torch.tensor([class_weights[i] for i in range(3)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def make_loader(X, y, prev, shuffle=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        if config.USE_LAG_LABEL_FEATURE and prev is not None:
            pt = torch.tensor(prev, dtype=torch.float32).unsqueeze(1)
            return DataLoader(TensorDataset(Xt, yt, pt), batch_size=batch_size, shuffle=shuffle)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_tr, y_tr, prev_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va, prev_va)
    test_loader  = make_loader(X_te, y_te, prev_te)

    best_val_f1    = 0.0
    epochs_no_impr = 0
    best_state     = None
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(yb)
        scheduler.step()
        train_loss = running_loss / len(y_tr)

        model.eval()
        val_loss, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(yb.cpu().numpy())
        val_loss /= len(y_va)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        train_losses.append(round(train_loss, 4))
        val_losses.append(round(val_loss, 4))

        if epoch % 10 == 0:
            logger.info(f"  [{model_name}] Epoch {epoch:3d}: "
                        f"train={train_loss:.4f}  val={val_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                logger.info(f"  [{model_name}] Early stop epoch {epoch} (best val_f1={best_val_f1:.4f})")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    test_preds, test_proba = [], []
    with torch.no_grad():
        for batch in test_loader:
            Xb = batch[0].to(DEVICE)
            probs = torch.softmax(model(Xb), dim=1).cpu().numpy()
            test_proba.extend(probs)
            test_preds.extend(probs.argmax(axis=1))

    test_preds  = np.array(test_preds)
    test_proba  = np.array(test_proba)
    metrics = evaluate(y_te, test_preds, test_proba, "Test")
    metrics["train_time_s"] = round(train_time, 1)
    print_metrics(model_name, metrics)

    torch.save(best_state, config.MODELS_SAVED_DIR / model_file)
    arch_attrs = ["input_size", "hidden_size", "num_layers", "num_classes", "d_model", "nhead", "bidirectional"]
    arch = {a: getattr(model, a, None) for a in arch_attrs if getattr(model, a, None) is not None}
    meta_extra = {"architecture": arch,
                  "training_history": {"train_losses": train_losses, "val_losses": val_losses}}
    _save_meta(model_name, meta_file, metrics, mean, std, extra=meta_extra)
    return metrics, model, test_proba


def _save_meta(model_name, meta_file, metrics, mean, std, extra=None):
    meta = {
        "model": model_name,
        "feature_names": config.MODEL_INPUT_FEATURES,
        "sequence_length": config.SEQUENCE_LENGTH_DAYS,
        "use_lag_label": config.USE_LAG_LABEL_FEATURE,
        "label_map": config.LABEL_MAP,
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "metrics_test": metrics,
    }
    if extra:
        meta.update(extra)
    with open(config.MODELS_SAVED_DIR / meta_file, "w") as f:
        json.dump(meta, f, indent=2)


# ==============================================================================
# 12. Ensemble (weighted softmax voting)
# ==============================================================================
def run_ensemble(y_te, probas: dict) -> dict:
    """Combine Transformer, XGBoost, LSTM probabilities with weights."""
    weights = {"Transformer": 0.50, "XGBoost": 0.30, "LSTM": 0.20}
    combined = sum(w * probas[m] for m, w in weights.items() if m in probas)
    y_pred = combined.argmax(axis=1)
    metrics = evaluate(y_te, y_pred, combined, "Test")
    print_metrics("Ensemble (TF50+XGB30+LSTM20)", metrics)
    return metrics


# ==============================================================================
# 13. LOFO
# ==============================================================================
def run_lofo(X, y, prev_label, meta, mean, std, class_weights):
    fields = sorted(meta["field"].unique())
    records = []
    device = "cuda" if USE_GPU_XGB else "cpu"
    logger.info("\n" + "=" * 55)
    logger.info("LOFO -- Leave-One-Field-Out (XGBoost + lag label)")
    logger.info("=" * 55)

    for held_out in fields:
        tr_mask = meta["field"].values != held_out
        te_mask = meta["field"].values == held_out

        X_tr = apply_scaler(X[tr_mask], mean, std)
        X_te = apply_scaler(X[te_mask], mean, std)
        y_tr, y_te = y[tr_mask], y[te_mask]
        pr_tr = prev_label[tr_mask] if prev_label is not None else None
        pr_te = prev_label[te_mask] if prev_label is not None else None

        X_tr_f = flatten_sequences(X_tr)
        X_te_f = flatten_sequences(X_te)
        if config.USE_LAG_LABEL_FEATURE and pr_tr is not None:
            X_tr_f = flatten_with_lag(X_tr_f, pr_tr)
            X_te_f = flatten_with_lag(X_te_f, pr_te)

        sw = np.array([class_weights[int(c)] for c in y_tr])
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            random_state=SEED, tree_method="hist", device=device, verbosity=0,
        )
        xgb.fit(X_tr_f, y_tr, sample_weight=sw)
        y_pred  = xgb.predict(X_te_f)
        y_proba = xgb.predict_proba(X_te_f)
        m = evaluate(y_te, y_pred, y_proba, held_out)

        records.append({
            "held_out_field": held_out,
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "roc_auc": m.get("roc_auc"),
            "healthy_f1": m["healthy_f1-score"],
            "moderate_f1": m["moderate_f1-score"],
            "severe_f1": m["severe_f1-score"],
        })
        logger.info(f"  {held_out:35s} macro_f1={m['macro_f1']:.4f}  severe_f1={m['severe_f1-score']:.4f}")

    df = pd.DataFrame(records)
    mean_mf1 = df["macro_f1"].mean()
    std_mf1  = df["macro_f1"].std()
    logger.info(f"LOFO Mean Macro F1 = {mean_mf1:.4f} +/- {std_mf1:.4f}")
    df.to_csv(RESULTS_DIR / "lofo_results.csv", index=False)
    return df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("  DROUGHT DETECTION -- FULL TRAINING PIPELINE v2")
    print(f"  Device: {DEVICE}  |  Seq len: {config.SEQUENCE_LENGTH_DAYS}d  |  Lag: {config.USE_LAG_LABEL_FEATURE}")
    print("=" * 60)

    step_preprocessing()

    X, y, prev_label, meta = step_load_sequences()

    # If old 30-day npz loaded but config is now 60d, warn and rebuild
    if X.shape[1] != config.SEQUENCE_LENGTH_DAYS:
        logger.warning(f"Loaded sequences have window={X.shape[1]}d but config says {config.SEQUENCE_LENGTH_DAYS}d. "
                       f"Delete data/processed/weather/sequences/pooled_sequences.npz and rerun.")

    train_idx, val_idx, test_idx = chronological_split(X, y, meta)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    pr_tr = prev_label[train_idx] if prev_label is not None else None
    pr_va = prev_label[val_idx]   if prev_label is not None else None
    pr_te = prev_label[test_idx]  if prev_label is not None else None

    mean, std = fit_scaler(X_tr)
    X_tr_s = apply_scaler(X_tr, mean, std)
    X_va_s = apply_scaler(X_va, mean, std)
    X_te_s = apply_scaler(X_te, mean, std)

    class_weights = compute_class_weights(y_tr)

    print("\n--- Class Distribution ---")
    for name, ys in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
        classes, counts = np.unique(ys, return_counts=True)
        total = len(ys)
        dist = {config.DROUGHT_CLASSES[int(c)]: f"{cnt} ({100*cnt/total:.1f}%)"
                for c, cnt in zip(classes, counts)}
        print(f"  {name}: {dist}")

    maj_m, pers_m = run_baselines(y_tr, y_te, test_idx)

    all_results = {}
    all_probas  = {}

    # Random Forest
    rf_m, _ = train_random_forest(
        X_tr_s, y_tr, X_te_s, y_te, pr_tr, pr_te, class_weights, mean, std)
    all_results["Random Forest"] = rf_m

    # XGBoost
    xgb_m, _ = train_xgboost(
        X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
        pr_tr, pr_va, pr_te, class_weights, mean, std)
    all_results["XGBoost"] = xgb_m

    # Load XGBoost proba for ensemble
    xgb_loaded = joblib.load(config.MODELS_SAVED_DIR / "xgboost.joblib")
    X_te_xgb = flatten_sequences(X_te_s)
    if config.USE_LAG_LABEL_FEATURE and pr_te is not None:
        X_te_xgb = flatten_with_lag(X_te_xgb, pr_te)
    all_probas["XGBoost"] = xgb_loaded.predict_proba(X_te_xgb)

    # LSTM
    seq_len   = X_tr_s.shape[1]
    n_features = X_tr_s.shape[2]
    lstm = DroughtLSTM(
        input_size=n_features, hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS, num_classes=3,
        dropout=config.LSTM_DROPOUT, bidirectional=True,
    )
    lstm_m, _, lstm_proba = train_pytorch_model(
        lstm, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
        pr_tr, pr_va, pr_te, class_weights,
        "LSTM (BiLSTM+Attn)", "drought_lstm.pt", "drought_lstm_metadata.json",
        mean, std, lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
    )
    all_results["LSTM"] = lstm_m
    all_probas["LSTM"]  = lstm_proba

    # Transformer
    transformer = DroughtTransformer(
        input_size=n_features, d_model=128, nhead=4,
        num_layers=3, dim_feedforward=256, dropout=0.2,
        num_classes=3, seq_len=seq_len,
    )
    tf_m, _, tf_proba = train_pytorch_model(
        transformer, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
        pr_tr, pr_va, pr_te, class_weights,
        "Transformer", "transformer.pt", "transformer_metadata.json",
        mean, std, lr=5e-4, batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
    )
    all_results["Transformer"] = tf_m
    all_probas["Transformer"]  = tf_proba

    # Ensemble
    ens_m = run_ensemble(y_te, all_probas)
    all_results["Ensemble"] = ens_m

    # Summary table
    print("\n" + "=" * 70)
    print("  FINAL MODEL COMPARISON")
    print("=" * 70)
    rows = [
        {"Model": k,
         "Accuracy":     v["accuracy"],
         "Balanced_Acc": v["balanced_accuracy"],
         "Macro_F1":     v["macro_f1"],
         "ROC_AUC":      v.get("roc_auc", "N/A"),
         "Healthy_F1":   v.get("healthy_f1-score", v.get("healthy_f1")),
         "Moderate_F1":  v.get("moderate_f1-score", v.get("moderate_f1")),
         "Severe_F1":    v.get("severe_f1-score", v.get("severe_f1"))}
        for k, v in all_results.items()
    ]
    df_results = pd.DataFrame(rows)
    print(df_results.to_string(index=False))
    df_results.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)

    # LOFO
    df_lofo = run_lofo(X, y, prev_label, meta, mean, std, class_weights)
    print(f"\nLOFO Summary:\n{df_lofo[['held_out_field','macro_f1','severe_f1']].to_string(index=False)}")

    best = max(all_results, key=lambda k: all_results[k]["macro_f1"])
    summary = {
        "best_model": best,
        "best_macro_f1": all_results[best]["macro_f1"],
        "persistence_macro_f1": pers_m["macro_f1"],
        "all_results": all_results,
        "lofo_mean_macro_f1": float(df_lofo["macro_f1"].mean()),
        "config": {
            "sequence_length": config.SEQUENCE_LENGTH_DAYS,
            "use_lag_label": config.USE_LAG_LABEL_FEATURE,
        },
    }
    with open(RESULTS_DIR / "best_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>>> BEST MODEL: {best} (Macro F1 = {all_results[best]['macro_f1']:.4f})")
    print(f">>> Persistence baseline: Macro F1 = {pers_m['macro_f1']:.4f}")
    print(f">>> All results saved to results/")


if __name__ == "__main__":
    main()
