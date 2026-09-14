"""
models/ml_models.py
====================
DroughtTransformer + helper for tree-based models.

DroughtTransformer:
  - Positional encoding (learned)
  - Multi-head self-attention (4 heads)
  - Feed-forward sublayer with GELU
  - Global average pooling + classification head

flatten_sequences:
  Converts (N, 30, 9) sequences into (N, 45) statistical feature vectors
  for Random Forest / XGBoost. Statistics: mean, std, min, max, linear trend
  per feature over the window. 9 features × 5 stats = 45 columns.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Statistical feature flattening for tree models
# ---------------------------------------------------------------------------

def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Convert (N, seq_len, n_features) → (N, n_features * 5) statistical
    feature matrix for tree-based models.

    Statistics per feature: mean, std, min, max, linear trend (slope).
    9 features × 5 stats = 45 columns — matches the 45 in saved RF metadata.
    """
    N, T, F = X.shape
    out = np.empty((N, F * 5), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    t_centered = t - t.mean()
    t_var = (t_centered ** 2).sum()

    for f in range(F):
        vals = X[:, :, f]                          # (N, T)
        out[:, f * 5 + 0] = vals.mean(axis=1)
        out[:, f * 5 + 1] = vals.std(axis=1)
        out[:, f * 5 + 2] = vals.min(axis=1)
        out[:, f * 5 + 3] = vals.max(axis=1)
        # Linear trend slope per sample
        deviations = vals - vals.mean(axis=1, keepdims=True)
        slopes = (deviations * t_centered).sum(axis=1) / (t_var + 1e-9)
        out[:, f * 5 + 4] = slopes

    return out


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class DroughtTransformer(nn.Module):
    """Transformer encoder for next-day drought prediction.

    Input:  (batch, sequence_length, input_size)
    Output: (batch, num_classes)  — logits
    """

    def __init__(
        self,
        input_size: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_classes: int = 3,
        seq_len: int = 30,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = LearnedPositionalEncoding(seq_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)    # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.input_norm(x)
        x = self.encoder(x)       # (batch, seq_len, d_model)
        x = x.mean(dim=1)         # Global average pooling
        return self.classifier(x)
