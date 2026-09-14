"""
models/lstm_model.py
====================
DroughtLSTM — Bidirectional LSTM with attention for 3-class drought prediction.

Architecture improvements over the baseline:
  - Bidirectional LSTM for richer temporal representations
  - Temporal attention mechanism (learns which days matter most)
  - Layer normalization for training stability
  - Residual projection
  - Proper dropout placement
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Soft attention over the time dimension.
    Produces a weighted sum of hidden states so the model can focus
    on the most drought-relevant days within the 30-day window."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_out)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)  # (batch, hidden_size)
        return context


class DroughtLSTM(nn.Module):
    """Bidirectional LSTM + Temporal Attention for next-day drought class prediction.

    Input:  (batch, sequence_length, input_size)  — raw weather features
    Output: (batch, num_classes)                  — logits (no softmax here)
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.directions
        self.attention = TemporalAttention(lstm_out_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*dirs)
        context = self.attention(lstm_out)  # (batch, hidden*dirs)
        logits = self.classifier(context)  # (batch, num_classes)
        return logits
