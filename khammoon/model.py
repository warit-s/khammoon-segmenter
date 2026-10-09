"""Char-level boundary taggers (torch; training only — inference is exported to ONNX)."""
from __future__ import annotations

import math

import torch
from torch import nn


class ConvBiLSTM(nn.Module):
    """Small student: char embedding -> conv stack -> BiLSTM -> boundary logit."""

    def __init__(self, vocab: int, emb: int = 64, conv: int = 128, hidden: int = 128, layers: int = 2,
                 kernels=(3, 5, 7), dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, conv, k, padding=k // 2) for k in kernels)
        self.lstm = nn.LSTM(conv * len(kernels), hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids).transpose(1, 2)
        x = torch.cat([torch.relu(c(x)) for c in self.convs], 1).transpose(1, 2)
        x, _ = self.lstm(self.drop(x))
        return self.out(self.drop(x)).squeeze(-1)


class DilatedCNN(nn.Module):
    """Fast CPU student: char embedding -> residual dilated conv blocks -> boundary logit."""

    def __init__(self, vocab: int, emb: int = 64, width: int = 256, dilations=(1, 2, 4, 8, 1, 2, 4, 8),
                 kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.inp = nn.Conv1d(emb, width, 1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(width, width, kernel, padding=d * (kernel // 2), dilation=d) for d in dilations)
        self.norms = nn.ModuleList(nn.BatchNorm1d(width) for _ in dilations)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.inp(self.emb(ids).transpose(1, 2))
        for conv, norm in zip(self.blocks, self.norms):
            x = x + self.drop(torch.relu(norm(conv(x))))
        return self.out(x).squeeze(1)


class Transformer(nn.Module):
    """Teacher: pre-LN transformer encoder with learned absolute positions."""

    def __init__(self, vocab: int, d: int = 512, layers: int = 8, heads: int = 8, ff: int = 2048,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        nn.init.normal_(self.emb.weight, std=1 / math.sqrt(d))
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.emb(ids) + self.pos(pos)[None]
        x = self.enc(x, src_key_padding_mask=ids == 0)
        return self.out(self.norm(x)).squeeze(-1)


def build(arch: str, vocab: int, **kw) -> nn.Module:
    return {"bilstm": ConvBiLSTM, "cnn": DilatedCNN, "transformer": Transformer}[arch](vocab, **kw)
