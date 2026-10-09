"""Pure-numpy forward pass of the ConvBiLSTM tagger (no onnxruntime / torch needed).

Weights come from `weights.npz`, written by scripts/export_onnx.py from the PyTorch
state_dict (same parameter names). Gate order follows torch.nn.LSTM: input, forget, cell, output.
A numba version of the recurrence was tried and was no faster than the batched numpy loop.
"""
from __future__ import annotations

import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_dir_numpy(xw: np.ndarray, whh_t: np.ndarray, reverse: bool) -> np.ndarray:
    """xw: (B, L, 4H) input projection incl. both biases; whh_t: (H, 4H). Returns (B, L, H)."""
    B, L, H4 = xw.shape
    H = H4 // 4
    h = np.zeros((B, H), dtype=xw.dtype)
    c = np.zeros((B, H), dtype=xw.dtype)
    out = np.empty((B, L, H), dtype=xw.dtype)
    steps = range(L - 1, -1, -1) if reverse else range(L)
    for t in steps:
        g = xw[:, t] + h @ whh_t
        i = _sigmoid(g[:, :H])
        f = _sigmoid(g[:, H:2 * H])
        cc = np.tanh(g[:, 2 * H:3 * H])
        o = _sigmoid(g[:, 3 * H:])
        c = f * c + i * cc
        h = o * np.tanh(c)
        out[:, t] = h
    return out


class NumpyConvBiLSTM:
    def __init__(self, path):
        w = np.load(path)
        self.w = {k: w[k].astype(np.float32) for k in w.files}
        self.kernels = sorted(int(k.split(".")[1]) for k in self.w if k.startswith("convs.") and k.endswith("weight"))
        self.layers = sum(1 for k in self.w if k.startswith("lstm.weight_ih_l") and not k.endswith("reverse"))
        # pre-transpose for row-major matmuls
        self.lstm = []
        for layer in range(self.layers):
            dirs = []
            for sfx in ("", "_reverse"):
                wih = self.w[f"lstm.weight_ih_l{layer}{sfx}"]
                whh = self.w[f"lstm.weight_hh_l{layer}{sfx}"]
                b = self.w[f"lstm.bias_ih_l{layer}{sfx}"] + self.w[f"lstm.bias_hh_l{layer}{sfx}"]
                dirs.append((np.ascontiguousarray(wih.T), np.ascontiguousarray(whh.T), b))
            self.lstm.append(dirs)
        self.conv = {}
        for n in self.kernels:
            W = self.w[f"convs.{n}.weight"]  # (C, E, k) -> (k*E, C) matching the im2col layout
            C, E, k = W.shape
            self.conv[n] = (np.ascontiguousarray(W.transpose(2, 1, 0).reshape(k * E, C)), self.w[f"convs.{n}.bias"], k)

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """ids: (B, L) int -> logits (B, L)."""
        x = self.w["emb.weight"][ids]  # (B, L, E)
        B, L, E = x.shape
        feats = []
        for n in self.kernels:
            W2, b, k = self.conv[n]
            p = k // 2
            xp = np.pad(x, ((0, 0), (p, p), (0, 0)))
            # im2col: (B*L, k*E) contiguous, one BLAS matmul per kernel size
            cols = np.concatenate([xp[:, j:j + L] for j in range(k)], axis=-1).reshape(B * L, k * E)
            feats.append(np.maximum(cols @ W2 + b, 0.0))
        x = np.concatenate(feats, axis=-1)  # (B*L, 3C)
        for dirs in self.lstm:
            outs = []
            for reverse, (wih_t, whh_t, b) in zip((False, True), dirs):
                xw = (x @ wih_t + b).reshape(B, L, -1)
                outs.append(_lstm_dir_numpy(np.ascontiguousarray(xw, dtype=np.float32), whh_t, reverse))
            x = np.concatenate(outs, axis=-1).reshape(B * L, -1)
        return (x @ self.w["out.weight"].T + self.w["out.bias"]).reshape(B, L)
