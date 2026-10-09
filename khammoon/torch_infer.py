"""Sliding-window inference with a torch model (used for evaluation and pseudo-labelling)."""
from __future__ import annotations

import numpy as np
import torch

from .chars import Vocab
from .tcc import candidates as cand_mask


def windows(n: int, win: int, ctx: int):
    """Yield (start, keep_from, keep_to) covering [0, n) with `ctx` context on each side."""
    if n <= win:
        yield 0, 0, n
        return
    step = win - 2 * ctx
    s = 0
    while True:
        lo = 0 if s == 0 else s + ctx
        if s + win >= n:
            yield s, lo, n
            return
        yield s, lo, s + win - ctx
        s += step


@torch.no_grad()
def predict_probs(model, vocab: Vocab, texts: list[str], win: int = 256, ctx: int = 64,
                  batch: int = 512, device="cuda", pad: bool = True) -> list[np.ndarray]:
    """pad=True wraps each text in "\n" as in training (documents are "\n"-delimited)."""
    model.eval()
    if pad:
        texts = ["\n" + t + "\n" for t in texts]
    probs = [np.zeros(len(t), dtype=np.float32) for t in texts]
    jobs = []
    for ti, t in enumerate(texts):
        ids = vocab.encode(t)
        for s, lo, hi in windows(len(t), win, ctx):
            jobs.append((ti, s, lo, hi, ids[s:s + win]))
    by_len: dict[int, list] = {}
    for j in jobs:
        by_len.setdefault(len(j[4]), []).append(j)
    for group in by_len.values():
        for b in range(0, len(group), batch):
            chunk = group[b:b + batch]
            x = torch.from_numpy(np.stack([j[4] for j in chunk]).astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                p = torch.sigmoid(model(x).float()).cpu().numpy()
            for k, (ti, s, lo, hi, _) in enumerate(chunk):
                probs[ti][lo:hi] = p[k, lo - s:hi - s]
    return [q[1:-1] for q in probs] if pad else probs


def decode(text: str, prob: np.ndarray, cand: np.ndarray, thr: float = 0.5) -> list[str]:
    cut = (cand == 2) | ((cand == 1) & (prob > thr))
    idx = np.flatnonzero(cut).tolist()
    if not idx or idx[0] != 0:
        idx = [0] + idx
    idx.append(len(text))
    return [text[a:b] for a, b in zip(idx, idx[1:]) if b > a]


def segment(model, vocab: Vocab, texts: list[str], thr: float = 0.5, **kw) -> list[list[str]]:
    kw.setdefault("device", next(model.parameters()).device)
    probs = predict_probs(model, vocab, texts, **kw)
    return [decode(t, p, cand_mask(t), thr) for t, p in zip(texts, probs)]
