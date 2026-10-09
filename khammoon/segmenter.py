"""Public segmenter: ONNX Runtime backend (fast) or pure-numpy backend (no extra dependency)."""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from .chars import Vocab
from .tcc import candidates

WIN, CTX = 256, 64


def _windows(n: int):
    if n <= WIN:
        yield 0, 0, n
        return
    step, s = WIN - 2 * CTX, 0
    while True:
        lo = 0 if s == 0 else s + CTX
        if s + WIN >= n:
            yield s, lo, n
            return
        yield s, lo, s + WIN - CTX
        s += step


class Segmenter:
    """Thai monomorpheme (คำมูล) segmenter.

    >>> seg = Segmenter()
    >>> seg("โรงเรียนสอนคนตาบอด")
    ['โรง', 'เรียน', 'สอน', 'คน', 'ตา', 'บอด']
    """

    def __init__(self, model_dir: str | Path | None = None, threshold: float | None = None,
                 backend: str = "auto", batch_size: int | None = None, providers: list[str] | None = None):
        """backend: "onnx" (needs onnxruntime, fastest), "numpy" (numpy only), or "auto"
        (onnx when onnxruntime is installed, else numpy). Both give the same segmentation."""
        d = Path(model_dir) if model_dir else Path(str(resources.files("khammoon") / "model"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.vocab = Vocab.load(d / "vocab.json")
        self.threshold = meta.get("threshold", 0.5) if threshold is None else threshold
        if backend == "auto":
            try:
                import onnxruntime  # noqa: F401

                backend = "onnx"
            except ImportError:
                backend = "numpy"
        if backend == "onnx":
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=providers or ["CPUExecutionProvider"])
            self._forward = lambda x: sess.run(None, {"ids": x})[0]
            self.batch_size = batch_size or 64
        elif backend == "numpy":
            from .numpy_model import NumpyConvBiLSTM

            self._forward = NumpyConvBiLSTM(d / "weights.npz")
            self.batch_size = batch_size or 16  # small batches are faster (cache-friendly)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

    def probabilities(self, texts: list[str]) -> list[np.ndarray]:
        # Training documents are always delimited by "\n"; pad each text the same way so the
        # first/last characters see the context they were trained with.
        texts = ["\n" + t + "\n" for t in texts]
        probs = [np.zeros(len(t), dtype=np.float32) for t in texts]
        jobs = []
        for ti, t in enumerate(texts):
            ids = self.vocab.encode(t)
            for s, lo, hi in _windows(len(t)):
                jobs.append((ti, s, lo, hi, ids[s:s + WIN]))
        # batch only windows of identical length: the model never saw padding in training
        by_len: dict[int, list] = {}
        for j in jobs:
            by_len.setdefault(len(j[4]), []).append(j)
        for L, group in by_len.items():
            for b in range(0, len(group), self.batch_size):
                chunk = group[b:b + self.batch_size]
                x = np.stack([j[4] for j in chunk]).astype(np.int64)
                logit = self._forward(x)
                p = 1.0 / (1.0 + np.exp(-logit))
                for k, (ti, s, lo, hi, _) in enumerate(chunk):
                    probs[ti][lo:hi] = p[k, lo - s:hi - s]
        return [p[1:-1] for p in probs]

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        out = []
        for t, p in zip(texts, self.probabilities(texts)):
            c = candidates(t)
            cut = (c == 2) | ((c == 1) & (p > self.threshold))
            idx = np.flatnonzero(cut).tolist() + [len(t)]
            out.append([t[a:b] for a, b in zip(idx, idx[1:])])
        return out

    def segment(self, text: str) -> list[str]:
        if not text:
            return []
        return self.segment_batch([text])[0]

    __call__ = segment


_default: Segmenter | None = None


def segment(text: str) -> list[str]:
    """Segment with the bundled model (loaded once)."""
    global _default
    if _default is None:
        _default = Segmenter()
    return _default.segment(text)
