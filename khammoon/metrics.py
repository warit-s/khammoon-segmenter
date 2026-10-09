"""Segmentation metrics.

All metrics are micro-averaged over a corpus. A segmentation is a list of tokens whose
concatenation equals the original text.

- ``boundary``: F1 over internal boundary offsets (every position between two tokens).
- ``boundary_nows``: same, but ignoring boundaries adjacent to whitespace. Whitespace
  boundaries are trivially recoverable, so this is the headline metric.
- ``token``: F1 over non-whitespace token spans (start, end); a token counts only if
  both of its edges are right.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def boundaries(tokens: list[str]) -> set[int]:
    out, pos = set(), 0
    for t in tokens[:-1]:
        pos += len(t)
        out.add(pos)
    return out


def spans(tokens: list[str]) -> set[tuple[int, int]]:
    out, pos = set(), 0
    for t in tokens:
        if t.strip():
            out.add((pos, pos + len(t)))
        pos += len(t)
    return out


@dataclass
class PRF:
    tp: int = 0
    n_pred: int = 0
    n_gold: int = 0

    def add(self, pred: set, gold: set) -> None:
        self.tp += len(pred & gold)
        self.n_pred += len(pred)
        self.n_gold += len(gold)

    @property
    def p(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def r(self) -> float:
        return self.tp / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.p * self.r / (self.p + self.r) if self.p + self.r else 0.0


@dataclass
class Scores:
    boundary: PRF = field(default_factory=PRF)
    boundary_nows: PRF = field(default_factory=PRF)
    token: PRF = field(default_factory=PRF)
    n: int = 0
    exact: int = 0

    def add(self, text: str, pred: list[str], gold: list[str]) -> None:
        if "".join(pred) != text:
            raise ValueError(f"prediction does not reconstruct text: {text[:80]!r}")
        if "".join(gold) != text:
            raise ValueError(f"gold does not reconstruct text: {text[:80]!r}")
        pb, gb = boundaries(pred), boundaries(gold)
        self.boundary.add(pb, gb)

        def not_ws(i: int) -> bool:
            return not text[i - 1].isspace() and not text[i].isspace()

        self.boundary_nows.add({i for i in pb if not_ws(i)}, {i for i in gb if not_ws(i)})
        self.token.add(spans(pred), spans(gold))
        self.n += 1
        self.exact += pb == gb

    def as_dict(self) -> dict[str, float]:
        d = {}
        for name in ("boundary_nows", "boundary", "token"):
            m = getattr(self, name)
            d[f"{name}_p"], d[f"{name}_r"], d[f"{name}_f1"] = m.p, m.r, m.f1
        d["sent_exact"] = self.exact / self.n if self.n else 0.0
        d["n"] = self.n
        return d


def evaluate(texts, preds, golds) -> dict[str, float]:
    s = Scores()
    for t, p, g in zip(texts, preds, golds):
        s.add(t, p, g)
    return s.as_dict()
