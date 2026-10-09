"""Character vocabulary and per-position features shared by training and inference."""
from __future__ import annotations

import json
import unicodedata

import numpy as np

# reserved ids
PAD, UNK_THAI, UNK_LATIN, UNK_DIGIT, UNK_PUNCT, UNK_SPACE, UNK_OTHER = range(7)
N_RESERVED = 7


def char_class(ch: str) -> int:
    o = ord(ch)
    if 0x0E00 <= o <= 0x0E7F:
        return UNK_THAI
    if ch.isspace():
        return UNK_SPACE
    if ch.isdigit():
        return UNK_DIGIT
    if ch.isalpha():
        return UNK_LATIN
    if unicodedata.category(ch)[0] in "PS":
        return UNK_PUNCT
    return UNK_OTHER


class Vocab:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.index = {c: i + N_RESERVED for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return N_RESERVED + len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        idx = self.index
        return np.fromiter((idx.get(c) or char_class(c) for c in text), dtype=np.int32, count=len(text))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.chars, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Vocab":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))


def labels_from_tokens(tokens: list[str]) -> np.ndarray:
    """y[i] = 1 if a token starts at character i (y[0] = 1)."""
    n = sum(len(t) for t in tokens)
    y = np.zeros(n, dtype=np.int8)
    pos = 0
    for t in tokens:
        y[pos] = 1
        pos += len(t)
    return y
