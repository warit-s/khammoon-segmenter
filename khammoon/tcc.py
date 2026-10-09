"""Thai Character Cluster (TCC) boundaries.

Ported from PyThaiNLP (``pythainlp/tokenize/tcc.py``, Apache-2.0), itself after
Theeramunkong et al. 2000; credits: Jakkrit TeCho (TCC), Wittawat Jitkrittum (grammar),
Korakot Chaovavanich (Python). Changed to match in place (``pattern.match(text, pos)``)
so it is linear in the text length, and to return a start-position mask.
"""
from __future__ import annotations

import re

import numpy as np

_RULES = (
    """\
c[ั]([่-๋]c)?
c[ั]([่-๋]c)?k
เc็ck
เcctาะk
เccีtยะk
เccีtย(?=[เ-ไก-ฮ]|$)k
เc[ิีุู]tย(?=[เ-ไก-ฮ]|$)k
เcc็ck
เcิc์ck
เcิtck
เcีtยะ?k
เcืtอะk
เcื
เctา?ะ?k
c[ึื]tck
c[ะ-ู]tk
c[ิุู]์
cรรc์
c็
ct[ะาำ]?k
แc็ck
แcc์k
แctะk
แcc็ck
แccc์k
โctะk
[เ-ไ]ctk
ก็
อึ
หึ
""".replace("k", "(cc?[d|ิ]?[์])?")
    .replace("c", "[ก-ฮ]")
    .replace("t", "[่-๋]?")
    .replace("d", "ูุ")
    .split()
)
_PAT = re.compile("|".join(_RULES))


def tcc_starts(text: str) -> np.ndarray:
    """int8 mask, 1 where a TCC starts (always 1 at position 0 and for non-Thai chars)."""
    n = len(text)
    out = np.zeros(n, dtype=np.int8)
    p, match = 0, _PAT.match
    while p < n:
        out[p] = 1
        m = match(text, p)
        p = m.end() if m and m.end() > p else p + 1
    return out


def candidates(text: str) -> np.ndarray:
    """0 = boundary impossible, 1 = model decides, 2 = forced (text start / whitespace edge)."""
    c = tcc_starts(text)
    if len(text):
        c[0] = 2
    for i, ch in enumerate(text):
        if ch.isspace():
            c[i] = 2
            if i + 1 < len(text):
                c[i + 1] = 2
    return c
