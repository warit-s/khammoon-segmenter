"""Off-the-shelf / lexicon baseline segmenters (used by baselines_test.py), and a runner on golden-dev.

usage: python scripts/baselines.py [--engines a,b,...]      (scores on data/ref2/best_devval.jsonl)
"""
import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khammoon.metrics import evaluate  # noqa: E402

THAI_RUN = re.compile(r"[\u0E01-\u0E4E]+")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def split_ws(tokens):
    """Split every whitespace char into its own token (silver convention)."""
    out = []
    for t in tokens:
        buf = ""
        for ch in t:
            if ch.isspace():
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
            else:
                buf += ch
        if buf:
            out.append(buf)
    return out


def silver_counts(min_count=2):
    c = collections.Counter()
    for cfg in ("best2009", "wangchan"):
        for r in load(ROOT / "data/silver" / f"{cfg}.jsonl"):
            for t in r["tokens"]:
                if THAI_RUN.fullmatch(t):
                    c[t] += 1
    return {w: n for w, n in c.items() if n >= min_count}


class Unigram:
    """Viterbi over Thai runs with silver unigram costs; non-Thai handled by newmm."""

    def __init__(self, counts, maxlen=25, unk_cost=20.0):
        total = sum(counts.values())
        self.cost = {w: -math.log(n / total) for w, n in counts.items()}
        self.maxlen, self.unk = maxlen, unk_cost

    def seg_thai(self, s):
        n = len(s)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(n):
            if best[i] == math.inf:
                continue
            for j in range(i + 1, min(n, i + self.maxlen) + 1):
                c = self.cost.get(s[i:j])
                if c is None and j == i + 1:
                    c = self.unk
                if c is not None and best[i] + c < best[j]:
                    best[j], back[j] = best[i] + c, i
        out, j = [], n
        while j > 0:
            out.append(s[back[j]:j])
            j = back[j]
        return out[::-1]

    def __call__(self, text):
        from pythainlp.tokenize import word_tokenize

        out, pos = [], 0
        for m in THAI_RUN.finditer(text):
            if m.start() > pos:
                out += word_tokenize(text[pos:m.start()], engine="newmm", keep_whitespace=True)
            out += self.seg_thai(m.group())
            pos = m.end()
        if pos < len(text):
            out += word_tokenize(text[pos:], engine="newmm", keep_whitespace=True)
        return out


def make_engine(name):
    from pythainlp.tokenize import syllable_tokenize, word_tokenize

    if name == "whitespace":
        return lambda t: [x for x in re.split(r"(\s)", t) if x]
    if name == "syllable":
        return lambda t: syllable_tokenize(t, engine="dict", keep_whitespace=True)
    if name in ("newmm", "attacut", "nlpo3", "longest"):
        return lambda t: word_tokenize(t, engine=name, keep_whitespace=True)
    if name == "newmm_silverdict":
        from pythainlp.util import Trie

        trie = Trie(silver_counts().keys())
        return lambda t: word_tokenize(t, engine="newmm", custom_dict=trie, keep_whitespace=True)
    if name == "unigram_silver":
        return Unigram(silver_counts())
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engines",
        default="whitespace,syllable,newmm,nlpo3,attacut,newmm_silverdict,unigram_silver",
    )
    args = ap.parse_args()
    data = {"best_devval": load(ROOT / "data/ref2/best_devval.jsonl")}
    results = {}
    print(f"{'engine':18} {'corpus':9} {'bF1_nows':>8} {'bF1':>7} {'tokF1':>7} {'P':>6} {'R':>6} {'sec':>6}")
    for name in args.engines.split(","):
        seg = make_engine(name)
        for cfg, rows in data.items():
            t0 = time.time()
            preds = [split_ws(seg(r["text"])) for r in rows]
            dt = time.time() - t0
            m = evaluate([r["text"] for r in rows], preds, [r["tokens"] for r in rows])
            results[f"{name}/{cfg}"] = {**m, "sec": dt}
            print(
                f"{name:18} {cfg:9} {m['boundary_nows_f1']:8.4f} {m['boundary_f1']:7.4f} {m['token_f1']:7.4f} "
                f"{m['boundary_nows_p']:6.3f} {m['boundary_nows_r']:6.3f} {dt:6.1f}",
                flush=True,
            )
    out = ROOT / "runs" / "baselines_dev.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
