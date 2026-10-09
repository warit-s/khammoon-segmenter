"""Turn silver (or any tokens jsonl) into flat numpy arrays for char-level training.

Writes data/prep/<name>.npz with
  ids  : int32 char ids (documents concatenated, separated by one '\n')
  y    : int8 1 if a token starts here
  cand : int8 1 if a boundary is allowed here (TCC start), 2 if forced (whitespace edge)
and data/prep/vocab.json (built from silver on the first run).

usage: python scripts/prep_silver.py [--min-count 20] [--inputs data/silver/best2009.jsonl,...] [--name silver]
"""
import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khammoon.chars import Vocab, labels_from_tokens  # noqa: E402
from khammoon.tcc import candidates as cand_mask  # noqa: E402


def one(tokens):
    text = "".join(tokens) + "\n"
    y = labels_from_tokens(tokens + ["\n"])
    return text, y, cand_mask(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="data/silver/best2009.jsonl,data/silver/wangchan.jsonl")
    ap.add_argument("--name", default="silver")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.005)
    args = ap.parse_args()
    out = ROOT / "data/prep"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in args.inputs.split(","):
        with open(ROOT / f, encoding="utf-8") as fh:
            rows += [json.loads(line)["tokens"] for line in fh]
    print("docs", len(rows))

    vpath = out / "vocab.json"
    if vpath.exists():
        vocab = Vocab.load(vpath)
    else:
        cnt = collections.Counter(ch for toks in rows for t in toks for ch in t)
        vocab = Vocab(sorted(c for c, n in cnt.items() if n >= args.min_count))
        vocab.save(vpath)
    print("vocab", len(vocab))

    rng = np.random.default_rng(0)
    is_val = rng.random(len(rows)) < args.val_frac
    with Pool(32) as pool:
        res = pool.map(one, rows, chunksize=256)
    for split, sel in (("train", ~is_val), ("val", is_val)):
        part = [r for r, s in zip(res, sel) if s]
        if not part:
            continue
        text = "".join(r[0] for r in part)
        np.savez(
            out / f"{args.name}_{split}.npz",
            ids=vocab.encode(text),
            y=np.concatenate([r[1] for r in part]),
            cand=np.concatenate([r[2] for r in part]),
        )
        print(split, len(part), "docs", len(text), "chars")


if __name__ == "__main__":
    main()
