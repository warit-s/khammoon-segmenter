"""Inter-annotator agreement on the `annotator_agreement` configuration of the released data.

Three annotators (A, B, C) segmented the same 400 sentences per corpus independently. Every
position between two characters is one item with two categories, boundary or not; the script
reports Fleiss' kappa over all items, boundary F1 between each pair of annotators pooled over all
positions, and the share of sentences on which all three, exactly two, or none agree (the numbers
in the technical report).

usage: python scripts/annotator_agreement.py [--dir data/ref2/annotator_agreement]
"""
import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATORS = ["a", "b", "c"]


def boundaries(tokens):
    out, pos = set(), 0
    for t in tokens[:-1]:
        pos += len(t)
        out.add(pos)
    return out


def fleiss_kappa(items, n_raters=3):
    n = len(items)
    p_j = [sum(r[c] for r in items) / (n * n_raters) for c in range(2)]
    p_bar = sum(sum(c * (c - 1) for c in r) / (n_raters * (n_raters - 1)) for r in items) / n
    p_e = sum(p * p for p in p_j)
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def f1(tp, n_a, n_b):
    if tp == 0:
        return 0.0
    p, r = tp / n_b, tp / n_a
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/ref2/annotator_agreement")
    args = ap.parse_args()
    for f in sorted((ROOT / args.dir).glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        items, agree = [], [0, 0, 0]
        pair = {p: [0, 0, 0] for p in itertools.combinations(ANNOTATORS, 2)}  # tp, |x|, |y|
        for r in rows:
            b = {k: boundaries(r[f"tokens_{k}"]) for k in ANNOTATORS}
            for pos in range(1, len(r["text"])):
                yes = sum(pos in b[k] for k in ANNOTATORS)
                items.append((yes, 3 - yes))
            for x, y in pair:
                c = pair[(x, y)]
                c[0] += len(b[x] & b[y]); c[1] += len(b[x]); c[2] += len(b[y])
            same = len({tuple(r[f"tokens_{k}"]) for k in ANNOTATORS})
            agree[{1: 0, 2: 1, 3: 2}[same]] += 1
        n = len(rows)
        print(f"{f.stem}: {n} sentences, {len(items):,} positions")
        print(f"  Fleiss' kappa {fleiss_kappa(items):.4f}")
        for (x, y), (tp, na, nb) in pair.items():
            print(f"  F1 {x.upper()}-{y.upper()} {f1(tp, na, nb):.4f}", end="")
        print(f"\n  all three agree {100*agree[0]/n:.1f}%  two of three {100*agree[1]/n:.1f}%  none {100*agree[2]/n:.1f}%")


if __name__ == "__main__":
    main()
