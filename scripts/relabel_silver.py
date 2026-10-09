"""Relabel silver with word-level conventions learnt from gold devtrain.

For every Thai string w that gold devtrain segments the same way in all of its >= --min-gold
occurrences, rewrite every silver occurrence of w (as one token, or as a 2-4 token run that
concatenates to w) to the gold segmentation.

With --patterns, first merge non-Thai spans that gold keeps whole but silver splits
(hyphenated Latin words such as E-Commerce / COVID-19, single-letter markers such as (ข),
and Latin possessives such as Lay's) when the span edges are already token boundaries.

usage: python scripts/relabel_silver.py --patterns [--gold data/ref2/best_devtrain.jsonl] [--out data/silver_c1]
"""
import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THAI = re.compile(r"[\u0E01-\u0E4E]{2,}")
MAXK = 4
KEEP_PATTERNS = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9])"
    r"|\([A-Za-z0-9\u0E01-\u0E2E]\)"
    r"|[A-Za-z]+[’']s(?![A-Za-z])"
)


def merge_patterns(toks, stats):
    text = "".join(toks)
    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t)
    edge = set(starts) | {pos}
    spans = [m.span() for m in KEEP_PATTERNS.finditer(text) if m.start() in edge and m.end() in edge]
    if not spans:
        return toks
    out, i = [], 0
    for a, b in spans:
        while starts[i] < a if i < len(starts) else False:
            out.append(toks[i])
            i += 1
        j = i
        while j < len(toks) and starts[j] < b:
            j += 1
        if j - i > 1:
            stats["pattern_merge"] += 1
        out.append(text[a:b])
        i = j
    out.extend(toks[i:])
    return out


def read(path):
    with open(ROOT / path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def gold_table(rows, min_gold):
    """w -> tuple(pieces) for strings gold always segments the same way."""
    seen = collections.defaultdict(collections.Counter)
    for r in rows:
        toks = r["tokens"]
        for i in range(len(toks)):
            s = ""
            for k in range(MAXK):
                if i + k >= len(toks):
                    break
                s += toks[i + k]
                if THAI.fullmatch(s):
                    seen[s][tuple(toks[i:i + k + 1])] += 1
    table = {}
    for w, c in seen.items():
        if len(c) == 1 and sum(c.values()) >= min_gold:
            table[w] = next(iter(c))
    return table


def relabel(toks, table, stats):
    out, i = [], 0
    while i < len(toks):
        # longest run of 1..MAXK tokens whose concatenation has a gold convention
        best = None
        s = ""
        for k in range(MAXK):
            if i + k >= len(toks):
                break
            s += toks[i + k]
            if s in table:
                best = (k + 1, table[s])
        if best is None:
            out.append(toks[i])
            i += 1
            continue
        k, pieces = best
        if tuple(toks[i:i + k]) != pieces:
            stats["changed"] += 1
            stats["merge" if len(pieces) < k else "split" if len(pieces) > k else "move"] += 1
        out.extend(pieces)
        i += k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gold", type=int, default=2)
    ap.add_argument("--out", default="data/silver_c1")
    ap.add_argument("--patterns", action="store_true")
    ap.add_argument("--silver", default="data/silver", help="directory with the silver to relabel")
    ap.add_argument("--gold", default="data/ref2/best_devtrain.jsonl",
                    help="comma-separated gold jsonl files the convention table is learnt from")
    args = ap.parse_args()
    gold = [r for f in args.gold.split(",") for r in read(f)]
    table = gold_table(gold, args.min_gold)
    print("gold-convention strings:", len(table), "multi-piece:", sum(len(v) > 1 for v in table.values()))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps({k: list(v) for k, v in table.items()}, ensure_ascii=False))
    for cfg in ("best2009", "wangchan"):
        stats = collections.Counter()
        rows = read(f"{args.silver}/{cfg}.jsonl")
        with open(out / f"{cfg}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                toks = merge_patterns(r["tokens"], stats) if args.patterns else r["tokens"]
                toks = relabel(toks, table, stats)
                assert "".join(toks) == r["text"]
                r["tokens"] = toks
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(cfg, dict(stats))


if __name__ == "__main__":
    main()
