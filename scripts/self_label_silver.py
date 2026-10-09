"""Re-segment the silver texts with a trained model (self-training / distillation labels).

With --base <dir>, only boundaries between two Thai letters are taken from the model; every
other boundary decision (next to Latin, digits, punctuation, whitespace) is kept from the
base silver labels.

usage: python scripts/self_label_silver.py runs/<name> --out data/silver_c2 [--thr 0.45] [--base data/silver_c1]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_ckpt import load_model  # noqa: E402
from khammoon.torch_infer import segment  # noqa: E402


def _is_thai_letter(ch):
    return "\u0E01" <= ch <= "\u0E4E"


def _cuts(tokens):
    out, pos = set(), 0
    for t in tokens[:-1]:
        pos += len(t)
        out.add(pos)
    return out


def merge_thai_boundaries(text, base_tokens, model_tokens):
    base, model = _cuts(base_tokens), _cuts(model_tokens)
    cuts = set()
    for i in range(1, len(text)):
        thai = _is_thai_letter(text[i - 1]) and _is_thai_letter(text[i])
        if (i in model) if thai else (i in base):
            cuts.add(i)
    idx = [0] + sorted(cuts) + [len(text)]
    return [text[a:b] for a, b in zip(idx, idx[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--base", default=None, help="silver dir whose labels are kept outside Thai-Thai boundaries")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, vocab, a = load_model(Path(args.run), args.ckpt)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for cfg in ("best2009", "wangchan"):
        src = ROOT / (args.base or "data/silver") / f"{cfg}.jsonl"
        rows = [json.loads(line) for line in open(src, encoding="utf-8")]
        with open(out / f"{cfg}.jsonl", "w", encoding="utf-8") as fh:
            for b in range(0, len(rows), args.chunk):
                chunk = rows[b:b + args.chunk]
                preds = segment(model, vocab, [r["text"] for r in chunk], thr=args.thr, win=a["win"],
                                ctx=a["win"] // 4, batch=2048)
                for r, p in zip(chunk, preds):
                    assert "".join(p) == r["text"]
                    if args.base:
                        p = merge_thai_boundaries(r["text"], r["tokens"], p)
                    r["tokens"] = p
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(cfg, len(rows), flush=True)


if __name__ == "__main__":
    main()
