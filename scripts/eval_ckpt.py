"""Evaluate a trained checkpoint on a reference split.

usage: python scripts/eval_ckpt.py runs/<name> [--split best_devval] [--ckpt best.pt] [--thr 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from khammoon.chars import Vocab  # noqa: E402
from khammoon.model import build  # noqa: E402
from train_char import load_ref, ref_eval  # noqa: E402


def load_model(run: Path, ckpt: str = "best.pt", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a = json.loads((run / "args.json").read_text())
    vocab = Vocab.load(ROOT / "data/prep/vocab.json")
    model = build(a["arch"], len(vocab), **json.loads(a["model_kw"])).to(device)
    model.load_state_dict(torch.load(run / ckpt, map_location=device))
    return model, vocab, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="best_devval")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, vocab, a = load_model(Path(args.run), args.ckpt)
    r = ref_eval(model, vocab, load_ref(args.split), win=a["win"], ctx=a["win"] // 4, thr=args.thr)
    for cfg in (k for k in r if k != "mean_bF1_nows"):
        m = r[cfg]
        print(f"{args.run} {args.split} {cfg:9} bF1_nows {m['boundary_nows_f1']:.4f} (P {m['boundary_nows_p']:.4f} "
              f"R {m['boundary_nows_r']:.4f}) bF1 {m['boundary_f1']:.4f} tokF1 {m['token_f1']:.4f}")
    print(f"mean bF1_nows {r['mean_bF1_nows']:.4f}")


if __name__ == "__main__":
    main()
