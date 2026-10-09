"""Export a trained checkpoint to the package model directory (ONNX + vocab + meta).

usage: python scripts/export_onnx.py runs/<name> [--out khammoon/model] [--thr 0.5]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_ckpt import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default="khammoon/model")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    model, vocab, a = load_model(Path(args.run), args.ckpt, device="cpu")
    model = model.float().cpu().eval()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(2, 256, dtype=torch.long)
    torch.onnx.export(model, (x,), out / "model.onnx", input_names=["ids"], output_names=["logits"],
                      dynamic_axes={"ids": {0: "batch", 1: "len"}, "logits": {0: "batch", 1: "len"}},
                      opset_version=17, dynamo=False)
    import numpy as np

    np.savez(out / "weights.npz", **{k: v.float().numpy() for k, v in model.state_dict().items()})
    shutil.copy(ROOT / "data/prep/vocab.json", out / "vocab.json")
    meta = {"run": args.run, "ckpt": args.ckpt, "arch": a["arch"], "model_kw": json.loads(a["model_kw"]),
            "win": a["win"], "threshold": args.thr}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, "size", (out / "model.onnx").stat().st_size)


if __name__ == "__main__":
    main()
