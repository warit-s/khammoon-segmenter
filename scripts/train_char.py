"""Train a char-level boundary tagger on prepared arrays (scripts/prep_silver.py) and keep the
checkpoint that scores best on golden-dev devval.

usage: python scripts/train_char.py --name c1 --data silver_c1 [--arch bilstm] [--steps 6000]
       [--aux silver_c2 --aux-frac 0.5]   (mix a second dataset into every batch)
       [--init runs/<name>/best.pt]        (fine-tune from a checkpoint)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khammoon.chars import Vocab  # noqa: E402
from khammoon.metrics import evaluate  # noqa: E402
from khammoon.model import build  # noqa: E402
from khammoon.torch_infer import segment  # noqa: E402


EVAL_SETS = {
    "best_devval": "data/ref2/best_devval.jsonl",      # golden-dev selection part (scripts/download_data.py)
    "best_devtrain": "data/ref2/best_devtrain.jsonl",  # golden-dev training part
    "test": "data/ref2/wangchan_test.jsonl",           # evaluation only: never select on it
    "guideline": "data/guideline_dev.jsonl",           # guideline examples (scripts/make_guideline_dev.py)
}


def load_ref(split):
    """`split`: one or more EVAL_SETS keys joined by '+'."""
    out = {}
    for s in split.split("+"):
        with open(ROOT / EVAL_SETS[s], encoding="utf-8") as fh:
            out[s] = [json.loads(line) for line in fh]
    return out


def ref_eval(model, vocab, ref, **kw):
    res = {}
    for cfg, rows in ref.items():
        preds = segment(model, vocab, [r["text"] for r in rows], **kw)
        res[cfg] = evaluate([r["text"] for r in rows], preds, [r["tokens"] for r in rows])
    res["mean_bF1_nows"] = float(np.mean([res[c]["boundary_nows_f1"] for c in ref]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="bilstm", choices=["bilstm", "cnn", "transformer"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, help="prepared dataset name in data/prep (e.g. silver_c1)")
    ap.add_argument("--model-kw", default="{}")
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--init", default=None, help="checkpoint to initialise from (fine-tuning)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-split", default="best_devval", help="evaluation set used for checkpoint selection")
    ap.add_argument("--aux", default=None, help="second prepared dataset mixed into every batch (e.g. silver_c1)")
    ap.add_argument("--aux-frac", type=float, default=0.0, help="fraction of each batch drawn from --aux")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    vocab = Vocab.load(ROOT / "data/prep/vocab.json")
    d = np.load(ROOT / "data/prep" / f"{args.data}_train.npz")
    dev = "cuda"
    # keep everything on the GPU: CPU gathers are the bottleneck on a busy host
    ids = torch.from_numpy(d["ids"]).to(dev)
    y = torch.from_numpy(d["y"]).to(dev)
    m = torch.from_numpy(d["cand"] == 1).to(dev)
    N = len(ids)
    print(f"train chars {N:,}")
    n_aux = int(round(args.batch * args.aux_frac)) if args.aux else 0
    if n_aux:
        da = np.load(ROOT / "data/prep" / f"{args.aux}_train.npz")
        aux_ids = torch.from_numpy(da["ids"]).to(dev)
        aux_y = torch.from_numpy(da["y"]).to(dev)
        aux_m = torch.from_numpy(da["cand"] == 1).to(dev)
        print(f"aux {args.aux}: {len(aux_ids):,} chars, {n_aux}/{args.batch} windows per batch")
    ref = load_ref(args.select_split)

    model = build(args.arch, len(vocab), **json.loads(args.model_kw)).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params {nparam:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / args.warmup) * max(0.0, 0.5 * (1 + np.cos(np.pi * s / args.steps))))
    ar = torch.arange(args.win, device=dev)
    best, log = -1.0, []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        st = torch.from_numpy(rng.integers(0, N - args.win, args.batch - n_aux)).to(dev)
        idx = st[:, None] + ar[None]
        xb, yb, mb = ids[idx].long(), y[idx].float(), m[idx].float()
        if n_aux:
            sa = torch.from_numpy(rng.integers(0, len(aux_ids) - args.win, n_aux)).to(dev)
            ia = sa[:, None] + ar[None]
            xb = torch.cat([xb, aux_ids[ia].long()])
            yb = torch.cat([yb, aux_y[ia].float()])
            mb = torch.cat([mb, aux_m[ia].float()])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(xb).float()
        loss = (F.binary_cross_entropy_with_logits(logit, yb, reduction="none") * mb).sum() / mb.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = ref_eval(model, vocab, ref, win=args.win, ctx=args.win // 4)
            r["step"] = step
            log.append(r)
            parts = " ".join(f"{c} {r[c]['boundary_nows_f1']:.4f} (R {r[c]['boundary_nows_r']:.4f})" for c in ref)
            print(f"  eval step {step}: {parts} mean {r['mean_bF1_nows']:.4f}", flush=True)
            if r["mean_bF1_nows"] > best:
                best = r["mean_bF1_nows"]
                torch.save(model.state_dict(), out / "best.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1))
    torch.save(model.state_dict(), out / "last.pt")
    print("best dev mean bF1_nows", best)


if __name__ == "__main__":
    main()
