"""Score off-the-shelf segmenters on the golden-test (Wangchan reference), in parallel.

usage: python scripts/baselines_test.py --engines attacut,nlpo3 [--workers 8]
Appends to runs/baselines_test.json
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from khammoon.metrics import evaluate  # noqa: E402

_seg = None


def _init(engine):
    global _seg
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch  # noqa: F401  (attacut)

    torch.set_num_threads(1)
    from baselines import make_engine

    _seg = make_engine(engine)


def _run(text):
    from baselines import split_ws

    return split_ws(_seg(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="attacut,nlpo3")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(ROOT / "data/ref2/wangchan_test.jsonl", encoding="utf-8")]
    texts, golds = [r["text"] for r in rows], [r["tokens"] for r in rows]
    out_path = ROOT / "runs/baselines_test.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for e in args.engines.split(","):
        t0 = time.time()
        with Pool(args.workers, initializer=_init, initargs=(e,)) as pool:
            preds = pool.map(_run, texts, chunksize=8)
        m = evaluate(texts, preds, golds)
        m["sec"] = time.time() - t0
        res[e] = m
        print(f"{e:12} P {m['boundary_nows_p']:.4f} R {m['boundary_nows_r']:.4f} F1 {m['boundary_nows_f1']:.4f} "
              f"({m['sec']:.0f}s)", flush=True)
        out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
