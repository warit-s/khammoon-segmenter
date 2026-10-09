"""CPU throughput of the segmenter on golden-dev devval (1 thread).

usage: python scripts/speed.py khammoon/model
"""
import json, sys, time
from khammoon.segmenter import Segmenter
from khammoon.metrics import evaluate
for d in sys.argv[1:]:
    seg = Segmenter(d)
    for cfg in ("best_devval",):
        rows = [json.loads(l) for l in open(f"data/ref2/{cfg}.jsonl")]
        t = time.time(); P = seg.segment_batch([r["text"] for r in rows]); dt = time.time() - t
        n = sum(len(r["text"]) for r in rows)
        m = evaluate([r["text"] for r in rows], P, [r["tokens"] for r in rows])
        print(f"{d} {cfg} bF1_nows {m['boundary_nows_f1']:.4f} {n/dt:,.0f} chars/s")
