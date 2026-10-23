"""Score the LLM pipeline that produced the silver labels on golden-dev and golden-test.

Reads its saved outputs from data/llm/ (config `llm_outputs` of typhoon-ai/khammoon-data, laid out by
scripts/download_data.py) and uses the same metric as the rest of the repository.

usage: python scripts/eval_llm_outputs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from khammoon.metrics import evaluate  # noqa: E402


def load(path):
    with open(ROOT / path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main():
    for name, pred, gold in (("golden-dev", "data/llm/best2009.jsonl", "data/ref2/best_all.jsonl"),
                             ("golden-test", "data/llm/wangchan.jsonl", "data/ref2/wangchan_test.jsonl")):
        p = {r["id"]: r["tokens"] for r in load(pred)}
        g = load(gold)
        m = evaluate([r["text"] for r in g], [p[r["id"]] for r in g], [r["tokens"] for r in g])
        print(f"{name:12} n={m['n']}  P {m['boundary_nows_p']:.4f}  R {m['boundary_nows_r']:.4f}  "
              f"F1 {m['boundary_nows_f1']:.4f}")


if __name__ == "__main__":
    main()
