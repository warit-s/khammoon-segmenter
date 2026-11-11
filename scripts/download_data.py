"""Download typhoon-ai/khammoon-data and lay it out for the training pipeline.

Writes
  data/silver/{best2009,wangchan}.jsonl       training text with the ORIGINAL LLM labels (`tokens_llm`)
                                              as `tokens` -- the input of relabel_silver.py
  data/silver_release/{best2009,wangchan}.jsonl  the released (corrected) labels
  data/ref2/best_all.jsonl                    dev set (BEST2009 reference, 2,252)
  data/ref2/best_devtrain.jsonl               dev rows with dev_split == devtrain (1,484)
  data/ref2/best_devval.jsonl                 dev rows with dev_split == devval (768)
  data/ref2/wangchan_test.jsonl               test set (Wangchan reference, 2,000) -- evaluation only
  data/llm/{best2009,wangchan}.jsonl          the LLM segmenter's output on the development / test texts
  data/ref2/annotator_agreement/{best2009,wangchan}.jsonl
                                              the three annotators' independent segmentations (400 + 400)

usage: python scripts/download_data.py [--repo typhoon-ai/khammoon-data] [--revision main]
(set HF_TOKEN while the dataset is private)
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"


def rows(repo, path, revision):
    f = hf_hub_download(repo, path, repo_type="dataset", revision=revision)
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def dump(it, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in it:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{path}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="typhoon-ai/khammoon-data")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    for cfg in ("best2009", "wangchan"):
        silver = list(rows(args.repo, f"llm_labelled/{cfg}.jsonl", args.revision))
        dump(({**{k: v for k, v in r.items() if k not in ("tokens", "tokens_llm")}, "tokens": r["tokens_llm"]}
              for r in silver), out / "silver" / f"{cfg}.jsonl")
        dump(({k: v for k, v in r.items() if k != "tokens_llm"} for r in silver),
             out / "silver_release" / f"{cfg}.jsonl")
    dev = list(rows(args.repo, "dev/best2009.jsonl", args.revision))
    dump(dev, out / "ref2/best_all.jsonl")
    for part in ("devtrain", "devval"):
        dump((r for r in dev if r["dev_split"] == part), out / f"ref2/best_{part}.jsonl")
    dump(rows(args.repo, "test/wangchan.jsonl", args.revision), out / "ref2/wangchan_test.jsonl")
    for cfg in ("best2009", "wangchan"):
        dump(rows(args.repo, f"llm_outputs/{cfg}.jsonl", args.revision), out / f"llm/{cfg}.jsonl")
    for cfg in ("best2009", "wangchan"):
        dump(rows(args.repo, f"annotator_agreement/{cfg}.jsonl", args.revision),
             out / f"ref2/annotator_agreement/{cfg}.jsonl")


if __name__ == "__main__":
    main()
