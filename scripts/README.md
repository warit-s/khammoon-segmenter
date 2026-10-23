# Reproducing khammoon 0.1.0

All commands run from the repository root with the training extras installed
(`pip install -e ".[train]"`). Training steps need a CUDA GPU (each run takes
minutes); everything else runs on a CPU. `HF_TOKEN` is needed while the dataset is private.

```bash
# 1. data: silver (original LLM labels), golden-dev (BEST2009), golden-test (Wangchan)
python scripts/download_data.py

# 2. relabel round 1: convention table from golden-dev devtrain + guideline Step-2 merges
python scripts/relabel_silver.py --patterns --gold data/ref2/best_devtrain.jsonl --out data/silver_c1
python scripts/prep_silver.py --inputs data/silver_c1/best2009.jsonl,data/silver_c1/wangchan.jsonl --name silver_c1
python scripts/prep_silver.py --inputs data/ref2/best_devtrain.jsonl --name best_devtrain --val-frac 0
python scripts/prep_silver.py --inputs data/ref2/best_all.jsonl --name best_all --val-frac 0

# 3. model on round-1 silver, fine-tuned on golden-dev devtrain
python scripts/train_char.py --arch bilstm --name c1 --data silver_c1 --steps 6000 --lr 2e-3 --eval-every 1000
python scripts/train_char.py --arch bilstm --name c1_ft --data best_devtrain --init runs/c1/best.pt \
    --steps 1200 --warmup 100 --lr 3e-4 --batch 64 --eval-every 100

# 4. relabel round 2: c1_ft re-predicts Thai-Thai boundaries of the silver
python scripts/self_label_silver.py runs/c1_ft --thr 0.45 --base data/silver_c1 --out data/silver_c2
python scripts/prep_silver.py --inputs data/silver_c2/best2009.jsonl,data/silver_c2/wangchan.jsonl --name silver_c2

# 5. model on round-2 silver; choose the mixing ratio / threshold on golden-dev devval
python scripts/train_char.py --arch bilstm --name c2 --data silver_c2 --steps 6000 --lr 2e-3 --eval-every 1000
python scripts/train_char.py --arch bilstm --name c2_ftmix5 --data best_devtrain --aux silver_c2 --aux-frac 0.5 \
    --init runs/c2/best.pt --steps 1200 --warmup 100 --lr 3e-4 --batch 64 --eval-every 100

# 6. release model: all of golden-dev + 50% silver per batch, fixed steps; export
python scripts/train_char.py --arch bilstm --name c2_rel --data best_all --aux silver_c2 --aux-frac 0.5 \
    --init runs/c2/best.pt --steps 1000 --warmup 100 --lr 3e-4 --batch 64 --eval-every 1000
python scripts/export_onnx.py runs/c2_rel --ckpt last.pt --thr 0.5

# 7. evaluation on golden-test (once) and baselines
python -c "import json; from khammoon import Segmenter; from khammoon.metrics import evaluate; \
r=[json.loads(l) for l in open('data/ref2/wangchan_test.jsonl')]; t=[x['text'] for x in r]; \
print(evaluate(t, Segmenter().segment_batch(t), [x['tokens'] for x in r]))"
pip install -e ".[baselines]"          # PyThaiNLP, AttaCut, nlpO3
python scripts/baselines_test.py --engines nlpo3,attacut
python scripts/eval_llm_outputs.py     # the LLM pipeline that produced the silver labels
```

`data/silver_c2` produced in step 4 is the `tokens` field of the released `silver` config
(`data/silver_release/` after step 1). Other tools: `eval_ckpt.py` (score a checkpoint on an
evaluation set), `speed.py` (CPU
throughput), `make_guideline_dev.py` (guideline examples → `tests/guideline_examples.jsonl`).
deepcut needs TensorFlow and was scored in a separate environment.
