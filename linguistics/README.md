# Linguistic basis for monomorphemic segmentation

These documents define what a correct segmentation *is* for this project. Everything
else in the repo (the `khammoon` segmenter, its training data and benchmarks) is downstream of
the monomorpheme (คำมูล) definition set out here.

## Files

- **`thai_word_segmentation_monomorphemic_approach.md`** — the linguistic rationale.
  Thai word-formation taxonomy (คำมูล, คำประสม, คำซ้อน, คำซ้ำ, คำสมาส/สนธิ), the factors
  that make boundaries ambiguous (internal cohesion, semantic transparency, subunit count,
  cross-boundary sharing, sound change, context), and the two operational tests for
  monomorpheme-hood: the **substitution test** (การหาคำมาแทนที่) and the **insertion test**
  (การหาคำมาแทรก).

- **`linguistic_guideline.md`** — the operational annotation spec compiled from the above.
  A sequential decision procedure (structural chunks → non-Thai / punctuation / symbols →
  Thai-character handling → the คำมูล-vs-compound decision across five categories +
  pronunciation test), with the atomic-integrity constraint
  (`''.join(tokenized.split('|')) == raw_text`) and 149 worked examples (102 in the rules + 47 few-shot exemplars).
  Every example satisfies the atomic-integrity constraint (five inconsistent examples were
  corrected on 2026-10-06; see the git history).

## How it connects to the code

- **Training data.** `linguistic_guideline.md` was the prompt contract for the two-pass LLM
  segmenter (Qwen3-30B-A3B) that produced the silver training data, and the annotation spec
  for the three human annotators of the reference sets. The LLM segmenter itself belongs to an
  earlier pipeline and is not part of this repository; its outputs are on Hugging Face
  (`Warit/thai-monomorpheme-llm-annotated`), the human labels in `Warit/thai-monomorpheme-reference`
  and, re-split for release, in `Warit/khammoon-data` (golden-dev = BEST2009, golden-test = Wangchan).
- **Rules taken directly from the guideline.** Step 2 (foreign text is split on whitespace
  only; heading markers such as `(๒)` are kept) is applied to the silver data by
  `scripts/relabel_silver.py --patterns`.
- **Tests.** The worked examples are extracted by `scripts/make_guideline_dev.py` into
  `tests/guideline_examples.jsonl` and used as a regression check of the released model
  (`tests/test_segmenter.py`). The two `ทำลาย (meaning: ...)` examples are skipped there because
  they need the context described in Step 3.3.
