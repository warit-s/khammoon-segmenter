# khammoon — Thai morpheme segmenter

khammoon splits Thai text into **morphemes** in the sense of Thai grammar (คำมูล, *khammoon*): the
smallest units that still carry meaning on their own, as defined by the annotation guideline in
[`linguistics/`](linguistics/).
It is a 4 MB character-level model that runs on a CPU with nothing but `numpy`; no GPU and no LLM.

**Package:** [`pip install khammoon`](https://pypi.org/project/khammoon/) · **Model:** [`typhoon-ai/khammoon`](https://huggingface.co/typhoon-ai/khammoon) · **Data:** [`typhoon-ai/khammoon-data`](https://huggingface.co/datasets/typhoon-ai/khammoon-data)

```python
from khammoon import segment

segment("นักเรียนโรงเรียนสอนคนตาบอดทำงานขะมักเขม้น")
# ['นัก', 'เรียน', 'โรง', 'เรียน', 'สอน', 'คน', 'ตา', 'บอด', 'ทำ', 'งาน', 'ขะมัก', 'เขม้น']
```

## What a morpheme is

Ordinary Thai word segmenters keep compounds together; khammoon splits them down to their
meaningful parts. `สินค้า` ("goods") is the compound สิน (wealth) + ค้า (trade), so it becomes
`สิน|ค้า`, whereas `ราคา` ("price") has no meaningful parts (รา + คา) and stays whole. Following
the guideline, Pali–Sanskrit compounds read back-to-front (`ประวัติศาสตร์`) and loanwords
(`วิปครีม`) are not split, and foreign-script text is split only at whitespace (`COVID-19`).

## Install

```bash
pip install .            # from this directory; needs only numpy
pip install ".[onnx]"    # adds onnxruntime: about 2.4x faster, identical output
pip install ".[train]"   # to train or evaluate (torch, pythainlp, ...)
```

| backend | dependencies | speed (1 CPU thread) |
|---|---|---|
| `onnx` | numpy + onnxruntime | ~30k characters/s |
| `numpy` | numpy | ~13k characters/s |

The default `backend="auto"` uses `onnx` when onnxruntime is installed and `numpy` otherwise.
Both backends produce the same segmentation on every sentence of the development set.

## Usage

```python
from khammoon import Segmenter

seg = Segmenter()                                     # loads the bundled model
seg("ไปที่ www.google.com เลย")                        # ['ไป', 'ที่', ' ', 'www.google.com', ' ', 'เลย']
seg.segment_batch(["ข้อความแรก", "ข้อความที่สอง"])     # batching is faster than one call per text
Segmenter(threshold=0.3)                              # lower threshold: more boundaries, higher recall
Segmenter(backend="numpy")                            # force the pure-numpy backend
```

```bash
khammoon "โรงเรียนสอนคนตาบอด"            # โรง|เรียน|สอน|คน|ตา|บอด
cat corpus.txt | khammoon --sep " "      # reads stdin line by line
```

Guarantees: `"".join(tokens) == text` always holds (characters are never added, removed or
normalised); every whitespace character is its own token; and boundaries are placed only between
Thai Character Clusters, so no token is misspelled (for example, none starts with the vowel `า`).

## Results

The evaluation set is the **test set**: the Wangchan reference, 2,000 documents segmented by three
trained annotators following the guideline and adjudicated (Fleiss κ 0.991). It was never used
during development: not for training, model selection or threshold choice. The metric is
boundary precision / recall / F1, pooled over all boundaries, ignoring boundaries adjacent to
whitespace (those are fixed by the input, and the human annotation treats runs of whitespace
inconsistently). Confidence intervals and the full discussion are in the technical report.

| method | precision | recall | F1 |
|---|---|---|---|
| PyThaiNLP `newmm` (BEST-style words) | 0.962 | 0.687 | 0.802 |
| nlpO3 (`newmm` in Rust) | 0.961 | 0.693 | 0.805 |
| deepcut | 0.972 | 0.875 | 0.921 |
| AttaCut | 0.981 | 0.874 | 0.925 |
| syllable segmentation (PyThaiNLP `syllable_tokenize`) | 0.821 | 0.991 | 0.898 |
| unigram Viterbi over the LLM-label lexicon | 0.975 | 0.881 | 0.925 |
| Qwen3-30B-A3B two-pass LLM segmenter that produced the training labels | 0.975 | 0.948 | 0.962 |
| the same BiLSTM trained on the uncorrected LLM labels | 0.977 | 0.948 | 0.962 |
| **khammoon 0.1.0** | **0.992** | **0.993** | **0.992** |

`newmm`, deepcut and AttaCut segment into BEST2009 *words*, not morphemes, so their recall is
low; the table measures distance to the morpheme definition, not the quality of those tools
at their own task (`scripts/baselines_test.py`). The Qwen3-30B-A3B row scores the saved outputs of
the same LLM segmenter that labelled the training text, run on the test set texts
(`scripts/eval_llm_outputs.py`). khammoon is about 30,000 times smaller than that LLM, runs on a
CPU, and makes about 5 times fewer errors (1 − F1: 0.008 vs 0.038). Models trained on its output
perform on par with those trained on word segmenters on POS tagging, NER and sentiment
classification; see the technical report.

The model has ~1.06M parameters (4.2 MB).

## Data

All data are released as [`typhoon-ai/khammoon-data`](https://huggingface.co/datasets/typhoon-ai/khammoon-data).

| config | source | role |
|---|---|---|
| `llm_labelled` | BEST2009 train + WangchanThaiInstruct train, segmented by an LLM (Qwen3-30B-A3B, two passes following the guideline) and then corrected; 180,206 documents, 70M characters | training |
| `dev` | development set: BEST2009 reference, 2,252 sentences, human-labelled | label correction, fine-tuning, model and threshold selection; split into dev-train (1,484) and dev-val (768), marked `devtrain` / `devval` in the `dev_split` field |
| `test` | test set: Wangchan reference, 2,000 documents, human-labelled | evaluation only |

Training rows whose text also occurs in either human set were removed before training. The
cross-domain split (news, novels and encyclopedia text for development; instruction data in
finance, legal, medical and retail for testing) is deliberate.

**All of the data is open, for developers and researchers to build on.** Besides the model, we
release the human-labelled development and test sets (a common benchmark with a fixed protocol), the 70M
characters of LLM-labelled text with both the original LLM labels and the corrected ones (training
data for new segmenters, and a record of where an LLM annotator goes wrong), and the LLM's outputs
on the reference sets (the strongest baseline, reproducible without a GPU). Use them to train your
own segmenter, compare against ours on the same test set, or study Thai morphology and
tokenization; the licenses of the source texts apply (see [License](#license)).

## Method

1. **Character tagging.** The model predicts, before every character, whether a boundary
   falls there. Boundaries are allowed only at Thai Character Cluster (TCC) edges and are forced
   at whitespace edges; a boundary is placed where the predicted probability exceeds the
   threshold (0.5).
2. **Label correction, round 1.** The LLM systematically under-splits frequent Thai compounds
   (`สินค้า`, `กฎหมาย`, `ข้อมูล`). We build a table of strings that dev-train always
   segments the same way (at least twice) and rewrite every occurrence in the LLM labels to match, and we
   merge foreign-script spans that the guideline keeps whole (`COVID-19`, `People's`) and
   parenthesised heading markers (`(๒)`).
3. **Train and fine-tune.** A BiLSTM (character embedding → convolutions with kernels 3/5/7 →
   2-layer BiLSTM, 256-character windows) is trained on the round-1 labels and fine-tuned on
   dev-train.
4. **Label correction, round 2.** The fine-tuned model re-segments the whole LLM-labelled corpus;
   only boundaries between two Thai letters are taken from it, all others keep the round-1
   labels. This spreads the human conventions over the 70M characters of LLM-labelled text.
5. **Mixed fine-tuning.** A new model is trained on the round-2 labels and fine-tuned with half
   of every batch from dev-train and half from the LLM-labelled text, so it does not forget what
   the development set lacks (it is 2009 text without `COVID-19` and splits every hyphen). The
   mixing ratio (0% or 50% LLM-labelled data, chosen by a rule fixed beforehand: the largest
   LLM-labelled share whose dev-val F1 is within 0.0005 of the best), the number of steps and the
   threshold were selected on dev-val; the released model was then fine-tuned on the whole
   development set with those values.
6. **Export** to ONNX (`model.onnx`) and to raw weights (`weights.npz`) for the numpy backend.

| dev-val F1 | before fine-tuning | human-only fine-tuning | 50% LLM-labelled fine-tuning |
|---|---|---|---|
| LLM labels after round 1 | 0.9916 | 0.9970 | 0.9957 |
| LLM labels after round 2 | 0.9955 | 0.9974 | 0.9972 (selected) |

Every step can be rerun from the released data; see [`scripts/README.md`](scripts/README.md). The technical report describing this work is linked from the model card on Hugging Face.

## Limitations

- Only one cross-domain test set; there is no human-annotated set for social-media or general
  web text yet.
- Most remaining errors are rare proper names and transliterations, and Pali–Sanskrit
  compounds that are over-split.
- The test set was evaluated three times. Each change of method was motivated by the
  development set or by examples outside the test set, never by test results: (1) F1 0.9916;
  (2) after fixing an inference bug found through a README example (text-final characters were
  mis-segmented because training documents are newline-delimited, and batches contained padding),
  0.9917; (3) after label-correction round 2 and mixed fine-tuning (prompted by `COVID-19` being
  split), P 0.9917 / R 0.9928 / F1 0.9923, the numbers in the table.
- The developer had seen error analyses on the Wangchan reference in an earlier experiment,
  before this split was fixed. The foreign-script merge rule follows Step 2 of the guideline but
  may have been influenced by what was seen.

## License

| part | license |
|---|---|
| code (all `.py` files, scripts, tests) | [Apache-2.0](LICENSE) |
| model weights (`khammoon/model/model.onnx`, `weights.npz`) | [CC BY-NC-SA 4.0](khammoon/model/LICENSE); no commercial use, because the training data carry NonCommercial terms |
| data ([`typhoon-ai/khammoon-data`](https://huggingface.co/datasets/typhoon-ai/khammoon-data)) | per row, inherited from the source: CC BY-NC-SA 3.0 (BEST2009), CC BY-NC 4.0 or CC BY-SA 4.0 (WangchanThaiInstruct) |

The wheel installed by pip bundles the model weights, so that part falls under CC BY-NC-SA 4.0;
see [NOTICE](NOTICE).

## Citation

If you use khammoon, the guideline or the data, please cite the technical report:

```bibtex
@techreport{khammoon2026,
  title       = {khammoon: From a Linguistic Guideline to a Thai Morpheme Segmenter},
  author      = {Sirichotedumrong, Warit and Na-Thalang, Adisai and Wittayasakpan, Chanakan and
                 Phatcharoen, Kritsadha and Pipatanakul, Kunat},
  institution = {Typhoon, SCB DataX},
  year        = {2026},
  url         = {https://github.com/warit-s/khammoon-segmenter}
}
```

## Credits

TCC rules ported from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) (Apache-2.0;
Theeramunkong et al. 2000). Training data from the BEST2009 corpus (NECTEC) and
WangchanThaiInstruct (VISTEC/AIResearch).
