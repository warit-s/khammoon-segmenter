import json
import random
from pathlib import Path

import pytest

from khammoon import Segmenter, segment
from khammoon.cli import main as cli_main
from khammoon.metrics import evaluate
from khammoon.tcc import candidates

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def seg():
    return Segmenter()


def test_basic(seg):
    assert seg("โรงเรียนสอนคนตาบอด") == ["โรง", "เรียน", "สอน", "คน", "ตา", "บอด"]
    assert segment("ไปที่ www.google.com เลย") == ["ไป", "ที่", " ", "www.google.com", " ", "เลย"]


@pytest.mark.parametrize("text", ["", " ", "\n\n", "ก", "abc", "๑๒๓", "😀😀", "ๆๆๆ", "​", "a\tb\r\nc"])
def test_edge_cases_roundtrip(seg, text):
    toks = seg(text)
    assert "".join(toks) == text
    assert all(toks)


def test_whitespace_is_its_own_token(seg):
    for t in seg("สวัสดี ครับ\nลาก่อน  นะ"):
        assert not t.strip() or t == t.strip()


def test_roundtrip_random(seg):
    rng = random.Random(0)
    alphabet = [chr(c) for c in range(0x0E01, 0x0E5C)] + list("abcXYZ0123 .,-()\n\"'%") + ["😀", "​"]
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 700))) for _ in range(200)]
    for text, toks in zip(texts, seg.segment_batch(texts)):
        assert "".join(toks) == text
        assert all(toks)


def test_long_text_matches_short_windows(seg):
    base = "ประเทศไทยมีพื้นที่ป่าไม้ลดลงอย่างต่อเนื่อง "
    long = base * 60  # far longer than one 256-char window
    toks = seg(long)
    assert "".join(toks) == long
    assert toks[: len(seg(base))] == seg(base)


def test_no_boundary_inside_tcc(seg):
    text = "เกี่ยวข้องกับการบริหารจัดการทรัพยากรธรรมชาติ"
    c = candidates(text)
    pos = 0
    for t in seg(text)[:-1]:
        pos += len(t)
        assert c[pos] != 0


def test_guideline_examples(seg):
    rows = [json.loads(line) for line in open(HERE / "guideline_examples.jsonl", encoding="utf-8")]
    preds = seg.segment_batch([r["text"] for r in rows])
    m = evaluate([r["text"] for r in rows], preds, [r["tokens"] for r in rows])
    # the guideline's worked examples are deliberately hard edge cases
    assert m["boundary_nows_f1"] > 0.85


def test_cli(capsys):
    assert cli_main(["--sep", "/", "โรงเรียน"]) == 0
    assert capsys.readouterr().out.strip() == "โรง/เรียน"


def test_numpy_backend_matches_onnx(seg):
    pytest.importorskip("onnxruntime")
    np_seg = Segmenter(backend="numpy")
    assert seg.backend == "onnx" and np_seg.backend == "numpy"
    rows = [json.loads(line) for line in open(HERE / "guideline_examples.jsonl", encoding="utf-8")]
    texts = [r["text"] for r in rows] + ["ประเทศไทยมีพื้นที่ป่าไม้ลดลง " * 30]
    assert np_seg.segment_batch(texts) == seg.segment_batch(texts)
