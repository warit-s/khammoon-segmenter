"""Extract the worked examples of linguistics/linguistic_guideline.md as a tiny human-labelled
dev set (the spec itself, not the reference). Examples whose output does not reconstruct the
input, that contain '|' in the input, or that carry a '(meaning: ...)' note are dropped.

Writes data/guideline_dev.jsonl
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = (ROOT / "linguistics/linguistic_guideline.md").read_text(encoding="utf-8").splitlines()
out, seen, dropped = [], set(), []
for i, line in enumerate(lines[:-1]):
    m, o = re.match(r"\s*Input: (.*)", line), re.match(r"\s*Output: (.*)", lines[i + 1])
    if not (m and o):
        continue
    text, toks = m.group(1), o.group(1).split("|")
    if "(meaning:" in text or "|" in text or "".join(toks) != text or "" in toks:
        dropped.append(text)
        continue
    if text in seen:
        continue
    seen.add(text)
    out.append({"id": len(out), "text": text, "tokens": toks})
with open(ROOT / "data/guideline_dev.jsonl", "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("kept", len(out), "dropped", len(dropped), dropped)
