"""Command line interface: segment arguments, or stdin line by line.

    khammoon "โรงเรียนสอนคนตาบอด"          -> โรง|เรียน|สอน|คน|ตา|บอด
    cat corpus.txt | khammoon --sep " "   (one output line per input line)
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .segmenter import Segmenter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="khammoon", description="Thai monomorpheme (คำมูล) segmenter")
    ap.add_argument("text", nargs="*", help="text to segment (default: read stdin line by line)")
    ap.add_argument("--sep", default="|", help="separator placed between tokens (default: |)")
    ap.add_argument("--threshold", type=float, default=None, help="boundary probability threshold")
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "numpy"],
                    help="inference backend (default: onnx if onnxruntime is installed, else numpy)")
    ap.add_argument("--batch-lines", type=int, default=256, help="stdin lines per model batch")
    ap.add_argument("--version", action="version", version=f"khammoon {__version__}")
    args = ap.parse_args(argv)

    seg = Segmenter(threshold=args.threshold, backend=args.backend)
    if args.text:
        print(args.sep.join(seg(" ".join(args.text))))
        return 0

    def flush(buf: list[str]) -> None:
        for toks in seg.segment_batch(buf):
            sys.stdout.write(args.sep.join(toks) + "\n")

    buf: list[str] = []
    for line in sys.stdin:
        buf.append(line.rstrip("\n"))
        if len(buf) >= args.batch_lines:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
