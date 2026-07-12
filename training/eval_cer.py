"""Measure CER of a CT2 whisper model over a labeled corpus split.

This is the adoption gate: run it on the held-out eval split for BOTH the current
production model and a fine-tuned candidate; adopt the candidate only if CER
improves meaningfully (rule of thumb: >10% relative).

    python training\\eval_cer.py --model models\\faster-whisper-small --corpus data\\corpus\\eval
    python training\\eval_cer.py --model models\\faster-whisper-large-v3-turbo --corpus data\\corpus\\eval

Works on CPU (this dev box). Text is normalized before scoring (punctuation and
all whitespace stripped — see courtstt/textnorm.py, including the number-orthography
caveat).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from courtstt.textnorm import normalize_for_cer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="CT2 model dir (or HF size name)")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Split folder containing metadata.csv + wav clips")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="keek-keek")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N clips")
    args = parser.parse_args()

    import jiwer
    from faster_whisper import WhisperModel

    meta = args.corpus / "metadata.csv"
    if not meta.exists():
        print(f"error: {meta} not found", file=sys.stderr)
        return 2
    with meta.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["transcription"].strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"model={args.model}  clips={len(rows)}  compute={args.compute_type}")
    model = WhisperModel(args.model, device="cpu",
                         compute_type=args.compute_type, cpu_threads=args.cpu_threads)

    refs, hyps = [], []
    total_audio = total_time = 0.0
    for i, row in enumerate(rows, 1):
        started = time.perf_counter()
        segments, info = model.transcribe(
            str(args.corpus / row["file_name"]),
            language=args.language, beam_size=args.beam_size,
            condition_on_previous_text=False,
        )
        hyp = " ".join(s.text.strip() for s in segments)
        total_time += time.perf_counter() - started
        total_audio += info.duration

        ref_n, hyp_n = normalize_for_cer(row["transcription"]), normalize_for_cer(hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
        clip_cer = jiwer.cer(ref_n, hyp_n) if ref_n else 0.0
        print(f"  [{i}/{len(rows)}] {row['file_name']}  CER={clip_cer:.3f}")

    overall = jiwer.cer(refs, hyps)
    rtf = total_audio / total_time if total_time else 0.0
    print(f"\nOVERALL CER = {overall:.4f}  ({overall * 100:.2f}%)  "
          f"| {total_audio:.0f}s audio in {total_time:.0f}s ({rtf:.1f}x realtime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
