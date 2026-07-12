"""Build a fine-tuning corpus from transcribed recordings.

Cuts each recording into per-segment clips (16 kHz mono WAV) using the segment
timestamps in the pipeline's JSON master, and writes the standard audiofolder
layout that train_lora.py and eval_cer.py consume:

    data/corpus/<split>/
    ├─ metadata.csv        # file_name,transcription
    └─ <recording>_seg0001.wav ...

Usage (audio file + its JSON master from the transcripts folder):
    python training\\build_corpus.py --audio D:\\rec\\hearing.wav ^
        --json D:\\rec\\transcripts\\hearing.json --split train

The metadata.csv rows contain the PIPELINE's text. THE HUMAN CORRECTION STEP IS
EDITING metadata.csv: open it (UTF-8), listen to each clip, fix the text. Only
corrected rows are trustworthy training data — track which files you've reviewed.

Skips segments shorter than --min-seconds (default 1.0) and longer than 30 s
(Whisper's training window).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0


def build(audio_path: Path, json_path: Path, corpus_dir: Path, split: str,
          min_seconds: float) -> int:
    import numpy as np
    import soundfile as sf
    from faster_whisper.audio import decode_audio  # handles wav/mp3/m4a/... -> 16k mono

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        print(f"no segments in {json_path}")
        return 0

    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    out_dir = corpus_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"
    new_file = not meta_path.exists()

    # Idempotency: never append duplicate rows for a recording already in the corpus
    # (re-running would silently double its training weight and clobber corrections).
    existing: set[str] = set()
    if not new_file:
        with meta_path.open(encoding="utf-8") as f:
            existing = {row["file_name"] for row in csv.DictReader(f)}
    if any(name.startswith(audio_path.stem + "_seg") for name in existing):
        print(f"SKIP: clips from '{audio_path.stem}' already exist in {meta_path} — "
              f"delete those rows and wav files first if you intend to rebuild them.")
        return 0

    written = 0
    with meta_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["file_name", "transcription"])
        for i, seg in enumerate(segments):
            duration = seg["end"] - seg["start"]
            text = seg["text"].strip()
            if duration < min_seconds or duration > MAX_SECONDS or not text:
                continue
            clip = audio[int(seg["start"] * SAMPLE_RATE): int(seg["end"] * SAMPLE_RATE)]
            clip_name = f"{audio_path.stem}_seg{i:04d}.wav"
            sf.write(out_dir / clip_name, np.asarray(clip), SAMPLE_RATE)
            writer.writerow([clip_name, text])
            written += 1

    print(f"{audio_path.name}: wrote {written} clip(s) to {out_dir} "
          f"(skipped {len(segments) - written})")
    print(f"NEXT: review/correct the text column of {meta_path} while listening to the clips.")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, required=True, help="Original recording")
    parser.add_argument("--json", type=Path, required=True, help="JSON master from the pipeline")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    parser.add_argument("--split", choices=["train", "eval"], default="train",
                        help="eval split is held out and NEVER trained on")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if not args.audio.is_file() or not args.json.is_file():
        print("error: audio or json file not found", file=sys.stderr)
        return 2
    build(args.audio, args.json, args.corpus, args.split, args.min_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
