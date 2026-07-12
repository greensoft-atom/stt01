from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from courtstt.config import Config, find_config_file, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args) -> Config:
    config_file = find_config_file(args.config)
    return load_config(config_file, args.profile, language=getattr(args, "language", None))


def _make_engine(cfg: Config):
    from courtstt.engines.fasterwhisper_engine import FasterWhisperEngine  # heavy import
    return FasterWhisperEngine(cfg)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="dev", help="Config profile: dev | target (default: dev)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")
    parser.add_argument("--language", default=None, help="Override language (default from config: ko)")
    parser.add_argument("-v", "--verbose", action="store_true")


def cmd_run(args) -> int:
    from courtstt.pipeline import run_batch

    if not args.in_dir.is_dir():
        print(f"error: input folder not found: {args.in_dir}", file=sys.stderr)
        return 2
    out_dir = args.out_dir or (args.in_dir / "transcripts")

    cfg = _load(args)
    engine = _make_engine(cfg)
    summary = run_batch(cfg, engine, args.in_dir, out_dir)

    print()
    total_flagged = sum(r.flagged for r in summary.done)
    print(f"Batch finished: {len(summary.done)} done, {len(summary.skipped)} skipped, "
          f"{len(summary.failed)} failed, {total_flagged} segments flagged for review")
    for result in summary.failed:
        print(f"  FAILED {result.name}: {result.error}")
    if total_flagged:
        print(f"Tip: run 'stt01 review \"{out_dir}\"' for a consolidated review worklist.")
    return 1 if summary.failed else 0


def cmd_single(args) -> int:
    from courtstt.pipeline import process_file

    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 2
    out_dir = args.out_dir or args.file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load(args)
    engine = _make_engine(cfg)
    result = process_file(cfg, engine, args.file, out_dir)
    print(f"\nTranscribed {result.name}: {result.segments} segments, "
          f"{result.flagged} flagged for review")
    for fmt, name in result.outputs.items():
        print(f"  {fmt}: {out_dir / name}")
    return 0


def cmd_check(args) -> int:
    """Environment self-check: config, model, glossary — run before trusting a batch."""
    problems = 0

    def ok(msg): print(f"  OK    {msg}")
    def bad(msg):
        nonlocal problems
        problems += 1
        print(f"  FAIL  {msg}")

    print("stt01 environment check")
    try:
        config_file = find_config_file(args.config)
        ok(f"config: {config_file}")
        cfg = load_config(config_file, args.profile)
        ok(f"profile '{cfg.profile_name}': model={cfg.model}, compute={cfg.compute_type}, "
           f"beam={cfg.beam_size}, threads={cfg.cpu_threads}, formats={list(cfg.output_formats)}")
    except Exception as exc:
        bad(f"config: {exc}")
        return 1

    model_dir = Path(cfg.model_path)
    if model_dir.is_dir() and (model_dir / "model.bin").exists():
        size_mb = (model_dir / "model.bin").stat().st_size / 1e6
        ok(f"model files present ({size_mb:.0f} MB): {model_dir}")
    else:
        bad(f"model not found at '{cfg.model_path}' — run: python scripts/download_models.py")

    prompt = cfg.load_glossary_prompt()
    if prompt:
        ok(f"glossary loaded ({len(prompt)} chars of initial_prompt)")
    else:
        bad(f"glossary empty or missing: {cfg.glossary_path}")

    corrections = cfg.load_corrections()
    ok(f"corrections: {len(corrections)} replacement pair(s)")

    try:
        import faster_whisper
        ok(f"faster-whisper {faster_whisper.__version__}")
    except ImportError as exc:
        bad(f"faster-whisper not importable: {exc}")

    if not problems and args.full:
        print("  ...   loading model + transcribing 1s of silence (full check)")
        try:
            import numpy as np
            engine = _make_engine(cfg)
            engine._model.transcribe(np.zeros(16000, dtype=np.float32), language=cfg.language)
            ok("model loads and runs")
        except Exception as exc:
            bad(f"model failed to run: {exc}")

    print(f"\n{'All checks passed.' if not problems else f'{problems} problem(s) found.'}")
    return 1 if problems else 0


def cmd_review(args) -> int:
    from courtstt.review import collect_flagged, write_report

    if not args.transcripts_dir.is_dir():
        print(f"error: folder not found: {args.transcripts_dir}", file=sys.stderr)
        return 2
    cfg = _load(args)
    items = collect_flagged(cfg, args.transcripts_dir)
    report = write_report(items, args.transcripts_dir)
    print(f"{len(items)} flagged segment(s). Report: {report}")
    return 0


def cmd_gui(args) -> int:
    from courtstt.gui import run_gui
    return run_gui(config_path=args.config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stt01",
        description="Offline Korean speech-to-text batch pipeline (court session recordings).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Transcribe every audio file in a folder")
    run.add_argument("--in", dest="in_dir", required=True, type=Path, help="Input folder with audio files")
    run.add_argument("--out", dest="out_dir", type=Path, default=None,
                     help="Output folder (default: <in>/transcripts)")
    _add_common(run)
    run.set_defaults(func=cmd_run)

    single = sub.add_parser("single", help="Transcribe one audio file")
    single.add_argument("file", type=Path, help="Audio file")
    single.add_argument("--out", dest="out_dir", type=Path, default=None,
                        help="Output folder (default: next to the audio file)")
    _add_common(single)
    single.set_defaults(func=cmd_single)

    check = sub.add_parser("check", help="Verify config, model files, and environment")
    check.add_argument("--full", action="store_true", help="Also load the model and run a 1s test")
    _add_common(check)
    check.set_defaults(func=cmd_check)

    review = sub.add_parser("review", help="Aggregate flagged segments into a review report")
    review.add_argument("transcripts_dir", type=Path, help="Transcripts folder (contains *.json)")
    _add_common(review)
    review.set_defaults(func=cmd_review)

    gui = sub.add_parser("gui", help="Open the graphical interface")
    gui.add_argument("--config", type=Path, default=None, help="Path to config.toml")
    gui.set_defaults(func=cmd_gui, verbose=False)

    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
