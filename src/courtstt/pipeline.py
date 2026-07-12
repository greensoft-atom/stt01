from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from courtstt.config import Config
from courtstt.engines.base import TranscriptionEngine
from courtstt.postprocess import format_timestamp
from courtstt.writers import write_outputs

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
LOG_NAME = "stt01.log"
PROGRESS_LOG_INTERVAL_SECONDS = 120.0  # of audio time, so hour-long files show life signs


def file_key(path: Path) -> str:
    """Cheap identity for resume: name + size + mtime (hashing hour-long audio is too slow)."""
    st = path.stat()
    return f"{path.name}|{st.st_size}|{int(st.st_mtime)}"


@dataclass
class FileResult:
    name: str
    status: str                 # "done" | "failed"
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    segments: int = 0
    flagged: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BatchSummary:
    done: list[FileResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[FileResult] = field(default_factory=list)
    cancelled: bool = False


class Manifest:
    """Processed-file ledger in the output directory; makes batch runs resumable."""

    def __init__(self, out_dir: Path):
        self._path = out_dir / MANIFEST_NAME
        self._data: dict[str, dict] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def is_done(self, key: str) -> bool:
        return self._data.get(key, {}).get("status") == "done"

    def record(self, key: str, **fields) -> None:
        self._data[key] = {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")


def find_audio_files(cfg: Config, in_dir: Path) -> list[Path]:
    return sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in cfg.input_extensions
    )


def attach_file_log(out_dir: Path) -> logging.Handler:
    """Mirror pipeline logs into <out_dir>/stt01.log so every batch leaves an audit trail."""
    handler = logging.FileHandler(out_dir / LOG_NAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def process_file(cfg: Config, engine: TranscriptionEngine, path: Path, out_dir: Path) -> FileResult:
    """Transcribe one file and write all outputs. Raises on failure."""
    from courtstt.postprocess import needs_review

    started = time.perf_counter()
    last_logged = 0.0

    def on_progress(position: float, total: float) -> None:
        nonlocal last_logged
        if position - last_logged >= PROGRESS_LOG_INTERVAL_SECONDS:
            last_logged = position
            log.info("  ... %s / %s", format_timestamp(position), format_timestamp(total))

    segments, info = engine.transcribe(path, on_progress=on_progress)
    elapsed = time.perf_counter() - started

    outputs = write_outputs(out_dir, path, segments, info, cfg)
    flagged = sum(1 for s in segments if needs_review(s, cfg))
    duration = info.get("duration") or 0.0
    rtf = (duration / elapsed) if elapsed > 0 else 0.0
    log.info("Done: %s (%.1fs audio in %.1fs, %.1fx realtime, %d segments, %d flagged for review)",
             path.name, duration, elapsed, rtf, len(segments), flagged)
    return FileResult(
        name=path.name, status="done",
        audio_seconds=round(duration, 1), elapsed_seconds=round(elapsed, 1),
        segments=len(segments), flagged=flagged, outputs=outputs,
    )


def run_batch(cfg: Config, engine: TranscriptionEngine, in_dir: Path, out_dir: Path,
              should_stop: Callable[[], bool] | None = None) -> BatchSummary:
    """Transcribe every audio file in in_dir. Resumable, per-file error isolation.

    should_stop is checked between files (used by the GUI cancel button).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handler = attach_file_log(out_dir)
    try:
        manifest = Manifest(out_dir)
        summary = BatchSummary()

        files = find_audio_files(cfg, in_dir)
        if not files:
            log.warning("No audio files found in %s (extensions: %s)",
                        in_dir, ", ".join(cfg.input_extensions))
            return summary

        log.info("Batch: %d file(s) from %s -> %s (profile=%s, model=%s)",
                 len(files), in_dir, out_dir, cfg.profile_name, cfg.model)
        for path in files:
            if should_stop and should_stop():
                log.warning("Cancelled before: %s", path.name)
                summary.cancelled = True
                break
            key = file_key(path)
            if manifest.is_done(key):
                log.info("Skip (already done): %s", path.name)
                summary.skipped.append(path.name)
                continue

            try:
                log.info("Transcribing: %s", path.name)
                result = process_file(cfg, engine, path, out_dir)
                manifest.record(key, status="done", **result.outputs,
                                audio_seconds=result.audio_seconds,
                                elapsed_seconds=result.elapsed_seconds,
                                flagged=result.flagged)
                summary.done.append(result)
            except Exception as exc:  # per-file isolation: one bad file must not kill the batch
                log.error("FAILED: %s — %s", path.name, exc)
                log.debug("%s", traceback.format_exc())
                manifest.record(key, status="failed", error=str(exc))
                summary.failed.append(FileResult(name=path.name, status="failed", error=str(exc)))

        return summary
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
