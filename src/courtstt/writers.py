from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from courtstt.config import Config
from courtstt.engines.base import Segment
from courtstt.postprocess import to_plain_text


def write_json(out_path: Path, source: Path, segments: list[Segment], info: dict) -> None:
    """Master output: full segments with timestamps and confidence. Never discard this."""
    payload = {
        "source_file": source.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "info": info,
        "segments": [s.to_dict() for s in segments],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def write_txt(out_path: Path, segments: list[Segment], cfg: Config) -> None:
    """Human deliverable, derived from the segments."""
    text = to_plain_text(segments, cfg, cfg.load_corrections())
    out_path.write_text(text, encoding="utf-8")


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def write_srt(out_path: Path, segments: list[Segment], cfg: Config) -> None:
    """Subtitle view — useful for reviewing the transcript against the audio in a player."""
    corrections = cfg.load_corrections()
    from courtstt.postprocess import apply_corrections

    blocks = []
    for i, seg in enumerate(segments, start=1):
        text = apply_corrections(seg.text, corrections)
        blocks.append(f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n{text}\n")
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def write_outputs(out_dir: Path, source: Path, segments: list[Segment],
                  info: dict, cfg: Config) -> dict[str, str]:
    """Write the JSON master plus every configured human-readable format.

    Returns {format: filename} for the manifest.
    """
    written: dict[str, str] = {}
    json_path = out_dir / (source.stem + ".json")
    write_json(json_path, source, segments, info)
    written["json"] = json_path.name

    if "txt" in cfg.output_formats:
        p = out_dir / (source.stem + ".txt")
        write_txt(p, segments, cfg)
        written["txt"] = p.name
    if "srt" in cfg.output_formats:
        p = out_dir / (source.stem + ".srt")
        write_srt(p, segments, cfg)
        written["srt"] = p.name
    return written
