"""Review report: aggregate low-confidence segments across a transcripts folder.

The ⚠ flags inside each .txt show *where* to re-listen; this report collects them
all in one place so a reviewer can work through a whole session batch top-down.
It also doubles as the fine-tuning data-collection worklist (see
docs/FINETUNING_ROADMAP.md): every corrected flagged segment is a training pair.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from courtstt.config import Config
from courtstt.engines.base import Segment
from courtstt.postprocess import format_timestamp, needs_review

log = logging.getLogger(__name__)

REPORT_NAME = "review_report.txt"


@dataclass
class FlaggedItem:
    source_file: str
    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None


def collect_flagged(cfg: Config, transcripts_dir: Path) -> list[FlaggedItem]:
    items: list[FlaggedItem] = []
    for json_path in sorted(transcripts_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for raw in data.get("segments", []):
            seg = Segment(
                start=raw["start"], end=raw["end"], text=raw["text"],
                avg_logprob=raw.get("avg_logprob"), no_speech_prob=raw.get("no_speech_prob"),
            )
            if needs_review(seg, cfg):
                items.append(FlaggedItem(
                    source_file=data.get("source_file", json_path.stem),
                    start=seg.start, end=seg.end, text=seg.text,
                    avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
                ))
    return items


def write_report(items: list[FlaggedItem], transcripts_dir: Path) -> Path:
    lines = [
        "stt01 review report — low-confidence segments requiring human verification",
        f"transcripts folder: {transcripts_dir}",
        f"flagged segments: {len(items)}",
        "",
    ]
    current_file = None
    for item in items:
        if item.source_file != current_file:
            current_file = item.source_file
            lines += [f"### {current_file}", ""]
        confidence = f"logprob={item.avg_logprob:.2f}" if item.avg_logprob is not None else ""
        lines.append(
            f"[{format_timestamp(item.start)} - {format_timestamp(item.end)}] {confidence}"
        )
        lines.append(f"    {item.text}")
        lines.append("")
    out = transcripts_dir / REPORT_NAME
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
