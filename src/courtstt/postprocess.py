from __future__ import annotations

from courtstt.config import Config
from courtstt.engines.base import Segment


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def needs_review(seg: Segment, cfg: Config) -> bool:
    """Low-confidence segments get flagged so a reviewer can jump to the audio position."""
    if seg.avg_logprob is not None and seg.avg_logprob < cfg.review_avg_logprob_below:
        return True
    if seg.no_speech_prob is not None and seg.no_speech_prob > cfg.review_no_speech_prob_above:
        return True
    return False


def apply_corrections(text: str, corrections: list[tuple[str, str]]) -> str:
    for wrong, right in corrections:
        text = text.replace(wrong, right)
    return text


def to_plain_text(segments: list[Segment], cfg: Config,
                  corrections: list[tuple[str, str]] | None = None) -> str:
    """Human-readable transcript: paragraphs on long pauses, ⚠ flags on low confidence.

    The .txt is a derived view — the JSON master (writers.write_json) keeps full detail.
    """
    corrections = corrections or []
    paragraphs: list[list[str]] = []
    current: list[str] = []
    prev_end: float | None = None

    for seg in segments:
        if prev_end is not None and seg.start - prev_end > cfg.paragraph_gap_seconds and current:
            paragraphs.append(current)
            current = []
        text = apply_corrections(seg.text, corrections)
        if needs_review(seg, cfg):
            text = f"⚠[{format_timestamp(seg.start)}] {text}"
        current.append(text)
        prev_end = seg.end

    if current:
        paragraphs.append(current)

    return "\n\n".join(" ".join(p) for p in paragraphs) + ("\n" if paragraphs else "")
