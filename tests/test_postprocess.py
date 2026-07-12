from pathlib import Path

from courtstt.config import Config
from courtstt.engines.base import Segment
from courtstt.postprocess import apply_corrections, format_timestamp, needs_review, to_plain_text


def make_cfg(**overrides) -> Config:
    defaults = dict(
        profile_name="test", model="x", compute_type="int8", beam_size=1, cpu_threads=1,
        project_root=Path("."),
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3725.9) == "01:02:05"


def test_paragraph_break_on_long_gap():
    cfg = make_cfg(paragraph_gap_seconds=2.0)
    segments = [
        Segment(0.0, 1.0, "첫 문장입니다.", avg_logprob=-0.2, no_speech_prob=0.0),
        Segment(1.2, 2.0, "이어지는 문장.", avg_logprob=-0.2, no_speech_prob=0.0),
        Segment(5.0, 6.0, "새 문단입니다.", avg_logprob=-0.2, no_speech_prob=0.0),
    ]
    text = to_plain_text(segments, cfg)
    paragraphs = text.strip().split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0] == "첫 문장입니다. 이어지는 문장."
    assert paragraphs[1] == "새 문단입니다."


def test_low_confidence_flagged_with_timestamp():
    cfg = make_cfg(review_avg_logprob_below=-1.0)
    seg = Segment(65.0, 66.0, "불명확한 부분", avg_logprob=-1.5, no_speech_prob=0.0)
    assert needs_review(seg, cfg)
    text = to_plain_text([seg], cfg)
    assert "⚠[00:01:05] 불명확한 부분" in text


def test_confident_segment_not_flagged():
    cfg = make_cfg()
    seg = Segment(0.0, 1.0, "명확한 문장", avg_logprob=-0.3, no_speech_prob=0.1)
    assert not needs_review(seg, cfg)


def test_apply_corrections():
    pairs = [("형법 재 삼백 조", "형법 제300조")]
    assert apply_corrections("피고인은 형법 재 삼백 조 위반", pairs) == "피고인은 형법 제300조 위반"


def test_empty_segments_gives_empty_text():
    assert to_plain_text([], make_cfg()) == ""
