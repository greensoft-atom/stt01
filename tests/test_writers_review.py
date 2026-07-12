import json
from pathlib import Path

from courtstt.engines.base import Segment
from courtstt.review import collect_flagged, write_report
from courtstt.writers import _srt_time, write_json, write_srt


def test_srt_time_format():
    assert _srt_time(0.0) == "00:00:00,000"
    assert _srt_time(3661.5) == "01:01:01,500"


def test_write_srt(cfg, tmp_path):
    segments = [Segment(0.0, 1.5, "판결을 선고합니다.")]
    out = tmp_path / "a.srt"
    write_srt(out, segments, cfg)
    content = out.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,500\n판결을 선고합니다." in content


def test_review_report_collects_only_flagged(cfg, tmp_path):
    segments = [
        Segment(0.0, 1.5, "명확한 문장.", avg_logprob=-0.2, no_speech_prob=0.05),
        Segment(4.5, 6.0, "불명확한 문장.", avg_logprob=-1.4, no_speech_prob=0.1),
    ]
    write_json(tmp_path / "a.json", Path("a.wav"), segments, {"duration": 6.0})
    # manifest.json must be ignored by the collector
    (tmp_path / "manifest.json").write_text(json.dumps({"x": {"status": "done"}}), encoding="utf-8")

    items = collect_flagged(cfg, tmp_path)
    assert len(items) == 1
    assert items[0].text == "불명확한 문장."
    assert items[0].source_file == "a.wav"

    report = write_report(items, tmp_path)
    body = report.read_text(encoding="utf-8")
    assert "flagged segments: 1" in body
    assert "[00:00:04 - 00:00:06]" in body
    assert "불명확한 문장." in body
