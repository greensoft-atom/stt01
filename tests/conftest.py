from pathlib import Path

import pytest

from courtstt.config import Config
from courtstt.engines.base import Segment


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        profile_name="test", model="fake-model", compute_type="int8",
        beam_size=1, cpu_threads=1, project_root=tmp_path,
        input_extensions=(".wav",), output_formats=("txt", "srt"),
    )


class FakeEngine:
    """Deterministic engine for pipeline tests. Fails on files named fail*."""

    def __init__(self, segments: list[Segment] | None = None):
        self.segments = segments if segments is not None else [
            Segment(0.0, 1.5, "판결을 선고합니다.", avg_logprob=-0.2, no_speech_prob=0.05),
            Segment(4.5, 6.0, "피고인은 무죄.", avg_logprob=-1.4, no_speech_prob=0.1),
        ]
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path, on_progress=None):
        self.calls.append(audio_path)
        if audio_path.name.startswith("fail"):
            raise RuntimeError("corrupt audio")
        if on_progress:
            for s in self.segments:
                on_progress(s.end, 6.0)
        return list(self.segments), {"engine": "fake", "model": "fake-model", "duration": 6.0,
                                     "language": "ko", "language_probability": 1.0}


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


def make_wav(path: Path) -> Path:
    """Tiny valid-enough placeholder; the fake engine never reads it."""
    path.write_bytes(b"RIFF0000WAVE")
    return path
