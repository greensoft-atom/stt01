from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Segment:
    """One transcribed span of audio. Times are seconds from file start."""

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
        }


class TranscriptionEngine(Protocol):
    """Engine interface. Implementations: faster-whisper (v1); sherpa-onnx SenseVoice (bench/v2)."""

    def transcribe(self, audio_path: Path, on_progress=None) -> tuple[list[Segment], dict]:
        """Return (segments, info). info holds at least 'duration' (seconds).

        on_progress, if given, is called as on_progress(seconds_transcribed, total_seconds)
        while transcription advances through the file.
        """
        ...
