from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from courtstt.config import Config
from courtstt.engines.base import Segment

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, float], None]  # (seconds_transcribed, total_seconds)


class FasterWhisperEngine:
    """CPU int8 faster-whisper engine with hallucination guards always on.

    Guards (see docs/ASR_DESIGN_NOTES.md §5):
    - vad_filter=True: silero VAD skips silence, the main hallucination trigger
    - condition_on_previous_text=False: breaks repetition loops on long files
    """

    def __init__(self, cfg: Config):
        from faster_whisper import WhisperModel  # deferred: heavy import

        self._cfg = cfg
        # hotwords, NOT initial_prompt: initial_prompt only biases the first 30 s
        # window; hotwords re-bias every window, which is what a glossary needs
        # on hour-long recordings (we run with condition_on_previous_text=False).
        self._hotwords = cfg.load_glossary_prompt()
        log.info("Loading model '%s' (compute_type=%s, cpu_threads=%d)",
                 cfg.model_path, cfg.compute_type, cfg.cpu_threads)
        self._model = WhisperModel(
            cfg.model_path,
            device="cpu",
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
        )

    def transcribe(self, audio_path: Path,
                   on_progress: ProgressFn | None = None) -> tuple[list[Segment], dict]:
        cfg = self._cfg
        raw_segments, info = self._model.transcribe(
            str(audio_path),
            language=cfg.language,
            beam_size=cfg.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            hotwords=self._hotwords,
        )
        segments: list[Segment] = []
        for s in raw_segments:  # generator: transcription happens during iteration
            if s.text.strip():
                segments.append(Segment(
                    start=s.start,
                    end=s.end,
                    text=s.text.strip(),
                    avg_logprob=s.avg_logprob,
                    no_speech_prob=s.no_speech_prob,
                ))
            if on_progress:
                on_progress(s.end, info.duration)
        return segments, {
            "engine": "faster-whisper",
            "model": cfg.model,
            "duration": info.duration,
            "language": info.language,
            "language_probability": info.language_probability,
        }
