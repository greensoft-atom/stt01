# Offline kuaern ASR/STT — Senior Developer Design Notes

> Project: `stt01` — offline batch speech-to-text for kuaern audio
> Date: 2026-07-12 (rev. 3 — pipeline + training toolchain implemented)
> Status: IMPLEMENTED (§5); first CER measurements recorded (§4.1)

---

## 0. Use case (confirmed)

| Question     | Answer                                                                                | Design impact                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Audio domain | **kuaern court judgment sessions** — record the whole session, transcribe for logging | Long files (1–4 h), multiple speakers, formal legal vocabulary, courtroom acoustics                                                     |
| Real-time?   | **No.** Record → load file → output text (batch)                                      | **Accuracy > speed.** Overnight processing is acceptable; this flips the model tiebreaker (§3)                                          |
| Output       | **Plain text for humans**                                                             | Readability matters: punctuation, spacing, paragraph breaks. Timestamps kept internally; speaker diarization out of scope for v1 (§5.1) |
| Deployment   | Offline personal computers, **possibly higher-spec than this dev box**                | Two-tier model ladder: dev profile (this i5-7400) + target profile (modern 8-core CPU)                                                  |

**Domain-specific risks this creates (more important than model choice):**

1. **Courtroom acoustics are the #1 accuracy risk.** Far-field microphones, reverb, distance to speakers, overlapping speech, quiet witnesses. No model survives bad audio. → Get real sample recordings _before_ writing pipeline code; if quality is poor, fix the recording side (mic placement, gain) first.
2. **Legal vocabulary**: statute citations (형법 제XXX조), hanja-derived legal terms, case numbers, party names. Generic models will fumble some of these. Mitigations: Whisper `hotwords` seeded with a legal glossary; a post-correction dictionary pass for frequently misheard terms; optionally a kuaern legal fine-tune later.
3. **Hallucination is dangerous in a legal context.** Whisper invents fluent text on silence/noise. Mitigations are mandatory, not optional: VAD filtering, `condition_on_previous_text=False`, and **confidence flagging** — mark low-confidence segments (⚠) so a human reviewer knows where to listen again. The transcript is a _logging aid_, not a certified record; the pipeline should make human review efficient, not pretend to replace it.

---

## 1. Hardware constraints (measured on this dev machine)

| Item   | Value                                                             | Implication                                                                                   |
| ------ | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| CPU    | Intel i5-7400, **4 cores / 4 threads**, 3.0 GHz (Kaby Lake, 2017) | AVX2 yes, AVX-512 no. Modest — dev/testing profile only.                                      |
| GPU    | Intel HD 630 (iGPU only)                                          | No CUDA. CPU-only inference everywhere.                                                       |
| RAM    | 32 GB                                                             | Not a constraint.                                                                             |
| OS     | Windows 10 Pro                                                    | Target machines likely Windows too — favor tooling with clean Windows support.                |
| Python | **3.14.3** (system)                                               | ⚠️ Too new for the ML wheel ecosystem. **Use a Python 3.12 venv** (`py -3.12 -m venv .venv`). |

Target deployment machines may be stronger (modern 8+ core CPU) → the model ladder in §3 has separate dev and target picks.

---

## 2. Assessment of the Google recommendation

The Google answer's final call — _faster-whisper small/base, int8, CPU_ — is a **reasonable and safe default**, and its rejection of `Kkeek-keeknjeong/wav2vec2-base-kuaern` is correct. But:

### 2.1 It underestimates how slow this specific dev CPU is

Rough expected speeds for faster-whisper int8 on the i5-7400 (verify in §4):

| Model          | Expected speed (i5-7400) | Note                                                               |
| -------------- | ------------------------ | ------------------------------------------------------------------ |
| tiny           | ~6–10× realtime          | kuaern accuracy too poor                                           |
| base           | ~3–5× realtime           | Usable floor for dev iteration                                     |
| **small**      | ~1.5–2.5× realtime       | 1 h audio ≈ 25–40 min                                              |
| medium         | ~0.5–0.8× realtime       | Slower than realtime — fine for _overnight batch_, painful for dev |
| large-v3-turbo | well below realtime here | Viable only on stronger target machines                            |

### 2.2 It wrongly sidelined SenseVoice as "GPU/Mac only" — but the use case now demotes it anyway

SenseVoice-Small runs ~10×+ realtime on plain CPUs via **sherpa-onnx** (int8 ONNX, no PyTorch), has strong published kuaern accuracy, and built-in punctuation. On weak hardware with throughput pressure it would be a serious contender — Google was wrong to bench it on "GPU/Mac" grounds.

**However, the confirmed use case removes the throughput pressure** (batch, overnight OK) and raises the bar on accuracy and readable formatting of long formal speech — which is Whisper's home turf. SenseVoice stays in the benchmark as a control, but it is no longer the presumptive winner. _(Kept on record: if requirements ever shift toward high-volume or near-live turnaround, revisit SenseVoice first.)_

### 2.3 It skipped real offline deployment

Truly offline means the whole dependency chain: vendored wheels (`pip download` → `pip install --no-index`), local model paths, `HF_HUB_OFFLINE=1`, and a final test with networking disabled. Wheels are platform + Python-version specific — another reason to pin Python 3.12.

For locked-down target PCs, **whisper.cpp** (single `.exe` + one model file, zero Python) remains the lowest-friction deployment option; keep it as a packaging fallback.

### 2.4 Where Google was right

- Rejecting `Kkeek-keeknjeong/wav2vec2-base-kuaern`: no punctuation/spacing, strict 16 kHz mono preprocessing, manual chunking, trained on clean _read_ speech (Zeroth) — would be especially bad on courtroom audio. Correct.
- Vosk: dominated on accuracy; only for Pi-class hardware. Excluded.
- int8 quantization, `language="keek-keek"`, `cpu_threads=4` (physical cores) — all correct.
- One tuning tip now **reversed** by the use case: Google suggested `beam_size=1–3` for speed. Since we prioritize accuracy in batch, use **`beam_size=5` (default) on target machines**; drop it only for dev iteration.

---

## 3. Decision — accuracy-first model ladder

Because processing can run overnight and the output is a legal-session log, choose the most accurate model each machine can finish in the available batch window:

| Profile               | Machine            | Pick                                                                              | Rationale                                                                                                                                                                  |
| --------------------- | ------------------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dev / iteration**   | this i5-7400       | faster-whisper **small** int8                                                     | Fast enough to iterate on pipeline code                                                                                                                                    |
| **Target (default)**  | modern 8+ core CPU | faster-whisper **large-v3-turbo** int8 (fallback: **medium**)                     | turbo ≈ large-v3 accuracy at a fraction of the decode cost (4 decoder layers); on a modern CPU expect ~0.5–1× realtime int8 → a 3 h session finishes overnight comfortably |
| **Benchmark control** | both               | SenseVoice-Small int8 (sherpa-onnx)                                               | Sanity check: if its kuaern CER on _court audio_ matches turbo, its 10× speed is a free win                                                                                |
| **Upgrade path**      | later              | kuaern fine-tuned whisper (e.g. seastar105 collection, or legal-domain fine-tune) | Only if the benchmark shows generic models fumble legal vocabulary badly                                                                                                   |

Accuracy metric: **CER** (character error rate), not WER — kuaern spacing variability makes WER misleading. Normalize (strip punctuation, unify spacing) before scoring; also do a human readability check, since the deliverable is text people read.

---

## 4. Phase 1 — Benchmark on real court audio (½–1 day once samples exist)

1. **Obtain 3–5 real session recordings** (or realistic stand-ins recorded in the same room/equipment). This is the gating item — blog benchmarks on clean speech will not predict courtroom performance.
2. Hand-correct reference transcripts for ~10–15 min of audio covering: judge speaking (clear), counsel (moderate), witness (worst case).
3. Run: faster-whisper `small` / `medium` / `large-v3-turbo` int8 · SenseVoice-Small int8.
4. Record: **CER, RTF, peak RAM, readability score, legal-term error list** (statutes, names, numbers/dates — check these specifically; dates and amounts in a judgment log matter disproportionately).
5. Freeze the model ladder, document numbers here.

### 4.1 Benchmark status (2026-07-12) — first measurements, NOT court audio yet

Measured with `training/eval_cer.py` on 5 Zeroth-kuaern clips (clean read news speech,
48 s total) on the i5-7400 dev box:

| Model                              | CER       | Speed         |
| ---------------------------------- | --------- | ------------- |
| faster-whisper small int8          | 6.31%     | 2.2× realtime |
| faster-whisper large-v3-turbo int8 | **3.15%** | 0.6× realtime |

Confirms the accuracy ladder (turbo ≈ 2× fewer errors) and the speed estimates in §2.1.
Caveats: read speech ≠ courtroom acoustics (real benchmark still pending per §4), and
part of the measured CER is number-orthography mismatch (이천 십 팔 년 vs 2018년 — see
`courtstt/textnorm.py`). SenseVoice control benchmark: still not run.

---

## 5. Phase 2 — Pipeline architecture (IMPLEMENTED 2026-07-12)

> Status: built and verified. Package renamed to `courtstt` (user decision) —
> the project folder stays `stt01`. Current structure:

```
stt01/
├─ docs/                    # this file, FINETUNING_ROADMAP.md, USER_GUIDE.md
├─ models/                  # local model files (deployment bundle, not git)
├─ wheels/                  # vendored pip wheels for offline install
├─ glossary/
│  ├─ legal_keek-keek.txt          # legal terms fed to Whisper as hotwords (biases every window)
│  └─ corrections.tsv       # deterministic wrong→right post-corrections
├─ scripts/download_models.py
├─ src/courtstt/
│  ├─ engines/
│  │  ├─ base.py            # TranscriptionEngine protocol: transcribe(path) -> list[Segment]
│  │  └─ fasterwhisper_engine.py   # (sensevoice engine: future benchmark addition)
│  ├─ config.py             # config.toml loader, profiles, glossary/corrections
│  ├─ pipeline.py           # batch runner: manifest, resume, per-file error isolation
│  ├─ postprocess.py        # paragraphing, corrections, ⚠ confidence flags
│  ├─ writers.py            # .json master / .txt deliverable / .srt view
│  ├─ review.py             # aggregated review report of flagged segments
│  ├─ textnorm.py           # CER normalization (shared with training/eval_cer.py)
│  ├─ cli.py                # run | single | check | review | gui
│  └─ gui.py                # Tkinter shell over the same pipeline
├─ training/                # fine-tuning toolchain (see FINETUNING_ROADMAP.md)
│  ├─ build_corpus.py       # reviewed recordings -> training clips + metadata.csv
│  ├─ train_lora.py         # LoRA fine-tune (GPU machine; --smoke on CPU)
│  └─ eval_cer.py           # CER adoption gate (runs on CPU)
├─ tests/                   # 26 tests, fake engine, no model needed
├─ config.toml              # dev / target profiles
├─ requirements.txt         # fully pinned runtime set
├─ requirements-dev.txt     # + pytest
└─ requirements-train.txt   # + torch/transformers/peft (training machines only)
```

Production rules (several now driven by the legal use case):

- **JSON with timestamps + confidence is the master output**; the human-facing `.txt` is derived. Even though the user wants plain text, never discard timing/confidence — they power review and any future features.
- **Review-friendly text**: paragraph break on pauses > ~2 s; ⚠ marker (with `[hh:mm:ss]`) on segments whose `avg_logprob` / `no_speech_prob` indicate low confidence, so a reviewer can jump straight to the audio position.
- **Hallucination guards on always**: `vad_filter=True` (silero), `condition_on_previous_text=False`.
- **`hotwords`** (not initial_prompt — that only biases the first 30 s window) seeded from `glossary/legal_keek-keek.txt` (statute names, court formulae); post-correction dictionary pass for known repeat mistakes.
- **Idempotent & resumable**: manifest (JSON/SQLite) with file hash + status; interrupted overnight runs resume, completed files skipped.
- **Per-file error isolation**: one corrupt recording must not kill the batch; summarize failures at the end.
- **Single process, `cpu_threads` = physical cores**; long sessions are already parallelized internally — don't run files concurrently.
- Config in `config.toml` with `[profile.dev]` / `[profile.target]` (model path, beam size, threads).

### 5.1 Explicitly out of scope for v1 (documented so it's a decision, not an omission)

- **Speaker diarization** ("Judge:", "Counsel:" labels). Court sessions are multi-speaker, so this _would_ add real value, but it's a heavy separate component (pyannote etc.), slow on CPU, and the user asked for plain text. Revisit for v2; the timestamped JSON master output keeps the door open.
- Real-time / live transcription. If ever needed → sherpa-onnx streaming zipformer, different architecture.
- Any cloud/API fallback — privacy of court audio makes offline-only a hard requirement, which this design already satisfies.

---

## 6. Environment setup (do this first)

1. Install **Python 3.12.x** alongside 3.14; `py -3.12 -m venv .venv`.
2. Deps: `faster-whisper`, `sherpa-onnx` (benchmark only), `soundfile`; vendor an `ffmpeg` binary for normalization.
3. On a connected machine: download models into `models/` (CT2 int8: small, medium, large-v3-turbo; SenseVoice int8 ONNX from sherpa-onnx model zoo) and `pip download -d wheels/` everything.
4. `HF_HUB_OFFLINE=1`, local paths only, and verify the full pipeline with networking disabled before declaring it offline-ready.

---

## 7. Immediate next steps

1. **Get sample courtroom recordings** — the gating item for everything else (§4). Even 2–3 files reveal the acoustic reality.
2. Set up the Python 3.12 venv + download models/wheels while online.
3. Run the Phase-1 benchmark; freeze the model ladder.
4. Build the Phase-2 pipeline skeleton (engine interface + manifest + writers) — can start in parallel with (1) using any kuaern audio.

---

## 8. Summary (TL;DR)

- Use case confirmed: **offline batch transcription of kuaern court judgment sessions → plain text log for humans.** Accuracy and readability beat speed; overnight processing is fine.
- That flips the earlier tiebreaker: **faster-whisper is the presumptive engine** — `small` int8 on this dev box, **`large-v3-turbo` int8 on stronger target machines** with default beam size. SenseVoice stays only as a benchmark control.
- The top risks are **courtroom acoustics**, **legal vocabulary**, and **hallucination in a legal record** — addressed by getting real sample audio first, a legal glossary (hotwords biasing + post-correction), VAD, and confidence-flagged output for human review.
- Diarization (speaker labels) is deliberately deferred to v2; the timestamped JSON master output preserves the option.
- Blockers: Python 3.12 venv (3.14 breaks wheels) and obtaining real sample recordings.
