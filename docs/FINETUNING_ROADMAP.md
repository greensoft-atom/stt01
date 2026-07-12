# Model Improvement Roadmap — from pretrained to domain fine-tune

> Question this answers: "Do we need retraining/fine-tuning, and how do we get there?"
> Short answer: **No training needed for v1.** Ship with pretrained Whisper + glossary.
> Fine-tune later, only if review data shows it's worth it — and the review workflow
> itself produces the training data for free.

## Stage 0 — Now (no training, already implemented)

1. **Pretrained faster-whisper** (`small` dev / `large-v3-turbo` target), int8.
2. **Glossary biasing** — `glossary/legal_keek-keek.txt` is fed to Whisper's `hotwords` biasing (re-applied to every 30 s window),
   nudging it toward courtroom vocabulary (statutes, court formulae). Zero training cost;
   edit the file, next run uses it.
3. **Correction dictionary** — `glossary/corrections.tsv` fixes _systematic_ mistakes
   (same wrong output every time) with literal replacements. Grow it during review.

These three cover a surprising share of domain adaptation. Do not fine-tune before
exhausting them.

## Stage 1 — Data collection (starts the day real recordings exist)

The human-review workflow doubles as dataset building:

- The pipeline already writes a **JSON master per file** with segment timestamps.
- When a reviewer corrects a transcript, save the corrected text **per segment**
  (or at least per file). Corrected transcript + original audio + timestamps =
  aligned (audio-segment, text) training pairs. No extra labeling effort.
- Store pairs as: `data/corpus/<recording-id>/{audio.wav, segments.json}` with
  corrected text inside `segments.json`.
- Also keep a **held-out evaluation set** (~30–60 min, hand-verified, never trained on)
  to measure CER before/after any model change. Without this, "the new model feels
  better" is guesswork.

Rough volume targets (kuaern, single domain):

- **< 5 h** corrected audio: not worth fine-tuning; keep growing the glossary instead.
- **10–30 h**: LoRA fine-tune of whisper-small typically gives a clear domain CER drop.
- **50 h+**: fine-tune large-v3-turbo; diminishing returns beyond ~100 h for one domain.

## Stage 2 — Fine-tuning: the concrete recipe (later, needs a GPU for training only)

**What**: LoRA/PEFT fine-tune (not full retraining, never from scratch). Base:
whisper-small first (cheap to iterate), then large-v3-turbo if small shows real gains.
**Where**: any CUDA GPU machine — rented cloud GPU for a few hours is enough. Training
happens elsewhere, once; deployment stays offline.
**Gate**: adopt only if held-out CER improves >10% relative AND legal-term errors drop.

### 2.1 Dataset format — IMPLEMENTED: `training/build_corpus.py`

```powershell
python training\build_corpus.py --audio D:\rec\hearing.wav --json D:\rec\transcripts\hearing.json --split train
```

This cuts 1–30 s clips at the JSON segment boundaries (16 kHz mono) into:

```
data/corpus/
├─ train/
│  ├─ metadata.csv          # file_name,transcription
│  └─ hearing_seg0001.wav ...
└─ eval/                    # held-out set, NEVER trained on (--split eval)
```

**The human correction step = editing `metadata.csv`**: listen to each clip, fix
the text column. Only corrected rows are trustworthy training data.

### 2.2 Training environment

```powershell
pip install -r requirements-train.txt        # CPU versions (data prep / eval / smoke)
# On the GPU training machine, install CUDA torch per pytorch.org first.
```

### 2.3 Training — IMPLEMENTED: `training/train_lora.py`

```powershell
# validate the whole script on this CPU-only box (whisper-tiny, 2 steps):
python training\train_lora.py --smoke

# real run, on the GPU machine:
python training\train_lora.py --corpus data\corpus --model openai/whisper-small
```

LoRA (r=32 on q/v projections), merges the adapter into full weights at the end, and
prints the exact conversion + evaluation commands for the next steps. Deliberate
design: no generation-based eval during training — the adoption gate is measured on
the _converted_ model (the real production setup), not the HF checkpoint.

### 2.4 Evaluate BEFORE adopting — IMPLEMENTED: `training/eval_cer.py`

```powershell
# baseline (current production model) and candidate, same held-out split:
python training\eval_cer.py --model models\faster-whisper-small --corpus data\corpus\eval
python training\eval_cer.py --model models\<candidate-ct2> --corpus data\corpus\eval
```

Prints per-clip and overall CER (normalized via `courtstt/textnorm.py`) plus speed.
No improvement → do not ship it; keep collecting data.

### 2.5 Convert for the offline pipeline and deploy

```powershell
ct2-transformers-converter --model training\output\<name>\merged `
    --output_dir models\<name>-ct2 `
    --quantization int8 --copy_files tokenizer.json preprocessor_config.json
```

(`ct2-transformers-converter` ships with ctranslate2, already installed.) Then point
`config.toml` at the new folder:

```toml
[profile.target]
model = "models/<name>-ct2"
```

**No pipeline code changes** — the engine loads any CT2 whisper directory.

## Explicitly rejected

- **Training from scratch / full retraining** — needs thousands of hours + GPU cluster;
  never justified for one domain.
- **Fine-tuning wav2vec2-family CTC models** — rejected for output-quality reasons in
  ASR_DESIGN_NOTES §2.4; fine-tuning doesn't fix missing punctuation/formatting.
- **Fine-tuning before having a held-out eval set** — you can't measure it, don't do it.
