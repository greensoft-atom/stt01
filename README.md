# stt01 — Offline kuaern STT for Court Session Recordings

Batch speech-to-text pipeline: record a court judgment session → drop the audio file
in a folder → get a human-readable kuaern transcript. Runs fully offline on CPU.

Naming: **stt01** is the project; the importable Python package is **`courtstt`**
(`src/courtstt/`). Run it as `python -m courtstt ...`; when installed normally, both
the `courtstt` and `stt01` commands are available.

Documentation:

- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** — step-by-step: setup, offline deployment,
  GUI and CLI usage, review workflow, troubleshooting. Start here to operate the app.
- [docs/ASR_DESIGN_NOTES.md](docs/ASR_DESIGN_NOTES.md) — design rationale & model choice
- [docs/FINETUNING_ROADMAP.md](docs/FINETUNING_ROADMAP.md) — model upgrade path with the
  concrete training recipe (dataset format, LoRA commands, CT2 conversion)

## Setup (once, with internet)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
python scripts\download_models.py small            # dev model (~500 MB)
python scripts\download_models.py large-v3-turbo   # target-machine model (~1.6 GB), optional
.venv\Scripts\python -m courtstt check --full      # verify everything works
```

> Note (this dev machine): `pip install -e .` fails due to an antivirus file-lock quirk
> on D:\DevWorks; the package is linked via `.venv\Lib\site-packages\stt01_src.pth`
> (pointing at `src\`) instead — that's why commands use `python -m courtstt` here.
> On other machines the normal editable install works fine.

For a fully offline target machine, vendor the wheels:

```powershell
.venv\Scripts\pip download -r requirements.txt -d wheels\
# offline machine:  pip install --no-index --find-links wheels\ -r requirements.txt
```

## Commands

```powershell
# On this dev machine run via:  .venv\Scripts\python -m courtstt <command>
# On a normally-installed machine, 'stt01' and 'courtstt' commands both work.

stt01 run --in D:\recordings\2026-07-12        # transcribe a whole folder
stt01 single D:\recordings\hearing.wav          # transcribe one file
stt01 review D:\recordings\2026-07-12\transcripts   # consolidated worklist of flagged segments
stt01 check [--full]                            # environment self-check (config/model/glossary)
stt01 gui                                       # graphical interface for non-technical users
```

Common options: `--profile dev|target` (default `dev`), `--language xx` (default `auto` —
the engine detects the spoken language per file),
`--config <path>`, `--out <dir>`, `-v`.

## Outputs (in `<input>\transcripts\` by default)

| File                | Purpose                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `<name>.txt`        | The human transcript: paragraphs split on pauses; low-confidence spots marked `⚠[hh:mm:ss]`                              |
| `<name>.json`       | **Master output**: per-segment timestamps + confidence. Keep it — powers review, future speaker labels, fine-tuning data |
| `<name>.srt`        | Optional subtitle view (enable `"srt"` in `output_formats`) — review the text against audio in any video player          |
| `review_report.txt` | From `stt01 review`: all flagged segments across the batch, grouped by file                                              |
| `manifest.json`     | Processing ledger: re-runs skip finished files; failed files retry                                                       |
| `stt01.log`         | Audit log of the batch run                                                                                               |

## Recommended workflow per session batch

1. Copy the session recordings into a dated folder.
2. `stt01 run --in <folder>` (or use the GUI). Interrupted? Just run again — it resumes.
3. `stt01 review <folder>\transcripts` → work through `review_report.txt`, listening at
   the given `[hh:mm:ss]` positions and correcting the `.txt`.
4. Repeated mistakes → add a line to `glossary\corrections.tsv` (`wrong<TAB>right`).
   New courtroom vocabulary → add to `glossary\legal_keek-keek.txt`. Both apply on the next run.

## Improving accuracy

Without retraining (do these first):

1. Grow `glossary/legal_keek-keek.txt` — terms listed there are fed to the model as
   hotwords, biasing recognition toward domain vocabulary in every 30 s window.
2. Grow `glossary/corrections.tsv` — deterministic `wrong<TAB>right` text fixes
   applied after transcription.

> **Both glossary files ship intentionally empty** (project decision, 2026-07-13).
> The pipeline runs normally with them empty — biasing and post-correction are
> simply inactive. Populate them from evidence only: when review of real
> recordings shows the model repeatedly missing or garbling a specific term, add
> that term. The mechanism activates automatically the moment a line appears.

Fine-tuning toolchain (`training/`, deps in `requirements-train.txt`):

- `build_corpus.py` — cut reviewed recordings into training clips + metadata.csv
- `train_lora.py` — LoRA fine-tune (GPU machine; `--smoke` validates on CPU)
- `eval_cer.py` — CER adoption gate on the held-out split (runs on CPU)

Full recipe: [docs/FINETUNING_ROADMAP.md](docs/FINETUNING_ROADMAP.md).

## Configuration (`config.toml`)

- `profile.dev` — whisper-small int8, tuned for this 4-core dev box.
- `profile.target` — large-v3-turbo int8 for stronger deployment machines
  (download the model first, adjust `cpu_threads` to the machine's physical cores).
- `output_formats` — `["txt"]` by default; add `"srt"` for the subtitle view.
- Review thresholds: `review_avg_logprob_below`, `review_no_speech_prob_above`.

## Tests

```powershell
.venv\Scripts\python -m pytest tests -q -p no:cacheprovider
```

(26 tests; `-p no:cacheprovider` avoids the antivirus file-lock quirk on this machine.)

## Architecture

`src/courtstt/engines/` isolates the ASR model behind a small protocol
(`transcribe(path) -> segments + info`) — the SenseVoice benchmark engine or a future
fine-tuned model drops in without touching the pipeline. `pipeline.py` owns batching/resume/isolation,
`postprocess.py` owns readability (paragraphs, ⚠ flags, corrections), `writers.py`
owns output formats, and both CLI and GUI are thin shells over the same pipeline.
