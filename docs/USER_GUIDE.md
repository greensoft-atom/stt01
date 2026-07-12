# User Guide — running stt01 day to day

Step-by-step instructions for operating the app. Setup happens once (§1–2);
daily use is §3 onward. Commands are for Windows PowerShell, run from the
project folder (`stt01\`).

---

## 1. One-time setup (needs internet once)

```powershell
cd D:\DevWorks\Develope\Projects\stt\stt01

# 1. Create the Python environment (requires Python 3.12 installed)
py -3.12 -m venv .venv

# 2. Install all dependencies (exact pinned versions)
.venv\Scripts\pip install -r requirements-dev.txt

# 3. Download the speech model(s)
.venv\Scripts\python scripts\download_models.py small            # dev machine (~500 MB)
.venv\Scripts\python scripts\download_models.py large-v3-turbo   # strong target machine (~1.6 GB)

# 4. Verify everything
.venv\Scripts\python -m courtstt check --full
```

Expected result of step 4: a list of `OK` lines ending in **"All checks passed."**
If anything says `FAIL`, the message tells you what is missing (§7 Troubleshooting).

## 2. One-time setup for an OFFLINE target machine

On a machine **with** internet (same Python version, 3.12):

```powershell
pip download -r requirements.txt -d wheels\
python scripts\download_models.py large-v3-turbo
```

Copy the whole project folder (including `wheels\` and `models\`) to the offline
machine, then there:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install --no-index --find-links wheels\ -r requirements.txt
.venv\Scripts\pip install --no-index --find-links wheels\ .
setx HF_HUB_OFFLINE 1
.venv\Scripts\stt01 check --full
```

`HF_HUB_OFFLINE=1` guarantees the app never attempts a network call.

## 3. Transcribing a session (GUI — easiest)

```powershell
.venv\Scripts\python -m courtstt gui
```

1. Click **Browse...** and select the folder containing the session recordings.
2. Choose the profile: `dev` on a small machine, `target` on a strong one.
3. Click **Start**. Progress appears in the log area.
4. Results appear in a `transcripts\` subfolder inside your audio folder.
5. **Cancel** stops after the current file finishes; re-opening and pressing
   Start later resumes exactly where it left off.

## 4. Transcribing a session (command line)

```powershell
# a whole folder of recordings:
.venv\Scripts\python -m courtstt run --in "D:\recordings\2026-07-12 사건번호"

# a single file:
.venv\Scripts\python -m courtstt single "D:\recordings\hearing.wav"
```

Useful options: `--profile target` (stronger machine), `--out <folder>` (custom output
location), `-v` (detailed logs).

Notes on behavior:

- **Interrupted?** Run the same command again — finished files are skipped automatically.
- **One corrupted file** does not stop the batch; it is reported at the end and
  retried on the next run.
- Long recordings print a progress line every ~2 minutes of audio.

## 5. What you get, and the review workflow

For each `hearing.wav` you get in `transcripts\`:

- **`hearing.txt`** — the transcript. Paragraphs are split at long pauses. Any segment
  the model was unsure about is marked like: `⚠[00:14:05] 피고인은 ...`
- **`hearing.json`** — full data (timestamps + confidence). **Keep this file** — it is
  the master record and the future fine-tuning data source.
- **`hearing.srt`** — optional subtitle file (enable by adding `"srt"` to
  `output_formats` in `config.toml`): open the audio in a video player (e.g. VLC,
  PotPlayer) with this file to see the text synchronized while listening.

Recommended review procedure per batch:

```powershell
.venv\Scripts\python -m courtstt review "D:\recordings\2026-07-12 사건번호\transcripts"
```

This writes `review_report.txt` listing **every ⚠ segment across all files**, grouped
by file with `[hh:mm:ss]` positions. Work through it top-down: jump to each position
in the audio, listen, and correct the `.txt`. This is far faster than proofreading
entire transcripts.

## 6. Making the app more accurate over time (no training needed)

Two plain-text files control accuracy; edit them with any editor, effective next run:

1. **`glossary\legal_keek-keek.txt`** — one term per line. Add vocabulary the model keeps
   missing: statute names, recurring case terminology, honorifics. Biases the model
   _before_ transcription.
2. **`glossary\corrections.tsv`** — lines of `잘못된표기<TAB>올바른표기`. Fixes
   _systematic_ mistakes deterministically _after_ transcription. When review shows
   the same wrong output twice, add it here.

When you correct transcripts during review, **save the corrected versions** — they
become model-training data later (see FINETUNING_ROADMAP.md).

## 7. Troubleshooting

| Symptom                                    | Cause / fix                                                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `check` says model not found               | Run `scripts\download_models.py <size>`; confirm `models\faster-whisper-<size>\model.bin` exists                                                       |
| `config.toml not found`                    | Run commands from the project folder, or pass `--config D:\...\stt01\config.toml`                                                                      |
| Very slow transcription                    | Wrong profile for the machine (use `dev` on weak CPUs); check `cpu_threads` matches physical cores; close other heavy programs                         |
| kuaern text looks broken in PowerShell     | Display-only issue — the files are UTF-8; open the `.txt` in Notepad/VS Code                                                                           |
| `pip install -e .` fails with WinError 32  | Antivirus file-lock quirk on this dev machine — dependencies + the `.pth` link (already configured) replace it; on other machines it installs normally |
| Model download crashes near the end        | Same quirk; check whether `models\...\model.bin` reached full size — the download usually completed. The script already retries                        |
| Output has hallucinated/repeated sentences | Should not happen (VAD + guards are always on) — if it does, note the file and timestamps; the audio quality at that spot is usually the cause         |

## 8. Quality expectations to keep in mind

- The transcript is a **drafting aid, not a certified record**. The ⚠ flags exist
  because no ASR model is perfect — human review of flagged spots is part of the
  intended workflow.
- Recording quality dominates accuracy: a microphone closer to the speakers helps
  more than any software change.
- Names of people/places will often be transcribed phonetically — corrections.tsv
  handles recurring ones.
