"""Download CTranslate2 whisper models into models/ for offline use.

Run on an internet-connected machine:
    python scripts/download_models.py small
    python scripts/download_models.py large-v3-turbo

Afterwards the pipeline runs with no network (config.toml points at models/...).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Use the classic HTTP download path instead of Xet chunk reconstruction.
# On this machine the Xet "Reconstructing" phase deadlocks against the
# antivirus/file-watcher (observed hung 40+ min at 0% CPU on 2026-07-12).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Systran publishes ready-made CT2 conversions of the official OpenAI weights.
REPOS = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def download(size: str, attempts: int = 8) -> Path:
    from huggingface_hub import snapshot_download

    repo = REPOS[size]
    target = MODELS_DIR / f"faster-whisper-{size}"
    print(f"Downloading {repo} -> {target}")
    # Antivirus on Windows briefly locks freshly written files, which breaks the
    # rename at the end of each file download (WinError 32). Downloads resume,
    # so retrying through the transient lock is safe and sufficient.
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(repo_id=repo, local_dir=target)
            break
        except PermissionError as exc:
            if attempt == attempts:
                raise
            print(f"Transient file lock (attempt {attempt}/{attempts}): {exc}. Retrying in 5s...")
            time.sleep(5)
    print(f"OK: {target}")
    return target


if __name__ == "__main__":
    sizes = sys.argv[1:] or ["small"]
    unknown = [s for s in sizes if s not in REPOS]
    if unknown:
        print(f"Unknown model(s): {unknown}. Choose from: {', '.join(REPOS)}")
        raise SystemExit(2)
    for s in sizes:
        download(s)
