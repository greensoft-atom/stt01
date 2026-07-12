"""Minimal Tkinter front-end so non-technical users can run batches.

Design: the GUI is a thin shell over pipeline.run_batch — all real logic stays in
the pipeline so CLI and GUI behavior never diverge. The worker runs in a thread;
log records reach the text widget through a queue (Tkinter is not thread-safe).
"""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from courtstt.config import find_config_file, load_config

log = logging.getLogger(__name__)


class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(self.format(record))


class App:
    def __init__(self, root: tk.Tk, config_path: Path | None):
        self.root = root
        self.config_file = find_config_file(config_path)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None

        root.title("stt01 — Offline kuaern Speech-to-Text")
        root.geometry("720x520")
        root.minsize(560, 400)

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # input folder row
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text="Input folder (audio):").pack(side=tk.LEFT)
        self.in_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.in_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(row1, text="Browse...", command=self._browse).pack(side=tk.LEFT)

        # profile row
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row2, text="Profile:").pack(side=tk.LEFT)
        self.profile_var = tk.StringVar(value="dev")
        profiles = self._read_profiles()
        self.profile_box = ttk.Combobox(row2, textvariable=self.profile_var,
                                        values=profiles, state="readonly", width=12)
        self.profile_box.pack(side=tk.LEFT, padx=6)
        ttk.Label(row2, text="Results are saved to the transcripts\\ subfolder of the input folder.").pack(side=tk.LEFT, padx=12)

        # buttons
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=(0, 6))
        self.start_btn = ttk.Button(row3, text="Start", command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        self.cancel_btn = ttk.Button(row3, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=6)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(row3, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        # log area
        self.log_text = scrolledtext.ScrolledText(frame, state=tk.DISABLED, wrap=tk.WORD,
                                                  font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(150, self._drain_log_queue)

    def _read_profiles(self) -> list[str]:
        import tomllib
        raw = tomllib.loads(self.config_file.read_text(encoding="utf-8"))
        return sorted(raw.get("profile", {}).keys()) or ["dev"]

    def _browse(self) -> None:
        folder = filedialog.askdirectory(title="Select the folder containing audio files")
        if folder:
            self.in_var.set(folder)

    def _start(self) -> None:
        in_dir = Path(self.in_var.get().strip() or ".")
        if not self.in_var.get().strip() or not in_dir.is_dir():
            messagebox.showerror("stt01", "Please select an input folder.")
            return
        self.cancel_event.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.profile_box.config(state=tk.DISABLED)
        self.status_var.set("Transcribing...")
        self.worker = threading.Thread(target=self._work, args=(in_dir,), daemon=True)
        self.worker.start()

    def _work(self, in_dir: Path) -> None:
        handler = QueueLogHandler(self.log_queue)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        try:
            cfg = load_config(self.config_file, self.profile_var.get())
            from courtstt.engines.fasterwhisper_engine import FasterWhisperEngine
            from courtstt.pipeline import run_batch

            engine = FasterWhisperEngine(cfg)
            summary = run_batch(cfg, engine, in_dir, in_dir / "transcripts",
                                should_stop=self.cancel_event.is_set)
            flagged = sum(r.flagged for r in summary.done)
            outcome = "Cancelled" if summary.cancelled else "Finished"
            self.log_queue.put(
                f"=== {outcome}: {len(summary.done)} done, {len(summary.skipped)} skipped, "
                f"{len(summary.failed)} failed, {flagged} segment(s) flagged for review ===")
        except Exception as exc:
            log.error("GUI batch failed: %s", exc)
            self.log_queue.put(f"Error: {exc}")
        finally:
            logging.getLogger().removeHandler(handler)
            self.log_queue.put("__DONE__")

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling (stops after the current file)")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__DONE__":
                    self.start_btn.config(state=tk.NORMAL)
                    self.cancel_btn.config(state=tk.DISABLED)
                    self.profile_box.config(state="readonly")
                    self.status_var.set("Idle")
                    continue
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, line + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log_queue)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel("stt01", "Transcription is still running. Quit anyway?"):
                return
            self.cancel_event.set()
        self.root.destroy()


def run_gui(config_path: Path | None = None) -> int:
    root = tk.Tk()
    App(root, config_path)
    root.mainloop()
    return 0
