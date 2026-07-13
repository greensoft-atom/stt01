from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_OUTPUT_FORMATS = ("txt", "srt")  # json master is always written


@dataclass
class Config:
    # profile
    profile_name: str
    model: str
    compute_type: str
    beam_size: int
    cpu_threads: int
    # common — None means the engine auto-detects the spoken language per file
    language: str | None = None
    glossary_path: Path | None = None
    corrections_path: Path | None = None
    paragraph_gap_seconds: float = 2.0
    review_avg_logprob_below: float = -1.0
    review_no_speech_prob_above: float = 0.6
    input_extensions: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    output_formats: tuple[str, ...] = ("txt",)
    project_root: Path = field(default_factory=Path.cwd)

    @property
    def model_path(self) -> str:
        """Resolve the model against the project root if it is a relative directory."""
        p = self.project_root / self.model
        return str(p) if p.exists() else self.model

    def load_glossary_prompt(self) -> str | None:
        """Build the initial_prompt string from the glossary file."""
        if not self.glossary_path or not self.glossary_path.exists():
            return None
        terms = [
            line.strip()
            for line in self.glossary_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not terms:
            return None
        # Whisper's prompt window is 224 tokens; stay safely under it.
        prompt = ""
        for term in terms:
            candidate = f"{prompt}, {term}" if prompt else term
            if len(candidate) > 400:
                break
            prompt = candidate
        return prompt

    def load_corrections(self) -> list[tuple[str, str]]:
        """Load WRONG<TAB>RIGHT replacement pairs."""
        if not self.corrections_path or not self.corrections_path.exists():
            return []
        pairs: list[tuple[str, str]] = []
        for line in self.corrections_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0]:
                pairs.append((parts[0], parts[1]))
        return pairs


def find_config_file(explicit: Path | None = None) -> Path:
    """Locate config.toml: explicit arg > current directory > project root (dev checkeek-keekut)."""
    candidates = [explicit] if explicit else []
    candidates += [
        Path.cwd() / "config.toml",
        Path(__file__).resolve().parents[2] / "config.toml",
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError(
        "config.toml not found. Run from the project folder or pass --config <path>."
    )


def load_config(config_file: Path, profile: str, language: str | None = None) -> Config:
    raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
    common = raw.get("common", {})
    profiles = raw.get("profile", {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise KeyError(f"profile '{profile}' not found in {config_file} (available: {available})")
    prof = profiles[profile]
    root = config_file.parent

    formats = tuple(common.get("output_formats", ["txt"]))
    invalid = [f for f in formats if f not in VALID_OUTPUT_FORMATS]
    if invalid:
        raise ValueError(f"invalid output_formats {invalid}; choose from {list(VALID_OUTPUT_FORMATS)}")

    def _path(key: str) -> Path | None:
        value = common.get(key)
        return (root / value) if value else None

    # "auto" (or empty) -> None: the engine auto-detects the spoken language from
    # the first seconds of each file instead of being told a fixed code. Any other
    # value is passed to the engine verbatim and must be a code the engine accepts.
    lang: str | None = language or common.get("language", "auto")
    if not lang or str(lang).strip().lower() == "auto":
        lang = None

    return Config(
        profile_name=profile,
        model=prof["model"],
        compute_type=prof.get("compute_type", "int8"),
        beam_size=int(prof.get("beam_size", 5)),
        cpu_threads=int(prof.get("cpu_threads", 4)),
        language=lang,
        glossary_path=_path("glossary"),
        corrections_path=_path("corrections"),
        paragraph_gap_seconds=float(common.get("paragraph_gap_seconds", 2.0)),
        review_avg_logprob_below=float(common.get("review_avg_logprob_below", -1.0)),
        review_no_speech_prob_above=float(common.get("review_no_speech_prob_above", 0.6)),
        input_extensions=tuple(common.get("input_extensions", [".wav", ".mp3", ".m4a", ".flac", ".ogg"])),
        output_formats=formats,
        project_root=root,
    )
