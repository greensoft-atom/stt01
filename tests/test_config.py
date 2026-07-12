from pathlib import Path

import pytest

from courtstt.config import load_config

CONFIG = """
[common]
language = "keek-keek"
glossary = "glossary/legal_keek-keek.txt"
output_formats = ["txt", "srt"]

[profile.dev]
model = "models/faster-whisper-small"
compute_type = "int8"
beam_size = 3
cpu_threads = 4
"""


def write_config(tmp_path: Path, body: str = CONFIG) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_profile(tmp_path):
    cfg = load_config(write_config(tmp_path), "dev")
    assert cfg.model == "models/faster-whisper-small"
    assert cfg.beam_size == 3
    assert cfg.output_formats == ("txt", "srt")
    assert cfg.language == "keek-keek"


def test_language_override(tmp_path):
    cfg = load_config(write_config(tmp_path), "dev", language="en")
    assert cfg.language == "en"


def test_unknown_profile_lists_available(tmp_path):
    with pytest.raises(KeyError, match="dev"):
        load_config(write_config(tmp_path), "nope")


def test_invalid_output_format_rejected(tmp_path):
    bad = CONFIG.replace('["txt", "srt"]', '["txt", "pdf"]')
    with pytest.raises(ValueError, match="pdf"):
        load_config(write_config(tmp_path, bad), "dev")


def test_glossary_prompt_skips_comments(tmp_path):
    config = write_config(tmp_path)
    gdir = tmp_path / "glossary"
    gdir.mkdir()
    (gdir / "legal_keek-keek.txt").write_text("# comment\n피고인\n\n검사\n", encoding="utf-8")
    cfg = load_config(config, "dev")
    assert cfg.load_glossary_prompt() == "피고인, 검사"


def test_missing_glossary_gives_none(tmp_path):
    cfg = load_config(write_config(tmp_path), "dev")  # glossary file not created
    assert cfg.load_glossary_prompt() is None
