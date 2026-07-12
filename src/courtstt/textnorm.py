"""Text normalization for CER scoring.

Korean ASR accuracy is measured as CER (character error rate) on normalized text:
punctuation and whitespace differences must not count as recognition errors.

Known limitation: number orthography is NOT unified — "이천 십 팔 년" vs "2018년"
counts as an error even though both are correct readings. Keep reference
transcripts in the same number style you want the model to produce (digits,
matching Whisper's default behavior), or accept the small inflation.
"""
from __future__ import annotations

import unicodedata


def normalize_for_cer(text: str) -> str:
    """Lowercase, strip all punctuation/symbols and ALL whitespace, NFC-compose.

    Whitespace is removed entirely because Korean spacing is inconsistent between
    writers; CER over spaced text would mostly measure spacing style, not recognition.
    """
    text = unicodedata.normalize("NFC", text).lower()
    kept = [
        ch for ch in text
        if not unicodedata.category(ch).startswith(("P", "S", "Z", "C"))
    ]
    return "".join(kept)
