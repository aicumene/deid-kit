# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Lightweight document-language detection.

The vault needs one decision from it: is this text predominantly Cyrillic ('ru') or not ('en')?
That decides which NER language the residual detector runs in and which script a residual
person span must be written in (see ``deidkit.vault._admit``). Heuristic and dependency-free;
a caller that knows the language passes it to ``tokenize(..., language=...)`` instead.
"""

from __future__ import annotations

_CYRILLIC_RANGE = ("Ѐ", "ӿ")
_CYRILLIC_THRESHOLD = 0.30  # share of letters that are Cyrillic → call it Russian


def detect_language(text: str) -> str:
    """Return an ISO 639-1 code: 'ru' if the text is predominantly Cyrillic, else 'en'."""
    if not text:
        return "en"
    cyr = lat = 0
    for ch in text:
        if _CYRILLIC_RANGE[0] <= ch <= _CYRILLIC_RANGE[1]:
            cyr += 1
        elif ch.isascii() and ch.isalpha():
            lat += 1
    letters = cyr + lat
    if letters == 0:
        return "en"
    return "ru" if (cyr / letters) >= _CYRILLIC_THRESHOLD else "en"


__all__ = ["detect_language"]
