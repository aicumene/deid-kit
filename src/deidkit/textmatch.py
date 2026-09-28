# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The pipeline-invariant comparison form of a text — the base of every fold in this package.

WHAT NORMALISATION FOLDS, AND WHY EXACTLY THIS SET. The rule is: fold what a PIPELINE can
change without a human changing a word, and nothing else. Text reaches a de-identifier through
PDF extraction (one document can be extracted two different ways), OCR, and copy-paste out of a
viewer. Those stages routinely alter:

  whitespace          line wrapping, double spaces, NBSP        collapse to one space
  case                headings, ALL-CAPS forms                  casefold (not .lower(): the
                                                                German ß must fold to "ss")
  compatibility forms ligatures ﬁ/ﬃ, fullwidth digits, №        unicodedata NFKC
  invisible controls  soft hyphen U+00AD, ZWSP U+200B           strip category Cf
  punctuation shapes  ‘’“” vs '", ‐‑–—− vs -                    fold to the ASCII form
  line-break hyphens  "physi-\\ncally" from a justified column  drop hyphen-before-newline

This is NOT a loosening toward paraphrase. Every fold is applied to BOTH sides, none of them
maps one word onto a different word, and none maps a non-Latin script onto Latin — NFKC does
not touch Cyrillic homoglyphs. The same function can therefore serve a verbatim quote check
(does this quote appear in that source?) without turning it into a fuzzy one, and the name
matcher in :mod:`deidkit.namefold` builds on exactly this form, so the two never disagree about
what "the same text" is.

Still deliberately absent: stemming, punctuation *stripping*, edit distance, synonym folding.

Pure: stdlib only.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")

# A hyphen immediately before a line break is a typesetting artefact, not a character of the
# word. Dropped BEFORE whitespace collapse, which is the only point the newline still exists.
_LINEBREAK_HYPHEN = re.compile(r"[-‐‑‒–—−]\s*\n\s*")

# Shapes a pipeline swaps freely. Folded to the ASCII form on both sides.
_PUNCT_FOLD = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
    0x2212: "-",                                   # MINUS SIGN
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x2032: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"', 0x2033: '"',
    0x00AB: '"', 0x00BB: '"',                      # guillemets
}


def normalize(text: str) -> str:
    """Pipeline-invariant comparison form, applied to BOTH sides of every comparison.

    See the module docstring for the argument behind each fold. Order matters: de-hyphenate
    while the newline is still there, NFKC before stripping format characters (NFKC creates
    none but resolves ligatures/fullwidth into ASCII the later steps can see), then fold
    punctuation shapes, then collapse whitespace, then casefold.
    """
    s = text or ""
    s = _LINEBREAK_HYPHEN.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = s.translate(_PUNCT_FOLD)
    return _WS.sub(" ", s).strip().casefold()


__all__ = ["normalize"]
