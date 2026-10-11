# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The Rust fold (``deidkit_fold``) answers as the Python one does: on every code point, on random strings
of the characters the fold treats specially, and where it must not start. A name the two folds key
differently would be missed — in clear on someone else's server — so the check is exhaustive, not a
sample. Skipped where the module is not installed."""

import os
import random
import unicodedata

import pytest

deidkit_fold = pytest.importorskip("deidkit_fold")

from deidkit import namefold as nf  # noqa: E402
from deidkit.textmatch import _LINEBREAK_HYPHEN, _PUNCT_FOLD  # noqa: E402


def python(fn, *args):
    """``fn`` with the native fold switched off for the call."""
    saved, nf.NATIVE = nf.NATIVE, None
    try:
        return fn(*args)
    finally:
        nf.NATIVE = saved


def both(fn, *args):
    return fn(*args), python(fn, *args)


def same(text: str) -> None:
    for fn in (nf.fold, nf.confusable, nf.key, nf.fold_haystack):
        native, py = both(fn, text)
        assert native == py, (fn.__name__, text)
    folded = python(nf.fold, text)
    for args in ((folded.text,), (folded.text, folded)):
        native, py = both(nf.strip_marks, *args)
        assert native == py, ("strip_marks", text)


def test_the_native_fold_is_the_one_in_use():
    if os.environ.get("DEIDKIT_NATIVE") == "0":
        pytest.skip("DEIDKIT_NATIVE=0 keeps the Python fold")
    assert nf.NATIVE is not None, "deidkit_fold is installed but does not start: built for another interpreter?"
    assert deidkit_fold.UNIDATA_VERSION == unicodedata.unidata_version
    assert nf.fold("ﬁ Ａ") == nf.Folded(text="fi a", starts=(0, 0, 1, 2), ends=(1, 1, 2, 3), raw_len=3)


CODE_POINTS = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]


@pytest.mark.parametrize("fn", [nf.fold_haystack, nf.key], ids=["fold_haystack", "key"])
def test_every_code_point_folds_the_same(fn):
    """In runs of neighbours (marks after marks, whitespace runs, hyphens) and alone after a letter."""
    for at in range(0, len(CODE_POINTS), 1024):
        run = "".join(CODE_POINTS[at:at + 1024])
        alone = "".join("x" + ch + " " for ch in CODE_POINTS[at:at + 1024])
        for text in (run, alone):
            native, py = both(fn, text)
            assert native == py, (fn.__name__, hex(ord(CODE_POINTS[at])))


# What the fold treats specially, and its neighbours.
SPECIAL = list(
    "abcxyzABCXYZ0129.,;:'\"()-_/\\ \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0     　"
    "‐‑‒–—―−\xad​‌‍⁠﻿\U000e0001"
    "̸̧́̈̇⃝ःि्ִّ"
    "İıßẞΣσςΐΰŉǅǈǋǲﬁﬂﬃﬀﬆＡｚ１①№™½²ⅫÅK"
    "аеойёЁйЙсрхуѕіјԁѵԛԝ"
    "ΗΕαβγεζηικμνορτυχϲάέ"
    "éÉñÑłŁøØđĐðÐþÞħĦŧŦšžčćőűåäöü"
    "‘’‚‛“”„‟′″«»"
    "한글각가각"
    "कषक्षक्षمُحَمَّد"
    "👍🏽‍♀️中文日本語ｶﾞ"
)


def test_random_strings_of_the_special_characters_fold_the_same():
    rnd = random.Random(20261010)
    for _ in range(4000):
        same("".join(rnd.choice(SPECIAL) for _ in range(rnd.randint(0, 60))))


@pytest.mark.parametrize("text", [
    "", " ", "\n", "-\n", "x-\n  y", "x‐ \n\t y", "x -\n", "á", "́a", "́́", "İstanbul",
    "TARVELO DENİZCİLİK", "Straße", "ΟΔΥΣΣΕΥΣ", "Sea ﬁnder", "ΗΕ 123", "  leading and trailing  ",
    "Ms Voss Ltd", "physi-\ncally", "a\x1cb\x1dc", "J̵o̵h̵n",
])
def test_the_cases_the_fold_was_written_for(text):
    same(text)


def test_a_lone_surrogate_goes_the_python_way():
    for text in ("a\ud800b", "\udfff", "Ada\ud83d Brenner"):
        same(text)


def test_a_module_for_another_unicode_version_or_pattern_does_not_start():
    args = (unicodedata.unidata_version, _LINEBREAK_HYPHEN.pattern, r"\s", dict(_PUNCT_FOLD), {"а": "a"},
            {"ł": "l"}, ["й"])
    deidkit_fold.Fold(*args)
    with pytest.raises(ValueError, match="Unicode"):
        deidkit_fold.Fold("1.0.0", *args[1:])
    with pytest.raises(ValueError, match="patterns"):
        deidkit_fold.Fold(args[0], r"-\n", *args[2:])
    with pytest.raises(ValueError, match="one character"):
        deidkit_fold.Fold(*args[:4], {"ae": "a"}, *args[5:])


def test_deidkit_native_0_keeps_the_python(monkeypatch):
    monkeypatch.setenv("DEIDKIT_NATIVE", "0")
    assert nf._native() is None
    monkeypatch.delenv("DEIDKIT_NATIVE")
    assert nf._native() is not None
