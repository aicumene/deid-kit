# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The residual detector over Presidio, without Presidio: a stub engine stands in for it.

What the vault needs from a detector is ``(span, kind)`` pairs; what this adapter adds is the
language policy (NER only in a language a model exists for), the mapping of Presidio's labels
onto token kinds, the LOCATION veto, and failing open.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from deidkit.detect import VETO_KIND, PresidioDetector
from deidkit.lang import detect_language


class _Engine:
    def __init__(self, results):
        self.results, self.calls = results, []

    def analyze(self, *, text, language, entities, score_threshold):
        self.calls.append((language, set(entities), score_threshold))
        return self.results


def _hit(text, span, entity_type):
    start = text.index(span)
    return SimpleNamespace(start=start, end=start + len(span), entity_type=entity_type)


def test_presidio_labels_become_token_kinds_and_a_location_is_only_a_veto():
    text = "Mail ann.lee@example.org, call +44 20 7946 0000, meet Ann Lee in Marlow House."
    engine = _Engine([
        _hit(text, "ann.lee@example.org", "EMAIL_ADDRESS"),
        _hit(text, "+44 20 7946 0000", "PHONE_NUMBER"),
        _hit(text, "Ann Lee", "PERSON"),
        _hit(text, "Marlow House", "LOCATION"),
        _hit(text, "Marlow House", "PERSON"),
        _hit(text, "Mail", "DATE_TIME"),               # a label the vault never tokenises
    ])
    found = PresidioDetector(lambda: engine).detect(text, "en")
    assert found == [("ann.lee@example.org", "EMAIL"), ("+44 20 7946 0000", "PHONE"),
                     ("Ann Lee", "PERSON"), ("Marlow House", VETO_KIND),
                     ("Marlow House", "PERSON")]
    language, entities, threshold = engine.calls[0]
    assert language == "en" and "PERSON" in entities and "LOCATION" in entities
    assert threshold == 0.5


def test_a_span_that_carries_a_token_is_never_returned_and_duplicates_fold():
    text = "PERSON_1@example.org and Ann Lee, ANN LEE"
    engine = _Engine([_hit(text, "PERSON_1@example.org", "EMAIL_ADDRESS"),
                      _hit(text, "Ann Lee", "PERSON"), _hit(text, "ANN LEE", "PERSON")])
    assert PresidioDetector(lambda: engine).detect(text, "en") == [("Ann Lee", "PERSON")]


def test_the_detector_fails_open(caplog):
    def broken():
        raise RuntimeError("model not loaded")

    with caplog.at_level("WARNING", logger="deidkit.vault"):
        assert PresidioDetector(broken).detect("Ann Lee", "en") == []
    assert "vault: Presidio analyze failed (model not loaded)" in caplog.text


def test_without_presidio_installed_the_default_engine_fails_open(monkeypatch, caplog):
    from deidkit import presidio
    presidio.analyzer.cache_clear()
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    with caplog.at_level("WARNING", logger="deidkit.vault"):
        assert PresidioDetector().detect("Ann Lee", "en") == []
    assert "vault: Presidio analyze failed" in caplog.text
    presidio.analyzer.cache_clear()


def test_language_detection_is_a_cyrillic_share():
    cyrillic_word = "".join(chr(c) for c in (0x0434, 0x043E, 0x043C))     # a three-letter word
    assert detect_language("Ann Lee signed the letter.") == "en"
    assert detect_language(f"{cyrillic_word} {cyrillic_word} Lee") == "ru"
    assert detect_language("") == "en" and detect_language("12 34") == "en"
