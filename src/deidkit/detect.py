# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The residual PII detector — what the vault asks about text its seed graph does not cover.

THE CONTRACT (:class:`PiiDetector`). ``detect(text, language)`` returns de-duplicated
``(span_text, kind)`` pairs found in ``text``:

  * ``kind`` is a token kind the vault mints for — ``PERSON``, ``EMAIL``, ``PHONE``, ``IBAN``,
    ``CARD``, ``ID``, ``ACCOUNT``, ``IP`` (see :data:`PRESIDIO_KIND` for the mapping from
    Presidio's labels) — or :data:`VETO_KIND` for a span the detector read as a LOCATION.
    Locations are never tokenised; a veto only stops the same span becoming a person;
  * a span that contains one of the vault's own tokens (``PERSON_1``) must not be returned;
  * a detector that cannot run returns ``[]`` (fail-open: the seed graph has already been
    applied, and the crossing's eligibility check is the hard wall). One that raises aborts
    the tokenisation, which fails closed.

Every span is checked again by the vault before it is enrolled (``deidkit.vault._admit``): a
detector proposes, the vault decides.

:class:`PresidioDetector` is the implementation over Microsoft Presidio's ``AnalyzerEngine`` (or
anything with the same ``analyze`` method). It imports nothing itself; the engine comes from the
caller, or from :func:`deidkit.presidio.analyzer` with the ``presidio`` extra installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from deidkit import namefold as nf

# The vault's own logger: these messages are the vault's residual pass reporting on itself.
log = logging.getLogger("deidkit.vault")

# Presidio entity_type → token prefix. LOCATION and DATE_TIME are intentionally absent:
# jurisdictions and dates must survive the crossing for cross-border questions and deadline
# arithmetic to work.
PRESIDIO_KIND: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "IBAN_CODE": "IBAN",
    "CREDIT_CARD": "CARD",
    "US_SSN": "ID",
    "US_PASSPORT": "ID",
    "US_DRIVER_LICENSE": "ID",
    "US_BANK_NUMBER": "ACCOUNT",
    "MEDICAL_LICENSE": "ID",
    "IP_ADDRESS": "IP",
    "RU_PASSPORT": "ID",
    "RU_INN": "ID",
    "RU_OGRN": "ID",
    "RU_SNILS": "ID",
}
PRESIDIO_ENTITIES = list(PRESIDIO_KIND)

# A label we ASK Presidio for and never tokenise. Locations are preserved by design — that is
# the whole point of keeping jurisdictions readable — but the same span is sometimes ALSO
# offered as a PERSON, and a real vault ended up with a town ("Monte Alto") and a building
# ("Marlow House") as natural persons. Presidio's own location verdict is the cheapest
# gazetteer we have, so it is used as a VETO on the person path and for nothing else: no
# LOCATION ever becomes a token.
VETO_KIND = "__LOCATION__"
ANALYZED_ENTITIES = [*PRESIDIO_ENTITIES, "LOCATION"]

# Kinds whose detection is a REGEX, not a judgement: an IBAN either parses or doesn't. These
# are admitted from the residual pass on sight. Everything else — PERSON above all — has to
# pass ``namefold.admit_person``.
STRUCTURAL_KINDS = frozenset({"EMAIL", "PHONE", "IBAN", "CARD", "ID", "ACCOUNT", "IP"})


class PiiDetector(Protocol):
    """Residual PII detection. See the module docstring for the contract."""

    def detect(self, text: str, language: str) -> list[tuple[str, str]]:
        ...


class PresidioDetector:
    """:class:`PiiDetector` over a Presidio ``AnalyzerEngine``.

    ``analyzer`` is a zero-argument callable returning the engine (typically cached, since
    building one loads the NLP models); by default :func:`deidkit.presidio.analyzer`.
    """

    def __init__(self, analyzer: Callable[[], Any] | None = None) -> None:
        self._analyzer = analyzer

    def detect(self, text: str, language: str) -> list[tuple[str, str]]:
        """Return de-duplicated ``(span_text, kind)`` for residual PII. Fail-open: [] if Presidio
        is unavailable (the known-entity pass already ran; the gateway is the hard wall).

        Runs in the document's PRIMARY language only. Running the English NER over Cyrillic text
        (or vice-versa) is worthless — the wrong-language spaCy model flags every capitalized word
        as a proper noun and obliterates the document. Cross-script names of KNOWN parties are
        already covered deterministically by the entity + surname-alias seeding, so the residual
        pass only needs the primary-language NER plus the language-agnostic regex recognizers
        (emails, IBANs, passport/INN numbers)."""
        analyzer = self._analyzer
        if analyzer is None:
            try:
                from deidkit.presidio import analyzer
            except Exception as exc:  # noqa: BLE001
                log.warning("vault: Presidio unavailable, residual PII pass skipped (%s)", exc)
                return []
        # A language we hold no NER model for gets the language-agnostic REGEX recognisers only
        # (emails, IBANs, phone and ID numbers), never the English NER. Measured 23.09.2026 on a
        # German tenancy file: the English model read "Die Wohnung", "Eine Kündigung" and
        # "Ihr Recht" as people, the name splitter enrolled "Eine" and "Kündigung" as parts of a
        # name, and every "eine" in the scope crossed as PERSON_4 — "für Sie PERSON_4 Härte". Known
        # parties are enrolled deterministically from the scope's graph, so the NER's job here is
        # only the residue, and a residue it cannot read is better left to an independent payload
        # check than invented.
        if language in ("en", "ru"):
            lang, entities = language, ANALYZED_ENTITIES
        else:
            lang = "en"
            entities = [e for e, kind in PRESIDIO_KIND.items() if kind in STRUCTURAL_KINDS]
        try:
            results = analyzer().analyze(
                text=text, language=lang, entities=entities, score_threshold=0.5
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("vault: Presidio analyze failed (%s)", exc)
            return []
        seen: dict[tuple[str, str], tuple[str, str]] = {}
        for r in results:
            val = text[r.start:r.end]
            kind = VETO_KIND if r.entity_type == "LOCATION" else PRESIDIO_KIND.get(r.entity_type)
            if kind and val.strip() and not nf.TOKEN_IN_TEXT.search(val):
                seen.setdefault((nf.key(val), kind), (val, kind))
        return list(seen.values())


__all__ = ["ANALYZED_ENTITIES", "PRESIDIO_ENTITIES", "PRESIDIO_KIND", "PiiDetector",
           "PresidioDetector", "STRUCTURAL_KINDS", "VETO_KIND"]
