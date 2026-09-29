# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A residual detector made of patterns only: e-mail addresses, IBANs, payment cards, phone numbers.

It recognises nothing by judgement, so it has no model to load and no false people to invent. It
is the detector for text where name recognition does harm: source code, logs, command output, in
which a named-entity model reads identifiers, class names and library names as people. Names in
such text are caught by the vault's seed graph, the people and organisations enrolled for the
scope beforehand, not by this detector.

Each pattern is checked beyond its shape where the value carries a check: an IBAN must pass
ISO 13616 mod 97, and a card number the Luhn check and a card network's prefix and length. A
phone number must be written in international form (a leading ``+`` and 8 to 15 digits). Bare
digit runs are left alone: in code and logs they are far more often ports, counters, versions
and timestamps than telephone numbers. Addresses that name a function rather than a person
(``noreply@``, ``postmaster@``) are not reported.

The detector implements :class:`deidkit.detect.PiiDetector`: ``detect(text, language)`` returns
de-duplicated ``(span, kind)`` pairs, and never a span that contains one of the vault's tokens.
"""

from __future__ import annotations

import re

from deidkit import namefold as nf

_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}(?![A-Za-z0-9-])")
# Two letters, two check digits, then 11 to 30 alphanumerics, optionally in groups of four.
_IBAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}(?![A-Za-z0-9])")
_CARD = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
_PHONE = re.compile(r"(?<![\w+])\+\d(?:[ .()-]{0,2}\d){7,14}(?!\d)")

#: Addresses that name a function, not a person. Not tokenised.
_ROLE_LOCAL_PARTS = frozenset({"noreply", "no-reply", "donotreply", "do-not-reply", "mailer-daemon",
                               "postmaster"})


def iban_ok(value: str) -> bool:
    s = value.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in moved)
    return int(digits) % 97 == 1


def _longest_iban(match: str) -> str | None:
    """The longest prefix of a match that is a valid IBAN. The pattern is greedy and can take in
    a following word in capitals ("… 0130 00 EUR"); a valid IBAN inside it is still found."""
    for end in range(len(match), 14, -1):
        cut = match[:end].rstrip()
        if cut != match[:end] or (end < len(match) and match[end] != " "):
            continue
        if iban_ok(cut):
            return cut
    return None


# Issuer prefixes and lengths of the major card networks: a Luhn-valid digit run alone is one
# number in ten, which in logs and code is mostly timestamps and identifiers.
_CARD_NETWORKS = (
    (re.compile(r"^4"), (13, 16, 19)),                                   # Visa
    (re.compile(r"^(?:5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720))"), (16,)),   # Mastercard
    (re.compile(r"^3[47]"), (15,)),                                      # American Express
    (re.compile(r"^(?:6011|65|64[4-9])"), (16, 19)),                      # Discover
    (re.compile(r"^35(?:2[89]|[3-8]\d)"), (16, 19)),                     # JCB
    (re.compile(r"^3(?:0[0-5]|[68])"), (14,)),                            # Diners Club
)


def card_ok(value: str) -> bool:
    digits = "".join(c for c in value if c.isdigit())
    return luhn_ok(digits) and any(p.match(digits) and len(digits) in lengths
                                   for p, lengths in _CARD_NETWORKS)


def luhn_ok(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _blank_escapes(text: str) -> str:
    """The text with each backslash escape (``\\n``, ``\\t``, ``\\u00a0``) replaced by spaces of the
    same length, so that offsets hold and ``\\nlena.voss@…`` is not read as ``nlena.voss@…``."""
    if "\\" not in text:
        return text
    out = list(text)
    i = 0
    while i < len(text) - 1:
        if text[i] == "\\":
            nxt = text[i + 1]
            if nxt in "ntrfbv":
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if nxt in "uU" and i + 5 < len(text) and all(c in "0123456789abcdefABCDEF"
                                                          for c in text[i + 2:i + 6]):
                out[i:i + 6] = [" "] * 6
                i += 6
                continue
            i += 2                      # an escaped character (\\\\, \\") is skipped whole
            continue
        i += 1
    return "".join(out)


def _email_is_personal(value: str) -> bool:
    return value.rpartition("@")[0].lower() not in _ROLE_LOCAL_PARTS


class RegexDetector:
    """:class:`~deidkit.detect.PiiDetector` over patterns only. ``kinds`` narrows what it reports."""

    KINDS = ("EMAIL", "IBAN", "CARD", "PHONE")

    def __init__(self, kinds: tuple[str, ...] | None = None) -> None:
        self.kinds = tuple(kinds or self.KINDS)

    def detect(self, text: str, language: str) -> list[tuple[str, str]]:
        found: dict[tuple[str, str], tuple[str, str]] = {}
        raw, text = text, _blank_escapes(text)

        def add(span: str, kind: str) -> None:
            if kind in self.kinds and span.strip() and not nf.TOKEN_IN_TEXT.search(span):
                found.setdefault((nf.key(span), kind), (span, kind))

        def at(m) -> str:
            return raw[m.start():m.end()]

        for m in _EMAIL.finditer(text):
            if _email_is_personal(at(m)):
                add(at(m), "EMAIL")
        for m in _IBAN.finditer(text):
            iban = _longest_iban(at(m))
            if iban:
                add(iban, "IBAN")
        for m in _CARD.finditer(text):
            if card_ok(at(m)):
                add(at(m), "CARD")
        for m in _PHONE.finditer(text):
            if 8 <= sum(c.isdigit() for c in at(m)) <= 15:
                add(at(m), "PHONE")
        return list(found.values())


__all__ = ["RegexDetector", "card_ok", "iban_ok", "luhn_ok"]
