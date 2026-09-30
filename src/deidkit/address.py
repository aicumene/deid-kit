# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Street addresses as tokens — for text that crosses to a cloud model.

The vault deliberately keeps places readable ("we hide *who*, not *where/when*"): a jurisdiction,
an institution, a city are what cross-border questions and deadline arithmetic run on. A **street
address** is a different thing — the home of a natural person, personal data in its own right and
the strongest quasi-identifier a document carries. Measured 23.09.2026 on a demo document set: with
every name tokenised, the cloud still read "ORG_1 - Seeufer 4 - 20095 Hamburg" and the flat
"Musterweg 12, 2. OG links" in the letter, the lease, the title and the file name. A
property-management GmbH plus a street in Hamburg is one line in a public register.

So on the cloud path the street and house number (with the floor/unit that follows) and the postal
code become ``ADDRESS_n``; the city and the country stay, because they are context, not identity.

Tokens come from the vault's own derivation (``HMAC(scope salt, kind ‖ value)``), so the same
address is the same token in every request of a scope and nothing new has to be stored. Detection
leans towards recall: a false positive hides a few words of an address-like phrase, a false
negative is a person's home in the cloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: German street-type endings, as a suffix of the street word ("Musterweg", "Hauptstraße")
#: or as a separate word ("Am Alten Hafen", "An der Alster").
_DE_SUFFIX = (r"(?:straße|strasse|str\.|weg|platz|allee|ufer|damm|ring|gasse|chaussee|pfad|steig|"
              r"markt|brücke|hof|kamp|stieg|twiete|kai|park|berg|feld|tor|graben|wall)")
#: A house number, optionally a range ("12-14", "5a/7") written WITHOUT spaces: a letterhead writes
#: "Musterweg 12 - 22303 Hamburg", and a spaced dash there separates the street from the postcode.
#: A letter suffix counts only when it is a single letter ("12a", "12 a"), never the start of the
#: next word ("12 als …"), and the optional space is consumed only together with it.
_DE_NUMBER = r"\d{1,4}(?:\s?[a-zA-Z](?![\wäöüß]))?(?:[-–/]\d{1,4}(?:[a-zA-Z](?![\wäöüß]))?)?(?!\d)"
#: The floor and side ("2. OG links") are NOT part of the token: "Musterweg 12" in the letterhead and
#: "Musterweg 12, 2. OG links" in the lease are one flat, and two tokens would hide exactly that from
#: the model (measured 23.09.2026). A floor on its own identifies nobody; it stays readable.
_DE_UNIT = ""

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "Musterweg 12, 2. OG links", "Seeufer 4", "Hauptstr. 5a" — and in a file name, where the
    # number is joined by a dash or an underscore: "01-Mietvertrag-Musterweg-12-Kuendigung.md"
    # crossed whole on 30.09.2026 while the same street with a space became a token.
    ("de_street", re.compile(
        r"(?<![\wÄÖÜäöüß])[A-ZÄÖÜ][\wÄÖÜäöüß\-]*" + _DE_SUFFIX + r"(?:\s+|[-_])" + _DE_NUMBER + _DE_UNIT,
        re.UNICODE)),
    # "Große Bleichen 12", "Am Alten Hafen 3", "An der Alster 72"
    ("de_street_words", re.compile(
        r"(?<![\wÄÖÜäöüß])(?:(?:Am|An der|An den|Auf dem|Auf der|Im|In der|In den|Zum|Zur|Hinter der|Unter den)\s+"
        r"[A-ZÄÖÜ][\wÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][\wÄÖÜäöüß\-]+)?|Große[nrs]?\s+[A-ZÄÖÜ][\wÄÖÜäöüß\-]+)\s+"
        + _DE_NUMBER + _DE_UNIT, re.UNICODE)),
    # "221B Baker Street", "10 Downing St", "1 Canada Square"
    ("en_street", re.compile(
        r"(?<![\w])\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'’\-]+\s+){1,3}"
        r"(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Court|Ct\.?|Place|Pl\.?|Square|Sq\.?|"
        r"Terrace|Crescent|Close|Way|Gardens|Row|Mews|Walk|Hill|Boulevard|Blvd\.?|Parade|Grove|Wharf)\b")),
    # "Flat 3, 12 Park Lane" — the flat before the street
    ("en_unit", re.compile(r"(?<![\w])(?:Flat|Apartment|Apt\.?|Unit|Suite)\s+\d{1,4}[A-Za-z]?\b")),
    # UK postcode "SW1A 2AA", "EC4Y 0DT"
    ("uk_postcode", re.compile(r"(?<![\w])[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}(?![\w])")),
    # German postal code before a city on the SAME line: "20095 Hamburg" — the digits only. Five
    # digits exactly: a four-digit run before a capitalised word is, in German, usually a YEAR — the
    # letter's own date "28. August 2026\n\nKündigung …" crossed as "28. August ADDRESS_…" on
    # 23.09.2026 — and an amount before its unit ("12000 Euro") is not a place either.
    ("postcode_city", re.compile(
        r"(?<![\w\d.,§])(\d{5})(?=[ \t]+(?!(?:Euro|EUR|Dollar|USD|GBP|CHF|Pfund|Franken|Stück|Anteil|Aktie|"
        r"Tag|Monat|Jahr|Seite|Quadratmeter|Mal)\w*)[A-ZÄÖÜ][a-zäöüß]{2,})")),
    # Austrian and Swiss codes are four digits — the same shape as a year — so only with the country
    # prefix a letterhead uses: "A-1010 Wien", "CH-8001 Zürich".
    ("postcode_prefixed", re.compile(r"(?<![\w])(?:A|AT|CH)-\d{4}(?![\d])")),
)


@dataclass
class AddressRedaction:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)   # token → address as written
    found: list[tuple[str, str]] = field(default_factory=list)   # (pattern, surface)


def find_addresses(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping ``(start, end, pattern)`` spans, longest first where two overlap."""
    spans: list[tuple[int, int, str]] = []
    for name, pat in _PATTERNS:
        for m in pat.finditer(text or ""):
            s, e = (m.start(1), m.end(1)) if m.groups() else (m.start(), m.end())
            spans.append((s, e, name))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[tuple[int, int, str]] = []
    for s, e, name in spans:
        if out and s < out[-1][1]:
            continue
        out.append((s, e, name))
    return out


def tokenize_addresses(text: str, *, derive, taken: set[str] | None = None,
                       known: dict[str, str] | None = None) -> AddressRedaction:
    """Replace every street address in ``text`` with a stable ``ADDRESS_n`` token.

    ``derive(surface, taken) -> token`` is the vault's HMAC derivation bound to the scope salt;
    ``known`` (normalised surface → token) keeps one address one token across the pieces of a
    single request even where the derivation would be asked twice.
    """
    taken = set(taken or ())
    # NOT copied: the caller shares one ``known`` across every piece of one request. A copy made each
    # repeat look new, and because the address's own token was then "taken", the derivation moved on
    # to the next — one file name crossed with four different tokens (23.09.2026).
    known = known if known is not None else {}
    out, pos = [], 0
    red = AddressRedaction(text=text or "")
    for s, e, name in find_addresses(text or ""):
        surface = text[s:e]
        norm = " ".join(surface.split()).casefold()
        tok = known.get(norm)
        if tok is None:
            tok = derive(surface, taken)
            taken.add(tok)
            known[norm] = tok
        red.mapping.setdefault(tok, surface)
        red.found.append((name, surface))
        out.append(text[pos:s])
        out.append(tok)
        pos = e
    out.append((text or "")[pos:])
    red.text = "".join(out)
    return red


def initials_of(name: str) -> str | None:
    """"MARTIN KESSLER" → "MK", "Dr. Ada Brenner" → "AB"; None for a one-part name.

    Forms of address are skipped, so a title never becomes an initial."""
    from deidkit import namefold as nf
    parts = [p for p in nf.name_parts(name or "") if p[:1].isalpha() and not nf.is_honorific(p)
             and not nf.is_initial(p)]
    if not 2 <= len(parts) <= 4:
        return None
    return "".join(p[0].upper() for p in parts)


def tokenize_initials(text: str, people: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Replace the initials of the scope's known people ("MK", "M.K.", "M. K.") with their token.

    Measured 23.09.2026: with every name tokenised, a signature line still carried the signatory's
    initials. Only UPPERCASE initials standing alone count, only for people this scope knows —
    an abbreviation that happens to share the letters of nobody in the scope is left alone.
    Returns the text and ``{token: name}`` for the tokens it used.
    """
    used: dict[str, str] = {}
    out = text or ""
    for token, name in people.items():
        ini = initials_of(name)
        if not ini:
            continue
        pat = re.compile(r"(?<![\w.])" + r"\.?\s?".join(re.escape(ch) for ch in ini) + r"\.?(?![\w])")
        new, n = pat.subn(token, out)
        if n:
            out = new
            used[token] = name
    return out, used


__all__ = ["AddressRedaction", "find_addresses", "initials_of", "tokenize_addresses", "tokenize_initials"]
