# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Street addresses and initials — the quasi-identifiers that survive name tokenisation.

A street and house number (and the postal code) become ``ADDRESS_…``; the city and the country
stay. The initials of a scope's known people become that person's token. Both are measured
leaks: with every name tokenised, a letterhead still carried the street, and a signature line
the signatory's initials.
"""

from __future__ import annotations

import pytest

from deidkit import address
from deidkit.vault import _derive_token


@pytest.mark.parametrize("text,expected", [
    ("Frau PERSON_1 - Musterweg 12 - 22303 Hamburg", ["Musterweg 12", "22303"]),
    ("über die Wohnung Musterweg 12, 2. OG links, wegen", ["Musterweg 12"]),
    ("ORG_1 - Seeufer 4 - 20095 Hamburg", ["Seeufer 4", "20095"]),
    ("Hauptstr. 5a", ["Hauptstr. 5a"]),
    ("An der Alster 72", ["An der Alster 72"]),
    ("Flat 3, 221B Baker Street, London NW1 6XE", ["Flat 3", "221B Baker Street", "NW1 6XE"]),
    ("§ 12a Abs. 1 der Satzung am 28. August 2026 für 1.200 Anteile", []),
    ("Hamburg, 28. August 2026\n\nJahresabrechnung", []),                   # a year is not a postcode
    ("eine Kaution von 12000 Euro und 2400 Euro", []),                      # nor is an amount
    ("A-1010 Wien, CH-8001 Zürich", ["A-1010", "CH-8001"]),
])
def test_street_addresses_are_found_and_section_numbers_are_not(text, expected):
    got = [text[s:e] for s, e, _ in address.find_addresses(text)]
    assert got == expected


def test_one_address_is_one_token_across_the_pieces_of_a_request():
    calls = []

    def derive(surface, taken):
        calls.append(surface)
        return f"ADDRESS_{10_000_000 + len(calls)}"

    known = {}
    a = address.tokenize_addresses("Mietvertrag Musterweg 12 + Anlage.pdf, p. 1", derive=derive, known=known)
    b = address.tokenize_addresses("Mietvertrag Musterweg 12 + Anlage.pdf, p. 2", derive=derive, known=known)
    c = address.tokenize_addresses("die Wohnung Musterweg 12, 2. OG links", derive=derive, known=known)
    assert len(calls) == 1                                  # derived once, reused twice
    tok = next(iter(a.mapping))
    assert tok in b.text and tok in c.text and "2. OG links" in c.text


def test_address_tokens_are_stable_and_reversible():
    seen = {}

    def derive(surface, taken):
        return seen.setdefault(" ".join(surface.split()).casefold(), f"ADDRESS_{10_000_000 + len(seen)}")

    red = address.tokenize_addresses("Musterweg 12, 22303 Hamburg; again Musterweg 12.", derive=derive)
    assert "Musterweg" not in red.text and "22303" not in red.text and "Hamburg" in red.text
    toks = list(red.mapping)
    assert len(toks) == 2 and red.text.count(toks[0]) == 2     # the same address, the same token
    restored = red.text
    for t, v in red.mapping.items():
        restored = restored.replace(t, v)
    assert restored == "Musterweg 12, 22303 Hamburg; again Musterweg 12."


def test_address_tokens_come_from_the_vaults_salted_derivation():
    """The derivation the vault uses for a scope, so an address is one token in every request
    of that scope, and nothing new has to be stored."""
    salt = b"\x07" * 32

    def derive(surface, taken):
        return _derive_token(salt, "ADDRESS", surface, taken)

    one = address.tokenize_addresses("Seeufer 4", derive=derive)
    two = address.tokenize_addresses("Seeufer  4", derive=derive)
    assert one.text == two.text and one.text.startswith("ADDRESS_") and len(one.text) == 16


@pytest.mark.parametrize("name,ini", [("MARTIN KESSLER", "MK"), ("Dr. Ada Brenner", "AB"),
                                      ("Herrn Paul Winter", "PW"), ("Madonna", None)])
def test_initials_of_a_name_skip_titles(name, ini):
    assert address.initials_of(name) == ini


def test_the_initials_of_known_people_become_their_token():
    people = {"PERSON_48170392": "MARTIN KESSLER", "PERSON_31415926": "Dr. Ada Brenner"}
    text = "Signed: MK\nWitness: M. K.\nAlso A.B. — but MKG, AW and 'mk' stay."
    out, used = address.tokenize_initials(text, people)
    assert out.count("PERSON_48170392") == 2 and "PERSON_31415926" in out
    assert "MKG" in out and "AW" in out and "'mk'" in out
    assert set(used) == {"PERSON_48170392", "PERSON_31415926"}
