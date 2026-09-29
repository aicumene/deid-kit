# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A person written surname first ("Brenner, Ada") is one token, and a list of names is not read
across. MEASURED 29.09.2026: in ``04 Brenner, Ada - note.md`` the surname matched and the given
name crossed in clear."""

import re

import pytest

from deidkit.patterns import RegexDetector
from deidkit.proxy.engine import ScopeEngine
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

PERSON = re.compile(r"PERSON_\d+")


@pytest.fixture
def tokenise():
    seeds = InMemorySeedSource()
    for name in ("Ada Brenner", "Tom Brenner", "Eva Smith", "Jörg Müller"):
        seeds.add_entity("case-1", SeedEntity("individual", name))
    eng = ScopeEngine(InMemoryTokenStore(), "case-1", seeds=seeds, detector=RegexDetector())

    async def run(*texts: str) -> list[str]:
        return [r.text for r in await eng.tokenize_all(list(texts))]
    return run


async def test_surname_first_is_one_token_and_the_given_name_does_not_cross(tokenise):
    path, capitals, umlauts, natural = await tokenise(
        "letters/04 Brenner, Ada - note.md", "BRENNER Ada signed the lease.",
        "MUELLER, JOERG - Vollmacht.pdf", "Ada Brenner signed.")
    assert re.fullmatch(r"letters/04 PERSON_\d+ - note\.md", path)
    assert re.fullmatch(r"PERSON_\d+ signed the lease\.", capitals)
    assert re.fullmatch(r"PERSON_\d+ - Vollmacht\.pdf", umlauts)
    assert PERSON.findall(path) == PERSON.findall(capitals) == PERSON.findall(natural)


async def test_a_list_of_names_is_not_read_across(tokenise):
    natural, inverted, mixed = await tokenise(
        "Tenants: Ada Brenner, Tom Brenner.", "Tenants: Brenner, Ada; Brenner, Tom.",
        "Tom Brenner, Eva Smith and Ada Brenner.")
    ada, tom = PERSON.findall(natural)
    assert ada != tom and PERSON.findall(inverted) == [ada, tom]
    assert PERSON.findall(mixed)[0] == tom and PERSON.findall(mixed)[2] == ada
    assert len(set(PERSON.findall(mixed))) == 3
    assert not re.search(r"\b(Ada|Tom|Eva)\b", natural + inverted + mixed)


async def test_look_alikes_stay_as_written(tokenise):
    statute, adam = await tokenise("The ADA requires access.", "Brenner, Adam is another tenant.")
    assert statute == "The ADA requires access."
    assert re.fullmatch(r"PERSON_\d+, Adam is another tenant\.", adam)   # someone else: only the surname
