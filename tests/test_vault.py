# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Anonymization vault — reversible per-scope pseudonymization, through the public API.

The guarantees under test:
  1. Real identities (people, organisations) are replaced with stable tokens; the SAME entity
     always gets the SAME token, and tokenize→detokenize round-trips losslessly.
  2. Jurisdictions and dates are deliberately PRESERVED — they must survive the crossing for
     cross-border questions and deadline arithmetic to work.
  3. Generic role references ('the husband') are never tokenized — they're already
     de-identified.
  4. Detokenize replaces longer tokens first, so ORG_12 isn't corrupted by the ORG_1 rule.

The store and the seed graph are the in-memory implementations; the residual detector is a
stub, so the tests are deterministic and need no NER model.
"""

from __future__ import annotations

import re
import uuid

from deidkit import vault
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

M = uuid.uuid4()


class Detector:
    """A residual detector that returns what it was told to, and records what it saw."""

    def __init__(self, found=()):
        self.found = list(found)
        self.seen: list[str] = []

    def detect(self, text, language):
        self.seen.append(text)
        return list(self.found)


def _setup(entities=(), parties=None):
    store, seeds = InMemoryTokenStore(), InMemorySeedSource()
    for e in entities:
        seeds.add_entity(M, e)
    seeds.set_parties(M, parties)
    return store, seeds


def _ent(etype, name, identifier=""):
    return SeedEntity(type=etype, name=name, identifier=identifier)


async def test_roundtrip_hides_identities_keeps_jurisdiction():
    store, seeds = _setup(
        entities=[_ent("individual", "Henry Zielinski"), _ent("company", "Orvalis Shipping")],
        parties=[{"name": "Henry Zielinski", "role": "Director"},
                 {"name": "the husband", "role": "Spouse"}],
    )
    text = "Can Henry Zielinski enforce against Orvalis Shipping in Cyprus on 2024-03-01?"
    red = await vault.tokenize(store, M, text, seeds=seeds, detector=Detector(), language="en")

    # Identities gone; jurisdiction + date preserved.
    assert "Henry Zielinski" not in red.text
    assert "Orvalis Shipping" not in red.text
    assert "Cyprus" in red.text
    assert "2024-03-01" in red.text
    assert "PERSON_1" in red.text and "ORG_1" in red.text   # no salt yet: counter tokens
    assert red.redacted_count >= 2

    # Lossless re-hydration on the trusted plane.
    back = await vault.detokenize(store, M, red.text, mapping=red.mapping)
    assert "Henry Zielinski" in back
    assert "Orvalis Shipping" in back


async def test_generic_roles_never_tokenized():
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")],
                          parties=[{"name": "the husband", "role": "Spouse"}])
    await vault.tokenize(store, M, "the husband disputes Henry Zielinski", seeds=seeds,
                         detector=Detector(), language="en")
    reals = {r.real_value.lower() for r in store.token_rows(M)}
    assert "the husband" not in reals
    assert "henry zielinski" in reals


async def test_bare_surname_is_tokenized_via_alias():
    """A document often uses the bare surname without the full name. The vault seeds
    surname/given-name aliases from known PERSON entities so it's still caught."""
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")])
    red = await vault.tokenize(store, M, "The assets of Zielinski are frozen.", seeds=seeds,
                               detector=Detector(), language="en")
    assert "Zielinski" not in red.text
    back = await vault.detokenize(store, M, red.text, mapping=red.mapping)
    assert "Zielinski" in back


async def test_org_names_are_not_split_into_parts():
    """ORG names must NOT be split — tokenizing 'Silver'/'Bay' would gut common words."""
    store, seeds = _setup(entities=[_ent("company", "Silver Bay Shipping Ltd")])
    red = await vault.tokenize(store, M, "He visited the Silver Mine in Bay County.",
                               seeds=seeds, detector=Detector(), language="en")
    # The standalone common words survive (only the full org name would be tokenized).
    assert "Silver Mine" in red.text
    assert "Bay County" in red.text


async def test_same_entity_gets_stable_token_across_calls():
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")])
    r1 = await vault.tokenize(store, M, "re Henry Zielinski", seeds=seeds, detector=Detector(),
                              language="en")
    r2 = await vault.tokenize(store, M, "again Henry Zielinski", seeds=seeds,
                              detector=Detector(), language="en")
    tok1 = next(t for t, v in r1.mapping.items() if v == "Henry Zielinski")
    tok2 = next(t for t, v in r2.mapping.items() if v == "Henry Zielinski")
    assert tok1 == tok2
    # Only one row for the one entity, despite two tokenize calls.
    assert sum(1 for r in store.token_rows(M) if r.normalized == "henry zielinski") == 1


async def test_residual_pii_is_vaulted():
    store, seeds = _setup()
    red = await vault.tokenize(store, M, "contact jane@example.com please", seeds=seeds,
                               detector=Detector([("jane@example.com", "EMAIL")]), language="en")
    assert "jane@example.com" not in red.text
    assert "EMAIL_1" in red.text
    back = await vault.detokenize(store, M, red.text, mapping=red.mapping)
    assert "jane@example.com" in back


async def test_detokenize_replaces_longer_tokens_first():
    # No store lookups needed: mapping is supplied directly.
    store = InMemoryTokenStore()
    mapping = {"ORG_1": "Acme", "ORG_12": "Globex"}
    out = await vault.detokenize(store, M, "ORG_12 and ORG_1", mapping=mapping)
    assert out == "Globex and Acme"


async def test_residual_pass_runs_on_original_text():
    """Regression: the residual pass must see the ORIGINAL text, never the token-substituted
    form — otherwise the NER re-flags PERSON_1 and mints PERSON_12 → 'PERSON_1'."""
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")])
    det = Detector()
    await vault.tokenize(store, M, "the file of Henry Zielinski", seeds=seeds, detector=det,
                         language="en")
    assert "Henry Zielinski" in det.seen[0]
    assert "PERSON_1" not in det.seen[0]
    # And no row should ever hold a token-shaped real value.
    assert all(not vault._TOKEN_SHAPE.match(r.real_value) for r in store.token_rows(M))


async def test_empty_text_is_noop():
    store = InMemoryTokenStore()
    red = await vault.tokenize(store, M, "", language="en")
    assert red.text == ""
    assert red.redacted_count == 0


async def test_without_a_detector_the_known_graph_is_still_tokenised(caplog):
    """Fail-open on enrichment: no residual detector means no residual pass — logged — while
    the seed graph is applied as always."""
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")])
    with caplog.at_level("WARNING", logger="deidkit.vault"):
        red = await vault.tokenize(store, M, "Henry Zielinski and jane@example.com",
                                   seeds=seeds, language="en")
    assert "Henry Zielinski" not in red.text and "jane@example.com" in red.text
    assert "residual PII pass skipped" in caplog.text


# ── the salted, derived token ────────────────────────────────────────────────

async def test_with_a_salt_tokens_are_derived_from_the_referent():
    """Once a scope has a salt, a new referent's token is HMAC(salt, kind ‖ value): the same
    person gets the same token however the vault is rebuilt, and another scope — another salt —
    gives him an unrelated one."""
    tokens = []
    for salt in (b"\x01" * 32, b"\x01" * 32, b"\x02" * 32):
        store, seeds = _setup(entities=[_ent("company", "Orvalis Shipping"),
                                        _ent("individual", "Henry Zielinski")])
        store.add_salt(M, salt)
        red = await vault.tokenize(store, M, "Henry Zielinski of Orvalis Shipping",
                                   seeds=seeds, detector=Detector(), language="en")
        tok = next(t for t, v in red.mapping.items() if v == "Henry Zielinski")
        assert re.fullmatch(r"PERSON_\d{8}", tok)
        tokens.append(tok)
    assert tokens[0] == tokens[1] != tokens[2]


async def test_a_read_path_never_creates_a_salt():
    """``tokenize`` may be a read in the caller's terms; only ``ensure_salt`` writes one."""
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")])
    await vault.tokenize(store, M, "Henry Zielinski", seeds=seeds, detector=Detector(),
                         language="en")
    assert await store.load_salt(M) is None
    salt = await vault.ensure_salt(store, M)
    assert len(salt) == 32 and await store.load_salt(M) == salt
    assert await vault.ensure_salt(store, M) == salt          # created once


async def test_existing_counter_tokens_do_not_move_when_a_salt_arrives():
    """Tokens issued before the salt stay what they are: re-issuing them in the new shape is
    exactly the silent redirection of history the derivation exists to prevent."""
    store, seeds = _setup(entities=[_ent("individual", "Henry Zielinski")])
    before = await vault.tokenize(store, M, "Henry Zielinski", seeds=seeds, detector=Detector(),
                                  language="en")
    await vault.ensure_salt(store, M)
    after = await vault.tokenize(store, M, "Henry Zielinski", seeds=seeds, detector=Detector(),
                                 language="en")
    assert before.text == after.text == "PERSON_1"
