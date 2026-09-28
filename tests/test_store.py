# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The plug-in points: the token store contract, the seed source, and the read-only paths.

A deployment implements :class:`TokenStore` over its own storage. What the vault relies on is
pinned here: loads return fresh lists of the SAME live rows, adds return the row, a unit of work
that changed nothing does not commit, and the read paths never write.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from deidkit import vault
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore
from harness import M, Scope, alias_row, ent, token_row


async def test_loads_return_fresh_lists_of_the_same_live_rows():
    store = InMemoryTokenStore()
    row = store.add_token(M, token="PERSON_1", kind="PERSON", real_value="Ann Lee",
                          normalized="ann lee", status="active", attributes=None)
    a, b = await store.load_tokens(M), await store.load_tokens(M)
    assert a == b == [row] and a is not b and a[0] is row
    a.append("scratch")                                   # the vault appends to what it loaded
    assert await store.load_tokens(M) == [row]
    assert await store.load_tokens(uuid.uuid4()) == []    # scopes do not see each other
    assert await store.load_salt(M) is None
    store.add_salt(M, b"s" * 32)
    assert await store.load_salt(M) == b"s" * 32


async def test_a_unit_of_work_that_changed_nothing_does_not_commit():
    s = Scope(entities=[ent("individual", "Ann Lee")])
    await s.seed()
    assert s.store.commits == 1
    await s.seed()
    assert s.store.commits == 1


async def test_a_store_with_its_own_row_class_works():
    """The vault reads and edits rows by attribute name only; an ORM model or any object with
    the same attributes is a row."""

    class Store(InMemoryTokenStore):
        def add_token(self, scope_id, **kw):
            row = SimpleNamespace(canonical_token=None, **kw)
            self.token_rows(scope_id).append(row)
            return row

        def add_alias(self, scope_id, **kw):
            row = SimpleNamespace(**kw)
            self.alias_rows(scope_id).append(row)
            return row

    store, seeds = Store(), InMemorySeedSource({M: [SeedEntity("individual", "Ada Brenner")]})
    red = await vault.tokenize(store, M, "Ada Brenner and Brenner", seeds=seeds, language="en")
    assert red.text == "PERSON_1 and PERSON_1"
    assert all(isinstance(r, SimpleNamespace) for r in store.token_rows(M))


async def test_parties_are_seeded_by_role():
    """A party whose role says company is enrolled whole; an individual is split into parts."""
    s = Scope(parties=[{"name": "Harbor Homes GmbH", "role": "Landlord company"},
                       {"name": "Ada Brenner", "role": "Individual"},
                       {"name": "the tenant", "role": "Individual"}])
    await s.seed()
    kinds = {r.real_value: r.kind for r in s.rows}
    assert kinds == {"Harbor Homes GmbH": "ORG", "Ada Brenner": "PERSON"}
    idx = await s.index()
    assert idx.token_for("Brenner") == s.token_of("Ada Brenner")
    assert idx.token_for("Harbor") is None


def _recorded(store: InMemoryTokenStore) -> tuple:
    return (len(store.token_rows(M)), len(store.alias_rows(M)), store.commits,
            [(r.status, r.canonical_token, dict(r.attributes or {})) for r in store.token_rows(M)])


async def test_the_read_paths_never_write():
    s = Scope(entities=[ent("individual", "Ann Lee", role="Director")])
    s.detections = [("Lena Kraus", "PERSON")]
    await s.tokenize("Ann Lee met Lena Kraus.", language="en")
    before = _recorded(s.store)
    assert await vault.residual_surfaces(s.store, M, "Lena Kraus and Ann Lee") == ["Ann Lee",
                                                                                   "Lena Kraus"]
    assert set((await vault.active_people(s.store, M)).values()) == {"Ann Lee", "Lena Kraus"}
    await vault.token_attributes(s.store, M)
    await vault.detokenize(s.store, M, "PERSON_1 and PERSON_2")
    await vault.graph_residual_surfaces(s.store, M, "Ann Lee")
    assert _recorded(s.store) == before


async def test_the_graph_index_holds_only_what_the_seed_graph_knows():
    """For a de-identified fragment the residual guesses stay out — whatever their kind — and
    only what the seed graph knows is matched: its people, organisations and their identifiers,
    and never a bare form of address as a match key."""
    iban = "DE02120300000000202051"
    s = Scope(entities=[ent("individual", "Paul Winter", role="Tenant", attributes={"iban": iban})])
    s.detections = [("Lena Kraus", "PERSON"), ("mail@example.org", "EMAIL")]
    await s.tokenize(f"Paul Winter, Lena Kraus, mail@example.org, {iban}", language="en")
    s.aliases.append(alias_row(s.token_of("Paul Winter"), "Herrn", "herrn", "fragment"))

    red = await vault.graph_tokenize(
        s.store, M, f"Herrn Paul Winter and Lena Kraus: mail@example.org, {iban}", seeds=s.seeds)
    assert red.text.startswith("Herrn PERSON_") and "Lena Kraus" in red.text
    assert "mail@example.org" in red.text and iban not in red.text
    assert await vault.graph_residual_surfaces(s.store, M, "Lena Kraus and Paul Winter") == [
        "Paul Winter"]


async def test_token_attributes_answer_for_a_retired_token_with_its_canonical_row():
    s = Scope(rows=[
        token_row("PERSON_1", "PERSON", "Ann Lee", attributes={"roles": ["Director"]}),
        token_row("PERSON_2", "PERSON", "Ann", status="retired", canonical_token="PERSON_1"),
    ])
    attrs = await vault.token_attributes(s.store, M, ["PERSON_2", "PERSON_9"])
    assert attrs == {"PERSON_2": ("PERSON", {"roles": ["Director"]})}
