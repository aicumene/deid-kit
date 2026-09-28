# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The SQLite token store keeps what the vault learned across processes."""

import os
import stat

from deidkit import vault
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.sqlite_store import SQLiteTokenStore


async def _tokenize(store, text):
    seeds = InMemorySeedSource()
    seeds.add_entity("case-1", SeedEntity("individual", "Ada Brenner", role="Director"))
    seeds.add_entity("case-1", SeedEntity("company", "Harrowgate Freight Ltd"))
    await vault.ensure_salt(store, "case-1")
    return await vault.tokenize(store, "case-1", text, seeds=seeds)


async def test_tokens_salt_and_aliases_survive_a_reopen(tmp_path):
    path = tmp_path / "vault.sqlite"
    store = SQLiteTokenStore(path)
    first = await _tokenize(store, "Ada Brenner signed for Harrowgate Freight Ltd.")
    await store.commit()
    salt = await store.load_salt("case-1")
    store.close()

    again = SQLiteTokenStore(path)
    assert await again.load_salt("case-1") == salt
    assert len(await again.load_tokens("case-1")) == len(await store.load_tokens("case-1"))
    second = await vault.tokenize(again, "case-1", "Ada Brenner signed for Harrowgate Freight Ltd.",
                                  seed=False)
    assert second.text == first.text
    assert "Brenner" not in second.text and "Harrowgate" not in second.text


async def test_in_place_edits_are_saved(tmp_path):
    path = tmp_path / "vault.sqlite"
    store = SQLiteTokenStore(path)
    await _tokenize(store, "Ada Brenner")
    row = (await store.load_tokens("case-1"))[0]
    row.status = "retired"
    await store.commit()
    store.close()
    assert (await SQLiteTokenStore(path).load_tokens("case-1"))[0].status == "retired"


async def test_the_file_is_readable_by_its_owner_only(tmp_path):
    path = tmp_path / "vault.sqlite"
    SQLiteTokenStore(path).close()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


async def test_the_restore_table(tmp_path):
    store = SQLiteTokenStore(tmp_path / "vault.sqlite")
    assert store.restore_get("t:x") is None
    store.restore_put("t:x", "PERSON_1 signed.")
    assert store.restore_get("t:x") == "PERSON_1 signed."
