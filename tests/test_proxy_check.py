# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""``deid-proxy check`` finds a known value that crossed, even glued to an escape."""

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from deidkit import vault                                   # noqa: E402
from deidkit.proxy.check import run                         # noqa: E402
from deidkit.proxy.cli import seed_files                    # noqa: E402
from deidkit.seeds import InMemorySeedSource, SeedEntity    # noqa: E402
from deidkit.sqlite_store import SQLiteTokenStore           # noqa: E402


async def _vault(path):
    store = SQLiteTokenStore(path)
    seeds = InMemorySeedSource()
    seeds.add_entity("case-1", SeedEntity("individual", "Lena Voss"))
    seeds.add_entity("case-1", SeedEntity("company", "Harrowgate Freight Ltd"))
    await vault.ensure_salt(store, "case-1")
    await vault.seed_from_scope(store, "case-1", seeds)
    await store.commit()
    store.close()


def _record(path, text):
    row = {"dir": "out", "body": {"input": [{"type": "message", "content": [
        {"type": "input_text", "text": "Output: " + json.dumps({"output": text})},
        {"type": "input_text", "text": "see the filename argument"}]}]}}
    path.write_text(json.dumps(row) + "\n")


def test_a_value_glued_to_an_escape_is_reported(tmp_path, capsys):
    asyncio.run(_vault(tmp_path / "v.sqlite"))
    _record(tmp_path / "r.jsonl", "Dear Ms PERSON_1,\n\nHarrowgate Freight Ltd asks.")
    assert run(tmp_path / "r.jsonl", tmp_path / "v.sqlite", ["case-1"]) == 1
    assert "LEAK" in capsys.readouterr().out


def test_noise_inside_other_words_is_shown_masked_and_passes(tmp_path, capsys):
    asyncio.run(_vault(tmp_path / "v.sqlite"))
    _record(tmp_path / "r.jsonl", "Dear Ms PERSON_1,\n\nORG_2 asks.")
    assert run(tmp_path / "r.jsonl", tmp_path / "v.sqlite", ["case-1"]) == 0
    out = capsys.readouterr().out
    assert "fi****me" in out and "Lena" not in out and "Harrowgate" not in out


def test_seed_files_from_a_directory(tmp_path):
    (tmp_path / "b.toml").write_text('scope = "b"\n')
    (tmp_path / "a.toml").write_text('scope = "a"\n')
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in seed_files([], str(tmp_path))] == ["a.toml", "b.toml"]
