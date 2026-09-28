# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A :class:`~deidkit.store.TokenStore` in one SQLite file, for one process on one machine.

The vault's contract asks for live rows edited in place and saved at ``commit()``. This store
keeps the rows in memory exactly as :class:`~deidkit.store.InMemoryTokenStore` does, reads the
file once when it opens, and writes each scope's rows back in one transaction at every commit.
Rewriting a scope is cheap at the sizes a vault reaches (thousands of rows), and it saves the
in-place edits (a retired token, a moved alias) without tracking which row changed.

The file holds the real values and the salts: it is created readable by its owner only. One
process at a time: two processes writing one file would each overwrite the other's scope.

It also keeps one table that only the local proxy uses: ``restore``, the model's own words as
it wrote them, keyed by a hash of the version handed to the agent. It holds tokens only, never a
real value. The proxy's audit and record are separate files.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from deidkit.store import AliasRow, InMemoryTokenStore, ScopeId, TokenRow

_SCHEMA = """
create table if not exists tokens (
    scope text not null, seq integer not null, token text not null, kind text not null,
    real_value text not null, normalized text not null, status text not null,
    canonical_token text, attributes text,
    primary key (scope, seq));
create table if not exists aliases (
    scope text not null, seq integer not null, token text not null, surface text not null,
    normalized text not null, source text not null,
    primary key (scope, seq));
create table if not exists salts (scope text primary key, salt blob not null);
create table if not exists restore (key text primary key, value text not null, at real not null);
"""


class SQLiteTokenStore(InMemoryTokenStore):
    """A :class:`~deidkit.store.TokenStore` persisted to one SQLite file. Scope ids are strings."""

    def __init__(self, path: str | os.PathLike) -> None:
        super().__init__()
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists()
        self._db = sqlite3.connect(self.path)
        if fresh:
            os.chmod(self.path, 0o600)
        self._db.executescript(_SCHEMA)
        self._load()

    def _load(self) -> None:
        for scope, token, kind, real, norm, status, canon, attrs in self._db.execute(
                "select scope, token, kind, real_value, normalized, status, canonical_token, "
                "attributes from tokens order by scope, seq"):
            self.token_rows(scope).append(TokenRow(
                scope_id=scope, token=token, kind=kind, real_value=real, normalized=norm,
                status=status, canonical_token=canon,
                attributes=json.loads(attrs) if attrs else None))
        for scope, token, surface, norm, source in self._db.execute(
                "select scope, token, surface, normalized, source from aliases order by scope, seq"):
            self.alias_rows(scope).append(AliasRow(scope_id=scope, token=token, surface=surface,
                                                   normalized=norm, source=source))
        for scope, salt in self._db.execute("select scope, salt from salts"):
            self._salts[scope] = bytes(salt)

    # ── TokenStore ───────────────────────────────────────────────────────────
    def add_token(self, scope_id: ScopeId, **kw) -> TokenRow:
        return super().add_token(_scope(scope_id), **kw)

    def add_alias(self, scope_id: ScopeId, **kw) -> AliasRow:
        return super().add_alias(_scope(scope_id), **kw)

    def add_salt(self, scope_id: ScopeId, salt: bytes) -> None:
        super().add_salt(_scope(scope_id), salt)

    async def commit(self) -> None:
        await super().commit()
        with self._db:
            for scope, rows in self._tokens.items():
                self._db.execute("delete from tokens where scope = ?", (scope,))
                self._db.executemany(
                    "insert into tokens values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(scope, i, r.token, r.kind, r.real_value, r.normalized, r.status,
                      r.canonical_token, json.dumps(r.attributes) if r.attributes else None)
                     for i, r in enumerate(rows)])
            for scope, rows in self._aliases.items():
                self._db.execute("delete from aliases where scope = ?", (scope,))
                self._db.executemany(
                    "insert into aliases values (?, ?, ?, ?, ?, ?)",
                    [(scope, i, r.token, r.surface, r.normalized, r.source)
                     for i, r in enumerate(rows)])
            self._db.executemany("insert or replace into salts values (?, ?)",
                                 list(self._salts.items()))

    # ── the proxy's restore table ────────────────────────────────────────────
    def restore_get(self, key: str) -> str | None:
        row = self._db.execute("select value from restore where key = ?", (key,)).fetchone()
        return row[0] if row else None

    def restore_put(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute("insert or replace into restore values (?, ?, ?)",
                             (key, value, time.time()))

    def close(self) -> None:
        self._db.close()


def _scope(scope_id: ScopeId) -> str:
    if not isinstance(scope_id, str):
        raise TypeError(f"SQLiteTokenStore scopes are strings, got {type(scope_id).__name__}")
    return scope_id


__all__ = ["SQLiteTokenStore"]
