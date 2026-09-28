# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""One scope's vault in memory, for the tests: the store, the seed graph and a scripted detector.

``Scope`` is the in-memory counterpart of a deployment: an :class:`InMemoryTokenStore`, an
:class:`InMemorySeedSource`, and itself as the residual detector — ``detections`` is what the
"NER" hands the vault, which is the only part of the pipeline this package does not own.
All names, companies, vessels and identifiers in the tests are invented.
"""

from __future__ import annotations

import uuid

from deidkit import vault
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import AliasRow, InMemoryTokenStore, TokenRow

M = uuid.UUID("00000000-0000-4000-8000-000000000001")


def ent(etype: str, name: str, identifier: str = "", **kw) -> SeedEntity:
    return SeedEntity(type=etype, name=name, identifier=identifier, **kw)


def token_row(token: str, kind: str, value: str, *, status: str = "active", **kw) -> TokenRow:
    """A row as an older vault would hold it (counter tokens, no alias rows)."""
    return TokenRow(scope_id=M, token=token, kind=kind, real_value=value,
                    normalized=" ".join(value.split()).casefold(), status=status, **kw)


def alias_row(token: str, surface: str, normalized: str, source: str) -> AliasRow:
    return AliasRow(scope_id=M, token=token, surface=surface, normalized=normalized, source=source)


class Scope:
    def __init__(self, entities=(), parties=None, rows=(), aliases=()) -> None:
        self.store = InMemoryTokenStore()
        self.seeds = InMemorySeedSource()
        for e in entities:
            self.seeds.add_entity(M, e)
        self.seeds.set_parties(M, parties)
        self.store.token_rows(M).extend(rows)
        self.store.alias_rows(M).extend(aliases)
        #: What the residual detector returns: ``[(span, kind)]``.
        self.detections: list[tuple[str, str]] = []
        #: What it was asked: ``[(text, language)]``.
        self.asked: list[tuple[str, str]] = []

    # ── the PiiDetector protocol ─────────────────────────────────────────────
    def detect(self, text: str, language: str) -> list[tuple[str, str]]:
        self.asked.append((text, language))
        return list(self.detections)

    # ── the store's rows ─────────────────────────────────────────────────────
    @property
    def rows(self) -> list[TokenRow]:
        return self.store.token_rows(M)

    @rows.setter
    def rows(self, new) -> None:
        self.store.token_rows(M)[:] = list(new)

    @property
    def aliases(self) -> list[AliasRow]:
        return self.store.alias_rows(M)

    def token_of(self, value: str) -> str | None:
        return {r.real_value: r.token for r in self.rows}.get(value)

    # ── the vault, bound to this scope ───────────────────────────────────────
    async def tokenize(self, text: str, **kw):
        return await vault.tokenize(self.store, M, text, seeds=self.seeds, detector=self, **kw)

    async def detokenize(self, text: str, **kw) -> str:
        return await vault.detokenize(self.store, M, text, **kw)

    async def seed(self) -> None:
        await vault.seed_from_scope(self.store, M, self.seeds)

    async def reconcile(self, **kw) -> dict:
        return await vault.reconcile_scope(self.store, M, seeds=self.seeds, **kw)

    async def index(self, salt: bytes | None = None):
        return await vault._index_of(self.store, M, salt)
