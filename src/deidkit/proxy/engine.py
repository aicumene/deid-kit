# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""One scope's de-identification for the proxy: many texts in, one consistent request out.

A request to a model carries the whole conversation, re-sent on every turn. Three things follow.

* **Every piece is tokenised against the same vault.** Tokenising a piece can enrol a name the
  detector found in it; a piece tokenised before that enrolment would still carry the name. So
  the pieces are tokenised again until the vault stops learning (in practice one extra pass,
  and only on the request where something new appeared).
* **The history costs nothing and never changes.** A piece is tokenised once and cached by its
  SHA-256 together with the vault's size; the next turn finds it in the cache and sends exactly
  the same bytes, which keeps the provider's prompt cache and its reasoning checks intact. A new
  enrolment changes the size, and with it every key, which re-tokenises the history once.
* **One mapping per request.** ``vault.merge_mappings`` over all pieces, plus every token that
  appears in what is sent: de-tokenisation reverses exactly the tokens this request sent.

The vault is not safe for concurrent use; the engine serialises all work on a scope.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict

from deidkit import namefold as nf
from deidkit import vault
from deidkit.gateway import Redaction

_CACHE_SIZE = 50_000
_TOKEN = re.compile(r"(?<![0-9A-Za-z])[A-Z]+_\d+(?![0-9A-Za-z])")


class VaultDidNotSettle(RuntimeError):
    """Re-tokenising the request kept enrolling new values; nothing is sent."""


class ScopeEngine:
    def __init__(self, store, scope: str, *, seeds=None, detector=None,
                 language: str | None = None, glossary: bool = False) -> None:
        self.store = store
        self.scope = scope
        self.seeds = seeds
        self.detector = detector
        self.language = language
        self.glossary = glossary
        self._cache: OrderedDict[tuple[str, tuple[int, int]], Redaction] = OrderedDict()
        self._lock = asyncio.Lock()
        self._ready = False

    async def _prepare(self) -> None:
        if not self._ready:
            await vault.ensure_salt(self.store, self.scope)
            await vault.seed_from_scope(self.store, self.scope, self.seeds)
            await self.store.commit()
            self._ready = True

    async def _version(self) -> tuple[int, int]:
        return (len(await self.store.load_tokens(self.scope)),
                len(await self.store.load_aliases(self.scope)))

    async def _one(self, text: str) -> Redaction:
        if not text or not text.strip():
            return Redaction(text=text or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hit = self._cache.get((digest, await self._version()))
        if hit is not None:
            self._cache.move_to_end((digest, await self._version()))
            return hit
        red = await vault.tokenize(self.store, self.scope, text, detector=self.detector,
                                   language=self.language, seed=False, glossary=self.glossary)
        self._cache[(digest, await self._version())] = red
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return red

    async def tokenize_all(self, texts: list[str]) -> list[Redaction]:
        """Tokenise every piece of one request against one settled vault."""
        async with self._lock:
            await self._prepare()
            for _ in range(4):
                before = await self._version()
                reds = [await self._one(t) for t in texts]
                if await self._version() == before:
                    return reds
            raise VaultDidNotSettle(f"scope {self.scope!r}: the vault kept learning")

    async def mapping_for(self, reds: list[Redaction], sent: str) -> dict[str, str]:
        """The request's mapping: the merged per-piece mappings, plus every other token of the
        scope that appears in what is sent (restored model text carries tokens of its own)."""
        mapping = vault.merge_mappings(reds)
        seen = {m.group(0) for m in _TOKEN.finditer(nf.deconfuse_ascii(sent))} - set(mapping)
        if seen:
            async with self._lock:
                rows = {r.token: r for r in await self.store.load_tokens(self.scope)}
            for tok in seen:
                row = rows.get(tok)
                if row is None:
                    continue
                while row.canonical_token and row.canonical_token in rows and \
                        rows[row.canonical_token] is not row:
                    row = rows[row.canonical_token]
                mapping[tok] = row.real_value
        return mapping

    async def detokenize(self, text: str, mapping: dict[str, str]) -> str:
        return await vault.detokenize(self.store, self.scope, text, mapping=mapping)

    async def residual(self, text: str) -> list[str]:
        """Enrolled real values still present in ``text`` (read-only)."""
        async with self._lock:
            return await vault.residual_surfaces(self.store, self.scope, text)


__all__ = ["ScopeEngine", "VaultDidNotSettle"]
